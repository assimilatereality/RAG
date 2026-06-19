#=============================================================
# File: src/verses_rag/retrieval/reranker.py
# =============================================================
"""
Cross-encoder reranker (SPEC §4.5 stage 3, D6).

LLM-agnostic by design: operates purely on (query, candidate_text) pairs after
hybrid retrieval, before grading/generation. Works identically whether hits came
from QdrantStore or InMemoryStore — it only needs each hit's payload text.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 by default (fast, ~80MB, well-proven
for passage reranking). BAAI/bge-reranker-base is a config-swap upgrade.

Run a self-check:
    uv run python -m verses_rag.retrieval.reranker
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("reranker")

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RankedHit:
    """A retrieval hit after reranking.

    `payload` is carried through untouched so downstream nodes (grade, generate)
    keep full access to metadata for citations. `retrieval_score` preserves the
    original RRF/fusion score; `rerank_score` is the cross-encoder logit.
    """

    chunk_id: str
    payload: dict
    retrieval_score: float
    rerank_score: float

    @property
    def content(self) -> str:
        return self.payload.get("content", "")


class Reranker:
    """Cross-encoder reranker. Lazy model load — import costs nothing."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device  # None -> sentence-transformers picks (mps on Apple Silicon)
        self._model = None

    @classmethod
    def from_settings(cls, cfg) -> "Reranker":
        """Build from a RerankSettings block (config/settings.py)."""
        return cls(model_name=cfg.model, device=cfg.device)

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading cross-encoder %s …", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        hits: Sequence,            # store Hit objects: .chunk_id/.payload/.score (duck-typed)
        top_k: Optional[int] = None,
    ) -> list[RankedHit]:
        """Score (query, hit content) pairs and return hits sorted by rerank score.

        Mirrors hybrid_retriever_copy.py's stage 3, generalized: candidates arrive
        as store hits, not raw strings, and metadata survives the trip.
        """
        if not hits:
            return []

        # Single candidate: nothing to rank against; pass through with a neutral score.
        if len(hits) == 1:
            h = hits[0]
            return [RankedHit(h.chunk_id, h.payload, float(h.score), 0.0)]

        model = self._ensure_model()
        pairs = [(query, h.payload.get("content", "")) for h in hits]
        scores = model.predict(pairs)

        ranked = [
            RankedHit(
                chunk_id=h.chunk_id,
                payload=h.payload,
                retrieval_score=float(h.score),
                rerank_score=float(s),
            )
            for h, s in zip(hits, scores, strict=True)
        ]
        ranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return ranked[:top_k] if top_k else ranked


def main():
    """Self-check without Qdrant: fabricate hits and rerank."""
    from dataclasses import dataclass as _dc

    @_dc
    class FakeHit:
        chunk_id: str
        payload: dict
        score: float

    hits = [
        FakeHit("a", {"content": "The LORD is my shepherd; I shall not want.",
                      "book": "Psalms"}, 0.71),
        FakeHit("b", {"content": "And God said, Let there be light: and there was light.",
                      "book": "Genesis"}, 0.69),
        FakeHit("c", {"content": "For God so loved the world, that he gave his only begotten Son.",
                      "book": "John"}, 0.65),
    ]
    rr = Reranker()
    for r in rr.rerank("God's love for humanity", hits, top_k=3):
        print(f"[{r.rerank_score:7.3f}] (retr {r.retrieval_score:.2f}) "
              f"{r.payload['book']:8} {r.content[:55]}")


if __name__ == "__main__":
    main()
