# =============================================================
# File: src/verses_rag/graph/nodes/rerank.py
# =============================================================
"""
Rerank node — fourth node in the RAG graph (SPEC §4.5 stage 3, §4.6).

Takes the raw candidates from the retrieve node and runs them through the
cross-encoder reranker, returning a smaller ranked_docs list for grading.

Dependency injected via functools.partial at graph-build time (Option A):
    reranker — Reranker instance (retrieval/reranker.py)

Graph wiring example:
    from functools import partial
    rerank = partial(rerank_node, reranker=reranker)

Run self-check (no store or embedders needed):
    uv run python -m verses_rag.graph.nodes.rerank
"""

from __future__ import annotations

import logging
from typing import Any

from verses_rag.graph.state import GraphState

log = logging.getLogger("graph.rerank")


def rerank_node(
    state: GraphState,
    *,
    reranker: Any,
    settings=None,
) -> GraphState:
    """LangGraph node: cross-encoder rerank candidates → ranked_docs.

    Returns a partial state update containing only `ranked_docs`.
    Passes through gracefully if candidates is empty.
    """
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    query = state.get("query", "").strip()
    candidates = state.get("candidates", [])

    if not candidates:
        log.warning("rerank_node received no candidates")
        return {"ranked_docs": []}  # type: ignore[return-value]

    top_k = settings.rerank.top_k
    ranked = reranker.rerank(query, candidates, top_k=top_k)

    log.info(
        "reranked %d candidates → %d ranked_docs (top rerank_score %.3f)",
        len(candidates),
        len(ranked),
        ranked[0].rerank_score if ranked else 0.0,
    )
    return {"ranked_docs": ranked}  # type: ignore[return-value]


# --- self-check --------------------------------------------------------------

def main():
    import hashlib
    from functools import partial

    from verses_rag.config.settings import get_settings
    from verses_rag.graph.nodes.retrieve import ScoredHit
    from verses_rag.retrieval.reranker import Reranker

    def _sha(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    # Fabricate candidates directly — no store or embedders needed.
    candidates = [
        ScoredHit(_sha("Genesis 1:1"), {
            "content": "In the beginning God created the heaven and the earth.",
            "book": "Genesis", "chapter": 1, "verse_start": 1, "verse_end": 1,
        }, score=0.0320),
        ScoredHit(_sha("John 3:16"), {
            "content": "For God so loved the world, that he gave his only begotten Son.",
            "book": "John", "chapter": 3, "verse_start": 16, "verse_end": 16,
        }, score=0.0328),
        ScoredHit(_sha("Psalms 23:1"), {
            "content": "The LORD is my shepherd; I shall not want.",
            "book": "Psalms", "chapter": 23, "verse_start": 1, "verse_end": 1,
        }, score=0.0315),
        ScoredHit(_sha("Romans 8:28"), {
            "content": "And we know that all things work together for good.",
            "book": "Romans", "chapter": 8, "verse_start": 28, "verse_end": 28,
        }, score=0.0310),
        ScoredHit(_sha("Proverbs 3:5"), {
            "content": "Trust in the LORD with all thine heart.",
            "book": "Proverbs", "chapter": 3, "verse_start": 5, "verse_end": 5,
        }, score=0.0164),
    ]

    s = get_settings()
    reranker = Reranker.from_settings(s.rerank)
    rerank = partial(rerank_node, reranker=reranker, settings=s)

    queries = [
        "God's love and care for his people",
        "The LORD as shepherd",
        "Trust and wisdom",
    ]

    print("=== rerank node self-check ===\n")
    for query in queries:
        state: GraphState = {  # type: ignore[misc]
            "query": query,
            "candidates": candidates,
        }
        result = rerank(state)
        ranked = result.get("ranked_docs", [])
        print(f"Query: {query!r}")
        for r in ranked:  # type: ignore[union-attr]
            book = r.payload.get("book", "?")
            ch   = r.payload.get("chapter", "?")
            vs   = r.payload.get("verse_start", "?")
            ref  = f"{book} {ch}:{vs}"
            print(f"  [{r.rerank_score:7.3f}] (retr {r.retrieval_score:.4f}) "
                  f"{ref:16} {r.content[:50]}")
        print()


if __name__ == "__main__":
    main()