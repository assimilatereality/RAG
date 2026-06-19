# =============================================================
# File: src/verses_rag/graph/nodes/retrieve.py
# =============================================================
"""
Retrieve node — third node in the RAG graph (SPEC §4.5, §4.6).

Performs hybrid retrieval (dense + sparse, RRF-fused) against the vector
store, applies the metadata filters from analyze_filters, and soft-penalises
archived hits before returning candidates to the rerank node.

Dependencies injected via functools.partial at graph-build time (Option A):
    store  — QdrantStore or InMemoryStore (duck-typed: .hybrid_query())
    dense  — DenseEmbedder  (.encode_query())
    sparse — SparseEmbedder (.encode_query())

Graph wiring example:
    from functools import partial
    retrieve = partial(retrieve_node, store=store, dense=dense, sparse=sparse)

Run self-check (uses InMemoryStore — no Qdrant needed):
    uv run python -m verses_rag.graph.nodes.retrieve
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from verses_rag.graph.state import GraphState

log = logging.getLogger("graph.retrieve")

ARCHIVE_PENALTY = 0.5   # multiplier applied to archived hit scores (§5.3, metadata_filtering.py)


# --- score wrapper -----------------------------------------------------------

@dataclass
class ScoredHit:
    """Thin wrapper so we can adjust scores without mutating store objects."""
    chunk_id: str
    payload: dict[str, Any]
    score: float


def _wrap(hit: Any, score_override: float | None = None) -> ScoredHit:
    return ScoredHit(
        chunk_id=hit.chunk_id,
        payload=dict(hit.payload),
        score=score_override if score_override is not None else float(hit.score),
    )


# --- node entry point --------------------------------------------------------

def retrieve_node(
    state: GraphState,
    *,
    store: Any,
    dense: Any,
    sparse: Any,
    settings=None,
) -> GraphState:
    """LangGraph node: hybrid retrieval → soft-penalise archived → return candidates.

    Returns a partial state update containing only `candidates`.
    """
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    # Use reformulated query if transform_query ran, else original.
    query = (state.get("retrieval_query") or state.get("query", "")).strip()
    if not query:
        log.error("retrieve_node received empty query")
        return {"candidates": [], "error": "empty query"}  # type: ignore[return-value]

    # Separate pipeline-level flags from store-level filters.
    raw_filters = dict(state.get("filters", {}))
    downweight_archived = raw_filters.pop("downweight_archived", False)

    cfg = settings.retrieval
    top_k = cfg.bm25_k + cfg.dense_k   # fetch broad candidate set; rerank narrows it

    # Encode query for both retrievers.
    q_dense = dense.encode_query(query)
    q_sparse = sparse.encode_query(query)

    # Hybrid query — store handles RRF fusion and dedup internally.
    store_filters = raw_filters if raw_filters else None
    hits = store.hybrid_query(q_dense, q_sparse, top_k=top_k, filters=store_filters)

    # Wrap and apply archive penalty.
    candidates: list[ScoredHit] = []
    for h in hits:
        status = (h.payload or {}).get("status", "active")
        penalty = ARCHIVE_PENALTY if (downweight_archived and status == "archived") else 1.0
        candidates.append(_wrap(h, score_override=float(h.score) * penalty))

    # Re-sort after potential score adjustments.
    candidates.sort(key=lambda c: c.score, reverse=True)

    log.info(
        "retrieved %d candidates (top score %.4f) | filters=%s",
        len(candidates),
        candidates[0].score if candidates else 0.0,
        raw_filters or "none",
    )
    return {"candidates": candidates}  # type: ignore[return-value]


# --- self-check (InMemoryStore — no Qdrant needed) ---------------------------

def main():
    import hashlib
    from functools import partial

    from verses_rag.config.settings import get_settings
    from verses_rag.embeddings import DenseEmbedder, SparseEmbedder
    from verses_rag.stores import ChunkWithVectors, InMemoryStore

    def _sha(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    VERSES = [
        (_sha("Genesis 1:1"),  "In the beginning God created the heaven and the earth.",
         "Genesis", 1, 1, 1, "OT", "active"),
        (_sha("John 3:16"),    "For God so loved the world, that he gave his only begotten Son.",
         "John", 3, 16, 16, "NT", "active"),
        (_sha("Psalms 23:1"),  "The LORD is my shepherd; I shall not want.",
         "Psalms", 23, 1, 1, "OT", "active"),
        (_sha("Romans 8:28"),  "And we know that all things work together for good.",
         "Romans", 8, 28, 28, "NT", "active"),
        (_sha("Proverbs 3:5"), "Trust in the LORD with all thine heart.",
         "Proverbs", 3, 5, 5, "OT", "archived"),   # archived — should be penalised
    ]

    s = get_settings()
    print("=== retrieve node self-check ===\n")
    print("Loading embedders (first run downloads models)…")
    dense = DenseEmbedder(s.embedding.dense_model)
    sparse = SparseEmbedder(s.embedding.sparse_model)

    print("Embedding passages…")
    texts = [v[1] for v in VERSES]
    d_vecs = dense.encode_passages(texts)
    s_vecs = sparse.encode_passages(texts)

    store = InMemoryStore()
    store.ensure_collection(dense_dim=dense.dim)
    store.upsert([
        ChunkWithVectors(
            chunk_id=cid,
            payload={
                "content": text, "book": book, "chapter": chapter,
                "verse_start": vs, "verse_end": ve,
                "testament": testament, "status": status, "source_type": "bible",
            },
            dense=dv,
            sparse=sv,
        )
        for (cid, text, book, chapter, vs, ve, testament, status), dv, sv
        in zip(VERSES, d_vecs, s_vecs, strict=True)
    ])

    cases = [
        ("bible", {"source_type": "bible", "downweight_archived": True},
         "God's love and care for his people"),
        ("bible", {"source_type": "bible", "book": "Psalms", "downweight_archived": True},
         "The LORD as shepherd"),
        ("bible", {"source_type": "bible", "downweight_archived": True},
         "Trust and wisdom"),   # Proverbs 3:5 is archived — should rank lower
    ]

    retrieve = partial(retrieve_node, store=store, dense=dense, sparse=sparse, settings=s)

    for route, filters, query in cases:
        print(f"\nQuery: {query!r}  filters={filters}")
        state: GraphState = {"query": query, "source_route": route, "filters": filters}  # type: ignore[misc]
        result = retrieve(state)
        for c in result.get("candidates", []):  # type: ignore[union-attr]
            status = c.payload.get("status", "?")
            book = c.payload.get("book", "?")
            ch = c.payload.get("chapter", "?")
            vs = c.payload.get("verse_start", "?")
            ve = c.payload.get("verse_end", "?")
            ref = f"{book} {ch}:{vs}" + (f"-{ve}" if ve != vs else "")
            marker = "⚠ archived" if status == "archived" else ""
            print(f"  [{c.score:.4f}] {ref:18} {c.payload['content'][:45]}  {marker}")


if __name__ == "__main__":
    main()