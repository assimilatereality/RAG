# =============================================================
# File: src/verses_rag/stores/memory_store.py
# =============================================================
"""
InMemoryStore — fast test double for VectorStoreClient (SPEC D1).

Stores vectors in plain dicts; no external dependencies. Implements the same
RRF fusion logic as QdrantStore so tests exercise real retrieval behavior.
Not for production use — no persistence, no concurrent access.
"""

from __future__ import annotations

import logging
import math

from verses_rag.stores.base import ChunkWithVectors, Hit, SparseVector

log = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant; matches Qdrant's default


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _sparse_dot(a: SparseVector, b: SparseVector) -> float:
    b_map = dict(zip(b.indices, b.values, strict=False))
    return sum(v * b_map.get(i, 0.0) for i, v in zip(a.indices, a.values, strict=False))


# =============================================================
# EDIT: replace the _matches function in memory_store.py
# =============================================================

def _matches(payload: dict, filters: dict) -> bool:
    """True if the payload satisfies all filter conditions.

    Most keys are exact-equality. The special `verse_range` key holds a
    (vs, ve) tuple and tests interval OVERLAP against the chunk's window —
    matching the Qdrant adapter's behavior so InMemory tests stay faithful:

        chunk.verse_start <= ve  AND  chunk.verse_end >= vs
    """
    for k, v in filters.items():
        if k == "verse_range":
            vs, ve = v
            c_start = payload.get("verse_start")
            c_end   = payload.get("verse_end")
            if c_start is None or c_end is None:
                return False
            if not (c_start <= ve and c_end >= vs):
                return False
        else:
            if payload.get(k) != v:
                return False
    return True


class InMemoryStore:
    """Lightweight VectorStoreClient test double."""

    def __init__(self) -> None:
        self._points: dict[str, ChunkWithVectors] = {}

    def ensure_collection(
        self, dense_dim: int, metric: str = "cosine", sparse: bool = True
    ) -> None:
        log.debug("InMemoryStore.ensure_collection — no-op")

    def upsert(self, chunks: list[ChunkWithVectors]) -> None:
        for c in chunks:
            self._points[c.chunk_id] = c
        log.debug(
            "InMemoryStore: upserted %d points (%d total)",
            len(chunks),
            len(self._points),
        )

    def hybrid_query(
        self,
        dense_vec: list[float],
        sparse_vec: SparseVector,
        filters: dict | None = None,
        top_k: int = 5,
        dense_k: int = 10,
        sparse_k: int = 10,
    ) -> list[Hit]:
        candidates = [
            c for c in self._points.values() if _matches(c.payload, filters or {})
        ]
        dense_ranked = sorted(
            candidates, key=lambda c: _cosine(dense_vec, c.dense), reverse=True
        )[:dense_k]
        sparse_ranked = sorted(
            candidates, key=lambda c: _sparse_dot(sparse_vec, c.sparse), reverse=True
        )[:sparse_k]

        scores: dict[str, float] = {}
        for rank, c in enumerate(dense_ranked):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1 / (_RRF_K + rank + 1)
        for rank, c in enumerate(sparse_ranked):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1 / (_RRF_K + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            Hit(chunk_id=cid, score=score, payload=self._points[cid].payload)
            for cid, score in ranked
        ]

    def delete(
        self,
        ids: list[str] | None = None,
        filter_dict: dict | None = None,
    ) -> None:
        if ids:
            for cid in ids:
                self._points.pop(cid, None)
        if filter_dict:
            to_remove = [
                cid
                for cid, c in self._points.items()
                if _matches(c.payload, filter_dict)
            ]
            for cid in to_remove:
                del self._points[cid]

    def stats(self) -> dict:
        return {"collection": "in-memory", "points_count": len(self._points)}
