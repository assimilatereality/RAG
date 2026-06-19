# =============================================================
# File: src/verses_rag/stores/base.py
# =============================================================
"""
VectorStoreClient protocol + shared data types (SPEC §4.4, D1).

App logic imports from here; Qdrant SDK types never leak into calling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SparseVector:
    """Neutral sparse vector — indices into a vocabulary + corresponding weights."""

    indices: list[int]
    values: list[float]


@dataclass
class ChunkWithVectors:
    """A chunk ready to upsert: its ID, payload, and both vector representations."""

    chunk_id: str
    payload: dict  # all BaseChunk fields as a flat dict
    dense: list[float]
    sparse: SparseVector


@dataclass
class Hit:
    """One retrieval result returned by hybrid_query."""

    chunk_id: str
    score: float
    payload: dict


@runtime_checkable
class VectorStoreClient(Protocol):
    """Interface every store implementation must satisfy (SPEC D1)."""

    def ensure_collection(
        self,
        dense_dim: int,
        metric: str = "cosine",
        sparse: bool = True,
    ) -> None: ...

    def upsert(self, chunks: list[ChunkWithVectors]) -> None: ...

    def hybrid_query(
        self,
        dense_vec: list[float],
        sparse_vec: SparseVector,
        filters: dict | None = None,
        top_k: int = 5,
        dense_k: int = 10,
        sparse_k: int = 10,
    ) -> list[Hit]: ...

    def delete(
        self,
        ids: list[str] | None = None,
        filter_dict: dict | None = None,
    ) -> None: ...

    def stats(self) -> dict: ...
