# =============================================================
# File: src/verses_rag/stores/qdrant_store.py
# =============================================================
"""
QdrantStore — the one real VectorStoreClient implementation (SPEC §4.4, D1).

Uses named vectors (dense + sparse) with RRF fusion via query_points().
The deprecated search() API was removed in qdrant-client 1.18 / server 1.18,
so query_points() is used throughout.

Chunk IDs are SHA256 hex strings (64 chars). Qdrant requires UUID or uint64
point IDs, so we derive a deterministic UUID from the first 32 hex chars.
The original chunk_id is always stored in the payload for round-tripping.
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointIdsList,
    PointStruct,
    Prefetch,
    Range,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import (
    SparseVector as QSparseVector,
)

from verses_rag.stores.base import ChunkWithVectors, Hit, SparseVector

log = logging.getLogger(__name__)

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"
BATCH_SIZE = 100  # upsert batch size; tune for corpus size

_DISTANCE: dict[str, Distance] = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


def _to_point_id(chunk_id: str) -> str:
    """Deterministic UUID derived from a SHA256 chunk_id (first 32 hex chars)."""
    return str(uuid.UUID(hex=chunk_id[:32]))


def _build_filter(filter_dict: dict) -> Filter | None:
    """Translate a neutral {field: value} dict to a Qdrant Filter.

    Most keys are exact-match (MatchValue). The special `verse_range` key
    holds a (vs, ve) tuple and is translated to an interval-OVERLAP condition:
    a chunk matches when its stored window [verse_start, verse_end] overlaps
    the requested [vs, ve] range — i.e.

        chunk.verse_start <= ve  AND  chunk.verse_end >= vs

    This is required because verses are stored as overlapping windows (e.g.
    14-18), so no chunk equals an arbitrary requested verse like 16; overlap
    matching surfaces the window that *contains* it.
    """
    if not filter_dict:
        return None

    conditions: list[Condition] = []

    for k, v in filter_dict.items():
        if k == "verse_range":
            vs, ve = v
            # chunk.verse_start <= ve
            conditions.append(
                FieldCondition(key="verse_start", range=Range(lte=float(ve)))
            )
            # chunk.verse_end >= vs
            conditions.append(
                FieldCondition(key="verse_end", range=Range(gte=float(vs)))
            )
        else:
            conditions.append(
                FieldCondition(key=k, match=MatchValue(value=v))
            )

    return Filter(must=conditions) if conditions else None


class QdrantStore:
    """Qdrant-backed VectorStoreClient with hybrid dense+sparse retrieval."""

    def __init__(self, url: str, collection_name: str) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection_name

    # ------------------------------------------------------------------
    def ensure_collection(
        self,
        dense_dim: int,
        metric: str = "cosine",
        sparse: bool = True,
    ) -> None:
        """Create the collection if it doesn't already exist (idempotent)."""
        if self._client.collection_exists(self._collection):
            log.info(
                "collection '%s' already exists — skipping create", self._collection
            )
            return
        distance = _DISTANCE.get(metric, Distance.COSINE)
        sparse_cfg = (
            {SPARSE_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False))}
            if sparse
            else None
        )
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_NAME: VectorParams(size=dense_dim, distance=distance)
            },
            sparse_vectors_config=sparse_cfg,
        )
        log.info(
            "created collection '%s' (dim=%d, metric=%s, sparse=%s)",
            self._collection,
            dense_dim,
            metric,
            sparse,
        )

    # ------------------------------------------------------------------
    def upsert(self, chunks: list[ChunkWithVectors]) -> None:
        """Upsert chunks in batches of BATCH_SIZE."""
        points = [
            PointStruct(
                id=_to_point_id(c.chunk_id),
                vector={
                    DENSE_NAME: c.dense,
                    SPARSE_NAME: QSparseVector(
                        indices=c.sparse.indices,
                        values=c.sparse.values,
                    ),
                },
                payload={**c.payload, "chunk_id": c.chunk_id},
            )
            for c in chunks
        ]
        for i in range(0, len(points), BATCH_SIZE):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + BATCH_SIZE],
            )
        log.info("upserted %d points to '%s'", len(points), self._collection)

    # ------------------------------------------------------------------
    def hybrid_query(
        self,
        dense_vec: list[float],
        sparse_vec: SparseVector,
        filters: dict | None = None,
        top_k: int = 5,
        dense_k: int = 10,
        sparse_k: int = 10,
    ) -> list[Hit]:
        """Hybrid retrieval: sparse + dense prefetch fused with RRF."""
        qdrant_filter = _build_filter(filters or {})
        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using=DENSE_NAME,
                    limit=dense_k,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=QSparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                    using=SPARSE_NAME,
                    limit=sparse_k,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [
            Hit(
                chunk_id=p.payload.get("chunk_id", str(p.id)) if p.payload else str(p.id),
                score=p.score,
                payload=p.payload or {},
            )
            for p in results.points
]

    # ------------------------------------------------------------------
    def delete(
        self,
        ids: list[str] | None = None,
        filter_dict: dict | None = None,
    ) -> None:
        if ids:
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=[_to_point_id(i) for i in ids]),
            )
        if filter_dict:
            f = _build_filter(filter_dict)
            if f:
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=f,
                )

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        info = self._client.get_collection(self._collection)
        return {
            "collection": self._collection,
            "points_count": info.points_count,
            "status": str(info.status),
        }
