# =============================================================
# File: src/verses_rag/embeddings/sparse.py
# =============================================================
"""
Sparse embedder using FastEmbed (SPEC §4.3, D4).

Default model: "Qdrant/bm25" — lexical BM25 with IDF weighting via FastEmbed.
This produces Qdrant-native sparse vectors directly, identical in behavior to
the rank_bm25 used in the reference files but scalable and prod-compatible.

SPLADE ("prithivida/Splade_PP_en_v1") is a drop-in upgrade for learned sparse
representations — set via config to A/B later (SPEC §4.3).

Both models return (indices, values) pairs suitable for Qdrant SparseVector.
"""

from __future__ import annotations

import logging
from functools import cached_property

from fastembed import SparseTextEmbedding

from verses_rag.stores.base import SparseVector

log = logging.getLogger(__name__)


class SparseEmbedder:
    """Wraps a FastEmbed sparse model, returning SparseVector instances."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self.model_name = model_name
        log.info("loading sparse model: %s", model_name)

    @cached_property
    def _model(self) -> SparseTextEmbedding:
        """Lazy-load: model files are not downloaded until first encode call."""
        return SparseTextEmbedding(model_name=self.model_name)

    def _to_sparse(self, embedding) -> SparseVector:
        """Convert a FastEmbed SparseEmbedding to our neutral SparseVector."""
        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

    def encode_passages(self, texts: list[str]) -> list[SparseVector]:
        """Encode document passages for indexing."""
        return [self._to_sparse(e) for e in self._model.embed(texts)]

    def encode_query(self, query: str) -> SparseVector:
        """Encode a single query string for retrieval."""
        results = list(self._model.query_embed(query))
        return self._to_sparse(results[0])
