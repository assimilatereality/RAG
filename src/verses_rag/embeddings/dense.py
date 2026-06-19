# =============================================================
# File: src/verses_rag/embeddings/dense.py
# =============================================================
"""
Dense embedder using SentenceTransformers (SPEC §4.3).

Default model: BAAI/bge-large-en-v1.5 (1024-dim, strong English retrieval).
Alternate:     intfloat/e5-large-v2    (1024-dim, same family).

IMPORTANT: the model + dimension must match the Qdrant collection definition.
Changing models after indexing requires a full re-index (SPEC §4.3 note).

BGE models expect queries prefixed with "Represent this sentence for searching
relevant passages: " — this is handled automatically by encode_query().
E5 models expect "query: " / "passage: " prefixes — same pattern.
"""

from __future__ import annotations

import logging
from functools import cached_property

from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "

_QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-large-en-v1.5": _BGE_QUERY_PREFIX,
    "BAAI/bge-base-en-v1.5": _BGE_QUERY_PREFIX,
    "intfloat/e5-large-v2": _E5_QUERY_PREFIX,
    "intfloat/e5-base-v2": _E5_QUERY_PREFIX,
}

_PASSAGE_PREFIXES: dict[str, str] = {
    "intfloat/e5-large-v2": _E5_PASSAGE_PREFIX,
    "intfloat/e5-base-v2": _E5_PASSAGE_PREFIX,
}


class DenseEmbedder:
    """Wraps a SentenceTransformer model with query/passage prefix handling."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        self.model_name = model_name
        self._query_prefix = _QUERY_PREFIXES.get(model_name, "")
        self._passage_prefix = _PASSAGE_PREFIXES.get(model_name, "")
        log.info("loading dense model: %s", model_name)

    @cached_property
    def _model(self) -> SentenceTransformer:
        """Lazy-load: model is not downloaded until first encode call."""
        return SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        d = self._model.get_embedding_dimension()
        if d is None:
            raise RuntimeError(
                f"Model '{self.model_name}' did not report an embedding dimension."
            )
        return d

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        """Encode document passages for indexing."""
        prefixed = (
            [f"{self._passage_prefix}{t}" for t in texts]
            if self._passage_prefix
            else texts
        )
        vecs = self._model.encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string for retrieval."""
        prefixed = f"{self._query_prefix}{query}" if self._query_prefix else query
        vec = self._model.encode(prefixed, normalize_embeddings=True)
        return vec.tolist()
