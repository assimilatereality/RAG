# =============================================================
# File: src/verses_rag/indexing/indexer.py
# =============================================================
"""
Full ingest → embed → upsert pipeline.

Chains the Phase 1 ingestion pipeline with Phase 2 embedding and Qdrant
upsert. This is the entry point for building or refreshing the index.

Run:
    uv run python -m verses_rag.indexing.indexer            # articles only
    uv run python -m verses_rag.indexing.indexer --bible    # articles + KJV
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("indexing.indexer")


# --- payload builder ---------------------------------------------------------

def _chunk_to_payload(chunk: Any) -> dict:
    """Convert a BaseChunk / Chunk / ArticleChunk to a Qdrant payload dict.

    Uses dataclasses.asdict and strips None values so the payload stays lean.
    """
    d = dataclasses.asdict(chunk)
    return {k: v for k, v in d.items() if v is not None and k != "content"}


# --- core embed + upsert -----------------------------------------------------

def embed_and_upsert(
    chunks:     list[Any],
    store:      Any,
    dense:      Any,
    sparse:     Any,
    batch_size: int = 64,
) -> int:
    """Embed chunks in batches and upsert to the vector store.

    Args:
        chunks:     list of BaseChunk / Chunk / ArticleChunk objects.
        store:      QdrantStore (ensure_collection already called).
        dense:      DenseEmbedder.
        sparse:     SparseEmbedder.
        batch_size: number of chunks per embedding batch.

    Returns:
        Total number of chunks upserted.
    """
    from verses_rag.stores import ChunkWithVectors

    total   = len(chunks)
    indexed = 0

    for i in range(0, total, batch_size):
        batch  = chunks[i : i + batch_size]
        texts  = [c.content for c in batch]
        d_vecs = dense.encode_passages(texts)
        s_vecs = sparse.encode_passages(texts)

        cwv = [
            ChunkWithVectors(
                chunk_id = c.chunk_id,
                payload  = {**_chunk_to_payload(c), "content": c.content},
                dense    = dv,
                sparse   = sv,
            )
            for c, dv, sv in zip(batch, d_vecs, s_vecs, strict=True)
        ]
        store.upsert(cwv)
        indexed += len(batch)
        log.info("indexed %d / %d chunks (%.0f%%)", indexed, total, indexed / total * 100)

    return indexed


# --- full pipeline -----------------------------------------------------------

def build_index(
    settings         = None,
    include_bible:     bool = True,
    treat_all_as_articles: bool = False,
    batch_size:        int  = 64,
) -> dict:
    """Run the full ingest → embed → upsert pipeline.

    Creates the Qdrant collection if it doesn't exist. Safe to re-run —
    upsert is idempotent (content-addressed chunk IDs).

    Returns a summary dict with chunk counts and timing.
    """
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    from verses_rag.embeddings           import DenseEmbedder, SparseEmbedder
    from verses_rag.stores               import QdrantStore
    from verses_rag.ingestion.ingest     import ingest

    t0 = time.perf_counter()

    # --- embedders ---
    log.info("loading embedders…")
    dense  = DenseEmbedder(settings.embedding.dense_model)
    sparse = SparseEmbedder(settings.embedding.sparse_model)

    # --- ingest (produces chunks, no I/O to Qdrant) ---
    log.info("ingesting corpus…")
    chunks, report = ingest(
        settings              = settings,
        include_bible         = include_bible,
        treat_all_as_articles = treat_all_as_articles,
    )
    log.info("ingestion complete: %s", report.summary())

    if not chunks:
        log.warning("no chunks produced — nothing to index")
        return {"chunks": 0, "elapsed_s": 0.0}

    # --- Qdrant collection ---
    log.info("connecting to Qdrant: %s", settings.qdrant.url)
    store = QdrantStore(settings.qdrant.url, settings.qdrant.collection_name)
    store.ensure_collection(dense_dim=dense.dim)

    # --- embed + upsert ---
    log.info("embedding and upserting %d chunks (batch_size=%d)…", len(chunks), batch_size)
    indexed = embed_and_upsert(chunks, store, dense, sparse, batch_size=batch_size)

    elapsed = time.perf_counter() - t0
    stats   = store.stats()
    log.info("index build complete in %.1fs — %s", elapsed, stats)

    return {
        "files_seen":         report.files_seen,
        "bible_chunks":       report.bible_json_chunks,
        "article_chunks":     report.article_chunks,
        "duplicates_removed": report.duplicates_removed,
        "total_indexed":      indexed,
        "qdrant_points":      stats.get("points_count", indexed),
        "elapsed_s":          round(elapsed, 1),
    }


# --- CLI entry point ---------------------------------------------------------

def main():
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    include_bible = "--bible" in sys.argv
    print(f"\nBuilding index  (bible={'yes' if include_bible else 'no'})\n")

    summary = build_index(include_bible=include_bible)

    print("\n" + "="*50)
    print("Index build complete")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()