# =============================================================
# File: src/verses_rag/api/app_state.py
# =============================================================
"""
Application state singleton for the FastAPI layer (SPEC §4.10).

Loads all heavyweight resources once at startup and holds them for the
lifetime of the process:
  - DenseEmbedder  (BGE-large-en-v1.5, ~1.3 GB)
  - SparseEmbedder (Qdrant/bm25)
  - Reranker       (ms-marco-MiniLM, ~80 MB)
  - QdrantStore    (connection to self-hosted Qdrant)
  - Compiled LangGraph

Thread-safe: initialization is protected by a lock so concurrent startup
requests don't double-load models.

Usage in FastAPI lifespan:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_app_state()
        yield
        reset_app_state()   # clean shutdown (tests)

Run self-check (requires Qdrant running):
    uv run python -m verses_rag.api.app_state
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("api.app_state")

_lock:      threading.Lock = threading.Lock()
_state:     "AppState | None"  = None


# --- state container ---------------------------------------------------------

@dataclass
class AppState:
    """All heavyweight resources held for the process lifetime."""

    settings:  Any   # verses_rag.config.settings.Settings
    dense:     Any   # DenseEmbedder
    sparse:    Any   # SparseEmbedder
    reranker:  Any   # Reranker
    store:     Any   # QdrantStore
    graph:     Any   # compiled LangGraph
    ready:     bool  = False
    init_s:    float = 0.0   # seconds taken to initialise


# --- init / teardown ---------------------------------------------------------

async def init_app_state(settings=None) -> AppState:
    """Initialise and return the global AppState.

    Safe to call multiple times — returns the existing instance after the
    first call. Runs synchronously inside the async lifespan so FastAPI
    holds the startup until models are loaded.
    """
    global _state
    with _lock:
        if _state is not None and _state.ready:
            return _state

        if settings is None:
            from verses_rag.config.settings import get_settings
            settings = get_settings()

        t0 = time.perf_counter()
        log.info("initialising app state…")

        from verses_rag.embeddings         import DenseEmbedder, SparseEmbedder
        from verses_rag.retrieval.reranker import Reranker
        from verses_rag.stores             import QdrantStore
        from verses_rag.graph.graph        import build_graph
        from verses_rag.eval.tracing       import configure_tracing

        configure_tracing(settings)

        log.info("loading dense embedder: %s", settings.embedding.dense_model)
        dense = DenseEmbedder(settings.embedding.dense_model)

        log.info("loading sparse embedder: %s", settings.embedding.sparse_model)
        sparse = SparseEmbedder(settings.embedding.sparse_model)

        log.info("loading reranker: %s", settings.rerank.model)
        reranker = Reranker.from_settings(settings.rerank)

        log.info("connecting to Qdrant: %s", settings.qdrant.url)
        store = QdrantStore(settings.qdrant.url, settings.qdrant.collection_name)

        log.info("building LangGraph…")
        graph = build_graph(store, dense, sparse, reranker, settings=settings)

        init_s = time.perf_counter() - t0
        log.info("app state ready in %.1fs", init_s)

        _state = AppState(
            settings = settings,
            dense    = dense,
            sparse   = sparse,
            reranker = reranker,
            store    = store,
            graph    = graph,
            ready    = True,
            init_s   = init_s,
        )
        return _state


def get_app_state() -> AppState:
    """Return the initialised AppState. Raises if init_app_state() not called."""
    if _state is None or not _state.ready:
        raise RuntimeError(
            "AppState not initialised. "
            "Call await init_app_state() during FastAPI lifespan startup."
        )
    return _state


def reset_app_state() -> None:
    """Clear the singleton (used in tests to force re-initialisation)."""
    global _state
    with _lock:
        _state = None
        log.info("app state reset")


# --- self-check --------------------------------------------------------------

def main():
    import asyncio

    async def _run():
        print("=== app_state self-check ===\n")
        print("Initialising (requires Qdrant running)…")
        try:
            state = await init_app_state()
            print(f"\n✓ ready={state.ready}  init_s={state.init_s:.1f}s")
            print(f"  dense dim : {state.dense.dim}")
            print(f"  store     : {state.store.stats()}")

            # Confirm get_app_state() returns the same instance.
            assert get_app_state() is state
            print("  singleton : ✓ same instance on second call")

            # Reset and confirm RuntimeError.
            reset_app_state()
            try:
                get_app_state()
            except RuntimeError as e:
                print(f"  reset     : ✓ RuntimeError after reset ({e})")

        except Exception as e:
            print(f"\n✗ init failed: {e}")
            print("  Is Qdrant running?  docker compose up -d")

    asyncio.run(_run())


if __name__ == "__main__":
    main()