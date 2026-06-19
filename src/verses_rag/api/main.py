# =============================================================
# File: src/verses_rag/api/main.py
# =============================================================
"""
FastAPI application for the RAG pipeline (SPEC §4.10).

Endpoints:
  GET  /health         — provider + graph readiness check
  POST /query          — synchronous: wait for full answer
  POST /query/stream   — streaming: SSE node-progress + answer chunks
  POST /ingest         — trigger ingestion pipeline

Run locally:
    uv run uvicorn verses_rag.api.main:app --reload --port 8000

Or via the module entry point:
    uv run python -m verses_rag.api.main
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from verses_rag.api.app_state import init_app_state, get_app_state
from verses_rag.api.schemas import (
    Citation,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ProviderHealthSchema,
    ProviderStatus,
    QueryRequest,
    QueryResponse,
    StreamEvent,
    StreamEventType,
)
from verses_rag.graph.nodes.generate import ABSTAIN_PHRASE
from verses_rag.graph.graph import run_query
from verses_rag.llm.router import check_judge_health, ProviderStatus as RouterStatus

log = logging.getLogger("api.main")


# --- lifespan ----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and the graph at startup; clean up on shutdown."""
    await init_app_state()
    yield
    log.info("shutting down")


# --- app ---------------------------------------------------------------------

app = FastAPI(
    title       = "Verses RAG",
    description = "Retrieval-augmented generation over KJV scripture and articles.",
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # personal app — open for Streamlit on localhost
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# --- helpers -----------------------------------------------------------------

def _to_citation_schema(c: dict) -> Citation:
    return Citation(
        ref             = c.get("ref", ""),
        content_snippet = c.get("content_snippet", ""),
        rerank_score    = float(c.get("rerank_score", 0.0)),
        source_type     = c.get("source_type", "unknown"),
    )


def _router_status_to_schema(ps) -> ProviderStatus:
    mapping = {
        RouterStatus.OK:       ProviderStatus.OK,
        RouterStatus.DEGRADED: ProviderStatus.DEGRADED,
        RouterStatus.DOWN:     ProviderStatus.DOWN,
    }
    return mapping.get(ps, ProviderStatus.DOWN)


def _sse(event: StreamEvent) -> str:
    """Format one StreamEvent as an SSE data line."""
    return f"data: {event.model_dump_json()}\n\n"


# --- endpoints ---------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Check provider health and graph readiness."""
    try:
        state   = get_app_state()
        ready   = state.ready
        s       = state.settings
    except RuntimeError:
        return HealthResponse(status="down", graph_ready=False)

    judge_health = check_judge_health(s)

    primary = ProviderHealthSchema(
        provider   = judge_health.primary.provider,
        status     = _router_status_to_schema(judge_health.primary.status),
        latency_ms = judge_health.primary.latency_ms,
        error      = judge_health.primary.error,
    )
    backup = ProviderHealthSchema(
        provider   = judge_health.backup.provider,
        status     = _router_status_to_schema(judge_health.backup.status),
        latency_ms = judge_health.backup.latency_ms,
        error      = judge_health.backup.error,
    )

    overall = (
        "ok"       if judge_health.any_available and ready else
        "degraded" if judge_health.any_available           else
        "down"
    )

    return HealthResponse(
        status      = overall,
        graph_ready = ready,
        primary     = primary,
        backup      = backup,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run a query synchronously and return the full answer."""
    state = get_app_state()
    t0    = time.perf_counter()

    try:
        result = run_query(
            req.query,
            state.graph,
            settings          = state.settings,
            use_decomposition = req.use_decomposition,
        )
    except Exception as e:
        log.exception("query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.perf_counter() - t0
    answer  = result.get("answer", ABSTAIN_PHRASE)
    verdict = result.get("verdict", "abstain")

    return QueryResponse(
        query       = req.query,
        answer      = answer,
        verdict     = verdict,
        citations   = [_to_citation_schema(c) for c in result.get("citations", [])],
        sub_queries = result.get("sub_queries", [req.query]),
        latency_s   = round(latency, 2),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Stream node-progress events then the final answer via SSE."""
    state = get_app_state()

    async def _generate() -> AsyncGenerator[str, None]:
        from verses_rag.agents.query_decomposer import decompose, merge_results
        from verses_rag.eval.tracing import make_run_config

        t0 = time.perf_counter()
        try:
            sub_queries = (
                decompose(req.query)
                if req.use_decomposition
                else [req.query]
            )

            status_map = {
                "route":           "Routing…",
                "analyze_filters": "Analysing filters…",
                "retrieve":        "Retrieving…",
                "rerank":          "Reranking…",
                "grade_documents": "Grading relevance…",
                "transform_query": "Refining query…",
                "generate":        "Generating answer…",
                "verify":          "Verifying answer…",
                "force_abstain":   "Preparing response…",
            }

            all_results = []
            for idx, sub_q in enumerate(sub_queries, 1):
                if len(sub_queries) > 1:
                    yield _sse(StreamEvent(
                        event = StreamEventType.STATUS,
                        data  = f"Sub-query {idx}/{len(sub_queries)}: {sub_q[:60]}",
                    ))

                final_state: dict = {}
                for chunk in state.graph.stream(
                    {"query": sub_q, "retry_count": 0},
                    stream_mode = "updates",
                ):
                    for node_name, node_output in chunk.items():
                        label = status_map.get(node_name, f"{node_name}…")
                        yield _sse(StreamEvent(event=StreamEventType.STATUS, data=label))
                        final_state.update(node_output)

                all_results.append(final_state)

            # Merge sub-query results.
            merged  = merge_results(all_results, sub_queries)
            answer  = merged.get("answer", ABSTAIN_PHRASE)
            verdict = merged.get("verdict", "abstain")
            cites   = merged.get("citations", [])

            yield _sse(StreamEvent(event=StreamEventType.ANSWER, data=answer))
            for c in cites:
                yield _sse(StreamEvent(
                    event = StreamEventType.CITATION,
                    data  = _to_citation_schema(c).model_dump(),
                ))
            yield _sse(StreamEvent(
                event = StreamEventType.DONE,
                data  = {"verdict": verdict,
                         "latency_s": round(time.perf_counter() - t0, 2)},
            ))

        except Exception as e:
            log.exception("stream query failed: %s", e)
            yield _sse(StreamEvent(event=StreamEventType.ERROR,
                                   data={"detail": str(e)}))

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Trigger the full ingest → embed → upsert pipeline."""
    state = get_app_state()
    try:
        from verses_rag.indexing.indexer import embed_and_upsert
        from verses_rag.ingestion.ingest import ingest as run_ingest

        chunks, report = run_ingest(
            settings              = state.settings,
            include_bible         = req.include_bible,
            treat_all_as_articles = req.treat_all_as_articles,
        )
        if chunks:
            state.store.ensure_collection(dense_dim=state.dense.dim)
            embed_and_upsert(chunks, state.store, state.dense, state.sparse)
    except Exception as e:
        log.exception("ingest failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        files_seen         = report.files_seen,
        articles_processed = report.articles_processed,
        bible_json_chunks  = report.bible_json_chunks,
        article_chunks     = report.article_chunks,
        duplicates_removed = report.duplicates_removed,
        total_chunks       = report.total_chunks,
    )


# --- entry point -------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("verses_rag.api.main:app", host="0.0.0.0", port=8000, reload=True)