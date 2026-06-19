# =============================================================
# File: src/verses_rag/api/schemas.py
# =============================================================
"""
Request / response schemas for the FastAPI layer (SPEC §4.10).

Four endpoints:
  POST /query         — synchronous: wait for full answer
  POST /query/stream  — streaming: SSE node-progress + answer chunks
  POST /ingest        — trigger ingestion pipeline
  GET  /health        — provider + graph readiness check

Run self-check (no server needed):
    uv run python -m verses_rag.api.schemas
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --- query -------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:             str  = Field(..., min_length=1, max_length=1000,
                                    description="The user's question.")
    use_decomposition: bool = Field(default=True,
                                    description="Decompose multi-part queries automatically.")


class Citation(BaseModel):
    ref:             str   = Field(..., description="Canonical reference, e.g. 'John 3:16'.")
    content_snippet: str   = Field(..., description="First ~120 chars of the chunk.")
    rerank_score:    float = Field(..., description="Cross-encoder logit (higher = more relevant).")
    source_type:     str   = Field(..., description="'bible' or 'article'.")


class QueryResponse(BaseModel):
    query:       str            = Field(..., description="The original query.")
    answer:      str            = Field(..., description="Grounded answer or abstention phrase.")
    verdict:     str            = Field(..., description="'pass' or 'abstain'.")
    citations:   list[Citation] = Field(default_factory=list)
    sub_queries: list[str]      = Field(default_factory=list,
                                        description="Sub-queries if decomposition ran.")
    latency_s:   float          = Field(..., description="Wall-clock seconds for the full pipeline.")


# --- streaming ---------------------------------------------------------------

class StreamEventType(str, Enum):
    STATUS   = "status"    # node-progress message, e.g. "Retrieving..."
    ANSWER   = "answer"    # answer text chunk (may arrive in pieces)
    CITATION = "citation"  # one Citation object
    DONE     = "done"      # pipeline complete; data = {"verdict": "..."}
    ERROR    = "error"     # unrecoverable error; data = {"detail": "..."}


class StreamEvent(BaseModel):
    """One SSE payload. Serialised as JSON in the `data:` field of each event."""
    event: StreamEventType
    data:  Any   = None   # str for STATUS/ANSWER/ERROR, Citation dict for CITATION,
                          # {"verdict": str, "latency_s": float} for DONE


# --- ingest ------------------------------------------------------------------

class IngestRequest(BaseModel):
    include_bible:         bool = Field(default=True,  description="Ingest KJV JSON via BibleProcessor.")
    treat_all_as_articles: bool = Field(default=False, description="Skip classifier; treat every file as article.")


class IngestResponse(BaseModel):
    files_seen:       int
    articles_processed: int
    bible_json_chunks:  int
    article_chunks:   int
    duplicates_removed: int
    total_chunks:     int


# --- health ------------------------------------------------------------------

class ProviderStatus(str, Enum):
    OK       = "ok"
    DEGRADED = "degraded"
    DOWN     = "down"


class ProviderHealthSchema(BaseModel):
    provider:   str
    status:     ProviderStatus
    latency_ms: float | None = None
    error:      str   | None = None


class HealthResponse(BaseModel):
    status:      str                         # "ok" | "degraded" | "down"
    graph_ready: bool
    primary:     ProviderHealthSchema | None = None
    backup:      ProviderHealthSchema | None = None


# --- self-check --------------------------------------------------------------

def main():
    import json

    print("=== schemas self-check ===\n")

    # QueryRequest validation
    req = QueryRequest(query="What does Genesis say about creation?")
    print(f"QueryRequest:  {req.model_dump()}")

    # QueryResponse round-trip
    resp = QueryResponse(
        query    = req.query,
        answer   = "In the beginning God created the heaven and the earth. [1]",
        verdict  = "pass",
        citations = [Citation(ref="Genesis 1:1",
                              content_snippet="In the beginning God created...",
                              rerank_score=2.6,
                              source_type="bible")],
        sub_queries = [],
        latency_s   = 4.2,
    )
    print(f"\nQueryResponse: {resp.model_dump_json(indent=2)}")

    # StreamEvent
    events = [
        StreamEvent(event=StreamEventType.STATUS,   data="Retrieving…"),
        StreamEvent(event=StreamEventType.ANSWER,   data="In the beginning"),
        StreamEvent(event=StreamEventType.CITATION, data={"ref": "Genesis 1:1",
                                                          "content_snippet": "...",
                                                          "rerank_score": 2.6,
                                                          "source_type": "bible"}),
        StreamEvent(event=StreamEventType.DONE,     data={"verdict": "pass",
                                                          "latency_s": 4.2}),
    ]
    print("\nStreamEvents:")
    for e in events:
        print(f"  {e.event.value:10} {json.dumps(e.data)}")

    # HealthResponse
    health = HealthResponse(
        status      = "ok",
        graph_ready = True,
        primary     = ProviderHealthSchema(provider="openai/gpt-4o-mini",
                                           status=ProviderStatus.OK,
                                           latency_ms=432.1),
        backup      = ProviderHealthSchema(provider="anthropic/claude-haiku-4-5-20251001",
                                           status=ProviderStatus.OK,
                                           latency_ms=797.3),
    )
    print(f"\nHealthResponse: {health.model_dump_json(indent=2)}")

    # Validation error test
    try:
        QueryRequest(query="")
    except Exception as e:
        print(f"\nValidation error on empty query: {type(e).__name__} ✓")


if __name__ == "__main__":
    main()