# =============================================================
# File: src/verses_rag/graph/state.py
# =============================================================
"""
LangGraph state for the RAG pipeline (SPEC §4.6).

GraphState is the single dict that flows through every node. Each node
receives the full state and returns a partial dict of only the keys it
changed — LangGraph merges them automatically.

Fields progress through the pipeline in this order:
    query           → set by caller (entry point)
    source_route    → route node
    filters         → analyze_filters node
    candidates      → retrieve node
    ranked_docs     → rerank node
    grade_verdict   → grade_documents node
    draft_answer    → generate node
    citations       → generate node
    verdict         → verify node
    answer          → verify node (final output)
    retry_count     → grade_documents / verify (bounded loop guard)
    error           → any node (signals abnormal exit)
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    `total=False` makes every key optional so nodes can return partial
    updates without repeating unchanged fields. The entry point must
    supply `query`; all other fields are populated as the graph runs.
    """

    # --- entry ---
    query: str                          # the user's question, unchanged throughout

    # --- route node ---
    source_route: str                   # "bible" | "article" | "mixed"

    # --- transform_query node (retry path) ---
    retrieval_query: str                # reformulated query for next retrieve attempt;
                                        # if absent, retrieve falls back to `query`

    # --- analyze_filters node ---
    filters: dict[str, Any]            # Qdrant-neutral filter dict (§5.3)

    # --- retrieve node ---
    candidates: list[Any]              # raw store Hit objects from hybrid_query

    # --- rerank node ---
    ranked_docs: list[Any]             # RankedHit objects, sorted by rerank_score

    # --- grade_documents node ---
    grade_verdict: str                  # "sufficient" | "insufficient"
    retry_count: int                    # incremented on each retrieve→grade loop

    # --- generate node ---
    draft_answer: str
    citations: list[dict[str, Any]]    # [{ref, content_snippet, score}, ...]

    # --- verify node ---
    verdict: str                        # "pass" | "fail" | "abstain"
    answer: str                         # final answer surfaced to the caller

    # --- error handling ---
    error: Optional[str]               # set by any node on unrecoverable failure