# =============================================================
# File: src/verses_rag/graph/nodes/transform_query.py
# =============================================================
"""
Transform-query node — retry helper in the RAG graph (SPEC §4.6).

When grade_documents returns "insufficient", this node reformulates the
query before the next retrieve attempt. It uses the top retrieved passage
as context so the LLM understands what was found and what was missed.

The reformulated query goes into `retrieval_query` (not `query`) so the
original user question is preserved for generation and verification.

Dependency injected via functools.partial (Option A):
    llm — RunnableWithFallbacks from get_llm("route")
"""

from __future__ import annotations

import logging
import re
from typing import Any

from verses_rag.graph.state import GraphState

log = logging.getLogger("graph.transform_query")

_TRANSFORM_PROMPT = """\
A search retrieved passages to answer a question, but they were not \
relevant enough.

Original question: {query}

Top retrieved passage (for context on what was found):
{top_passage}

Rewrite the question to be more specific and likely to retrieve the \
correct passages. Return ONLY the rewritten question, no explanation.
"""


def transform_query_node(
    state: GraphState,
    *,
    llm: Any,
    settings=None,
) -> GraphState:
    """LangGraph node: reformulate query → retrieval_query for next retrieve pass."""
    query       = state.get("query", "").strip()
    ranked_docs = state.get("ranked_docs", [])
    top_passage = ranked_docs[0].content[:200] if ranked_docs else "(none retrieved)"

    prompt = _TRANSFORM_PROMPT.format(query=query, top_passage=top_passage)

    try:
        response  = llm.invoke(prompt)
        raw       = response.content if hasattr(response, "content") else response
        new_query = " ".join(str(i) for i in raw) if isinstance(raw, list) else str(raw)
        new_query = re.sub(r"<think>.*?</think>", "", new_query, flags=re.DOTALL).strip()
        new_query = new_query.strip("\"'")

        if new_query and new_query != query:
            log.info("transformed query: %r → %r", query[:60], new_query[:60])
            return {"retrieval_query": new_query}  # type: ignore[return-value]
    except Exception as e:
        log.warning("transform_query LLM call failed: %s — using original", e)

    # Fall through: retry with original query unchanged.
    return {"retrieval_query": query}  # type: ignore[return-value]