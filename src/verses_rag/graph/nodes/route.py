# =============================================================
# File: src/verses_rag/graph/nodes/route.py
# =============================================================
"""
Route node — first node in the RAG graph (SPEC §4.6, §4.7).

Classifies the query as "bible", "article", or "mixed" and sets the
initial source_route on the state. This tells downstream nodes which
collection(s) to search and which metadata filters to derive.

Two-stage cascade (mirrors the document classifier's philosophy):
  1. Heuristic — fast, free, handles most cases:
       - Book names / verse references in the query  → bible
       - Author / site / article-specific keywords   → article
       - Both signals present                        → mixed
  2. LLM fallback — only for genuinely ambiguous queries, via
       get_llm("route") (OpenAI primary / Anthropic backup per D7).

The node is deliberately stateless — same input always produces the
same output, making it easy to unit-test without a running LangGraph.

Run self-check:
    uv run python -m verses_rag.graph.nodes.route
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from verses_rag.canon import KJV_BOOKS, extract_scripture_refs
from verses_rag.graph.state import GraphState

log = logging.getLogger("graph.route")

# --- heuristic signals -------------------------------------------------------

_BOOK_NAMES_LOWER = {b.lower() for b in KJV_BOOKS}

# Words that strongly suggest the user wants article content.
_ARTICLE_SIGNALS = frozenset([
    "article", "blog", "post", "author", "wrote", "writes", "written",
    "website", "site", "says", "according to", "commentary", "devotional",
    "sermon", "study",
])

# Compiled: whole-word match for any KJV book name (handles "Genesis", "1 Samuel", etc.)
_BOOK_RE = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in sorted(KJV_BOOKS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

MAX_RETRIES = 2   # LLM fallback attempts before defaulting to "mixed"


# --- heuristic classifier ----------------------------------------------------

def _has_bible_book_signal(query: str) -> bool:
    """True if the query contains a KJV book name that isn't part of a person's name.

    Filters out matches like 'John Piper' or 'Mark Driscoll' by checking
    whether the matched name is immediately followed by a capitalized word
    (the tell-tale pattern of a first name + surname).
    """
    for m in _BOOK_RE.finditer(query):
        rest = query[m.end():].lstrip()
        if rest and rest[0].isupper():
            continue    # looks like "John Piper", "Luke Skywalker", etc. — skip
        return True
    return False


def _heuristic_route(query: str) -> Optional[str]:
    """Return 'bible', 'article', 'mixed', or None (ambiguous → LLM needed)."""
    q = query.lower()

    has_book = _has_bible_book_signal(query)
    has_verse = bool(extract_scripture_refs(query))
    has_article = any(sig in q for sig in _ARTICLE_SIGNALS)

    bible_signal = has_book or has_verse
    article_signal = has_article

    if bible_signal and article_signal:
        return "mixed"
    if bible_signal:
        return "bible"
    if article_signal:
        return "article"
    return None   # ambiguous — needs LLM


# --- LLM fallback ------------------------------------------------------------

_ROUTE_PROMPT = """\
You are routing a user query to the correct document collection.

Collections:
  - "bible"   : questions about specific scripture passages, verses, or books
  - "article" : questions about blog posts, articles, authors, or commentary
  - "mixed"   : questions that involve both scripture and article content

Query: {query}

Reply with a JSON object and nothing else:
{{"route": "bible" | "article" | "mixed", "reason": "<one sentence>"}}
"""


def _llm_route(query: str, settings=None) -> str:
    """Ask the judge LLM to classify the query. Returns 'bible'/'article'/'mixed'."""
    from verses_rag.llm.router import get_llm

    llm = get_llm("route", settings)
    prompt = _ROUTE_PROMPT.format(query=query)

    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else response
            # content is str | list[str | dict] in LangChain — flatten to str.
            if isinstance(raw, list):
                text = " ".join(str(item) for item in raw)
            else:
                text = str(raw)
            # Strip any accidental markdown fences before parsing.
            text = re.sub(r"```(?:json)?|```", "", text).strip()
            data = json.loads(text)
            route = data.get("route", "").lower()
            if route in ("bible", "article", "mixed"):
                log.info("LLM route: %s — %s", route, data.get("reason", ""))
                return route
            log.warning("LLM returned unexpected route %r (attempt %d)", route, attempt + 1)
        except Exception as e:
            log.warning("LLM route attempt %d failed: %s", attempt + 1, e)

    log.warning("LLM routing failed after %d attempts; defaulting to 'mixed'", MAX_RETRIES)
    return "mixed"


# --- node entry point --------------------------------------------------------

def route_node(state: GraphState, settings=None) -> GraphState:
    """LangGraph node: classify query and set source_route.

    Returns a partial state update — only the keys this node sets.
    """
    query = state.get("query", "").strip()
    if not query:
        log.error("route_node received empty query")
        return {"source_route": "mixed", "error": "empty query"}  # type: ignore[return-value]

    route = _heuristic_route(query)
    method = "heuristic"

    if route is None:
        log.info("query ambiguous, falling back to LLM router")
        route = _llm_route(query, settings)
        method = "llm"

    log.info("route=%s (%s) | query=%r", route, method, query[:60])
    return {"source_route": route}  # type: ignore[return-value]


# --- self-check --------------------------------------------------------------

def main():
    cases = [
        # Expected heuristic hits
        ("What does Genesis say about creation?",           "bible"),
        ("Explain Romans 8:28 in context.",                 "bible"),
        ("Find articles written by John Piper.",            "article"),
        ("What blog posts discuss the Sermon on the Mount?","article"),
        ("What do articles say about John 3:16?",          "mixed"),
        # Ambiguous — should fall to LLM
        ("What is the meaning of grace?",                   "? (LLM)"),
        ("Tell me about redemption.",                       "? (LLM)"),
    ]

    print("=== route node self-check ===\n")
    print(f"{'Query':<52} {'Expected':<12} {'Got':<8} {'Method'}")
    print("-" * 85)

    for query, expected in cases:
        heuristic = _heuristic_route(query)
        method = "heuristic"
        if heuristic is None:
            method = "llm (skipped)"
            heuristic = "(ambiguous)"

        result = route_node({"query": query})  # type: ignore[arg-type]
        got = result.get("source_route", "?")
        marker = "✓" if expected.startswith("?") or got == expected else "✗"
        print(f"{marker} {query:<50} {expected:<12} {got:<8} {method}")


if __name__ == "__main__":
    main()