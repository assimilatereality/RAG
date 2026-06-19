# =============================================================
# File: src/verses_rag/graph/nodes/analyze_filters.py
# =============================================================
"""
Analyze-filters node — second node in the RAG graph (SPEC §4.6, §5.3).

Takes the query and source_route from state and derives a neutral metadata
filter dict that the retrieve node passes to the vector store.

Neutral means store-agnostic: {"book": "Genesis", "testament": "OT"} — the
QdrantStore adapter translates this to Qdrant Filter/FieldCondition internally.

Filter extraction is heuristic-only (regex + keywords). No LLM needed:
filters are best-effort retrieval hints, not correctness requirements.
If no signals are found, an empty dict is returned and retrieval proceeds
unfiltered — always safe.

Signal coverage (§5.3):
  Bible  : book name(s), testament (OT/NT), chapter, verse range, source_type
  Article: status (current/latest → active), author, source_type
  Mixed  : all of the above, no source_type filter

Run self-check:
    uv run python -m verses_rag.graph.nodes.analyze_filters
"""

from __future__ import annotations

import logging
import re
from typing import Any

from verses_rag.canon import KJV_BOOKS, BOOK_ORDER, OT_COUNT, extract_scripture_refs
from verses_rag.graph.state import GraphState

log = logging.getLogger("graph.analyze_filters")

# --- compiled patterns -------------------------------------------------------

# Longest-first so "Song of Solomon" beats "Song", "1 Samuel" beats "Samuel".
_BOOK_RE = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in sorted(KJV_BOOKS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_TESTAMENT_RE = re.compile(
    r"\b(old\s+testament|new\s+testament|OT|NT)\b", re.IGNORECASE
)

# "author: John Piper", "written by John Piper", "by John Piper"
_AUTHOR_RE = re.compile(
    r"(?:author[:\s]+|written\s+by\s+|by\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
)

_TEMPORAL_SIGNALS = frozenset(["current", "latest", "recent", "newest", "today"])

# Matches "chapter 13", "ch. 13", or a bare number immediately after a book name
# has already been identified. Used as a fallback when no full Book C:V ref exists.
_CHAPTER_RE = re.compile(
    r"\bchapter\s+(\d+)\b|\bch\.?\s*(\d+)\b",
    re.IGNORECASE,
)

# Matches "Genesis 13" — book name followed by a bare number (no colon).
# Kept separate from _VERSE_REF_RE which requires the colon.
_BOOK_CHAPTER_RE = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in sorted(KJV_BOOKS, key=len, reverse=True)) + r")"
    r"\s+(\d+)(?!:\d)",   # number not followed by :digit (that would be a verse ref)
    re.IGNORECASE,
)


# --- extractors --------------------------------------------------------------

def _extract_book_filters(query: str) -> dict[str, Any]:
    """Extract book / testament / chapter / verse filters from a query."""
    filters: dict[str, Any] = {}

    # Testament
    t_match = _TESTAMENT_RE.search(query)
    if t_match:
        raw = t_match.group(1).lower().replace(" ", "")
        filters["testament"] = "OT" if raw in ("oldtestament", "ot") else "NT"

    # Books — only set a book filter when exactly one book is named.
    book_matches = [
        m.group(1) for m in _BOOK_RE.finditer(query)
        if not (query[m.end():].lstrip() and query[m.end():].lstrip()[0].isupper())
    ]
    seen: set[str] = set()
    canonical: list[str] = []
    for raw in book_matches:
        canon = next((b for b in KJV_BOOKS if b.lower() == raw.lower()), None)
        if canon and canon not in seen:
            canonical.append(canon)
            seen.add(canon)

    if len(canonical) == 1:
        book = canonical[0]
        filters["book"] = book
        if "testament" not in filters:
            filters["testament"] = "OT" if BOOK_ORDER[book] <= OT_COUNT else "NT"

    # Verse references — full Book C:V or Book C:V-W (highest priority).
    refs = extract_scripture_refs(query)
    if refs:
        first = refs[0]
        ref_match = re.match(r"(.+?)\s+(\d+):(\d+)(?:-(\d+))?$", first)
        if ref_match:
            filters.setdefault("book", ref_match.group(1))
            filters["chapter"] = int(ref_match.group(2))
            vs = int(ref_match.group(3))
            ve = int(ref_match.group(4)) if ref_match.group(4) else vs
            filters["verse_range"] = (vs, ve)
        return filters  # full ref found — no need for chapter-only fallback

    # Chapter-only fallback: "Genesis 13", "Genesis chapter 13", "ch. 13".
    # Only applied when no full verse ref was found above.
    chapter: int | None = None

    # "Genesis 13" style — bare number after book name.
    bc_match = _BOOK_CHAPTER_RE.search(query)
    if bc_match:
        raw_book = bc_match.group(1)
        canon = next((b for b in KJV_BOOKS if b.lower() == raw_book.lower()), None)
        if canon:
            filters.setdefault("book", canon)
            if "testament" not in filters:
                filters["testament"] = "OT" if BOOK_ORDER[canon] <= OT_COUNT else "NT"
            chapter = int(bc_match.group(2))

    # "chapter 13" / "ch. 13" style — works even without a book name in the match.
    ch_match = _CHAPTER_RE.search(query)
    if ch_match and chapter is None:
        chapter = int(ch_match.group(1) or ch_match.group(2))

    if chapter is not None:
        filters["chapter"] = chapter

    return filters


def _extract_article_filters(query: str) -> dict[str, Any]:
    """Extract status and author filters from a query."""
    filters: dict[str, Any] = {}

    q_lower = query.lower()
    if any(sig in q_lower for sig in _TEMPORAL_SIGNALS):
        filters["status"] = "active"

    author_match = _AUTHOR_RE.search(query)
    if author_match:
        filters["author"] = author_match.group(1).strip()

    return filters


# --- node entry point --------------------------------------------------------

def analyze_filters_node(state: GraphState) -> GraphState:
    """LangGraph node: derive metadata filters from query + source_route.

    Returns a partial state update containing only `filters`.
    An empty dict is a valid result — retrieval proceeds unfiltered.
    """
    query = state.get("query", "").strip()
    route = state.get("source_route", "mixed")

    filters: dict[str, Any] = {}

    if route in ("bible", "mixed"):
        filters.update(_extract_book_filters(query))

    if route in ("article", "mixed"):
        filters.update(_extract_article_filters(query))

    # NO hard source_type filter. A "bible"-routed query must still be able to
    # retrieve a scripture-focused article (the whole article corpus is *about*
    # scripture), so we never exclude a corpus by type. The cross-encoder reranker
    # is the relevance arbiter across both corpora — it reliably separates a
    # relevant article from off-topic verses without a filter doing it bluntly.
    # `source_route` is still in state for any node that wants the routing signal.

    # Soft-weight archived content (applied at retrieval, not here).
    # The store adapter reads this key and multiplies score by 0.5 for archived hits.
    filters.setdefault("downweight_archived", True)

    log.info("route=%s filters=%s", route, filters)
    return {"filters": filters}  # type: ignore[return-value]


# --- self-check --------------------------------------------------------------

def main():
    cases = [
        ("bible",   "What does Genesis say about creation?"),
        ("bible",   "Explain Romans 8:28 in context."),
        ("bible",   "What is in the Old Testament about sacrifice?"),
        ("bible",   "Show me Psalm 23:1-6."),
        ("article", "Find the latest articles written by John Piper."),
        ("article", "What current posts discuss the Sermon on the Mount?"),
        ("mixed",   "What do articles say about John 3:16?"),
        ("mixed",   "Compare Genesis and Romans on the nature of sin."),
        ("bible",   "Tell me about Revelation 22:21."),
        ("bible",   "Is Beth-el mentioned in Genesis 13?"),
        ("bible",   "What happens in Genesis chapter 13?"),
        ("bible",   "Tell me about Genesis 13"),
    ]

    print("=== analyze_filters self-check ===\n")
    for route, query in cases:
        state: GraphState = {"query": query, "source_route": route}  # type: ignore[misc]
        result = analyze_filters_node(state)
        filters = result.get("filters", {})
        print(f"[{route:7}] {query[:55]:<55}")
        for k, v in filters.items():
            print(f"          {k}: {v}")
        print()


if __name__ == "__main__":
    main()