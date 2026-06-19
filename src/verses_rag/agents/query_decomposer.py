# =============================================================
# File: src/verses_rag/agents/query_decomposer.py
# =============================================================
"""
Query decomposer (SPEC §4.7).

Detects comparative or multi-part queries and splits them into focused
sub-queries that the RAG graph can handle individually. Results from
each sub-query are then merged into a single answer.

Examples of multi-part queries:
    "Compare what Genesis and Romans say about sin"
    → ["What does Genesis say about sin?",
       "What does Romans say about sin?"]

    "What do Psalms and Proverbs teach about wisdom?"
    → ["What does Psalms teach about wisdom?",
       "What does Proverbs teach about wisdom?"]

Single queries pass through unchanged:
    "What does John 3:16 mean?" → ["What does John 3:16 mean?"]

Two-stage detection (mirrors the classifier/router philosophy):
  1. Heuristic — fast, free: compare/contrast keywords + two book names.
  2. LLM fallback — only for ambiguous cases.

Merging strategy: sub-answers are concatenated with per-book headers.
Citations are deduplicated by ref. If all sub-queries abstain, the merged
result abstains.

Run self-check (heuristic cases free; LLM cases need API keys):
    uv run python -m verses_rag.agents.query_decomposer
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from verses_rag.canon import KJV_BOOKS
from verses_rag.graph.nodes.generate import ABSTAIN_PHRASE

log = logging.getLogger("agents.decomposer")

# --- heuristic signals -------------------------------------------------------

_COMPARE_SIGNALS = frozenset([
    "compare", "contrast", "difference between", "differences between",
    "both", "each", "versus", "vs", "vs.", "and both",
])

_BOOK_RE = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in sorted(KJV_BOOKS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_DECOMPOSE_PROMPT = """\
A user asked a question that may involve multiple distinct parts or books \
of the Bible.

Query: {query}

If this is a comparative or multi-part query involving two or more distinct \
books or topics, decompose it into separate focused sub-queries (one per book \
or topic). Each sub-query should be self-contained and answerable on its own.

If it is a single focused query, return it unchanged as the only sub-query.

Reply with a JSON object and nothing else:
{{"is_multi_part": true | false, "sub_queries": ["query1", "query2", ...]}}
"""


# --- heuristic decomposer ----------------------------------------------------

def _find_books(query: str) -> list[str]:
    """Return canonical book names found in the query (person-name guard applied)."""
    books, seen = [], set()
    for m in _BOOK_RE.finditer(query):
        rest = query[m.end():].lstrip()
        if rest and rest[0].isupper():
            continue   # "John Piper" pattern — skip
        canon = next((b for b in KJV_BOOKS if b.lower() == m.group(1).lower()), None)
        if canon and canon not in seen:
            books.append(canon)
            seen.add(canon)
    return books


def _books_are_paired(query: str, books: list[str]) -> bool:
    """True if two or more book names appear connected by 'and' in the query."""
    for i, b1 in enumerate(books):
        for b2 in books[i + 1:]:
            pat = rf'\b{re.escape(b1)}\b[^.]*\band\b[^.]*\b{re.escape(b2)}\b'
            if re.search(pat, query, re.IGNORECASE):
                return True
    return False


def _heuristic_decompose(query: str) -> list[str] | None:
    """Return sub-queries if heuristic is confident, else None (→ LLM)."""
    q_lower = query.lower()

    # Word-boundary check prevents "each" matching inside "teachings" etc.
    has_compare = any(
        re.search(rf'\b{re.escape(sig)}\b', q_lower)
        for sig in _COMPARE_SIGNALS
    )
    books        = _find_books(query)
    books_paired = _books_are_paired(query, books)

    if not ((has_compare or books_paired) and len(books) >= 2):
        return None

    # Extract topic from "about/on/regarding/concerning X" — safer than
    # stripping book names and signal words from the whole query.
    topic_match = re.search(
        r'\b(?:about|on|regarding|concerning)\s+(.+?)(?:\s*\?|$)',
        query, re.IGNORECASE,
    )
    if not topic_match:
        return None   # can't extract topic cleanly → fall to LLM

    topic = topic_match.group(1).strip("., ?")
    if not topic:
        return None

    return [f"What does {book} say about {topic}?" for book in books]


# --- LLM fallback ------------------------------------------------------------

def _llm_decompose(query: str, llm: Any) -> list[str]:
    """Ask the judge LLM whether to decompose and how."""
    prompt = _DECOMPOSE_PROMPT.format(query=query)
    try:
        response = llm.invoke(prompt)
        raw  = response.content if hasattr(response, "content") else response
        text = " ".join(str(i) for i in raw) if isinstance(raw, list) else str(raw)
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        data = json.loads(text)
        subs = data.get("sub_queries", [])
        if isinstance(subs, list) and all(isinstance(s, str) for s in subs) and subs:
            log.info(
                "LLM decompose: is_multi_part=%s, %d sub-queries",
                data.get("is_multi_part"), len(subs),
            )
            return subs
    except Exception as e:
        log.warning("LLM decompose failed (%s); treating as single query", e)
    return [query]


# --- public API --------------------------------------------------------------

def decompose(query: str, llm: Any | None = None) -> list[str]:
    """Return a list of sub-queries for the given query.

    Single queries return [query]. Multi-part queries return one sub-query
    per part. Always returns at least one element.

    Args:
        query: The user's question.
        llm:   Optional judge LLM for ambiguous cases. If None, heuristic
               only (ambiguous cases treated as single queries).
    """
    query = query.strip()
    if not query:
        return []

    result = _heuristic_decompose(query)
    if result is not None:
        log.info("heuristic decompose → %d sub-queries", len(result))
        return result

    if llm is not None:
        return _llm_decompose(query, llm)

    log.debug("no decomposition signal; treating as single query")
    return [query]


# --- merge -------------------------------------------------------------------

def merge_results(results: list[dict[str, Any]], sub_queries: list[str]) -> dict[str, Any]:
    """Merge multiple sub-query graph results into a single response.

    Strategy:
      - Single result: pass through unchanged.
      - Multiple results: label each answer with its sub-query, concatenate,
        deduplicate citations by ref.
      - All abstaining: return ABSTAIN_PHRASE.
    """
    if not results:
        return {"answer": ABSTAIN_PHRASE, "verdict": "abstain", "citations": []}

    if len(results) == 1:
        return results[0]

    passing = [
        (r, q) for r, q in zip(results, sub_queries)
        if r.get("verdict") == "pass"
    ]

    if not passing:
        return {"answer": ABSTAIN_PHRASE, "verdict": "abstain", "citations": []}

    parts, seen_refs, citations = [], set(), []
    for result, sub_q in passing:
        parts.append(f"**{sub_q}**\n{result.get('answer', '')}")
        for c in result.get("citations", []):
            if c["ref"] not in seen_refs:
                seen_refs.add(c["ref"])
                citations.append(c)

    return {
        "answer":   "\n\n".join(parts),
        "verdict":  "pass",
        "citations": citations,
    }


# --- self-check --------------------------------------------------------------

def main():
    heuristic_cases = [
        ("Compare what Genesis and Romans say about sin",          2),
        ("What do Psalms and Proverbs teach about wisdom?",        2),
        ("Compare Matthew and John on the resurrection",           2),
        ("What does John 3:16 mean?",                              1),  # single
        ("How should I trust God when life is hard?",              1),  # single
        ("What does the Bible say about God's love?",              1),  # single
    ]

    print("=== query decomposer self-check ===\n")
    print("--- heuristic (no API) ---\n")
    for query, expected_count in heuristic_cases:
        subs = decompose(query)   # no llm → heuristic only
        marker = "✓" if len(subs) == expected_count else "✗"
        print(f"{marker} [{len(subs)} sub-queries] {query[:55]}")
        for s in subs:
            print(f"    → {s}")
        print()

    # LLM cases — only if API keys available
    import os
    from verses_rag.config.settings import get_settings
    s = get_settings()
    if not s.openai_api_key:
        print("No API key — skipping LLM decompose cases.")
        return

    from verses_rag.llm.router import get_llm
    llm = get_llm("route", s)

    llm_cases = [
        "What similarities exist between the teachings of Isaiah and Jeremiah?",
        "How do the Gospels of Matthew and Luke differ in their birth narratives?",
    ]
    print("--- LLM fallback ---\n")
    for query in llm_cases:
        subs = decompose(query, llm=llm)
        print(f"[{len(subs)} sub-queries] {query}")
        for s in subs:
            print(f"    → {s}")
        print()


if __name__ == "__main__":
    main()