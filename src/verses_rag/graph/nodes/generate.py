# =============================================================
# File: src/verses_rag/graph/nodes/generate.py
# =============================================================
"""
Generate node — sixth node in the RAG graph (SPEC §4.6, §4.8).

Produces a grounded answer from ranked_docs using the local Ollama
generation model (qwen3:1.7b by default). Returns draft_answer and
citations for the verify node.

Key design choices:
  - Strict context-only prompt: the model is told to answer ONLY from
    the provided passages and to abstain explicitly if they're insufficient.
  - Passages numbered [1]..[N] so the model can reference them inline.
  - Citations built from the docs sent to the prompt — grading has already
    confirmed relevance, so all included docs are valid citations.
  - Qwen3 <think> blocks stripped before returning (SPEC R5).
  - Parse failures return a graceful abstention rather than crashing.
  - Verse-reference queries bypass LLM generation entirely and return
    chunk text verbatim (avoids paraphrasing/truncation for direct lookups).

Dependency injected via functools.partial (Option A):
    llm — ChatOllama from get_llm("generation")

Graph wiring example:
    from functools import partial
    from verses_rag.llm.router import get_llm
    generate = partial(generate_node, llm=get_llm("generation"))
"""

from __future__ import annotations

import logging
import re
from typing import Any

from verses_rag.graph.state import GraphState

log = logging.getLogger("graph.generate")

ABSTAIN_PHRASE = (
    "I don't have enough information in the provided passages "
    "to answer this question."
)

# How many ranked docs to include in the generation prompt.
_DOCS_TO_USE = 3

# Max chars per passage in the LLM prompt — raised to avoid mid-verse truncation.
# Verse windows can be ~600–800 chars for 5-verse KJV windows.
_PASSAGE_CHAR_LIMIT = 1200

_GENERATE_PROMPT = """\
You are a scripture and article research assistant.
Answer the question using ONLY the passages provided below.
Cite passages inline using their number, like [1] or [2].
IMPORTANT: Use only the passage reference labels provided (e.g. [1], [2]).
Do NOT invent or infer specific verse numbers beyond what the label states.
If the passages do not contain the answer, respond with exactly:
"{abstain}"

Passages:
{passages}

Question: {query}

Answer:"""

# ---------------------------------------------------------------------------
# Verse-lookup detection
# ---------------------------------------------------------------------------

# Matches the core reference pattern: optional book prefix with chapter:verse
# or chapter:verse-verse range. Anchored loosely so it works inside longer
# questions like "What does Romans 8:1-5 say?" or bare "Romans 8:1".
_BOOK_PAT = (
    r"(?:1|2|3\s+)?[A-Z][a-zA-Z]+"   # optional numeric prefix + book word
    r"(?:\s+[A-Za-z]+)*"              # multi-word books: "Song of Solomon"
)
_VERSE_REF_RE = re.compile(
    rf"\b({_BOOK_PAT})\s+(\d+):(\d+)(?:\s*[-–]\s*(\d+))?\b",
    re.IGNORECASE,
)

# Query prefixes that signal "look this up and quote it"
_LOOKUP_PREFIXES = re.compile(
    r"^\s*(?:what\s+(?:does|do|is|are)\s+|"
    r"quote\s+|show\s+(?:me\s+)?|"
    r"give\s+(?:me\s+)?|"
    r"read\s+|recite\s+|"
    r"what\s+(?:is|are)\s+the\s+(?:verse|passage|text|words?)\s+(?:of|in|at|for)\s+)?",
    re.IGNORECASE,
)


def _is_verse_lookup(query: str) -> bool:
    """True if the query is primarily asking for a specific verse or verse range.

    Matches:
      - "Romans 8:1"
      - "Romans 8:1-5"
      - "What does Romans 8:28 say?"
      - "Show me Psalms 23:1-6"
      - "Genesis 1:1-3"

    Does NOT match:
      - "What does the Bible say about God's love?" (no specific reference)
      - "Romans 8 and grace" (chapter only, no verse)
    """
    # Strip common lookup-intent prefixes before checking for the reference.
    stripped = _LOOKUP_PREFIXES.sub("", query).strip()
    # After stripping, the query should be essentially just the reference
    # (possibly with trailing punctuation or "say/mean/teach/tell us").
    m = _VERSE_REF_RE.search(stripped)
    if not m:
        return False
    # Allow trailing filler words but not substantial additional content.
    after_ref = stripped[m.end():].strip().rstrip("?.,!")
    filler = re.compile(
        r"^(?:say|mean|teach|tell\s+us|refer\s+to|talk\s+about|state)?$",
        re.IGNORECASE,
    )
    return bool(filler.match(after_ref))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_label(payload: dict[str, Any]) -> str:
    """Human-readable reference label for a chunk payload."""
    if payload.get("source_type") == "bible":
        book = payload.get("book", "")
        ch   = payload.get("chapter", "")
        vs   = payload.get("verse_start", "")
        ve   = payload.get("verse_end", "")
        ref  = f"{book} {ch}:{vs}"
        return ref + f"-{ve}" if ve and ve != vs else ref
    title = payload.get("title", payload.get("source_path", "passage"))
    return str(title)[:60]


def _format_passages(ranked_docs: list[Any]) -> str:
    lines = []
    for i, doc in enumerate(ranked_docs[:_DOCS_TO_USE], 1):
        ref = _ref_label(doc.payload)
        lines.append(f"[{i}] ({ref})\n{doc.content[:_PASSAGE_CHAR_LIMIT]}")
    return "\n\n".join(lines)


def _build_citations(ranked_docs: list[Any]) -> list[dict[str, Any]]:
    """Build citation records from the docs included in the prompt."""
    citations = []
    for doc in ranked_docs[:_DOCS_TO_USE]:
        citations.append({
            "ref":             _ref_label(doc.payload),
            "content_snippet": doc.content[:120],
            "rerank_score":    doc.rerank_score,
            "source_type":     doc.payload.get("source_type", "unknown"),
        })
    return citations


def _strip_think_blocks(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks (SPEC R5)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_text(response: Any) -> str:
    """Pull plain text from a LangChain response object."""
    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, list):
        text = " ".join(str(item) for item in raw)
    else:
        text = str(raw)
    text = _strip_think_blocks(text)

    if "\nAnswer:" in text:
        before, _, after = text.partition("\nAnswer:")
        after = after.strip()
        if after and ABSTAIN_PHRASE not in after:
            text = after
        else:
            text = before.strip()

    # Strip bare citation-only lines like "[1]" with nothing else on the line.
    text = re.sub(r"^\s*\[\d+\]\s*$", "", text, flags=re.MULTILINE)

    # Strip spurious abstain footers (SPEC R5 — footer case).
    if ABSTAIN_PHRASE in text:
        stripped = text.replace(ABSTAIN_PHRASE, "").strip()
        if stripped:
            text = stripped

    return text.strip()


# ---------------------------------------------------------------------------
# Verse-lookup bypass (no LLM call)
# ---------------------------------------------------------------------------

def _direct_verse_response(
    ranked_docs: list[Any],
    query: str,
) -> dict[str, Any]:
    """Return verbatim chunk text for a verse-reference lookup query.

    Selects the top-ranked Bible chunk(s), assembles their full text without
    any LLM summarization, and builds citation records. Multiple chunks are
    included when the requested range spans more than one window.
    """
    # Gather all Bible chunks from the top results.
    bible_docs = [
        d for d in ranked_docs
        if d.payload.get("source_type") == "bible"
    ]
    if not bible_docs:
        # Fall back — caller will run normal LLM generation.
        return {}

    # For verse-lookup, sort by verse_start so the requested range leads.
    # Rerank score is not the right ordering signal here — the chunk that
    # starts earliest in the chapter is the one the user asked for.
    bible_docs.sort(key=lambda d: (
        d.payload.get("chapter", 0),
        d.payload.get("verse_start", 0),
    ))

    # Include up to _DOCS_TO_USE so verse ranges spanning windows are covered.
    selected = bible_docs[:_DOCS_TO_USE]

    parts = []
    citations = []
    for doc in selected:
        ref = _ref_label(doc.payload)
        # Full content — no char truncation for direct lookups.
        parts.append(doc.content)
        citations.append({
            "ref":             ref,
            "content_snippet": doc.content[:120],
            "rerank_score":    doc.rerank_score,
            "source_type":     "bible",
        })

    answer = "\n\n".join(parts)
    log.info(
        "verse-lookup bypass: returning %d chunk(s) verbatim (%d chars)",
        len(selected), len(answer),
    )
    return {"draft_answer": answer, "citations": citations}


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

def generate_node(
    state: GraphState,
    *,
    llm: Any,
    settings=None,
) -> GraphState:
    """LangGraph node: grounded generation → draft_answer + citations.

    For verse-reference lookup queries, bypasses LLM generation and returns
    chunk text verbatim. For all other queries, uses the LLM with a
    context-only grounded prompt.
    """
    query       = state.get("query", "").strip()
    ranked_docs = state.get("ranked_docs", [])

    if not ranked_docs:
        log.warning("generate_node received no ranked docs — returning abstention")
        return {  # type: ignore[return-value]
            "draft_answer": ABSTAIN_PHRASE,
            "citations": [],
        }

    # --- Verse-lookup bypass ---
    if _is_verse_lookup(query):
        log.info("verse-lookup query detected — bypassing LLM generation")
        result = _direct_verse_response(ranked_docs, query)
        if result:
            return result  # type: ignore[return-value]
        # No Bible chunks found despite the reference query — fall through to LLM.
        log.warning("verse-lookup bypass found no Bible chunks — falling through to LLM")

    # --- Normal LLM generation ---
    prompt = _GENERATE_PROMPT.format(
        abstain=ABSTAIN_PHRASE,
        passages=_format_passages(ranked_docs),
        query=query,
    )

    try:
        response    = llm.invoke(prompt)
        draft       = _extract_text(response)
    except Exception as e:
        log.error("generation LLM call failed: %s", e)
        return {  # type: ignore[return-value]
            "draft_answer": ABSTAIN_PHRASE,
            "citations": [],
            "error": str(e),
        }

    citations = _build_citations(ranked_docs)

    log.info(
        "generated answer (%d chars) with %d citations",
        len(draft), len(citations),
    )
    return {  # type: ignore[return-value]
        "draft_answer": draft,
        "citations":    citations,
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def main():
    import hashlib
    from functools import partial

    from verses_rag.config.settings import get_settings
    from verses_rag.llm.router import get_llm
    from verses_rag.retrieval.reranker import RankedHit

    def _sha(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    def _hit(ref: str, content: str, book: str, ch: int, vs: int,
             ve: int, rerank: float) -> RankedHit:
        return RankedHit(
            chunk_id=_sha(ref),
            payload={"content": content, "book": book, "chapter": ch,
                     "verse_start": vs, "verse_end": ve, "source_type": "bible"},
            retrieval_score=0.03,
            rerank_score=rerank,
        )

    # --- Test _is_verse_lookup ---
    print("=== _is_verse_lookup tests ===")
    cases = [
        ("Romans 8:1",                          True),
        ("Romans 8:1-5",                        True),
        ("What does Romans 8:28 say?",          True),
        ("Show me Psalms 23:1-6",               True),
        ("Genesis 1:1-3",                       True),
        ("What does the Bible say about love?", False),
        ("Romans 8 and grace",                  False),
        ("Compare Genesis and Romans on sin",   False),
    ]
    all_ok = True
    for q, expected in cases:
        got = _is_verse_lookup(q)
        status = "✓" if got == expected else "✗"
        if got != expected:
            all_ok = False
        print(f"  {status} {q!r:45} -> {got} (expected {expected})")
    print(f"\n{'All passed' if all_ok else 'FAILURES above'}\n")

    # --- Test verse-lookup bypass (no Ollama needed) ---
    print("=== verse-lookup bypass ===")
    ranked_docs = [
        _hit("Romans 8:1-5",
             "There is therefore now no condemnation to them which are in Christ Jesus, "
             "who walk not after the flesh, but after the Spirit. For the law of the "
             "Spirit of life in Christ Jesus hath made me free from the law of sin and "
             "death. For what the law could not do, in that it was weak through the flesh, "
             "God sending his own Son in the likeness of sinful flesh, and for sin, "
             "condemned sin in the flesh: That the righteousness of the law might be "
             "fulfilled in us, who walk not after the flesh, but after the Spirit. For "
             "they that are after the flesh do mind the things of the flesh; but they "
             "that are after the Spirit the things of the Spirit.",
             "Romans", 8, 1, 5, rerank=-2.1),
    ]
    state: GraphState = {"query": "Romans 8:1-5", "ranked_docs": ranked_docs}  # type: ignore
    result = _direct_verse_response(ranked_docs, "Romans 8:1-5")
    print("Answer:")
    print(result.get("draft_answer", ""))

    # --- Test full node with Ollama (optional) ---
    print("\n=== full generate_node (requires Ollama) ===")
    s   = get_settings()
    llm = get_llm("generation", s)
    generate = partial(generate_node, llm=llm, settings=s)

    state2: GraphState = {  # type: ignore
        "query": "What does the Bible say about God's love?",
        "ranked_docs": ranked_docs,
    }
    try:
        result2 = generate(state2)
        print("Answer:")
        print(result2.get("draft_answer", ""))
        print("\nCitations:")
        for c in result2.get("citations", []):
            print(f"  [{c['rerank_score']:7.3f}] {c['ref']:20} {c['content_snippet'][:60]}")
    except Exception as e:
        print(f"Skipped (Ollama not running? {e})")


if __name__ == "__main__":
    main()