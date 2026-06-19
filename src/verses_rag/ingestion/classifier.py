"""
Document classifier for the ingestion pipeline (SPEC §4.2).

Decides whether a router-loaded document is an ARTICLE or BIBLE text via a cheap
heuristic cascade, with an optional pluggable LLM fallback for ambiguous cases
(D7; not wired by default).

IMPORTANT design correction vs. the spec's first sketch
-------------------------------------------------------
The spec suggested "presence of chapter:verse patterns -> Bible". That is backwards:
scripture text does NOT cite itself — a KJV verse contains words like "In the
beginning...", not the string "Genesis 1:1". The documents dense with reference
strings are ARTICLES *about* scripture. So reference density is an ARTICLE signal
(a scripture-FOCUSED article), never a Bible signal, and content alone cannot
reliably identify raw scripture text.

Combined with this build's architecture — the KJV is ingested from a structured
JSON via BibleProcessor, never through the loader router — the reliable approach is:
  - Bible is recognized only by PATH / FILENAME (e.g. a /bible/ folder or
    "Genesis.txt"). High precision, intentionally low recall.
  - Everything else defaults to ARTICLE (correct: the corpus is ~all articles).
  - Reference density is surfaced as METADATA (scripture_focused + count), not type.
  - A doc that classifies as "bible" is FLAGGED: there is no raw-scripture-text
    processor here (Bible = JSON only), so it should be reviewed, not chunked.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from verses_rag.canon import KJV_BOOKS, extract_scripture_refs

log = logging.getLogger("classifier")

_BOOK_STEMS = {b.lower() for b in KJV_BOOKS}
_BIBLE_PATH_PARTS = {"bible", "scripture", "kjv"}

# Tunable: refs per 1k words above which an article is "scripture-focused".
SCRIPTURE_FOCUS_PER_1K = 3.0
SCRIPTURE_FOCUS_MIN_REFS = 2


@dataclass
class Classification:
    source_type: str  # "article" | "bible"
    confidence: float  # 0..1
    reason: str
    scripture_ref_count: int = 0
    scripture_focused: bool = False


def _norm_stem(stem: str) -> str:
    s = stem.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def _filename_is_book(path: Path) -> bool:
    """True if the filename names a canonical book.

    Checks the normalized stem directly (catches '1 Samuel', '1_samuel') and a
    version with a leading order-number stripped (catches '01_genesis', '40 matthew'),
    accepting either — so a real numbered book like '1 Samuel' isn't broken by the
    digit-stripping that handles zero-padded ordering prefixes.
    """
    norm = _norm_stem(path.stem)
    if norm in _BOOK_STEMS:
        return True
    stripped = re.sub(r"^\d+\s*", "", norm).strip()
    return stripped in _BOOK_STEMS


def classify(
    text: str,
    source_path: str = "",
    llm_fallback: Callable[[str], str] | None = None,
) -> Classification:
    """Classify one document. `llm_fallback(text) -> 'bible'|'article'` is optional."""
    path = Path(source_path) if source_path else Path()

    # Always compute density — it feeds metadata regardless of the type decision.
    refs = extract_scripture_refs(text)
    n_refs = len(refs)
    words = max(len(text.split()), 1)
    per_1k = n_refs / words * 1000
    focused = per_1k >= SCRIPTURE_FOCUS_PER_1K and n_refs >= SCRIPTURE_FOCUS_MIN_REFS

    def article(conf: float, reason: str) -> Classification:
        return Classification("article", conf, reason, n_refs, focused)

    # --- Cascade: the only reliable cheap Bible signals are path & filename. ---
    if source_path:
        if any(p.lower() in _BIBLE_PATH_PARTS for p in path.parts):
            return Classification(
                "bible", 0.95, "path is a Bible/scripture folder", n_refs, focused
            )
        if path.stem and _filename_is_book(path):
            return Classification(
                "bible",
                0.9,
                f"filename matches canonical book '{path.stem}'",
                n_refs,
                focused,
            )

    # --- Content can't identify raw scripture (verses don't self-cite) -> article. ---
    if len(text.split()) < 50:
        return article(0.5, "too little text to judge; defaulting to article")

    # Optional LLM tie-breaker for genuinely ambiguous cases (rarely needed here).
    if llm_fallback is not None:
        try:
            verdict = llm_fallback(text).strip().lower()
            if verdict in ("bible", "article"):
                return Classification(verdict, 0.7, "LLM fallback", n_refs, focused)
        except Exception as e:
            log.warning("llm_fallback failed (%s); using heuristic default", e)

    reason = "prose with scripture references" if focused else "general prose"
    return article(0.7, reason)


def classify_doc(
    doc, llm_fallback: Callable[[str], str] | None = None
) -> Classification:
    """Convenience for a loader_router.LoadedDoc (duck-typed: .text, .source_path)."""
    return classify(doc.text, doc.source_path, llm_fallback=llm_fallback)


def main():
    cases = [
        (
            "/corpus/articles/healing_words.txt",
            "The Healing Power of Words\n\nProverbs 12:18 reminds us about the tongue. "
            "As James 3:5 says, the tongue is small but boasts great things. "
            + "This is an article that reflects on those verses at length. "
            * 6,
        ),
        (
            "/corpus/bible/Genesis.txt",
            "In the beginning God created the heaven and the earth.",
        ),
        (
            "/corpus/articles/quarterly_update.md",
            "Our team shipped three features this quarter and grew revenue. " * 10,
        ),
        ("/corpus/bible/01_Genesis.txt", "In the beginning..."),
    ]
    for src, txt in cases:
        c = classify(txt, src)
        print(
            f"{Path(src).name:24}  -> {c.source_type:7} ({c.confidence:.2f})  "
            f"refs={c.scripture_ref_count} focused={c.scripture_focused}"
        )
        print(f"{'':24}     {c.reason}")
        if c.source_type == "bible":
            print(
                f"{'':24}     ⚠ no raw-scripture-text processor (Bible = JSON); flag for review"
            )
        print()


if __name__ == "__main__":
    main()
