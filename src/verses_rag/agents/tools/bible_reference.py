# =============================================================
# File: src/verses_rag/agents/tools/bible_reference.py
# =============================================================
"""
Bible reference resolver tool (SPEC §4.7).

Wraps canon.extract_scripture_refs as a LangChain tool so agents can
normalize messy reference strings ("Gen 1:1", "Psa 23", "1 Cor 13:4")
to canonical form ("Genesis 1:1", "Psalms 23", "1 Corinthians 13:4")
before passing them to the retrieval layer.

Also exposes a standalone resolve() function for direct use without the
LangChain tool wrapper (e.g., in the graph's analyze_filters node).

Run self-check:
    uv run python -m verses_rag.agents.tools.bible_reference
"""

from __future__ import annotations

from langchain_core.tools import tool

from verses_rag.canon import extract_scripture_refs, BOOK_ORDER


# --- standalone resolver -----------------------------------------------------

def resolve(reference: str) -> str | None:
    """Normalize a Bible reference string to canonical form.

    Returns the canonical reference string (e.g. "Genesis 1:1") or None
    if the input cannot be parsed.

    Examples:
        resolve("Gen 1:1")         -> "Genesis 1:1"
        resolve("Psa 23:1-6")      -> "Psalms 23:1-6"
        resolve("1 Cor 13:4")      -> "1 Corinthians 13:4"
        resolve("not a reference") -> None
    """
    refs = extract_scripture_refs(reference)
    return refs[0] if refs else None


def resolve_many(text: str) -> list[str]:
    """Extract and normalize all Bible references found in a text string."""
    return extract_scripture_refs(text)


def is_valid_reference(reference: str) -> bool:
    """Return True if the string parses as a known canonical reference."""
    return resolve(reference) is not None


# --- LangChain tool ----------------------------------------------------------

@tool
def resolve_bible_reference(reference: str) -> str:
    """Normalize a Bible reference string to its canonical form.

    Use this tool when you have a potentially abbreviated or non-standard
    Bible reference and need the canonical form for retrieval.

    Args:
        reference: A Bible reference string, e.g. "Gen 1:1", "Psa 23",
                   "1 Cor 13:4-7", "Romans 8:28-30".

    Returns:
        The canonical reference string, or an error message if unparseable.
    """
    canonical = resolve(reference)
    if canonical:
        return canonical
    return f"Could not parse '{reference}' as a Bible reference. " \
           f"Use full book names (e.g. 'Genesis 1:1') or known abbreviations."


# --- self-check --------------------------------------------------------------

def main():
    cases = [
        # (input, expected_canonical_or_None)
        ("Gen 1:1",            "Genesis 1:1"),
        ("Psa 23:1-6",         "Psalms 23:1-6"),
        ("1 Cor 13:4",         "1 Corinthians 13:4"),
        ("Rom 8:28",           "Romans 8:28"),
        ("Rev 22:21",          "Revelation 22:21"),
        ("Song 2:1",           "Song of Solomon 2:1"),
        ("Jn 3:16",            "John 3:16"),
        ("not a reference",    None),
        ("Chapter 4 policy",   None),
        ("Genesis 1:1",        "Genesis 1:1"),  # already canonical
    ]

    print("=== bible reference resolver self-check ===\n")
    print(f"{'Input':<25} {'Expected':<25} {'Got':<25} {'Pass'}")
    print("-" * 85)

    for ref, expected in cases:
        got = resolve(ref)
        marker = "✓" if got == expected else "✗"
        print(f"{marker} {ref:<23} {str(expected):<25} {str(got):<25}")

    print("\n--- LangChain tool ---")
    print(resolve_bible_reference.invoke("Gen 1:1"))
    print(resolve_bible_reference.invoke("not a reference"))


if __name__ == "__main__":
    main()