# =============================================================
# File: src/verses_rag/eval/dataset.py
# =============================================================
"""
Evaluation dataset for the RAG pipeline (SPEC §4.9).

An EvalCase defines one test: what the query is, which chunks should be
retrieved, what the pipeline should ultimately do (pass or abstain), and
which source type it targets.

Two datasets are provided:
  SAMPLE_CASES  — works with the 6-verse InMemoryStore used in self-checks.
                  Use for fast CI-level smoke tests.
  KJV_CASES     — requires the full indexed KJV corpus. Use for real eval.

Extending: add EvalCase entries to KJV_CASES as you identify failure modes
or want coverage of specific books / query types.

Run self-check (no store or API needed):
    uv run python -m verses_rag.eval.dataset
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    """One evaluation example.

    Fields:
        id:               Unique string identifier (used in reports).
        query:            The user's question as it would be typed.
        relevant_refs:    Canonical references (e.g. "John 3:16") that
                          should appear in the retrieved candidates.
                          Matched against chunk payload book/chapter/verse.
        source_type:      Expected source_route: "bible" | "article" | "mixed".
        expected_verdict: "pass" if the pipeline should produce an answer,
                          "abstain" if it should abstain (e.g. no relevant
                          content in corpus).
        notes:            What this case is testing (for report readability).
    """

    id:               str
    query:            str
    relevant_refs:    list[str]
    source_type:      str
    expected_verdict: str          # "pass" | "abstain"
    notes:            str = ""


# ---------------------------------------------------------------------------
# Sample dataset — works with the 6-verse InMemoryStore smoke-test corpus
# ---------------------------------------------------------------------------

SAMPLE_CASES: list[EvalCase] = [
    EvalCase(
        id="S01",
        query="What does the Bible say about God's love?",
        relevant_refs=["John 3:16", "Romans 5:8"],
        source_type="bible",
        expected_verdict="pass",
        notes="Core retrieval: two directly relevant verses in corpus",
    ),
    EvalCase(
        id="S02",
        query="The LORD as shepherd",
        relevant_refs=["Psalms 23:1"],
        source_type="bible",
        expected_verdict="pass",
        notes="Exact-match retrieval with book filter",
    ),
    EvalCase(
        id="S03",
        query="Trust in the LORD with all your heart",
        relevant_refs=["Proverbs 3:5"],
        source_type="bible",
        expected_verdict="pass",
        notes="Thematic query close to KJV wording — tests dense retrieval on Proverbs 3:5",
    ),
    EvalCase(
        id="S04",
        query="What does the New Testament say about the resurrection of the dead?",
        relevant_refs=[],   # no resurrection verses in the 6-verse corpus
        source_type="bible",
        expected_verdict="abstain",
        notes="Correct abstention: relevant content not in small corpus",
    ),
    
]

# ---------------------------------------------------------------------------
# KJV dataset — requires full indexed corpus
# ---------------------------------------------------------------------------

KJV_CASES: list[EvalCase] = [
    # --- Bible retrieval ---
    EvalCase(
        id="K01",
        query="What does John 3:16 say?",
        relevant_refs=["John 3:16", "1 John 4:9"],
        source_type="bible",
        expected_verdict="pass",
        notes="Verse lookup by reference — tests bypass and retrieval of John 3:16",
    ),
    EvalCase(
        id="K02",
        query="What does Romans say about justification by faith?",
        relevant_refs=["Romans 3:28", "Romans 5:1"],
        source_type="bible",
        expected_verdict="pass",
        notes="Doctrinal query requiring semantic retrieval across Romans",
    ),
    EvalCase(
        id="K03",
        query="The fruit of the Spirit",
        relevant_refs=["Galatians 5:22"],
        source_type="bible",
        expected_verdict="pass",
        notes="Canonical phrase retrieval",
    ),
    EvalCase(
        id="K04",
        query="Show me Psalm 23:1-6",
        relevant_refs=["Psalms 23:1", "Psalms 23:2", "Psalms 23:3",
                       "Psalms 23:4", "Psalms 23:5", "Psalms 23:6"],
        source_type="bible",
        expected_verdict="pass",
        notes="Verse-range query. analyze_filters sets book=Psalms, chapter=23 "
              "(NOT verse_start/end — those would over-constrain). NOTE: an "
              "unfiltered retrieval probe will NOT surface Psalm 23 for this "
              "query; it only retrieves correctly *with* the book/chapter filter "
              "the pipeline applies. Refs are correct as-is.",
    ),
    EvalCase(
        id="K05",
        query="What does Proverbs say about the tongue?",
        relevant_refs=["Proverbs 12:18", "Proverbs 18:21"],
        source_type="bible",
        expected_verdict="pass",
        notes="Topical query within a single book",
    ),
    EvalCase(
        id="K06",
        query="Compare what Genesis and Romans say about sin",
        relevant_refs=["Genesis 3:6", "Romans 3:23", "Romans 5:12"],
        source_type="mixed",
        expected_verdict="pass",
        notes="Decomposition: two sub-queries, both should find relevant content",
    ),
    EvalCase(
        id="K07",
        query="What does the New Testament say about the resurrection?",
        relevant_refs=["1 Corinthians 15:20", "Romans 6:5"],
        source_type="bible",
        expected_verdict="pass",
        notes="Testament filter + topical query",
    ),
    # --- Article retrieval (requires article corpus indexed) ---
    EvalCase(
        id="K08",
        query="What does the article say about the importance of a single word?",
        relevant_refs=[],  # article refs not verse-based; verdict tested only
        source_type="article",
        expected_verdict="pass",
        notes="Article route: targets the 'One Powerful Word' article (Numbers 13 "
              "anchor). Tests article retrieval + citation. ADJUST query to match "
              "an article actually in your corpus.",
    ),
    # --- Expected abstentions ---
    EvalCase(
        id="K09",
        query="What does the Quran say about Jesus?",
        relevant_refs=[],
        source_type="bible",
        expected_verdict="abstain",
        notes="Out-of-corpus: Quran not indexed; pipeline should abstain",
    ),
    EvalCase(
        id="K10",
        query="What is the current stock price of Apple?",
        relevant_refs=[],
        source_type="mixed",
        expected_verdict="abstain",
        notes="Completely off-topic: pipeline should abstain cleanly",
    ),
    # --- Inference cases (calibrate the loosened verify prompt) ---
    # These require the answer to state a reasonable inference from the passage,
    # not just verbatim text. Before the verify-prompt loosening these tended to
    # FAIL verification (pedantic literal reading); after, they should PASS.
    EvalCase(
        id="K11",
        query="Did Noah build the ark as God commanded?",
        relevant_refs=["Genesis 6:14", "Genesis 6:22"],
        source_type="bible",
        expected_verdict="pass",
        notes="Inference: 'God instructed + narrative continues' → Noah built it. "
              "Genesis 6:22 ('thus did Noah') supports the inference explicitly.",
    ),
    EvalCase(
        id="K12",
        query="What did God create on the first day?",
        relevant_refs=["Genesis 1:3", "Genesis 1:5"],
        source_type="bible",
        expected_verdict="pass",
        notes="Inference: 'let there be light' + 'first day' → light created day one, "
              "though no single verse says 'on the first day God created light'.",
    ),
    EvalCase(
        id="K13",
        query="Was Abraham willing to sacrifice his son?",
        relevant_refs=["Genesis 22:9", "Genesis 22:10"],
        source_type="bible",
        expected_verdict="pass",
        notes="Inference: binding Isaac + stretching forth the knife → willingness, "
              "without a verse stating 'Abraham was willing'.",
    ),
    # --- Verse-lookup bypass cases ---
    EvalCase(
        id="VL01",
        query="Romans 8:1",
        relevant_refs=["Romans 8:1"],
        source_type="bible",
        expected_verdict="pass",
        notes="Single verse lookup — bypass should return verbatim chunk, no LLM generation",
    ),
    EvalCase(
        id="VL02",
        query="Romans 8:1-5",
        relevant_refs=["Romans 8:1"],
        source_type="bible",
        expected_verdict="pass",
        notes="Verse range lookup — Range filter overlap should surface the 1-5 window",
    ),
    EvalCase(
        id="VL03",
        query="Romans 8:28",
        relevant_refs=["Romans 8:28"],
        source_type="bible",
        expected_verdict="pass",
        notes="Single verse mid-window — verse 28 lives inside the 25-29 chunk; "
              "grader must not reject based on range label",
    ),
    EvalCase(
        id="VL04",
        query="What does Romans 8:28 say?",
        relevant_refs=["Romans 8:28"],
        source_type="bible",
        expected_verdict="pass",
        notes="Lookup-intent prefix — _is_verse_lookup should detect and bypass LLM",
    ),
    EvalCase(
        id="VL05",
        query="Genesis 1:1-3",
        relevant_refs=["Genesis 1:1"],
        source_type="bible",
        expected_verdict="pass",
        notes="Range lookup in Genesis — first chapter, first window",
    ),

    # --- Chapter-only filter cases ---
    EvalCase(
        id="CH01",
        query="Is Beth-el mentioned in Genesis 13?",
        relevant_refs=["Genesis 13:3"],
        source_type="bible",
        expected_verdict="pass",
        notes="Chapter-only filter — 'Genesis 13' should set chapter=13 filter",
    ),
    EvalCase(
        id="CH02",
        query="What happens in Genesis chapter 13?",
        relevant_refs=["Genesis 13:1"],
        source_type="bible",
        expected_verdict="pass",
        notes="Explicit 'chapter N' keyword — chapter filter extraction",
    ),
    EvalCase(
        id="CH03",
        query="Tell me about Genesis 13",
        relevant_refs=["Genesis 13:1"],
        source_type="bible",
        expected_verdict="pass",
        notes="Bare 'Book N' pattern — chapter filter without keyword",
    ),
]


def load_cases(dataset: str = "sample") -> list[EvalCase]:
    """Return the requested dataset.

    Args:
        dataset: "sample" (6-verse InMemoryStore) or "kjv" (full corpus).
    """
    if dataset == "sample":
        return list(SAMPLE_CASES)
    if dataset == "kjv":
        return list(KJV_CASES)
    raise ValueError(f"Unknown dataset {dataset!r}. Choose 'sample' or 'kjv'.")


# --- self-check --------------------------------------------------------------

def main():
    print("=== eval dataset self-check ===\n")
    for name, cases in [("sample", SAMPLE_CASES), ("kjv", KJV_CASES)]:
        print(f"--- {name} ({len(cases)} cases) ---")
        passes   = sum(1 for c in cases if c.expected_verdict == "pass")
        abstains = sum(1 for c in cases if c.expected_verdict == "abstain")
        print(f"  pass={passes}  abstain={abstains}")
        for c in cases:
            refs = ", ".join(c.relevant_refs[:3])
            if len(c.relevant_refs) > 3:
                refs += f" (+{len(c.relevant_refs)-3} more)"
            print(f"  [{c.id}] {c.query[:50]:<50}  refs=[{refs}]")
        print()


if __name__ == "__main__":
    main()