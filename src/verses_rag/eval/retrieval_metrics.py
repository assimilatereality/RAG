# =============================================================
# File: src/verses_rag/eval/retrieval_metrics.py
# =============================================================
"""
Retrieval metrics for the RAG eval harness (SPEC §4.9).

Three standard IR metrics computed against ranked hits:
  hit@k      — did at least one relevant chunk appear in the top k?
  recall@k   — what fraction of relevant refs appeared in the top k?
  MRR        — Mean Reciprocal Rank across a query set

All per-query functions are pure: they take a list of hit-like objects
(duck-typed: .payload dict) and a list of canonical ref strings, and
return a float or bool. No store or LLM required.

Ref matching: a chunk "covers" a reference if its book, chapter, and
verse window contain the reference's verse(s). Window chunks (verse_start
to verse_end) cover any single verse that falls within the window.

Run self-check (no store or API needed):
    uv run python -m verses_rag.eval.retrieval_metrics
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# --- ref parsing -------------------------------------------------------------

_REF_RE = re.compile(r"^(.+?)\s+(\d+):(\d+)(?:-(\d+))?$")


def _parse_ref(ref: str) -> tuple[str, int, int, int] | None:
    """Parse 'Book C:V' or 'Book C:V-W' → (book, chapter, verse_start, verse_end)."""
    m = _REF_RE.match(ref.strip())
    if not m:
        return None
    book = m.group(1)
    ch   = int(m.group(2))
    vs   = int(m.group(3))
    ve   = int(m.group(4)) if m.group(4) else vs
    return (book, ch, vs, ve)


def _chunk_covers_ref(payload: dict[str, Any], ref: str) -> bool:
    """True if the chunk payload's verse window covers the reference."""
    parsed = _parse_ref(ref)
    if not parsed:
        return False
    book, chapter, ref_vs, ref_ve = parsed
    return (
        payload.get("book", "").lower() == book.lower()
        and payload.get("chapter") == chapter
        and payload.get("verse_start", 0) <= ref_vs
        and payload.get("verse_end",   0) >= ref_ve
    )


# --- per-query metrics -------------------------------------------------------

def hit_at_k(hits: list[Any], relevant_refs: list[str], k: int) -> bool:
    """True if at least one relevant ref appears in the top-k hits."""
    if not relevant_refs:
        return False
    for hit in hits[:k]:
        for ref in relevant_refs:
            if _chunk_covers_ref(hit.payload, ref):
                return True
    return False


def recall_at_k(hits: list[Any], relevant_refs: list[str], k: int) -> float:
    """Fraction of relevant refs covered by the top-k hits (0.0–1.0)."""
    if not relevant_refs:
        return 0.0
    found = sum(
        1 for ref in relevant_refs
        if any(_chunk_covers_ref(h.payload, ref) for h in hits[:k])
    )
    return found / len(relevant_refs)


def reciprocal_rank(hits: list[Any], relevant_refs: list[str]) -> float:
    """1 / rank of the first relevant hit, or 0.0 if none found."""
    if not relevant_refs:
        return 0.0
    for rank, hit in enumerate(hits, start=1):
        if any(_chunk_covers_ref(hit.payload, ref) for ref in relevant_refs):
            return 1.0 / rank
    return 0.0


# --- aggregate metrics -------------------------------------------------------

@dataclass
class RetrievalReport:
    """Aggregated retrieval metrics across a query set."""

    n_cases:      int
    hit_at_1:     float   # fraction with a relevant hit in position 1
    hit_at_3:     float
    hit_at_5:     float
    recall_at_3:  float   # avg fraction of relevant refs found in top-3
    recall_at_5:  float
    mrr:          float   # mean reciprocal rank

    def __str__(self) -> str:
        return (
            f"n={self.n_cases}  "
            f"hit@1={self.hit_at_1:.3f}  hit@3={self.hit_at_3:.3f}  hit@5={self.hit_at_5:.3f}  "
            f"recall@3={self.recall_at_3:.3f}  recall@5={self.recall_at_5:.3f}  "
            f"MRR={self.mrr:.3f}"
        )


def compute_retrieval_report(
    results: list[tuple[list[Any], list[str]]],
) -> RetrievalReport:
    """Compute aggregate metrics from a list of (hits, relevant_refs) pairs.

    Only cases with non-empty relevant_refs are included in the averages —
    abstain / out-of-corpus cases contribute nothing to retrieval metrics.

    Args:
        results: list of (ranked_hits, relevant_refs) pairs, one per query.
    """
    scored = [(hits, refs) for hits, refs in results if refs]
    n = len(scored)
    if n == 0:
        return RetrievalReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return RetrievalReport(
        n_cases     = n,
        hit_at_1    = sum(hit_at_k(h, r, 1) for h, r in scored) / n,
        hit_at_3    = sum(hit_at_k(h, r, 3) for h, r in scored) / n,
        hit_at_5    = sum(hit_at_k(h, r, 5) for h, r in scored) / n,
        recall_at_3 = sum(recall_at_k(h, r, 3) for h, r in scored) / n,
        recall_at_5 = sum(recall_at_k(h, r, 5) for h, r in scored) / n,
        mrr         = sum(reciprocal_rank(h, r) for h, r in scored) / n,
    )


# --- self-check --------------------------------------------------------------

def main():
    from dataclasses import dataclass as _dc

    @_dc
    class FakeHit:
        payload: dict

    def _h(book: str, ch: int, vs: int, ve: int) -> FakeHit:
        return FakeHit({"book": book, "chapter": ch,
                        "verse_start": vs, "verse_end": ve})

    # ranked hits (best first)
    hits = [
        _h("John",    3, 14, 18),   # window covering 3:14-18 → covers 3:16
        _h("Romans",  5,  6, 10),   # window covering 5:6-10  → covers 5:8
        _h("Psalms", 23,  1,  5),   # window covering 23:1-5  → covers 23:1
        _h("Genesis", 1,  1,  5),
        _h("Proverbs",3,  3,  7),   # window covering 3:3-7   → covers 3:5
    ]

    cases = [
        ("John 3:16 in position 1",   ["John 3:16"],   True,  1.0, 1.0),
        ("Romans 5:8 in position 2",  ["Romans 5:8"],  False, 1.0, 1.0),
        ("Two refs, both present",    ["John 3:16", "Romans 5:8"], True, 1.0, 1.0),
        ("Ref not in hits at all",    ["Matthew 5:1"], False, 0.0, 0.0),
        ("Psalms 23:1 in position 3", ["Psalms 23:1"], False, 1.0, 1.0),
    ]

    print("=== retrieval metrics self-check ===\n")

    print("--- per-query ---")
    for desc, refs, exp_hit1, exp_recall3, exp_recall5 in cases:
        h1  = hit_at_k(hits, refs, 1)
        h3  = hit_at_k(hits, refs, 3)
        r3  = recall_at_k(hits, refs, 3)
        rr  = reciprocal_rank(hits, refs)
        ok1 = "✓" if h1 == exp_hit1 else "✗"
        ok3 = "✓" if abs(r3 - exp_recall3) < 0.01 else "✗"
        print(f"  {desc}")
        print(f"    hit@1={h1} {ok1}  hit@3={h3}  recall@3={r3:.2f} {ok3}  RR={rr:.3f}")

    print("\n--- aggregate (4 cases with refs) ---")
    agg_input = [
        (hits, ["John 3:16"]),
        (hits, ["Romans 5:8"]),
        (hits, ["Psalms 23:1"]),
        (hits, ["Matthew 5:1"]),   # miss
    ]
    report = compute_retrieval_report(agg_input)
    print(f"  {report}")


if __name__ == "__main__":
    main()