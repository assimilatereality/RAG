# =============================================================
# File: src/verses_rag/eval/generation_metrics.py
# =============================================================
"""
Generation metrics for the RAG eval harness (SPEC §4.9).

Two LLM-as-judge metrics:
  faithfulness    — are all claims in the answer supported by the retrieved
                    passages? (0.0–1.0)
  answer_relevance — does the answer actually address the question?
                    (0.0–1.0)

Both use the same judge LLM as the verify node (OpenAI primary / Anthropic
backup) to keep scoring consistent with the pipeline's own quality bar.
Scores are floats so they aggregate cleanly into means for the eval report.

A separate rule-based abstention check requires no LLM:
  is_abstention — True if the answer is the standard abstain phrase.

Run self-check (requires API keys in .env):
    uv run python -m verses_rag.eval.generation_metrics
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from verses_rag.graph.nodes.generate import ABSTAIN_PHRASE

log = logging.getLogger("eval.generation")

# --- prompts -----------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are evaluating a RAG system's answer for faithfulness to its sources.

Question: {query}

Retrieved passages:
{passages}

Answer: {answer}

Score the answer on faithfulness: does every factual claim in the answer \
have explicit support in the passages above? Do not use outside knowledge.

Score: 1.0 = fully faithful (every claim supported)
       0.5 = partially faithful (some claims unsupported)
       0.0 = unfaithful (answer contradicts or ignores the passages)

Reply with JSON only:
{{"score": 0.0–1.0, "reason": "<one sentence>"}}
"""

_RELEVANCE_PROMPT = """\
You are evaluating whether an answer addresses the question asked.

Question: {query}

Answer: {answer}

Score the answer on relevance: does it directly and completely address \
what was asked?

Score: 1.0 = fully relevant (answers the question completely)
       0.5 = partially relevant (addresses part of the question)
       0.0 = irrelevant (does not address the question)

Note: if the answer says it cannot find the information, score 0.5 \
(it acknowledged the question but couldn't answer it).

Reply with JSON only:
{{"score": 0.0–1.0, "reason": "<one sentence>"}}
"""


# --- helpers -----------------------------------------------------------------

def _format_passages(citations: list[dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(citations, 1):
        lines.append(f"[{i}] ({c.get('ref', '?')})\n{c.get('content_snippet', '')}")
    return "\n\n".join(lines) if lines else "(no passages)"


def _format_ranked_docs(ranked_docs: list[Any]) -> str:
    """Full passage text from ranked_docs — preferred over citation snippets,
    which are truncated to ~120 chars and make the judge under-score answers
    whose support runs past the cutoff."""
    lines = []
    for i, doc in enumerate(ranked_docs, 1):
        ref = ""
        payload = getattr(doc, "payload", {}) or {}
        if payload.get("source_type") == "bible":
            ref = f"{payload.get('book','')} {payload.get('chapter','')}:{payload.get('verse_start','')}"
        else:
            ref = str(payload.get("title", "passage"))[:60]
        content = getattr(doc, "content", "")
        lines.append(f"[{i}] ({ref})\n{content[:400]}")
    return "\n\n".join(lines) if lines else "(no passages)"


def _call_judge(llm: Any, prompt: str) -> tuple[float, str]:
    """Call judge LLM, return (score, reason). Returns (0.0, error) on failure."""
    try:
        response = llm.invoke(prompt)
        raw  = response.content if hasattr(response, "content") else response
        text = " ".join(str(i) for i in raw) if isinstance(raw, list) else str(raw)
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        data = json.loads(text)
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))   # clamp to [0, 1]
        return score, data.get("reason", "")
    except Exception as e:
        log.warning("judge call failed: %s", e)
        return 0.0, f"error: {e}"


# --- per-answer metrics ------------------------------------------------------

def is_abstention(answer: str) -> bool:
    """True if the answer is the standard abstain phrase (rule-based, no LLM)."""
    return ABSTAIN_PHRASE in answer.strip()


def faithfulness_score(
    query: str,
    answer: str,
    citations: list[dict[str, Any]],
    llm: Any,
    ranked_docs: list[Any] | None = None,
) -> tuple[float, str]:
    """Score faithfulness of answer against retrieved passages (0.0–1.0).

    Returns (score, reason). Abstentions score 1.0 — they make no claims.

    Prefers full `ranked_docs` text when provided; falls back to citation
    snippets otherwise. Full text matters: snippets are truncated to ~120
    chars and cause the judge to under-score answers grounded in the part
    of the passage past the cutoff.
    """
    if is_abstention(answer):
        return 1.0, "abstention makes no claims"

    passages = (
        _format_ranked_docs(ranked_docs)
        if ranked_docs
        else _format_passages(citations)
    )
    prompt = _FAITHFULNESS_PROMPT.format(
        query    = query,
        passages = passages,
        answer   = answer,
    )
    return _call_judge(llm, prompt)


def answer_relevance_score(
    query: str,
    answer: str,
    llm: Any,
) -> tuple[float, str]:
    """Score how well the answer addresses the question (0.0–1.0).

    Returns (score, reason). Abstentions score 0.5 per prompt instructions.
    """
    prompt = _RELEVANCE_PROMPT.format(query=query, answer=answer)
    return _call_judge(llm, prompt)


# --- aggregate ---------------------------------------------------------------

@dataclass
class GenerationReport:
    """Aggregated generation metrics across a query set."""

    n_cases:              int
    n_abstentions:        int
    mean_faithfulness:    float
    mean_relevance:       float
    abstention_rate:      float

    def __str__(self) -> str:
        return (
            f"n={self.n_cases}  abstentions={self.n_abstentions} "
            f"({self.abstention_rate:.1%})  "
            f"faithfulness={self.mean_faithfulness:.3f}  "
            f"relevance={self.mean_relevance:.3f}"
        )


def compute_generation_report(
    scores: list[tuple[float, float, bool]],
) -> GenerationReport:
    """Aggregate from a list of (faithfulness, relevance, is_abstention) tuples."""
    n = len(scores)
    if n == 0:
        return GenerationReport(0, 0, 0.0, 0.0, 0.0)
    abstentions = sum(1 for _, _, a in scores if a)
    return GenerationReport(
        n_cases           = n,
        n_abstentions     = abstentions,
        mean_faithfulness = sum(f for f, _, _ in scores) / n,
        mean_relevance    = sum(r for _, r, _ in scores) / n,
        abstention_rate   = abstentions / n,
    )


# --- self-check --------------------------------------------------------------

def main():
    from verses_rag.config.settings import get_settings
    from verses_rag.llm.router import get_llm

    s   = get_settings()
    llm = get_llm("verify", s)

    citations = [
        {"ref": "John 3:16",
         "content_snippet": "For God so loved the world, that he gave his only "
                            "begotten Son, that whosoever believeth in him should "
                            "not perish, but have everlasting life."},
        {"ref": "Romans 5:8",
         "content_snippet": "But God commendeth his love toward us, in that, while "
                            "we were yet sinners, Christ died for us."},
    ]

    query = "What does the Bible say about God's love?"

    cases = [
        (
            "faithful answer",
            "God's love is shown through sacrifice. [1] John 3:16 states that "
            "God gave His only Son for the world's salvation. [2] Romans 5:8 "
            "shows Christ died for us while we were still sinners.",
        ),
        (
            "hallucinated answer",
            "God's love is unconditional and eternal. The Bible teaches that "
            "God loves everyone equally regardless of their deeds, as stated "
            "in Psalm 136 where 'his mercy endureth for ever' is repeated.",
        ),
        (
            "abstention",
            ABSTAIN_PHRASE,
        ),
    ]

    print("=== generation metrics self-check ===\n")
    all_scores = []
    for label, answer in cases:
        f_score, f_reason = faithfulness_score(query, answer, citations, llm)
        r_score, r_reason = answer_relevance_score(query, answer, llm)
        abstain = is_abstention(answer)
        all_scores.append((f_score, r_score, abstain))
        print(f"[{label}]")
        print(f"  faithfulness : {f_score:.2f}  — {f_reason}")
        print(f"  relevance    : {r_score:.2f}  — {r_reason}")
        print(f"  abstention   : {abstain}")
        print()

    report = compute_generation_report(all_scores)
    print(f"Aggregate: {report}")


if __name__ == "__main__":
    main()