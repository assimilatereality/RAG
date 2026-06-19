# =============================================================
# File: src/verses_rag/graph/nodes/verify.py
# =============================================================
"""
Verify node — seventh and final node in the RAG graph (SPEC §4.6, D7).

Faithfulness / hallucination check: compares the draft_answer against
the retrieved passages and decides whether every claim is grounded.

This is the most safety-critical judge role (§4.8.1). It is the last
line of defence before an answer reaches the user, so it always uses
the API judge (OpenAI primary / Anthropic backup via .with_fallbacks()).

Verdicts:
    "pass"    → draft_answer is faithful; answer = draft_answer
    "fail"    → answer contains unsupported claims; answer = ABSTAIN_PHRASE
    "abstain" → draft already self-abstained; answer = ABSTAIN_PHRASE

A "fail" verdict does not loop back — the grade→retrieve retry loop
already ran. Verify is the terminal backstop: if the answer isn't
faithful here, the system abstains rather than hallucinating.

Dependency injected via functools.partial (Option A):
    llm — RunnableWithFallbacks from get_llm("verify")

Graph wiring example:
    from functools import partial
    from verses_rag.llm.router import get_llm
    verify = partial(verify_node, llm=get_llm("verify"))

Run self-check (requires API keys in .env):
    uv run python -m verses_rag.graph.nodes.verify
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from verses_rag.graph.state import GraphState
from verses_rag.graph.nodes.generate import ABSTAIN_PHRASE

log = logging.getLogger("graph.verify")

_VERIFY_PROMPT = """\
You are a faithfulness checker for a retrieval-augmented generation system.

Task: verify that the factual claims in the Answer are supported by the \
Passages. Judge against the passages only — do not use outside knowledge.

Passages:
{passages}

Answer: {draft_answer}

Rules:
- "pass" : every claim is supported by the passages, either explicitly OR as a \
reasonable, direct inference from what the passages state. For example, if a \
passage says God instructed someone to do something and the narrative continues, \
treating it as done is a reasonable inference. Minor restatements and natural \
summary are acceptable.
- "fail" : the answer asserts a specific fact that is NOT in the passages and is \
NOT a reasonable inference from them (an invented detail, an outside source, a \
name/number/claim with no basis), or it contradicts the passages.
- "abstain" : the answer already states it cannot answer the question.

Judge the substance, not the phrasing. Do not fail an answer merely because it \
words something differently than the passage, or because it states a natural \
consequence of what the passage explicitly says. Reserve "fail" for claims that \
genuinely lack any basis in the passages.

Reply with a JSON object and nothing else:
{{"verdict": "pass" | "fail" | "abstain", "reason": "<one sentence>"}}
"""


# --- helpers -----------------------------------------------------------------

def _format_passages(ranked_docs: list[Any]) -> str:
    """Full passage text from ranked_docs (preferred over citation snippets)."""
    lines = []
    for i, doc in enumerate(ranked_docs[:3], 1):
        ref = _ref_label(doc.payload)
        lines.append(f"[{i}] ({ref})\n{doc.content[:400]}")
    return "\n\n".join(lines)


def _format_citations_fallback(citations: list[dict[str, Any]]) -> str:
    """Fallback when ranked_docs not in state: use citation snippets."""
    lines = []
    for i, c in enumerate(citations[:3], 1):
        lines.append(f"[{i}] ({c.get('ref', '?')})\n{c.get('content_snippet', '')}")
    return "\n\n".join(lines)


def _ref_label(payload: dict[str, Any]) -> str:
    if payload.get("source_type") == "bible":
        book = payload.get("book", "")
        ch   = payload.get("chapter", "")
        vs   = payload.get("verse_start", "")
        ve   = payload.get("verse_end", "")
        ref  = f"{book} {ch}:{vs}"
        return ref + f"-{ve}" if ve and ve != vs else ref
    return str(payload.get("title", payload.get("source_path", "passage")))[:60]


def _call_llm(llm: Any, draft: str, passages: str) -> tuple[str, str]:
    """Call the judge LLM. Returns (verdict, reason)."""
    prompt = _VERIFY_PROMPT.format(passages=passages, draft_answer=draft)
    response = llm.invoke(prompt)

    raw  = response.content if hasattr(response, "content") else response
    text = " ".join(str(i) for i in raw) if isinstance(raw, list) else str(raw)
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    try:
        data    = json.loads(text)
        verdict = data.get("verdict", "").lower()
        reason  = data.get("reason", "")
        if verdict in ("pass", "fail", "abstain"):
            return verdict, reason
        log.warning("unexpected verdict %r; defaulting to fail", verdict)
    except (json.JSONDecodeError, AttributeError) as e:
        log.warning("failed to parse verify response (%s): %r", e, text[:120])

    return "fail", "could not parse LLM response"


# --- node entry point --------------------------------------------------------

def verify_node(
    state: GraphState,
    *,
    llm: Any,
    settings=None,
) -> GraphState:
    """LangGraph node: faithfulness check → verdict + final answer.

    Returns a partial state update: verdict and answer.
    """
    draft       = state.get("draft_answer", "").strip()
    ranked_docs = state.get("ranked_docs", [])
    citations   = state.get("citations", [])

    # --- fast path: already abstained ---
    if not draft or ABSTAIN_PHRASE in draft:
        log.info("verify=abstain (draft already abstained)")
        return {  # type: ignore[return-value]
            "verdict": "abstain",
            "answer":  ABSTAIN_PHRASE,
        }

    # --- build passages for the prompt ---
    passages = (
        _format_passages(ranked_docs)
        if ranked_docs
        else _format_citations_fallback(citations)
    )

    if not passages.strip():
        log.warning("verify: no passages available — failing safe")
        return {  # type: ignore[return-value]
            "verdict": "fail",
            "answer":  ABSTAIN_PHRASE,
        }

    # --- LLM faithfulness check ---
    verdict, reason = _call_llm(llm, draft, passages)
    log.info("verify=%s | reason=%s", verdict, reason)

    answer = draft if verdict == "pass" else ABSTAIN_PHRASE
    return {  # type: ignore[return-value]
        "verdict": verdict,
        "answer":  answer,
    }


# --- self-check --------------------------------------------------------------

def main():
    import hashlib
    from functools import partial

    from verses_rag.config.settings import get_settings
    from verses_rag.llm.router import get_llm
    from verses_rag.retrieval.reranker import RankedHit

    def _sha(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    def _hit(ref: str, content: str, book: str, ch: int, vs: int) -> RankedHit:
        return RankedHit(
            chunk_id=_sha(ref),
            payload={"content": content, "book": book, "chapter": ch,
                     "verse_start": vs, "verse_end": vs, "source_type": "bible"},
            retrieval_score=0.03,
            rerank_score=-3.6,
        )

    ranked_docs = [
        _hit("John 3:16",
             "For God so loved the world, that he gave his only begotten Son, "
             "that whosoever believeth in him should not perish, but have "
             "everlasting life.",
             "John", 3, 16),
        _hit("Romans 5:8",
             "But God commendeth his love toward us, in that, while we were "
             "yet sinners, Christ died for us.",
             "Romans", 5, 8),
        _hit("1 John 4:8",
             "He that loveth not knoweth not God; for God is love.",
             "1 John", 4, 8),
    ]

    s   = get_settings()
    llm = get_llm("verify", s)
    verify = partial(verify_node, llm=llm, settings=s)

    cases = [
        (
            "faithful answer — expect: pass",
            "The Bible teaches that God's love is demonstrated through sacrifice. "
            "[1] John 3:16 states God gave His only Son so believers have everlasting life. "
            "[2] Romans 5:8 shows Christ died for us while we were yet sinners. "
            "[3] 1 John 4:8 declares that God is love.",
        ),
        (
            "hallucinated answer — expect: fail",
            "The Bible says God's love is unconditional and that He loves all people equally "
            "regardless of their actions. Jesus taught that we should love our enemies, "
            "and Paul wrote extensively about agape love in his letter to the Corinthians.",
        ),
        (
            "pre-abstained — expect: abstain",
            ABSTAIN_PHRASE,
        ),
    ]

    print("=== verify node self-check ===\n")
    for note, draft in cases:
        state: GraphState = {  # type: ignore[misc]
            "draft_answer": draft,
            "ranked_docs":  ranked_docs,
            "citations":    [],
        }
        result = verify(state)
        verdict = result.get("verdict", "?")
        expected = note.split("expect: ")[1]
        marker = "✓" if verdict == expected else "✗"
        print(f"{marker} {note}")
        print(f"  verdict : {verdict}")
        ans = result.get("answer", "")
        print(f"  answer  : {ans[:100]}{'…' if len(ans) > 100 else ''}")
        print()


if __name__ == "__main__":
    main()