# =============================================================
# File: src/verses_rag/graph/nodes/grade_documents.py
# =============================================================
"""
Grade-documents node — fifth node in the RAG graph (SPEC §4.6, D7).

CRAG-style relevance gate: decides whether ranked_docs contain enough
context to answer the query, or whether to loop back for re-retrieval.

Two-stage cascade:
  1. Score threshold — if the best rerank score is below
     settings.grade.score_threshold, grade "insufficient" without an
     LLM call. Saves API cost for clearly bad retrievals.
  2. LLM judge — OpenAI primary / Anthropic backup (D7) grades the
     top-k docs as a set and returns a structured verdict.

State transitions (handled by graph edges, not this node):
  "sufficient"   → generate node
  "insufficient" + retry_count < max_retries → transform_query → retrieve
  "insufficient" + retry_count >= max_retries → abstain (set by graph)

Dependency injected via functools.partial (Option A):
    llm — RunnableWithFallbacks from get_llm("grade")

Graph wiring example:
    from functools import partial
    from verses_rag.llm.router import get_llm
    grade = partial(grade_node, llm=get_llm("grade"))

Run self-check (requires API keys in .env):
    uv run python -m verses_rag.graph.nodes.grade_documents
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from verses_rag.graph.state import GraphState

log = logging.getLogger("graph.grade")

# How many ranked docs to include in the grading prompt.
_DOCS_TO_GRADE = 3

_GRADE_PROMPT = """\
You are grading whether retrieved documents contain sufficient context to \
answer a user query.

Query: {query}

Retrieved documents (ranked by relevance):
{docs}

Task: decide if these documents together contain enough information to \
give a faithful, grounded answer to the query.

Rules:
- "sufficient" means the document content contains the information needed \
to answer the query. For scripture reference queries (e.g. "Romans 8:28"), \
a document is sufficient if its text contains the requested verse, even if \
the document's verse range label spans additional verses (e.g. a chunk \
labelled "Romans 8:25-29" contains Romans 8:28 in its text).
- "insufficient" means the documents are off-topic, too vague, or missing \
key information needed to answer. Do not mark a document insufficient \
merely because its range label does not exactly match the requested verse.

Reply with a JSON object and nothing else:
{{"verdict": "sufficient" | "insufficient", "reason": "<one sentence>"}}
"""


def _format_docs(ranked_docs: list[Any]) -> str:
    lines = []
    for i, doc in enumerate(ranked_docs[:_DOCS_TO_GRADE], 1):
        ref = _ref_label(doc.payload)
        lines.append(f"[{i}] ({ref}) {doc.content[:600]}")
    return "\n".join(lines)


def _ref_label(payload: dict[str, Any]) -> str:
    """Build a human-readable reference label from chunk payload."""
    src = payload.get("source_type", "")
    if src == "bible":
        book = payload.get("book", "")
        ch   = payload.get("chapter", "")
        vs   = payload.get("verse_start", "")
        ve   = payload.get("verse_end", "")
        ref  = f"{book} {ch}:{vs}"
        return ref + f"-{ve}" if ve and ve != vs else ref
    # article
    title = payload.get("title", payload.get("source_path", "article"))
    return str(title)[:60]


def _call_llm(llm: Any, query: str, ranked_docs: list[Any]) -> tuple[str, str]:
    """Call the judge LLM. Returns (verdict, reason)."""
    prompt = _GRADE_PROMPT.format(
        query=query,
        docs=_format_docs(ranked_docs),
    )
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else response
    text = " ".join(str(item) for item in raw) if isinstance(raw, list) else str(raw)
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    try:
        data = json.loads(text)
        verdict = data.get("verdict", "").lower()
        reason  = data.get("reason", "")
        if verdict in ("sufficient", "insufficient"):
            return verdict, reason
        log.warning("unexpected verdict %r from LLM; defaulting to insufficient", verdict)
    except (json.JSONDecodeError, AttributeError) as e:
        log.warning("failed to parse LLM grade response (%s): %r", e, text[:120])

    return "insufficient", "could not parse LLM response"


# --- node entry point --------------------------------------------------------

def grade_node(
    state: GraphState,
    *,
    llm: Any,
    settings=None,
) -> GraphState:
    """LangGraph node: grade ranked_docs relevance → grade_verdict + retry_count.

    Returns a partial state update: grade_verdict and retry_count.
    """
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    query       = state.get("query", "").strip()
    ranked_docs = state.get("ranked_docs", [])
    retry_count = state.get("retry_count", 0)
    cfg         = settings.grade

    # --- fast path: no docs ---
    if not ranked_docs:
        log.info("grade=insufficient (no ranked docs)")
        return {  # type: ignore[return-value]
            "grade_verdict": "insufficient",
            "retry_count": retry_count + 1,   # must increment or loop never terminates
        }

    # --- fast path: score threshold ---
    top_score = ranked_docs[0].rerank_score
    if top_score < cfg.score_threshold:
        log.info(
            "grade=insufficient (top rerank score %.3f < threshold %.3f)",
            top_score, cfg.score_threshold,
        )
        return {  # type: ignore[return-value]
            "grade_verdict": "insufficient",
            "retry_count": retry_count + 1,   # must increment or loop never terminates
        }

    # --- LLM judge ---
    verdict, reason = _call_llm(llm, query, ranked_docs)
    log.info("grade=%s | reason=%s", verdict, reason)

    if verdict == "insufficient":
        retry_count += 1

    return {  # type: ignore[return-value]
        "grade_verdict": verdict,
        "retry_count": retry_count,
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

    def _hit(ref: str, content: str, book: str, ch: int, vs: int,
             rerank: float, retr: float = 0.03) -> RankedHit:
        return RankedHit(
            chunk_id=_sha(ref),
            payload={"content": content, "book": book, "chapter": ch,
                     "verse_start": vs, "verse_end": vs, "source_type": "bible"},
            retrieval_score=retr,
            rerank_score=rerank,
        )

    good_docs = [
        _hit("John 3:16",   "For God so loved the world, that he gave his only begotten Son.",
             "John", 3, 16, rerank=-3.6),
        _hit("Romans 5:8",  "But God commendeth his love toward us, in that, while we were "
             "yet sinners, Christ died for us.", "Romans", 5, 8, rerank=-6.2),
        _hit("Psalms 23:1", "The LORD is my shepherd; I shall not want.",
             "Psalms", 23, 1, rerank=-8.1),
    ]
    poor_docs = [
        _hit("Genesis 1:1", "In the beginning God created the heaven and the earth.",
             "Genesis", 1, 1, rerank=-11.2),
        _hit("Genesis 1:2", "And the earth was without form, and void.",
             "Genesis", 1, 2, rerank=-11.8),
    ]

    s   = get_settings()
    llm = get_llm("grade", s)
    grade = partial(grade_node, llm=llm, settings=s)

    cases = [
        ("What does the Bible say about God's love?",          good_docs, "expect: sufficient"),
        ("What does the Bible say about the creation of man?", poor_docs, "expect: insufficient"),
    ]

    print("=== grade_documents self-check ===\n")
    for query, docs, note in cases:
        state: GraphState = {  # type: ignore[misc]
            "query": query,
            "ranked_docs": docs,
            "retry_count": 0,
        }
        result = grade(state)
        verdict = result.get("grade_verdict", "?")
        retries = result.get("retry_count", 0)
        marker  = "✓" if (
            (note == "expect: sufficient"   and verdict == "sufficient") or
            (note == "expect: insufficient" and verdict == "insufficient")
        ) else "✗"
        print(f"{marker} {note}")
        print(f"  query   : {query}")
        print(f"  verdict : {verdict}  retry_count={retries}")
        print()


if __name__ == "__main__":
    main()


# =============================================================
# ADD to: src/verses_rag/config/settings.py
# =============================================================
# New nested block — place alongside BibleChunkingSettings etc.
# Add `grade: GradeSettings = GradeSettings()` to the Settings class.
# Env overrides:
#   GRADE__SCORE_THRESHOLD=-9.0
#   GRADE__MAX_RETRIES=1
#
# class GradeSettings(BaseModel):
#     """Relevance grading knobs (grade_documents node)."""
#     score_threshold: float = -8.0
#     # Rerank score below which we skip the LLM and grade insufficient directly.
#     # -8.0 is a reasonable default for ms-marco-MiniLM logits; tune via eval (Phase 5).
#     max_retries: int = 2
#     # Max retrieve→grade loops before the graph forces abstention.
#
# # on Settings:
#     grade: GradeSettings = GradeSettings()