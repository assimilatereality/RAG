# =============================================================
# File: src/verses_rag/eval/runner.py
# =============================================================
"""
Evaluation runner for the RAG pipeline (SPEC §4.9).

Runs a list of EvalCases through the compiled graph, collects retrieval
and generation metrics for each, and produces a summary report.

Usage:
    from verses_rag.eval.runner import run_eval, EvalConfig

    config = EvalConfig(dataset="sample")
    report = run_eval(graph, dense, sparse, reranker, config=config)
    print(report.summary())

Run end-to-end self-check (InMemoryStore, needs API keys + Ollama):
    uv run python -m verses_rag.eval.runner
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from verses_rag.eval.dataset import EvalCase, load_cases
from verses_rag.eval.retrieval_metrics import (
    RetrievalReport,
    compute_retrieval_report,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from verses_rag.eval.generation_metrics import (
    GenerationReport,
    compute_generation_report,
    faithfulness_score,
    answer_relevance_score,
    is_abstention,
)
from verses_rag.graph.graph import run_query
from verses_rag.eval.tracing import configure_tracing, make_run_config

log = logging.getLogger("eval.runner")


# --- config ------------------------------------------------------------------

@dataclass
class EvalConfig:
    dataset:      str   = "sample"   # "sample" or "kjv"
    k_values:     list[int] = field(default_factory=lambda: [1, 3, 5])
    score_gen:    bool  = True        # run generation metrics (needs API)
    verbose:      bool  = True        # print per-case results


# --- per-case result ---------------------------------------------------------

@dataclass
class CaseResult:
    case:              EvalCase
    ranked_docs:       list[Any]
    answer:            str
    verdict:           str
    citations:         list[dict]
    latency_s:         float

    # retrieval
    hit_at_1:          bool
    hit_at_3:          bool
    hit_at_5:          bool
    recall_at_3:       float
    recall_at_5:       float
    rr:                float

    # generation
    faithfulness:      float = 0.0
    relevance:         float = 0.0
    faithfulness_note: str   = ""
    relevance_note:    str   = ""

    @property
    def verdict_correct(self) -> bool:
        """True if pipeline verdict matches expected_verdict."""
        actual = "abstain" if is_abstention(self.answer) else "pass"
        return actual == self.case.expected_verdict


# --- full eval report --------------------------------------------------------

@dataclass
class EvalReport:
    config:       EvalConfig
    case_results: list[CaseResult]
    retrieval:    RetrievalReport
    generation:   GenerationReport
    total_s:      float

    def summary(self) -> str:
        n      = len(self.case_results)
        passes = sum(1 for r in self.case_results if r.verdict_correct)
        lines  = [
            f"\n{'='*60}",
            f"Eval report — dataset={self.config.dataset}  n={n}",
            f"{'='*60}",
            f"Verdict accuracy : {passes}/{n} ({passes/n:.1%})",
            f"Retrieval        : {self.retrieval}",
            f"Generation       : {self.generation}",
            f"Total time       : {self.total_s:.1f}s  "
            f"({self.total_s/n:.1f}s/case)",
            "",
            "Per-case:",
        ]
        for r in self.case_results:
            ok     = "✓" if r.verdict_correct else "✗"
            abst   = "abstain" if is_abstention(r.answer) else "pass   "
            lines.append(
                f"  {ok} [{r.case.id}] {r.case.query[:42]:<42} "
                f"verdict={abst}  hit@3={r.hit_at_3}  "
                f"RR={r.rr:.2f}  faith={r.faithfulness:.2f}  "
                f"rel={r.relevance:.2f}  {r.latency_s:.1f}s"
            )
        lines.append("="*60)
        return "\n".join(lines)


# --- runner ------------------------------------------------------------------

def run_eval(
    graph:    Any,
    *,
    llm:      Any,
    config:   EvalConfig | None = None,
    settings: Any               = None,
    cases:    list[EvalCase] | None = None,
) -> EvalReport:
    """Run evaluation on a dataset.

    Args:
        graph:    Compiled graph from build_graph().
        llm:      Judge LLM for generation metrics (get_llm("verify")).
        config:   EvalConfig; defaults to EvalConfig().
        settings: Settings override.
        cases:    Override the dataset with an explicit list of EvalCases.
    """
    if config is None:
        config = EvalConfig()
    if cases is None:
        cases = load_cases(config.dataset)
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    case_results: list[CaseResult] = []
    retrieval_inputs: list[tuple[list, list[str]]] = []
    gen_scores: list[tuple[float, float, bool]]    = []

    t_start = time.perf_counter()
    tracing_active = configure_tracing(settings)
    if tracing_active:
        log.info("LangSmith tracing active for this eval run")

    for case in cases:
        log.info("eval [%s] %s", case.id, case.query[:60])
        t0 = time.perf_counter()

        run_cfg = make_run_config(
            f"eval-{case.id}",
            tags     = ["eval", config.dataset],
            metadata = {"query": case.query, "expected": case.expected_verdict},
        )
        result = run_query(
            case.query, graph,
            settings=settings,
            use_decomposition=True,
            run_config=run_cfg,
        )

        latency = time.perf_counter() - t0
        ranked  = result.get("ranked_docs", [])
        answer  = result.get("answer", "")
        verdict = result.get("verdict", "")
        cites   = result.get("citations", [])

        # --- retrieval metrics ---
        h1  = hit_at_k(ranked, case.relevant_refs, 1)
        h3  = hit_at_k(ranked, case.relevant_refs, 3)
        h5  = hit_at_k(ranked, case.relevant_refs, 5)
        r3  = recall_at_k(ranked, case.relevant_refs, 3)
        r5  = recall_at_k(ranked, case.relevant_refs, 5)
        rr  = reciprocal_rank(ranked, case.relevant_refs)
        retrieval_inputs.append((ranked, case.relevant_refs))

        # --- generation metrics ---
        f_score, f_note, rel_score, rel_note = 0.0, "", 0.0, ""
        if config.score_gen and answer:
            f_score, f_note  = faithfulness_score(
                case.query, answer, cites, llm, ranked_docs=ranked
            )
            rel_score, rel_note = answer_relevance_score(case.query, answer, llm)
        abstain = is_abstention(answer)
        gen_scores.append((f_score, rel_score, abstain))

        cr = CaseResult(
            case=case, ranked_docs=ranked,
            answer=answer, verdict=verdict, citations=cites,
            latency_s=latency,
            hit_at_1=h1, hit_at_3=h3, hit_at_5=h5,
            recall_at_3=r3, recall_at_5=r5, rr=rr,
            faithfulness=f_score, relevance=rel_score,
            faithfulness_note=f_note, relevance_note=rel_note,
        )
        case_results.append(cr)

        if config.verbose:
            ok = "✓" if cr.verdict_correct else "✗"
            print(f"  {ok} [{case.id}] {case.query[:50]:<50}  "
                  f"{'abstain' if abstain else 'pass':7}  "
                  f"hit@3={h3}  RR={rr:.2f}  "
                  f"faith={f_score:.2f}  {latency:.1f}s")

    total_s = time.perf_counter() - t_start

    return EvalReport(
        config       = config,
        case_results = case_results,
        retrieval    = compute_retrieval_report(retrieval_inputs),
        generation   = compute_generation_report(gen_scores),
        total_s      = total_s,
    )


# --- self-check --------------------------------------------------------------

def main():
    import hashlib
    from functools import partial

    from verses_rag.config.settings import get_settings
    from verses_rag.embeddings       import DenseEmbedder, SparseEmbedder
    from verses_rag.retrieval.reranker import Reranker
    from verses_rag.stores           import ChunkWithVectors, InMemoryStore
    from verses_rag.graph.graph      import build_graph
    from verses_rag.llm.router       import get_llm

    def _sha(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    VERSES = [
        (_sha("Genesis 1:1"),  "In the beginning God created the heaven and the earth.",
         "Genesis", 1, 1, 1, "OT"),
        (_sha("John 3:16"),    "For God so loved the world, that he gave his only begotten Son, "
         "that whosoever believeth in him should not perish, but have everlasting life.",
         "John", 3, 16, 16, "NT"),
        (_sha("Psalms 23:1"),  "The LORD is my shepherd; I shall not want.",
         "Psalms", 23, 1, 1, "OT"),
        (_sha("Romans 8:28"),  "And we know that all things work together for good to them "
         "that love God, to them who are the called according to his purpose.",
         "Romans", 8, 28, 28, "NT"),
        (_sha("Proverbs 3:5"), "Trust in the LORD with all thine heart; and lean not unto "
         "thine own understanding.",
         "Proverbs", 3, 5, 5, "OT"),
        (_sha("Romans 5:8"),   "But God commendeth his love toward us, in that, while we were "
         "yet sinners, Christ died for us.",
         "Romans", 5, 8, 8, "NT"),
    ]

    s = get_settings()
    print("=== eval runner self-check ===\n")
    print("Loading embedders, reranker, graph…")

    dense    = DenseEmbedder(s.embedding.dense_model)
    sparse   = SparseEmbedder(s.embedding.sparse_model)
    reranker = Reranker.from_settings(s.rerank)

    texts  = [v[1] for v in VERSES]
    d_vecs = dense.encode_passages(texts)
    s_vecs = sparse.encode_passages(texts)

    store = InMemoryStore()
    store.ensure_collection(dense_dim=dense.dim)
    store.upsert([
        ChunkWithVectors(
            chunk_id=cid,
            payload={"content": text, "book": book, "chapter": ch,
                     "verse_start": vs, "verse_end": ve, "testament": t,
                     "source_type": "bible", "status": "active"},
            dense=dv, sparse=sv,
        )
        for (cid, text, book, ch, vs, ve, t), dv, sv
        in zip(VERSES, d_vecs, s_vecs, strict=True)
    ])

    graph = build_graph(store, dense, sparse, reranker, settings=s)
    llm   = get_llm("verify", s)

    print("\nRunning sample dataset (4 cases)…\n")
    config = EvalConfig(dataset="sample", score_gen=True, verbose=True)
    report = run_eval(graph, llm=llm, config=config, settings=s)
    print(report.summary())


if __name__ == "__main__":
    main()