# =============================================================
# File: src/verses_rag/eval/ab_test.py
# =============================================================
"""
A/B testing framework for the RAG pipeline (SPEC §4.9, D7).

Runs the same eval cases through two pipeline configurations and
produces a side-by-side comparison. The primary use case is testing
whether a cheaper/local model can replace an API judge role without
measurable quality loss — e.g. swapping gpt-4o-mini for a local model
on the grade or verify role.

Typical A/B scenarios:
  - grade threshold:  score_threshold=-8.0 vs -6.0 (stricter)
  - judge model:      gpt-4o-mini vs claude-haiku (primary vs backup)
  - rerank top_k:     5 vs 3 (speed vs quality tradeoff)
  - generation model: qwen3:1.7b vs a larger local model

Usage:
    from verses_rag.eval.ab_test import ABConfig, run_ab_test

    config_a = settings_with_overrides(grade__score_threshold=-8.0)
    config_b = settings_with_overrides(grade__score_threshold=-6.0)
    result = run_ab_test(
        graph_a, graph_b,
        llm=llm, ab_config=ABConfig(label_a="threshold=-8", label_b="threshold=-6"),
        settings_a=config_a, settings_b=config_b,
    )
    print(result.summary())

Run self-check (InMemoryStore, needs API keys + Ollama):
    uv run python -m verses_rag.eval.ab_test
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from verses_rag.eval.dataset  import EvalCase, load_cases
from verses_rag.eval.runner   import EvalConfig, EvalReport, run_eval

log = logging.getLogger("eval.ab_test")


# --- config ------------------------------------------------------------------

@dataclass
class ABConfig:
    label_a:     str = "config_a"
    label_b:     str = "config_b"
    dataset:     str = "sample"
    score_gen:   bool = True
    verbose:     bool = False    # suppress per-case noise; summary is enough


# --- result ------------------------------------------------------------------

@dataclass
class ABResult:
    config:   ABConfig
    report_a: EvalReport
    report_b: EvalReport

    def summary(self) -> str:
        a, b = self.report_a, self.report_b
        ra, rb = a.retrieval, b.retrieval
        ga, gb = a.generation, b.generation
        n      = a.retrieval.n_cases or 1

        acc_a = sum(1 for r in a.case_results if r.verdict_correct) / len(a.case_results)
        acc_b = sum(1 for r in b.case_results if r.verdict_correct) / len(b.case_results)

        def _delta(va: float, vb: float) -> str:
            d = vb - va
            return f"{d:+.3f}"

        lines = [
            f"\n{'='*70}",
            f"A/B Test — {self.config.label_a!r} vs {self.config.label_b!r}",
            f"Dataset: {self.config.dataset}  "
            f"n_cases={len(a.case_results)}",
            f"{'='*70}",
            f"{'Metric':<28} {'A':>10} {'B':>10} {'Δ (B-A)':>10}",
            f"{'-'*70}",
            f"{'Verdict accuracy':<28} {acc_a:>10.3f} {acc_b:>10.3f} {_delta(acc_a, acc_b):>10}",
            f"{'--- Retrieval ---':<28}",
            f"{'  hit@1':<28} {ra.hit_at_1:>10.3f} {rb.hit_at_1:>10.3f} {_delta(ra.hit_at_1, rb.hit_at_1):>10}",
            f"{'  hit@3':<28} {ra.hit_at_3:>10.3f} {rb.hit_at_3:>10.3f} {_delta(ra.hit_at_3, rb.hit_at_3):>10}",
            f"{'  recall@3':<28} {ra.recall_at_3:>10.3f} {rb.recall_at_3:>10.3f} {_delta(ra.recall_at_3, rb.recall_at_3):>10}",
            f"{'  MRR':<28} {ra.mrr:>10.3f} {rb.mrr:>10.3f} {_delta(ra.mrr, rb.mrr):>10}",
            f"{'--- Generation ---':<28}",
            f"{'  faithfulness':<28} {ga.mean_faithfulness:>10.3f} {gb.mean_faithfulness:>10.3f} {_delta(ga.mean_faithfulness, gb.mean_faithfulness):>10}",
            f"{'  relevance':<28} {ga.mean_relevance:>10.3f} {gb.mean_relevance:>10.3f} {_delta(ga.mean_relevance, gb.mean_relevance):>10}",
            f"{'  abstention rate':<28} {ga.abstention_rate:>10.1%} {gb.abstention_rate:>10.1%} {_delta(ga.abstention_rate, gb.abstention_rate):>10}",
            f"{'--- Latency ---':<28}",
            f"{'  total (s)':<28} {a.total_s:>10.1f} {b.total_s:>10.1f} {_delta(a.total_s, b.total_s):>10}",
            f"{'  per case (s)':<28} {a.total_s/max(len(a.case_results),1):>10.1f} {b.total_s/max(len(b.case_results),1):>10.1f}",
            f"{'='*70}",
        ]

        # Per-case diff: flag cases where verdict changed.
        changed = [
            (ra_, rb_)
            for ra_, rb_ in zip(a.case_results, b.case_results)
            if ra_.verdict_correct != rb_.verdict_correct
            or abs(ra_.faithfulness - rb_.faithfulness) > 0.2
        ]
        if changed:
            lines.append("Notable differences:")
            for ra_, rb_ in changed:
                lines.append(
                    f"  [{ra_.case.id}] {ra_.case.query[:45]:<45}"
                    f"  verdict: A={ra_.verdict_correct} B={rb_.verdict_correct}"
                    f"  faith: A={ra_.faithfulness:.2f} B={rb_.faithfulness:.2f}"
                )
        else:
            lines.append("No notable per-case differences.")

        lines.append("="*70)
        return "\n".join(lines)


# --- runner ------------------------------------------------------------------

def run_ab_test(
    graph_a:    Any,
    graph_b:    Any,
    *,
    llm:        Any,
    ab_config:  ABConfig | None = None,
    settings_a: Any = None,
    settings_b: Any = None,
    cases:      list[EvalCase] | None = None,
) -> ABResult:
    """Run the same eval cases through two pipeline configurations.

    Args:
        graph_a/b:   Compiled graphs (from build_graph with different settings).
        llm:         Judge LLM for generation metrics.
        ab_config:   Labels and dataset selection.
        settings_a/b: Settings for each configuration.
        cases:       Override the dataset with an explicit list.
    """
    if ab_config is None:
        ab_config = ABConfig()

    eval_cfg = EvalConfig(
        dataset   = ab_config.dataset,
        score_gen = ab_config.score_gen,
        verbose   = ab_config.verbose,
    )

    log.info("A/B: running config A (%s)…", ab_config.label_a)
    print(f"\n--- Running A: {ab_config.label_a} ---")
    report_a = run_eval(
        graph_a, llm=llm, config=eval_cfg,
        settings=settings_a, cases=cases,
    )

    log.info("A/B: running config B (%s)…", ab_config.label_b)
    print(f"\n--- Running B: {ab_config.label_b} ---")
    report_b = run_eval(
        graph_b, llm=llm, config=eval_cfg,
        settings=settings_b, cases=cases,
    )

    return ABResult(config=ab_config, report_a=report_a, report_b=report_b)


# --- self-check --------------------------------------------------------------

def main():
    """A/B test: grade score_threshold=-8.0 (default) vs -6.0 (stricter).

    Stricter threshold means more queries fail the fast-path grade check,
    leading to more retries and potentially more abstentions.
    Expected: config B has higher abstention rate, possibly lower accuracy.
    """
    import copy
    import hashlib

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

    s_a = get_settings()

    # Config B: stricter grade threshold — deep copy so mutation doesn't affect s_a.
    # get_settings() is @lru_cache so s_b = get_settings() returns the same object.
    s_b = s_a.model_copy(deep=True)
    s_b.grade.score_threshold = -6.0   # stricter than default -8.0

    print("=== A/B test self-check ===")
    print(f"  A: grade threshold = {s_a.grade.score_threshold}")
    print(f"  B: grade threshold = {s_b.grade.score_threshold}")

    dense    = DenseEmbedder(s_a.embedding.dense_model)
    sparse   = SparseEmbedder(s_a.embedding.sparse_model)
    reranker = Reranker.from_settings(s_a.rerank)

    texts  = [v[1] for v in VERSES]
    d_vecs = dense.encode_passages(texts)
    sv_s   = sparse.encode_passages(texts)

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
        in zip(VERSES, d_vecs, sv_s, strict=True)
    ])

    # Both graphs share the same store/embedders; only settings differ.
    graph_a = build_graph(store, dense, sparse, reranker, settings=s_a)
    graph_b = build_graph(store, dense, sparse, reranker, settings=s_b)
    llm     = get_llm("verify", s_a)

    result = run_ab_test(
        graph_a, graph_b,
        llm=llm,
        ab_config=ABConfig(
            label_a  = f"threshold={s_a.grade.score_threshold}",
            label_b  = f"threshold={s_b.grade.score_threshold}",
            dataset  = "sample",
            score_gen = True,
            verbose  = True,
        ),
        settings_a = s_a,
        settings_b = s_b,
    )
    print(result.summary())


if __name__ == "__main__":
    main()