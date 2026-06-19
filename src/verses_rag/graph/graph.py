# =============================================================
# File: src/verses_rag/graph/graph.py
# =============================================================
"""
RAG graph assembly (SPEC §4.6, D3).

Wires all seven nodes into a LangGraph StateGraph with:
  - Linear path: route → analyze_filters → retrieve → rerank → grade
  - Conditional: grade → generate (sufficient) or transform_query (retry)
                        → force_abstain (max retries exceeded)
  - Retry loop:  transform_query → retrieve (bounded by grade.max_retries)
  - Terminal:    generate → verify → END
                 force_abstain → END

Public API:
    graph = build_graph(store, dense, sparse, reranker, settings)
    result: GraphState = graph.invoke({"query": "..."})
    print(result["answer"])

Run end-to-end self-check (InMemoryStore, needs API keys + Ollama):
    uv run python -m verses_rag.graph.graph
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from verses_rag.graph.state import GraphState
from verses_rag.graph.nodes.route           import route_node
from verses_rag.graph.nodes.analyze_filters import analyze_filters_node
from verses_rag.graph.nodes.retrieve        import retrieve_node
from verses_rag.graph.nodes.rerank          import rerank_node
from verses_rag.graph.nodes.grade_documents import grade_node
from verses_rag.graph.nodes.transform_query import transform_query_node
from verses_rag.graph.nodes.generate        import generate_node, ABSTAIN_PHRASE
from verses_rag.graph.nodes.verify          import verify_node
from verses_rag.agents.query_decomposer     import decompose, merge_results

log = logging.getLogger("graph")


# --- force-abstain terminal node ---------------------------------------------

def _force_abstain_node(state: GraphState) -> GraphState:
    """Terminal node: max retries exceeded → set abstain answer."""
    log.info("max retries exceeded — forcing abstention")
    return {"verdict": "abstain", "answer": ABSTAIN_PHRASE}  # type: ignore[return-value]


# --- graph builder -----------------------------------------------------------

def build_graph(
    store: Any,
    dense: Any,
    sparse: Any,
    reranker: Any,
    settings=None,
):
    """Build and compile the RAG StateGraph.

    Args:
        store:    QdrantStore or InMemoryStore
        dense:    DenseEmbedder
        sparse:   SparseEmbedder
        reranker: Reranker
        settings: Settings override; defaults to get_settings()

    Returns:
        A compiled LangGraph runnable. Call .invoke({"query": "..."}).
    """
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    from verses_rag.llm.router import get_llm

    # --- bind dependencies via partial ---------------------------------------
    route          = partial(route_node,           settings=settings)
    analyze        = analyze_filters_node           # no extra deps
    retrieve       = partial(retrieve_node,        store=store, dense=dense,
                                                   sparse=sparse, settings=settings)
    rerank         = partial(rerank_node,           reranker=reranker, settings=settings)
    grade          = partial(grade_node,            llm=get_llm("grade",      settings),
                                                   settings=settings)
    transform      = partial(transform_query_node, llm=get_llm("route",      settings),
                                                   settings=settings)
    generate       = partial(generate_node,        llm=get_llm("generation", settings),
                                                   settings=settings)
    verify         = partial(verify_node,           llm=get_llm("verify",     settings),
                                                   settings=settings)

    # --- conditional edge: grade → next node ---------------------------------
    def _grade_router(state: GraphState) -> str:
        verdict = state.get("grade_verdict", "insufficient")
        retries = state.get("retry_count", 0)
        if verdict == "sufficient":
            return "generate"
        if retries >= settings.grade.max_retries:
            return "force_abstain"
        return "transform_query"

    # --- assemble graph ------------------------------------------------------
    g = StateGraph(GraphState)

    g.add_node("route",           route)
    g.add_node("analyze_filters", analyze)
    g.add_node("retrieve",        retrieve)
    g.add_node("rerank",          rerank)
    g.add_node("grade_documents", grade)
    g.add_node("transform_query", transform)
    g.add_node("generate",        generate)
    g.add_node("verify",          verify)
    g.add_node("force_abstain",   _force_abstain_node)

    g.set_entry_point("route")

    # linear backbone
    g.add_edge("route",           "analyze_filters")
    g.add_edge("analyze_filters", "retrieve")
    g.add_edge("retrieve",        "rerank")
    g.add_edge("rerank",          "grade_documents")

    # conditional branch after grading
    g.add_conditional_edges(
        "grade_documents",
        _grade_router,
        {
            "generate":       "generate",
            "transform_query":"transform_query",
            "force_abstain":  "force_abstain",
        },
    )

    # retry loop
    g.add_edge("transform_query", "retrieve")

    # generation → verify → done
    g.add_edge("generate",      "verify")
    g.add_edge("verify",        END)
    g.add_edge("force_abstain", END)

    return g.compile()


# --- public query runner -----------------------------------------------------

def run_query(
    query: str,
    graph: Any,
    *,
    settings=None,
    use_decomposition: bool = True,
    run_config: dict[str, Any] | None = None,
) -> dict:
    """Run a query through the compiled RAG graph.

    Handles decomposition of multi-part queries automatically:
      - Single queries run directly through the graph.
      - Multi-part queries (e.g. "Compare Genesis and Romans on sin") are
        split into sub-queries, each run independently, then merged.

    Args:
        query:            The user's question.
        graph:            Compiled graph from build_graph().
        settings:         Settings override; defaults to get_settings().
        use_decomposition: Set False to skip decomposition (useful for
                          testing individual nodes or benchmarking).
        run_config:       Optional LangChain RunnableConfig dict for
                          LangSmith tracing (from make_run_config()).

    Returns:
        A dict with keys: answer, verdict, citations, source_route,
        grade_verdict, and the full final GraphState.
    """
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    sub_queries = (
        decompose(query)           # heuristic only; LLM decompose opt-in later
        if use_decomposition
        else [query]
    )

    results = []
    for sub_q in sub_queries:
        log.info("running sub-query: %r", sub_q[:80])
        cfg   = run_config or {}
        state = graph.invoke({"query": sub_q, "retry_count": 0}, config=cfg)
        results.append(state)

    merged = merge_results(results, sub_queries)
    merged["sub_queries"] = sub_queries   # carry through for callers
    return merged

def main():
    import hashlib

    from verses_rag.config.settings import get_settings
    from verses_rag.embeddings       import DenseEmbedder, SparseEmbedder
    from verses_rag.retrieval.reranker import Reranker
    from verses_rag.stores           import ChunkWithVectors, InMemoryStore

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
    print("=== graph end-to-end self-check ===\n")

    print("Loading embedders and reranker…")
    dense    = DenseEmbedder(s.embedding.dense_model)
    sparse   = SparseEmbedder(s.embedding.sparse_model)
    reranker = Reranker.from_settings(s.rerank)

    print("Embedding and indexing passages…")
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
                     "source_type": "bible", "status": "active", "translation": "KJV"},
            dense=dv, sparse=sv,
        )
        for (cid, text, book, ch, vs, ve, t), dv, sv
        in zip(VERSES, d_vecs, s_vecs, strict=True)
    ])

    print("Building graph…")
    graph = build_graph(store, dense, sparse, reranker, settings=s)

    queries = [
        ("single",     "What does the Bible say about God's love?"),
        ("multi-part", "Compare what Genesis and Romans say about sin"),
    ]

    for kind, query in queries:
        print(f"\n{'='*60}")
        print(f"[{kind}] Query: {query}")
        print("="*60)
        result = run_query(query, graph, settings=s)
        subs = result.get("sub_queries", [query])
        if len(subs) > 1:
            print(f"Decomposed into {len(subs)} sub-queries:")
            for sq in subs:
                print(f"  • {sq}")
        print(f"\nVerdict : {result.get('verdict')}")
        print(f"\nAnswer  :\n{result.get('answer', '')}")
        print(f"\nCitations:")
        for c in result.get("citations", []):
            print(f"  {c['ref']:20} {c['content_snippet'][:60]}")


if __name__ == "__main__":
    main()