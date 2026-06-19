"""Run the KJV eval dataset against the live Qdrant index."""

from verses_rag.config.settings import get_settings
from verses_rag.embeddings import DenseEmbedder, SparseEmbedder
from verses_rag.retrieval.reranker import Reranker
from verses_rag.stores import QdrantStore
from verses_rag.graph.graph import build_graph
from verses_rag.llm.router import get_llm
from verses_rag.eval.runner import run_eval, EvalConfig


def main():
    s = get_settings()

    # Load against the REAL Qdrant index (not InMemoryStore).
    dense    = DenseEmbedder(s.embedding.dense_model)
    sparse   = SparseEmbedder(s.embedding.sparse_model)
    reranker = Reranker.from_settings(s.rerank)
    store    = QdrantStore(s.qdrant.url, s.qdrant.collection_name)

    graph = build_graph(store, dense, sparse, reranker, settings=s)
    llm   = get_llm("verify", s)

    config = EvalConfig(dataset="kjv", score_gen=True, verbose=True)
    report = run_eval(graph, llm=llm, config=config, settings=s)
    print(report.summary())


if __name__ == "__main__":
    main()