"""
Inspect what each KJV_CASES query actually retrieves from the live index.

Prints the top reranked chunks per query so relevant_refs in dataset.py can
be set from ground truth instead of guesses. Read-only — no eval, no scoring.

Run:
    uv run python scripts/inspect_retrieval.py
"""

from verses_rag.config.settings import get_settings
from verses_rag.embeddings import DenseEmbedder, SparseEmbedder
from verses_rag.retrieval.reranker import Reranker
from verses_rag.stores import QdrantStore
from verses_rag.eval.dataset import KJV_CASES


def _ref(payload: dict) -> str:
    if payload.get("source_type") == "bible":
        b  = payload.get("book", "?")
        ch = payload.get("chapter", "?")
        vs = payload.get("verse_start", "?")
        ve = payload.get("verse_end", "?")
        return f"{b} {ch}:{vs}" + (f"-{ve}" if ve != vs else "")
    return f"[article] {str(payload.get('title', '?'))[:40]}"


def main():
    s        = get_settings()
    dense    = DenseEmbedder(s.embedding.dense_model)
    sparse   = SparseEmbedder(s.embedding.sparse_model)
    reranker = Reranker.from_settings(s.rerank)
    store    = QdrantStore(s.qdrant.url, s.qdrant.collection_name)

    top_k = s.retrieval.bm25_k + s.retrieval.dense_k

    print("\n" + "=" * 70)
    print("Actual retrieval per KJV case (top 5 after rerank)")
    print("=" * 70)

    for case in KJV_CASES:
        # Use the raw query (no decomposition) to see direct retrieval.
        q_dense  = dense.encode_query(case.query)
        q_sparse = sparse.encode_query(case.query)
        hits     = store.hybrid_query(q_dense, q_sparse, top_k=top_k, filters=None)
        ranked   = reranker.rerank(case.query, hits, top_k=5)

        print(f"\n[{case.id}] {case.query}")
        print(f"   dataset refs : {case.relevant_refs or '(none)'}")
        print(f"   actual top 5 :")
        for r in ranked:
            print(f"      [{r.rerank_score:7.3f}] {_ref(r.payload)}")


if __name__ == "__main__":
    main()