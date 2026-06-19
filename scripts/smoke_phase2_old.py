"""
Phase 2 end-to-end smoke test.

Exercises the full Phase 2 loop:
  1. Create Qdrant collection (idempotent)
  2. Embed a small set of KJV verses — dense + sparse
  3. Upsert into Qdrant
  4. Run a hybrid query and print ranked results
  5. Confirm InMemoryStore returns results for the same query

Run:
    uv run python scripts/smoke_phase2.py

First run downloads BGE-large-en-v1.5 (~1.3 GB) and Qdrant/bm25 (small).
Subsequent runs use the cached models and skip straight to embedding.
"""

import hashlib

from verses_rag.config.settings import get_settings
from verses_rag.embeddings import DenseEmbedder, SparseEmbedder
from verses_rag.stores import ChunkWithVectors, InMemoryStore, QdrantStore

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


VERSES = [
    (_sha("Genesis 1:1"),   "In the beginning God created the heaven and the earth.",
     "Genesis", "OT"),
    (_sha("John 3:16"),     "For God so loved the world, that he gave his only begotten Son.",
     "John", "NT"),
    (_sha("Psalms 23:1"),   "The LORD is my shepherd; I shall not want.",
     "Psalms", "OT"),
    (_sha("Romans 8:28"),   "And we know that all things work together for good to them that love God.",
     "Romans", "NT"),
    (_sha("Proverbs 3:5"),  "Trust in the LORD with all thine heart; and lean not unto thine own understanding.",
     "Proverbs", "OT"),
]

QUERY = "God's love and care for his people"


def make_payload(chunk_id: str, text: str, book: str, testament: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "content": text,
        "book": book,
        "testament": testament,
        "source_type": "bible",
        "translation": "KJV",
    }


def run():
    s = get_settings()

    print("=== Phase 2 smoke test ===\n")

    # 1. Embedders (lazy — models download here on first run)
    print("Loading embedders…")
    dense = DenseEmbedder(s.embedding.dense_model)
    sparse = SparseEmbedder(s.embedding.sparse_model)
    print(f"  dense dim: {dense.dim}")

    # 2. Embed passages
    print("\nEmbedding passages…")
    texts = [v[1] for v in VERSES]
    dense_vecs = dense.encode_passages(texts)
    sparse_vecs = sparse.encode_passages(texts)

    chunks = [
        ChunkWithVectors(
            chunk_id=cid,
            payload=make_payload(cid, text, book, testament),
            dense=dvec,
            sparse=svec,
        )
        for (cid, text, book, testament), dvec, svec
        in zip(VERSES, dense_vecs, sparse_vecs)
    ]
    print(f"  {len(chunks)} chunks ready")

    # 3. Qdrant — create collection + upsert
    print("\nUpserting into Qdrant…")
    store = QdrantStore(s.qdrant.url, s.qdrant.collection_name)
    store.ensure_collection(dense_dim=dense.dim)
    store.upsert(chunks)
    print(f"  stats: {store.stats()}")

    # 4. Hybrid query against Qdrant
    print(f"\nQuery: '{QUERY}'")
    q_dense = dense.encode_query(QUERY)
    q_sparse = sparse.encode_query(QUERY)

    print("\nQdrant results:")
    for hit in store.hybrid_query(q_dense, q_sparse, top_k=3):
        print(f"  [{hit.score:.4f}] {hit.payload['book']:12} {hit.payload['content'][:60]}…")

    # 5. Same query against InMemoryStore
    mem = InMemoryStore()
    mem.ensure_collection(dense_dim=dense.dim)
    mem.upsert(chunks)
    print("\nInMemoryStore results:")
    for hit in mem.hybrid_query(q_dense, q_sparse, top_k=3):
        print(f"  [{hit.score:.4f}] {hit.payload['book']:12} {hit.payload['content'][:60]}…")

    print("\n=== smoke test passed ===")


if __name__ == "__main__":
    run()
