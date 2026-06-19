# verses-rag

A retrieval-augmented generation system over the KJV Bible and a small corpus of theological articles. Ask questions about scripture, look up specific verses or passages, or query the article corpus. Answers are grounded in retrieved text and verified for faithfulness before being returned.

Built for personal use. Not a general-purpose chatbot.

---

## What it does

- **Hybrid retrieval** — combines dense semantic search (BGE-large-en-v1.5) and sparse keyword search (BM25 via FastEmbed) with RRF fusion, backed by Qdrant
- **Cross-encoder reranking** — ms-marco-MiniLM reranks candidates before grading
- **Verse-lookup bypass** — direct reference queries (e.g. `Romans 8:28`, `Genesis 1:1-3`) skip LLM generation and return verbatim KJV text
- **Graded retrieval loop** — an LLM judge grades retrieved documents; insufficient results trigger query reformulation and a retry (up to `max_retries`)
- **Faithfulness verification** — a second LLM judge checks the generated answer against the source passages before it reaches the user
- **Query decomposition** — multi-part queries (e.g. "Compare Genesis and Romans on sin") are split into sub-queries and results merged
- **Streaming** — the API supports SSE streaming with per-node progress events

---

## Corpus

| Source | Count |
|---|---|
| KJV Bible verses (windowed chunks, 5 verses, overlap 1) | 7,924 chunks |
| Theological articles | 27 chunks |
| **Total indexed** | **7,951 points** |

Bible text is KJV only (public domain) in json sourced from `/Volumes/X10 Pro/RAG`. Articles are sourced from `/Volumes/X10 Pro/RAG/articles`.

---

## Stack

| Component | Choice |
|---|---|
| Vector store | Qdrant (self-hosted, Docker) |
| Dense embeddings | BGE-large-en-v1.5 (1024-dim, SentenceTransformers) |
| Sparse embeddings | Qdrant/bm25 (FastEmbed) |
| Reranker | ms-marco-MiniLM-L-6-v2 (CrossEncoder) |
| Generation | Ollama — qwen3:1.7b (local) |
| Grading / verification | OpenAI primary, Anthropic backup |
| Orchestration | LangGraph (state machine) + LangChain (LLM wrappers, text splitting) |
| Agents | Query decomposer, Bible reference resolver (LangChain tools) |
| API | FastAPI |
| UI | Streamlit |
| Tracing | LangSmith (opt-in) |

### Model roles

| Role | Model | Why |
|---|---|---|
| Generation | Ollama — qwen3:1.7b | Local, free to run, good enough for grounded summarisation |
| Grading | OpenAI primary, Anthropic backup | Judgment-critical; small local models are unreliable here |
| Verification | OpenAI primary, Anthropic backup | Last line of defence before answer reaches user |
| Query routing / reformulation | OpenAI primary, Anthropic backup | Needs reliable instruction-following |

LangSmith tracing is opt-in. Set `LANGSMITH_API_KEY` in `.env` and `LANGSMITH__ENABLED=true` to trace runs. Each eval case gets its own named run.

---

## Graph flow

```
query
  │
  ▼
route ──────────────────────────────────────────────────────────┐
  │ (bible / article / mixed)                                    │
  ▼                                                             │
analyze_filters                                                  │
  │ (book, chapter, verse_range, testament, status)             │
  ▼                                                             │
retrieve                                                         │
  │ (hybrid dense+sparse, RRF fusion, metadata filters)        │
  ▼                                                             │
rerank                                                           │
  │ (cross-encoder ms-marco-MiniLM)                            │
  ▼                                                             │
grade_documents ──── insufficient ──► transform_query ──────────┘
  │                  (retry loop,      (query reformulation,
  │ sufficient       max 2 retries)     LLM-assisted)
  │
  ├── max retries exceeded ──► force_abstain ──► END
  │
  ▼
generate
  │ (verse-lookup: verbatim bypass, no LLM)
  │ (other queries: Ollama qwen3:1.7b, context-only prompt)
  ▼
verify
  │ (OpenAI/Anthropic faithfulness check)
  ├── pass    ──► answer = draft_answer
  └── fail    ──► answer = abstain
       │
       ▼
      END
```

---

## Running locally

### Prerequisites

- Docker Desktop
- Ollama with `qwen3:1.7b` pulled: `ollama pull qwen3:1.7b`
- `.env` file with `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and optionally `LANGSMITH_API_KEY`

A minimal `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=ls__...        # optional
LANGSMITH__ENABLED=false         # set true to enable tracing
```

### Start the stack

```bash
docker compose up -d --build
```

This starts three services: Qdrant (port 6333), FastAPI (port 8000), Streamlit (port 8501).

Qdrant data persists in `./qdrant_storage`. The corpus volume is not mounted by default — the running app queries Qdrant directly.

### Check it's up

```bash
curl http://localhost:8000/health
```

### Query via API

```bash
# Verse lookup
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Romans 8:28"}' | python -m json.tool

# Verse range
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Romans 8:1-5"}' | python -m json.tool

# Thematic
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Proverbs say about the tongue?"}' | python -m json.tool
```

### Query via UI

Open `http://localhost:8501` in a browser.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Provider and graph readiness |
| POST | `/query` | Synchronous query, returns full answer |
| POST | `/query/stream` | SSE streaming with node-progress events |
| POST | `/ingest` | Trigger ingest → embed → upsert pipeline |

### Query request body

```json
{
  "query": "What does John 3:16 say?",
  "use_decomposition": true
}
```

Set `use_decomposition: false` to skip query splitting (useful for single-verse lookups where decomposition adds latency).

---

## Re-indexing

To re-ingest the corpus inside Docker, uncomment the corpus volume in `docker-compose.yml` then:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"include_bible": true, "treat_all_as_articles": false}' | python -m json.tool
```

Or run locally (faster, no volume needed):

```bash
uv run python -m verses_rag.indexing.indexer
```

---

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run python -m pytest tests/ -v

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Run eval against live index
uv run python - <<'EOF'
from verses_rag.config.settings import get_settings
from verses_rag.embeddings import DenseEmbedder, SparseEmbedder
from verses_rag.retrieval.reranker import Reranker
from verses_rag.stores.qdrant_store import QdrantStore
from verses_rag.graph.graph import build_graph
from verses_rag.llm.router import get_llm
from verses_rag.eval.runner import run_eval, EvalConfig
s = get_settings()
dense = DenseEmbedder(s.embedding.dense_model)
sparse = SparseEmbedder(s.embedding.sparse_model)
reranker = Reranker.from_settings(s.rerank)
store = QdrantStore(s.qdrant.url, s.qdrant.collection_name)
graph = build_graph(store, dense, sparse, reranker, settings=s)
llm = get_llm('verify', s)
report = run_eval(graph, llm=llm, config=EvalConfig(dataset='kjv', verbose=True), settings=s)
print(report.summary())
EOF
```

---

## Known limitations

- **Single-verse queries** return the containing chunk window (e.g. `Romans 8:1` returns verses 1–5). This is by design — chunks are 5-verse windows.
- **Generation model** (qwen3:1.7b) is small and occasionally needs output sanitisation for reasoning blocks and abstain footers. See `generate.py` `_extract_text()`.
- **Latency** is high for thematic queries (~10–25s) due to the grade→verify LLM calls. Verse lookups bypass generation and return in ~2s.
- **Article corpus** is small (27 chunks). Article queries work but coverage is limited to what's in `/Volumes/X10 Pro/RAG/articles`.