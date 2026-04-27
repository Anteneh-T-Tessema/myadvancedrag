# Advanced RAG Studio

> **Production-grade Retrieval-Augmented Generation** — moving beyond naive single-vector pipelines to deterministic, forensic-quality AI retrieval.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Is

Most RAG tutorials show a single-vector pipeline: embed a query → nearest-neighbor search → stuff into prompt. In production this fails badly:

- Short queries don't match dense technical paragraphs (vocabulary mismatch)
- Fixed-size chunking destroys paragraph coherence
- Vector search can't match exact entity names, serial numbers, or acronyms
- Every query hits the same expensive retrieval path regardless of intent

This project implements **four orthogonal architectural improvements**, each independently switchable, with a full interactive dashboard for live exploration:

| Technique | Problem Solved |
|---|---|
| **Semantic Routing** | Intent triage — route SQL queries away from vector DB |
| **HyDE** | Vocabulary mismatch — embed hypothetical doc, not raw query |
| **Hybrid Search + RRF** | Precision — dense semantic + BM25 exact match, fused via RRF |
| **Parent-Child Chunking** | Context loss — retrieve child precision, serve parent context |

---

## Live Dashboard

The system ships with a full dark-mode **React-free** interactive dashboard:

- **Query Explorer** — run queries through every pipeline stage with live trace
- **Pipeline Trace** — see routing decision, HyDE expansion, search stats, RRF scores per chunk
- **Ingest Documents** — choose chunking strategy, watch chunks get indexed in real time
- **Semantic Router Lab** — test any query against all route centroids, see confidence scores
- **HyDE Transform** — compare raw query vs. hypothetical document side-by-side
- **Pipeline Config** — tune every parameter live (no restart needed)
- **Index Stats** — monitor chunk counts, BM25 status, per-document breakdown

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1 · Semantic Router              │
│  Embedding classifier → route target    │
│  calculator | sql | code | legal | ...  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 2 · HyDE Transformation          │
│  LLM generates hypothetical document    │
│  Embed hypothetical → search vector     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 3 · Hybrid Search                │
│  Dense (cosine) ──┐                     │
│  BM25 sparse  ────┼──▶ RRF fusion       │
│  RRF(d) = Σ 1/(k + r(d))               │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 4 · Auto-Merge (Parent-Child)    │
│  Child match → promoted to Parent chunk │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 5 · LLM Generation (Ollama)      │
│  Grounded answer over retrieved context │
└─────────────────────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module map and decision trees.

---

## Core Modules

### `core/router.py` — Semantic Router
```python
router = SemanticRouter(confidence_threshold=0.45)
router.build()  # computes route centroids from example utterances

decision = router.route("What is the Q3 total revenue?")
# RouterDecision(target=RouteTarget.SQL_AGENT, confidence=0.81)
```

Routes: `vector_db`, `sql_agent`, `code_agent`, `legal_agent`, `calculator`, `conversational`, `hybrid`

### `core/hyde.py` — HyDE Transformer
```python
hyde = HyDETransformer(model="llama3.2", temperature=0.3)
result = hyde.generate_hypothetical("How does BM25 work?")
# hypothetical_doc = "BM25 is a probabilistic retrieval function that..."
# Use this paragraph-shaped vector for search instead of the 5-word query
```

### `core/hybrid_search.py` — Hybrid Search Engine
```python
engine = HybridSearchEngine(embed_model_name="all-MiniLM-L6-v2", rrf_k=60)
engine.add_chunks(chunk_dicts)

results = engine.search(
    query="Explain RRF fusion",
    use_hybrid=True,        # Dense + BM25
    auto_merge_parents=True # Child → Parent promotion
)
# results[0].rrf_score, .dense_score, .sparse_score, .dense_rank, .sparse_rank
```

### `core/chunker.py` — Advanced Chunker
```python
chunker = AdvancedChunker(similarity_threshold=0.5)

# Strategy 1: Topic-aware semantic chunks
chunks = chunker.semantic_chunk(text, source="contract.pdf")

# Strategy 2: Parent-child hierarchy
chunks = chunker.parent_child_chunk(text, source="whitepaper.pdf")
# Returns parents (512 tok) and children (150 tok) linked by parent_id

# Strategy 3: Fixed with rich metadata
chunks = chunker.fixed_chunk(text, extra_metadata={"dept": "legal"})
```

### `core/pipeline.py` — Orchestrator
```python
pipeline = AdvancedRAGPipeline(
    use_router=True,
    use_hyde=True,
    use_hybrid=True,
    use_auto_merge=True,
    llm_model="llama3.2",
)

pipeline.ingest("Your document text...", source="my_doc.pdf", strategy="semantic")

result = pipeline.query("How does hybrid search work?")
# result["pipeline_stages"] → full trace of every stage
# result["retrieved_chunks"] → RRF-ranked chunks with all scores
# result["answer"] → grounded LLM-generated response
```

---

## Chunking Strategies Compared

### Semantic Chunking
Embeds each sentence individually, finds topic shifts via cosine similarity drops:
```
sentence₁ · sentence₂  →  sim=0.87  (same topic, same chunk)
sentence₂ · sentence₃  →  sim=0.31  (topic change → new chunk)
```

### Parent-Child Retrieval
```
Document
├── Parent Chunk 0  (512 tokens) ← fed to LLM
│   ├── Child 0.0  (150 tokens) ← retrieved by vector search
│   ├── Child 0.1  (150 tokens)
│   └── Child 0.2  (150 tokens)
└── Parent Chunk 1  (512 tokens)
    ├── Child 1.0
    └── Child 1.1
```
When Child 0.1 is retrieved, the system automatically serves Parent 0 to the LLM — full context, precise retrieval.

---

## Reciprocal Rank Fusion (RRF)

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

| Variable | Meaning |
|---|---|
| `d` | Document |
| `R` | Set of rankers (dense, BM25) |
| `r(d)` | Rank of document *d* in ranker *r* |
| `k=60` | Smoothing constant (prevents rank-1 dominance) |

**Why rank-based?** Cosine similarity scores (-1 to 1) and BM25 scores (unbounded frequency counts) cannot be directly compared or averaged. RRF is score-agnostic — it only cares about **positions**.

---

## REST API (Quick Reference)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Status, models, index size |
| `GET/POST` | `/api/config` | Get/set pipeline configuration |
| `POST` | `/api/ingest` | Ingest a document |
| `POST` | `/api/ingest/demo` | Load 7-doc RAG demo corpus |
| `POST` | `/api/query` | Run the full RAG pipeline |
| `POST` | `/api/router/test` | Test routing for a query |
| `GET` | `/api/router/routes` | List all configured routes |
| `POST` | `/api/hyde/transform` | Generate hypothetical document |
| `GET` | `/api/stats` | Full index statistics |
| `POST` | `/api/stats/clear` | Clear index |
| `GET` | `/api/models` | List Ollama models |

Full reference: [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)

---

## Vector Store Recommendations

| Store | Architecture | Best For |
|---|---|---|
| **Qdrant** | Rust, local/cloud | Agent swarms, payload filtering, forensic RAG |
| **pgvector** | PostgreSQL extension | ERP, relational + vector hybrid queries |
| **Milvus** | Distributed | Billions of vectors, enterprise throughput |
| **Pinecone** | Managed cloud | Zero-ops prototyping |

This system's `HybridSearchEngine` uses an **in-memory index** (sentence-transformers + rank-bm25) — swap it for Qdrant or pgvector for production persistence.

---

## Embedding Model Recommendations

| Model | Context | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 256 tok | Fast dev, default |
| `nomic-embed-text` | 8192 tok | Long docs, runs via Ollama |
| `BGE-M3` | 8192 tok | Multi-lingual, SOTA open-source |
| Jina Embeddings | 8192 tok | Code + technical documentation |

---

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.ai)** running locally with at least one model:
  ```bash
  ollama pull llama3.2
  ```
- Dependencies (auto-installed by `start.sh`):
  ```
  flask, flask-cors, sentence-transformers, rank-bm25, numpy, ollama
  ```

---

## Quick Start

```bash
git clone https://github.com/Anteneh-T-Tessema/myadvancedrag.git
cd myadvancedrag
chmod +x start.sh && ./start.sh
```

Then open: **http://localhost:7891/static/index.html**

1. Click **"Load Demo Corpus"** to index 7 RAG reference documents
2. Type a query in **Query Explorer** → **Run Pipeline**
3. Watch the full pipeline trace: routing → HyDE → hybrid search → RRF → answer

---

## Project Structure

```
.
├── core/
│   ├── pipeline.py       # Main orchestrator
│   ├── router.py         # Semantic intent router
│   ├── hyde.py           # HyDE query transformer
│   ├── hybrid_search.py  # Dense + BM25 + RRF engine
│   └── chunker.py        # Semantic/Parent-Child/Fixed chunking
├── api/
│   └── server.py         # Flask REST API
├── static/
│   ├── index.html        # Dashboard SPA
│   ├── css/dashboard.css # Design system
│   └── js/dashboard.js   # Pipeline trace UI
├── docs/
│   ├── ARCHITECTURE.md   # Full architecture diagrams
│   ├── API_REFERENCE.md  # REST endpoint reference
│   └── TECHNIQUES.md     # Technique deep-dives
├── requirements.txt
├── start.sh
└── README.md
```

---

## Design Principles

1. **Every stage is independently toggleable** — compare naive vs. advanced retrieval by toggling HyDE or routing on/off
2. **Full pipeline trace on every query** — every latency, every score, every routing decision is surfaced in the UI
3. **Local-first** — all LLM inference via Ollama, all embeddings via sentence-transformers; no external API calls required
4. **Forensic-grade metadata** — every chunk carries document lineage, timestamps, and strategy tags for pre-retrieval filtering
5. **Swap-ready** — `HybridSearchEngine` is drop-in replaceable with Qdrant or pgvector without changing the pipeline API

---

## Author

**Anteneh Tessema** — AI Engineer  
[GitHub](https://github.com/Anteneh-T-Tessema)

---

## License

MIT — see [LICENSE](LICENSE)
