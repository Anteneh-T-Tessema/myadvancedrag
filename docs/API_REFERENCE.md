# REST API Reference

Base URL: `http://localhost:7891/api`

---

## Health & Status

### `GET /health`
Returns API status, available Ollama models, and index size.

**Response**
```json
{
  "status": "ok",
  "ollama_models": ["llama3.2:latest", "nomic-embed-text:latest"],
  "index_size": 42,
  "documents_count": 7
}
```

---

## Pipeline Configuration

### `GET /config`
Returns current pipeline configuration.

### `POST /config`
Updates pipeline configuration (recreates the pipeline instance).

**Body**
```json
{
  "embed_model": "all-MiniLM-L6-v2",
  "llm_model": "llama3.2",
  "use_router": true,
  "use_hyde": true,
  "use_hybrid": true,
  "use_auto_merge": true,
  "top_k": 6,
  "rrf_k": 60,
  "hyde_temperature": 0.3,
  "router_threshold": 0.45,
  "chunking_strategy": "semantic"
}
```

---

## Document Ingestion

### `POST /ingest`
Ingest a single document into the vector index.

**Body**
```json
{
  "text": "Your document content here...",
  "source": "my_document.pdf",
  "strategy": "semantic",
  "doc_id": "optional-uuid",
  "extra_metadata": { "department": "legal", "date": "2025-01-01" }
}
```

**Response**
```json
{
  "doc_id": "abc123...",
  "source": "my_document.pdf",
  "strategy": "semantic",
  "chunk_count": 8,
  "token_count": 1204,
  "latency_ms": 234.5,
  "ingested_at": "2025-04-27T11:00:00Z"
}
```

### `POST /ingest/demo`
Loads the built-in 7-document RAG reference corpus (great for testing).

---

## Querying

### `POST /query`
Execute the full Advanced RAG pipeline.

**Body**
```json
{
  "query": "How does Reciprocal Rank Fusion work?",
  "generate_answer": true,
  "metadata_filter": { "source": "hybrid_search_rrf.txt" }
}
```

**Response**
```json
{
  "query_id": "uuid",
  "original_query": "How does RRF work?",
  "pipeline_stages": [
    {
      "stage": "semantic_routing",
      "result": {
        "matched_route": "semantic_retrieval",
        "target": "vector_db",
        "confidence": 0.72,
        "fallback": false,
        "all_scores": { "semantic_retrieval": 0.72, "code_generation": 0.31 }
      }
    },
    {
      "stage": "hyde_transformation",
      "result": {
        "hypothetical_doc": "Reciprocal Rank Fusion (RRF) is a...",
        "latency_ms": 1200,
        "success": true
      }
    },
    {
      "stage": "hybrid_search",
      "results_count": 6,
      "latency_ms": 12.4
    },
    {
      "stage": "llm_generation",
      "result": {
        "model": "llama3.2",
        "latency_ms": 3100,
        "context_chunks_used": 6
      }
    }
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "uuid",
      "content": "BM25 is a...",
      "doc_id": "uuid",
      "chunk_type": "semantic",
      "rrf_score": 0.031746,
      "dense_score": 0.8821,
      "sparse_score": 14.23,
      "dense_rank": 1,
      "sparse_rank": 2
    }
  ],
  "answer": "Reciprocal Rank Fusion (RRF) combines...",
  "total_latency_ms": 4512.3
}
```

---

## Semantic Router

### `POST /router/test`
Test routing classification for a query without running retrieval.

**Body**: `{ "query": "What is total Q3 revenue?" }`

**Response**
```json
{
  "matched_route": "numerical_sql",
  "target": "sql_agent",
  "confidence": 0.81,
  "fallback": false,
  "all_scores": { "numerical_sql": 0.81, "conversational": 0.22 }
}
```

### `GET /router/routes`
List all configured route definitions with examples.

---

## HyDE Transformation

### `POST /hyde/transform`
Generate a hypothetical document for a query (standalone, no retrieval).

**Body**: `{ "query": "Explain attention mechanisms" }`

**Response**
```json
{
  "original_query": "Explain attention mechanisms",
  "hypothetical_doc": "Attention mechanisms in transformer...",
  "model_used": "llama3.2",
  "latency_ms": 1850.2,
  "success": true,
  "error": null
}
```

---

## Index Management

### `GET /stats`
Full index statistics including per-document breakdown.

### `POST /stats/clear`
Clear all indexed chunks (non-destructive to config).

### `GET /models`
List available local Ollama models.
