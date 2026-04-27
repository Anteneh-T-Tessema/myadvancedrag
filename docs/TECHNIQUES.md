# Advanced RAG Techniques — Deep Dive

This document provides rigorous technical explanations of each RAG technique implemented in this system, suitable for AI engineering interviews, research, and production reviews.

---

## 1. Semantic Routing

### The Problem
A monolithic RAG pipeline treats every query identically — embedding it, searching the vector DB, and prompting the LLM. This is catastrophically inefficient:
- A request for `"15% of 240,000"` doesn't need vector search at all.
- A legal clause analysis shouldn't use a general-purpose retriever.
- Casual greetings waste expensive LLM inference on dense-context prompts.

### The Solution
An embedding-based intent classifier routes queries **before** any expensive computation occurs.

### Implementation

```python
# For each route, compute centroid of example utterances
embeddings = model.encode(route.examples, normalize_embeddings=True)
route._centroid = np.mean(embeddings, axis=0)
route._centroid /= np.linalg.norm(route._centroid)

# At runtime
q_emb = model.encode([query], normalize_embeddings=True)[0]
scores = { route.name: float(np.dot(q_emb, route._centroid)) for route in routes }
best = max(scores, key=scores.get)
```

### Design Choices
- **Centroid-based** (not per-example softmax): Fast, zero extra inference, works with any embedding model.
- **Configurable threshold**: Below `router_threshold`, falls back to `HYBRID` — avoids mis-routing edge cases.
- **Extensible**: Add routes at runtime with `router.add_route(Route(...))`.

### Route Targets
| Target | Use Case |
|---|---|
| `vector_db` | Semantic knowledge retrieval |
| `sql_agent` | Aggregations, exact filters |
| `code_agent` | Code generation / debugging |
| `legal_agent` | Forensic/compliance reasoning |
| `calculator` | Pure arithmetic (no LLM) |
| `conversational` | Lightweight chat |
| `hybrid` | Fallback — full pipeline |

---

## 2. HyDE — Hypothetical Document Embeddings

### The Problem
Embedding a 6-word query and comparing it to a 300-word technical paragraph creates a **structural mismatch** in vector space. Dense models learn from paragraph-to-paragraph similarity; query-to-paragraph similarity is a harder distribution.

### The Solution
Before touching the index, use a fast local LLM to generate a *hypothetical* document — a plausible-sounding answer even if factually wrong. The **structural patterns, vocabulary, and syntax** of the hypothetical document will closely match the target corpus.

```
Query: "How does BM25 work?"
        ↓  (embed directly → poor match)
Vector: [0.02, -0.14, 0.87, ...]  ← query-shaped

        ↓  HyDE transform
"BM25 is a probabilistic retrieval function that ranks documents
based on term frequency weighted by inverse document frequency..."
        ↓  (embed this)
Vector: [0.31, 0.08, 0.91, ...]  ← paragraph-shaped → better match
```

### Why It Works
Even a factually incorrect hypothetical document uses:
- The correct **domain vocabulary** (BM25, IDF, term frequency)
- The correct **sentence structure** (technical, declarative)
- The correct **granularity** (paragraph-length)

This dramatically shifts the search vector into the same region of embedding space as the ground-truth document.

### Key Parameters
| Parameter | Effect |
|---|---|
| `temperature=0.3` | Low randomness — consistent, factual-sounding output |
| `max_tokens=256` | Keeps hypothetical doc paragraph-length, not essay-length |
| `fallback_to_original=True` | Graceful degradation if Ollama is unavailable |

---

## 3. Advanced Chunking Strategies

### The Naive Approach Problem
Fixed 500-token windows routinely:
- Cut sentences mid-thought
- Separate a code block from its description
- Mix unrelated topics into one chunk (diluting the embedding)

### Strategy A: Semantic Chunking

**Mechanism**: Embed every sentence individually. Compute cosine similarity between consecutive sentence pairs. A drop below `similarity_threshold` signals a topic change → chunk boundary.

```python
embeddings = model.encode(sentences, normalize_embeddings=True)
for i in range(len(sentences) - 1):
    sim = cosine_similarity(embeddings[i], embeddings[i+1])
    if sim < self.similarity_threshold:
        breakpoints.append(i + 1)
```

**Result**: Chunks are semantically coherent — each chunk covers exactly one idea, making its embedding dense and precise.

### Strategy B: Parent-Child Retrieval (Auto-Merging)

**Mechanism**: Two-level chunk hierarchy.
- **Child chunks**: ~150 tokens (sentence-level) → used for vector search
- **Parent chunks**: ~512 tokens (paragraph-level) → fed to LLM

When a child chunk is retrieved, the system **automatically promotes** it to its parent before generation:

```python
if chunk.chunk_type == "child" and chunk.parent_id:
    parent = index.find(chunk.parent_id)
    use parent for LLM context
```

**Tradeoff Resolved**: Precision of retrieval + breadth of generation context.

### Strategy C: Fixed + Metadata Tagging

Every chunk, regardless of strategy, receives structured metadata:

```python
{
    "doc_id": "uuid",
    "source": "legal_contract_v2.pdf",
    "strategy": "semantic",
    "ingested_at": "2025-04-27T11:00:00Z",
    "chunk_index": 3,
    "total_chunks": 12,
    "version": "1.0"
}
```

This enables **pre-retrieval filtering** — before searching, narrow the candidate set by metadata constraints (date ranges, department, document lineage). Critical for forensic and compliance systems.

---

## 4. Hybrid Search with Reciprocal Rank Fusion

### The Fundamental Tension

| | Dense Vector | BM25 Sparse |
|---|---|---|
| **Strength** | Semantic similarity ("revenue drop" ≈ "financial loss") | Exact keyword matching (serial numbers, API names) |
| **Weakness** | Fails on rare/exact terms | Fails on paraphrased queries |
| **Score type** | Cosine similarity (-1 to 1) | TF-IDF frequency (unbounded) |

Neither is sufficient alone. Scores are **incomparable** — you can't average them.

### Reciprocal Rank Fusion (RRF)

RRF solves the score incompatibility by operating on **rank positions** instead of raw scores:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

```python
for doc_id in all_retrieved:
    score = 0.0
    if doc_id in dense_map:
        score += 1.0 / (k + dense_map[doc_id])   # dense rank contribution
    if doc_id in sparse_map:
        score += 1.0 / (k + sparse_map[doc_id])  # bm25 rank contribution
    rrf_scores[doc_id] = score
```

**The constant k=60**: Prevents the top-ranked document from receiving a disproportionately high score. A document ranked #1 gets `1/61 ≈ 0.016`, not `1/1 = 1.0`.

**Documents appearing in both lists** receive contributions from both rankers → naturally promoted. Documents in only one list still appear but with lower RRF scores.

### Why This Beats Score Normalization
- No calibration needed — works across any two rankers
- Robust to outliers (one extremely high BM25 score doesn't dominate)
- Proven in TREC benchmarks to outperform simple score fusion

---

## 5. Production Considerations

### For Legal / Forensic Systems (LegalOS pattern)
- Use **metadata filtering** with `metadata_filter` to enforce document boundary conditions
- Route queries through `legal_agent` for forensic reasoning chains
- Use **parent-child chunking** — child precision, parent context
- Deploy with **Qdrant** (payload filtering) or **pgvector** (SQL + vector joins)

### For Agent Swarms
- Each agent specializes on a route target (legal, code, SQL)
- Shared vector index with metadata-partitioned namespaces
- Router acts as the **orchestrator triage layer** — no agent sees irrelevant context

### For ERP / Structured Data (pgvector pattern)
```sql
-- Single query: semantic similarity + relational filter
SELECT content, embedding <=> $1 AS distance
FROM documents
WHERE department = 'Finance'
  AND updated_at > '2025-07-01'
ORDER BY distance
LIMIT 10;
```

### Local Embedding Model Recommendations
| Model | Context | Best For |
|---|---|---|
| `all-MiniLM-L6-v2` | 256 tokens | Fast dev, small docs |
| `nomic-embed-text` | 8192 tokens | Long legal/code docs |
| `BGE-M3` | 8192 tokens | Multi-lingual, max accuracy |
| Jina Embeddings | 8192 tokens | Code + technical docs |
