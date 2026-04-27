# Tutorial: Hybrid Search & RRF Deep Dive

Hybrid Search is the "Golden Path" for RAG accuracy. It combines semantic understanding (Dense) with keyword precision (Sparse).

## ⚖️ The Balanced Retrieval Problem
- **Vector Search** is great for concepts but fails on acronyms like "HNSW-9" or unique IDs.
- **Keyword Search (BM25)** is great for "HNSW-9" but fails if the user says "Fast vector indexing" instead.

---

## 🛠️ Step-by-Step: Validating Fusion

### 1. Ingest a Mixed Document
Ingest a document that has both heavy jargon and clear conceptual definitions.

### 2. Configure the Toggles
In the **Query Explorer**, ensure **Hybrid Search** is toggled ON.
- **Behind the scenes**: The system runs two searches in parallel.
- **Fusion**: It uses **Reciprocal Rank Fusion (RRF)** to merge the results.

### 3. Analyze the Chunk Scores
Run a query and look at a retrieved **Chunk Card**. You will see:
- **Dense Score**: The semantic similarity.
- **BM25 Score**: The keyword frequency score.
- **RRF Score**: The final merged rank.

### 4. Tuning RRF K
Go to **Pipeline Config** and adjust the **RRF K** parameter (default 60).
- **Lower K (e.g., 20)**: Heavily favors the top 1-2 results from either ranker.
- **Higher K (e.g., 100)**: Smoothes the ranks, allowing lower-ranked documents to surface if they appear in both lists.

## 🎓 Summary
Hybrid search ensures your RAG system never "ignores" an exact keyword match while still being smart enough to understand the user's intent. It is the single most important technique for production stability.
