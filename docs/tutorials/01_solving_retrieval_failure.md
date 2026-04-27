# Tutorial: Solving Retrieval Failure with Advanced RAG

This guide walks you through a common "Naive RAG" failure and shows how the Advanced RAG Studio uses **HyDE** and **Semantic Chunking** to solve it.

## 🔴 The Problem: The "Vocabulary Gap"
Imagine you are building a RAG system for a technical engineering manual.
- **The Document says**: *"The HNSW algorithm utilizes hierarchical graphs to ensure logarithmic search time for high-dimensional vectors."*
- **The User asks**: *"How do I make vector search faster?"*

In a **Naive RAG** system, a vector search might fail because the words "faster" and "make" are very different from "hierarchical graphs" and "logarithmic."

---

## 🟢 The Solution: Step-by-Step

### 1. Ingest with Semantic Chunking
Instead of a fixed window, use the **Semantic** strategy in the "Ingest Documents" view. This ensures that the explanation of HNSW remains a single, cohesive unit.
- **Action**: Paste your technical doc and select `Semantic`.

### 2. Enable the HyDE Transform
In the "Query Explorer," ensure the **HyDE Transform** toggle is ON.
- **What happens**: The system first asks a local LLM: *"Generate a technical paragraph about making vector search faster."*
- **The LLM replies**: *"Optimizing vector retrieval often involves using approximate nearest neighbor (ANN) algorithms like HNSW or IVF to reduce search latency..."*

### 3. The Retrieval Match
Now, the system embeds the **LLM's response**, not your short query. 
- The generated response contains keywords like `HNSW` and `latency`.
- These match the technical document perfectly.
- **Result**: You get the correct answer even though you didn't use the technical jargon in your query.

---

## 🛠️ Experiment: Try it yourself
1. Go to the **Ingest Documents** view.
2. Load the **Demo Corpus**.
3. Go to **Query Explorer**.
4. Ask: *"How does the system handle different topics?"*
5. Look at the **Pipeline Trace**:
    - See the **Semantic Router** identifying the "hybrid" route.
    - See **HyDE** expanding your short question into a detailed technical paragraph.
    - See **Hybrid Search** finding chunks from the "Advanced RAG reference" doc.

## 🎓 Summary
Advanced RAG isn't about better LLMs; it's about **better retrieval context**. By using this studio, you've moved from "hoping for a match" to "engineering a match."
