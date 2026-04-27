# System Architecture

## Overview

Advanced RAG Studio is a **production-grade Retrieval-Augmented Generation** platform built to move beyond naive single-vector pipelines. It implements four orthogonal improvements — semantic routing, HyDE query transformation, hybrid search with RRF, and parent-child retrieval — each independently togglable.

---

## High-Level Pipeline

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           STAGE 1: Semantic Router          │
│  Embedding-based intent classifier          │
│  Routes → vector_db / sql_agent /           │
│           code_agent / legal_agent /        │
│           calculator / conversational       │
└──────────────────────┬──────────────────────┘
                       │ (if retrieval needed)
                       ▼
┌─────────────────────────────────────────────┐
│        STAGE 2: HyDE Transformation         │
│  LLM generates hypothetical document        │
│  Embeds hypothetical doc → search vector    │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│         STAGE 3: Hybrid Search              │
│                                             │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Dense Vector   │  │   BM25 Sparse    │  │
│  │  (cosine sim)   │  │  (term freq)     │  │
│  └────────┬────────┘  └────────┬─────────┘  │
│           │                   │             │
│           └──────────┬────────┘             │
│                      ▼                      │
│            Reciprocal Rank Fusion           │
│         RRF(d) = Σ 1/(k + r(d))            │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│         STAGE 4: Auto-Merge                 │
│  Child chunks → promoted to Parent          │
│  (full paragraph context for LLM)           │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│         STAGE 5: LLM Generation             │
│  Grounded answer via local Ollama model     │
│  Context = retrieved chunks injected        │
└─────────────────────────────────────────────┘
```

---

## Module Map

```
advancedrag/
├── core/
│   ├── __init__.py          # Package
│   ├── pipeline.py          # Orchestrator — wires all stages
│   ├── router.py            # Semantic Router (embedding classifier)
│   ├── hyde.py              # HyDE query transformer (Ollama)
│   ├── hybrid_search.py     # Dense + BM25 + RRF search engine
│   └── chunker.py           # Semantic / Parent-Child / Fixed chunking
├── api/
│   └── server.py            # Flask REST API (all endpoints)
├── static/
│   ├── index.html           # Dashboard SPA
│   ├── css/dashboard.css    # Dark-mode design system
│   └── js/dashboard.js      # Pipeline trace UI
├── docs/
│   ├── ARCHITECTURE.md      # This file
│   ├── API_REFERENCE.md     # REST endpoint reference
│   └── TECHNIQUES.md        # Deep-dive on each RAG technique
├── requirements.txt
├── start.sh
└── README.md
```

---

## Chunking Strategy Comparison

| Strategy | Retrieval Unit | Context Unit | Best For |
|---|---|---|---|
| **Semantic** | Paragraph (topic-coherent) | Same | General documents, articles |
| **Parent-Child** | Sentence (child) | Paragraph (parent) | Technical docs, long-form content |
| **Fixed + Metadata** | N-token window | Same | Baseline, structured data |

---

## Semantic Router Decision Tree

```
Query arrives
     │
     ▼
Embed query → compare to route centroids
     │
     ├── similarity > threshold?
     │       YES → dispatch to matched RouteTarget
     │       NO  → fallback to HYBRID
     │
     ├── RouteTarget.CALCULATOR     → safe eval (no LLM)
     ├── RouteTarget.CONVERSATIONAL → lightweight LLM chat
     ├── RouteTarget.SQL_AGENT      → [future: SQL executor]
     ├── RouteTarget.CODE_AGENT     → [future: code interpreter]
     ├── RouteTarget.LEGAL_AGENT    → [future: forensic chain]
     └── RouteTarget.VECTOR_DB      → full retrieval pipeline
```

---

## RRF Formula

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

- **d** = document
- **R** = set of rankers (dense, sparse)
- **r(d)** = rank of document *d* in ranker *r*
- **k** = smoothing constant (default: 60)

The constant *k* prevents the highest-ranked document from dominating the fusion. Using rank positions instead of raw scores makes the fusion **score-agnostic** — cosine similarity (range: -1 to 1) and BM25 scores (unbounded) become directly comparable.
