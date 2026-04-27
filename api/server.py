"""
Advanced RAG — Flask REST API
Exposes the full pipeline via JSON endpoints consumable by the frontend.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS

from core.pipeline import AdvancedRAGPipeline
from core.router import SemanticRouter
from core.hyde import HyDETransformer
from core.hardware import HardwareInspector, POPULAR_OLLAMA_MODELS
import ollama as ollama_client

app = Flask(__name__, static_folder="../static", static_url_path="/static")
CORS(app)

# ─── Global Pipeline Instance ─────────────────────────────────────────────────
_pipeline: AdvancedRAGPipeline = None


def get_pipeline() -> AdvancedRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AdvancedRAGPipeline()
    return _pipeline


# ─── Health ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    p = get_pipeline()
    models = p.list_available_models()
    stats = p.get_stats()
    return jsonify({
        "status": "ok",
        "ollama_models": models,
        "index_size": stats["index_stats"]["total_chunks"],
        "documents_count": stats["documents_ingested"],
    })


# ─── Config ───────────────────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def get_config():
    p = get_pipeline()
    return jsonify(p.config)


@app.route("/api/config", methods=["POST"])
def update_config():
    """Recreate the pipeline with new config."""
    global _pipeline
    data = request.get_json(force=True)
    allowed = {
        "embed_model", "llm_model", "use_router", "use_hyde",
        "use_hybrid", "use_auto_merge", "top_k", "rrf_k",
        "hyde_temperature", "router_threshold", "chunking_strategy",
    }
    kwargs = {k: v for k, v in data.items() if k in allowed}
    _pipeline = AdvancedRAGPipeline(**kwargs)
    return jsonify({"status": "ok", "config": _pipeline.config})


# ─── Ingest ───────────────────────────────────────────────────────────────────

@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Ingest a document.
    Body: { text, source?, doc_id?, strategy?, extra_metadata? }
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    result = get_pipeline().ingest(
        text=text,
        source=data.get("source", "api_upload"),
        doc_id=data.get("doc_id"),
        extra_metadata=data.get("extra_metadata"),
        strategy=data.get("strategy"),
    )
    return jsonify(result)


@app.route("/api/ingest/demo", methods=["POST"])
def ingest_demo():
    """Load the built-in demo corpus for quick testing."""
    pipeline = get_pipeline()
    demo_docs = _get_demo_corpus()
    results = []
    for doc in demo_docs:
        r = pipeline.ingest(
            text=doc["text"],
            source=doc["source"],
            strategy=doc.get("strategy", "semantic"),
        )
        results.append(r)
    return jsonify({"loaded": len(results), "documents": results})


# ─── Query ────────────────────────────────────────────────────────────────────

@app.route("/api/query", methods=["POST"])
def query():
    """
    Execute the Advanced RAG pipeline.
    Body: { query, generate_answer?, metadata_filter? }
    """
    data = request.get_json(force=True)
    q = data.get("query", "").strip()
    if not q:
        return jsonify({"error": "query is required"}), 400

    result = get_pipeline().query(
        query=q,
        generate_answer=data.get("generate_answer", True),
        metadata_filter=data.get("metadata_filter"),
    )
    return jsonify(result)


# ─── Router ───────────────────────────────────────────────────────────────────

@app.route("/api/router/test", methods=["POST"])
def test_router():
    """Test routing for a query without triggering full retrieval."""
    data = request.get_json(force=True)
    q = data.get("query", "").strip()
    if not q:
        return jsonify({"error": "query is required"}), 400

    p = get_pipeline()
    if not p._router_built:
        p.router.build()
        p._router_built = True
    decision = p.router.route(q)
    return jsonify(decision.to_dict())


@app.route("/api/router/routes", methods=["GET"])
def get_routes():
    """List all configured routes."""
    p = get_pipeline()
    return jsonify(p.router.get_routes_info())


# ─── HyDE ─────────────────────────────────────────────────────────────────────

@app.route("/api/hyde/transform", methods=["POST"])
def hyde_transform():
    """Transform a query using HyDE (without full retrieval)."""
    data = request.get_json(force=True)
    q = data.get("query", "").strip()
    if not q:
        return jsonify({"error": "query is required"}), 400

    p = get_pipeline()
    result = p.hyde.generate_hypothetical(q)
    return jsonify(result)


# ─── Stats ────────────────────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(get_pipeline().get_stats())


@app.route("/api/stats/clear", methods=["POST"])
def clear_index():
    """Clear the vector index (keeps config)."""
    p = get_pipeline()
    p.search_engine.clear()
    p.documents = []
    return jsonify({"status": "cleared"})


# ─── Models ───────────────────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({"models": get_pipeline().list_available_models()})


@app.route("/api/hardware", methods=["GET"])
def get_hardware():
    inspector = HardwareInspector()
    profile = inspector.inspect()
    return jsonify(profile.to_dict())


@app.route("/api/models/popular", methods=["GET"])
def get_popular_models():
    return jsonify(POPULAR_OLLAMA_MODELS)


@app.route("/api/models/pull", methods=["POST"])
def pull_model():
    data = request.get_json(force=True)
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "model name is required"}), 400

    try:
        # Use the ollama python client to pull the model
        # Note: In a real production app, this should be async/streaming
        # but for this demo we'll trigger the pull.
        ollama_client.pull(model_name)
        return jsonify({"status": "success", "message": f"Started pulling {model_name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Demo Corpus ──────────────────────────────────────────────────────────────

def _get_demo_corpus():
    return [
        {
            "source": "rag_fundamentals.txt",
            "strategy": "semantic",
            "text": """
Retrieval-Augmented Generation (RAG) is a paradigm that combines a retrieval component with a generative language model.
Instead of relying solely on the model's parametric memory, RAG retrieves relevant documents from an external knowledge base and conditions the generator on them.
This dramatically reduces hallucination rates and allows the system to cite sources.

The basic RAG pipeline consists of three stages: indexing, retrieval, and generation.
During indexing, documents are split into chunks, converted to vector embeddings, and stored in a vector database.
During retrieval, the user query is embedded and the nearest chunks are fetched via approximate nearest neighbor search.
During generation, the retrieved chunks are injected into the prompt context and the LLM generates a grounded answer.

Advanced RAG moves beyond this naive architecture by introducing query transformation, semantic routing, hybrid search, and parent-child retrieval strategies.
These improvements address the fundamental weaknesses of basic RAG: vocabulary mismatch, context loss from naive chunking, and the inability to handle structured or exact-match queries.
""",
        },
        {
            "source": "hyde_explanation.txt",
            "strategy": "semantic",
            "text": """
Hypothetical Document Embeddings (HyDE) is a query transformation technique that dramatically improves retrieval recall for ambiguous or technical queries.
The core insight is that embedding a short user question and comparing it against long document paragraphs creates a structural mismatch in the vector space.
Dense similarity works best when comparing objects of similar granularity.

HyDE addresses this by prompting a fast language model to generate a hypothetical document that looks like a plausible answer to the query.
Even if factually incorrect, this hallucinated document uses domain-appropriate vocabulary, syntactic patterns, and jargon that align with the target document corpus.
The system then embeds this hypothetical document rather than the original short query, using its dense vector for similarity search.

Research has shown that HyDE improves retrieval metrics like Recall@K and NDCG by a significant margin on technical and scientific corpora.
It is particularly effective for highly specific technical documentation where user queries are tersely phrased but target documents are densely written.
""",
        },
        {
            "source": "hybrid_search_rrf.txt",
            "strategy": "semantic",
            "text": """
Hybrid search combines dense vector retrieval with sparse lexical search to achieve both semantic understanding and precise keyword matching.
Dense retrieval excels at conceptual similarity — understanding that "financial losses" and "revenue decline" are semantically related.
However, dense models struggle with exact entity matching: specific serial numbers, API endpoints, legal citations, or technical acronyms.

BM25 (Best Match 25) is a classical sparse retrieval algorithm based on term frequency and inverse document frequency.
It assigns high scores to documents containing rare query terms that appear frequently in the document relative to the corpus.
BM25 is deterministic, highly interpretable, and blazingly fast compared to dense retrieval.

Reciprocal Rank Fusion (RRF) merges the ranked lists from dense and sparse retrieval into a unified ranking.
The formula RRF(d) = sum(1 / (k + r(d))) for each ranker r gives weight based on document rank position rather than raw scores.
This avoids the incompatibility problem between cosine similarity scores and BM25 frequency scores.
The smoothing constant k (typically 60) prevents outlier high-ranked documents from dominating the final ranking.
""",
        },
        {
            "source": "semantic_routing.txt",
            "strategy": "semantic",
            "text": """
Semantic routing is the architectural triage layer of an advanced RAG system.
When a query enters the system, a router classifies its intent and directs it to the most appropriate pipeline, agent, or datastore.
This prevents the expensive mistake of running every query through the same heavy retrieval and generation pipeline.

The routing mechanism uses embedding similarity: the query is compared against centroid embeddings computed from labeled example utterances.
Each route has a set of examples representing queries of that intent type.
If the query's embedding is sufficiently close to a route centroid (above the confidence threshold), the query is dispatched to that route's target.

Common route targets include: vector database retrieval, SQL agents, code execution agents, legal reasoning agents, and conversational handlers.
A query asking for Q3 revenue should go to a SQL agent, not the vector database.
A query about debugging a Python function should route to a code agent with a code-specialized model.
This specialization improves accuracy, reduces hallucination, and dramatically reduces per-query latency.
""",
        },
        {
            "source": "parent_child_chunking.txt",
            "strategy": "parent_child",
            "text": """
Parent-child retrieval, also called auto-merging retrieval, is a chunking strategy that separates the granularity used for retrieval from the granularity used for generation context.
The fundamental tension in RAG chunking is that smaller chunks produce more precise vector matches but provide insufficient context for the LLM to reason effectively.
Larger chunks provide rich context but reduce vector search precision because the embedding must capture more information.

Parent-child retrieval resolves this tension by creating two levels of chunks from the same document.
Child chunks are very fine-grained — often individual sentences — optimized for precise semantic vector matching.
Parent chunks are larger — paragraphs or sections — optimized for providing coherent context to the generation model.

When a child chunk is retrieved during search, the system automatically fetches its parent chunk instead for the generation stage.
This gives the LLM the surrounding context needed for accurate, deterministic reasoning while preserving the precision of child-level vector matching.
The parent-child relationship is tracked through metadata fields linking each child to its parent identifier.

Semantic chunking takes a different approach: instead of fixed token boundaries, it analyzes sentence-level embeddings to find natural topic shifts.
A sharp drop in cosine similarity between consecutive sentence embeddings signals a change in topic, triggering a chunk boundary.
This preserves semantic coherence within each chunk, ensuring that related ideas are never artificially separated by an arbitrary token count.
""",
        },
        {
            "source": "vector_stores_comparison.txt",
            "strategy": "fixed",
            "text": """
Vector databases are specialized storage systems optimized for high-dimensional embedding vectors.
They use approximate nearest neighbor (ANN) algorithms to enable sub-millisecond similarity search over millions of vectors.
The most common ANN algorithm is HNSW (Hierarchical Navigable Small World), which builds a multi-layer graph for logarithmic-time search.

Qdrant is a Rust-based vector database optimized for local deployment and agent swarm architectures.
Its primary advantage is advanced payload filtering: you can combine vector similarity search with strict metadata constraints in a single query.
This is essential for forensic and legal systems where results must be bounded by document lineage, date ranges, or clearance levels.
Qdrant runs efficiently on Apple Silicon and NVIDIA GPUs, making it ideal for local AI development.

pgvector is a PostgreSQL extension that stores vector embeddings alongside relational data.
Its killer feature is the ability to combine vector similarity search with standard SQL predicates in a single query.
For ERP systems and other highly relational domains, this means you can filter by business logic (department, date, status) while simultaneously ranking by semantic similarity.
The ACID compliance guarantees of PostgreSQL make pgvector the right choice for enterprise systems requiring deterministic, auditable retrieval.

Milvus is designed for enterprise-scale deployments with billions of vectors and high-throughput requirements.
Pinecone is a fully managed cloud service offering zero-ops infrastructure, suitable for rapid prototyping where data sovereignty is not a concern.
For production systems handling sensitive legal or financial data, self-hosted options like Qdrant or pgvector are strongly preferred.
""",
        },
        {
            "source": "embedding_models.txt",
            "strategy": "semantic",
            "text": """
Embedding models convert text into dense numerical vectors that capture semantic meaning.
The choice of embedding model is foundational to RAG system performance — it determines the quality of the semantic space in which all retrieval operates.

BGE-M3 from BAAI is among the most capable open-source embedding models available.
It supports multi-lingual retrieval across over 100 languages, multi-granularity encoding from sentences to full documents, and three representation modes: dense, sparse, and multi-vector ColBERT style.
BGE-M3 achieves state-of-the-art performance on BEIR benchmarks and is an excellent choice for complex retrieval tasks requiring high precision.

Nomic Embed (nomic-embed-text) is highly optimized for local execution and natively supported by Ollama.
Its 8192-token context window allows embedding large chunks of code, legal text, or technical documentation without truncation.
This makes it particularly valuable for parent-child retrieval where parent chunks may be several hundred tokens.

Jina Embeddings are specifically optimized for code and technical documentation retrieval.
They are trained on a mix of code repositories, API documentation, and technical manuals, giving them a structural advantage when the knowledge base contains complex software artifacts.

For local, privacy-preserving deployments on Apple Silicon, models in the 100–400M parameter range like all-MiniLM-L6-v2 offer an excellent speed-accuracy tradeoff.
For maximum accuracy in production, BGE-M3 or Nomic Embed run via Ollama are recommended.
""",
        },
    ]


if __name__ == "__main__":
    print("🚀 Advanced RAG API starting on http://localhost:7891")
    app.run(host="0.0.0.0", port=7891, debug=False)
