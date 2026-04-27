"""
Semantic Router — Intent-based query triage layer.
Routes queries to the optimal retrieval pipeline, agent, or datastore
based on embedding-similarity to intent categories.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class RouteTarget(str, Enum):
    """Supported route targets in the Advanced RAG system."""
    VECTOR_DB       = "vector_db"          # Standard semantic retrieval
    SQL_AGENT       = "sql_agent"          # Structured / numerical queries
    CALCULATOR      = "calculator"         # Pure math / computation
    CODE_AGENT      = "code_agent"         # Code generation / execution
    LEGAL_AGENT     = "legal_agent"        # Forensic / legal reasoning
    SUMMARIZER      = "summarizer"         # Document summarization
    CONVERSATIONAL  = "conversational"     # Chitchat / short Q&A
    HYBRID          = "hybrid"             # Default: full hybrid pipeline


@dataclass
class Route:
    """Definition of a single named route with example utterances."""
    name: str
    target: RouteTarget
    description: str
    examples: List[str]
    _centroid: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target.value,
            "description": self.description,
            "examples": self.examples,
        }


@dataclass
class RouterDecision:
    """Result of routing a query."""
    query: str
    matched_route: str
    target: RouteTarget
    confidence: float
    all_scores: Dict[str, float]
    fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "matched_route": self.matched_route,
            "target": self.target.value,
            "confidence": round(self.confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "fallback": self.fallback,
        }


# ─── Default Route Library ───────────────────────────────────────────────────

DEFAULT_ROUTES: List[Route] = [
    Route(
        name="numerical_sql",
        target=RouteTarget.SQL_AGENT,
        description="Queries requiring aggregation, filtering, or exact numerical lookups.",
        examples=[
            "What is the total revenue for Q3 2024?",
            "How many invoices were issued last month?",
            "List all transactions above $10,000 in the finance department.",
            "What is the average contract value by region?",
            "Count the number of active users by subscription tier.",
            "Show me all records where status is pending and amount > 5000.",
        ],
    ),
    Route(
        name="code_generation",
        target=RouteTarget.CODE_AGENT,
        description="Queries about code, APIs, programming, debugging, or software architecture.",
        examples=[
            "Write a Python function to parse JWT tokens.",
            "How do I implement a BM25 retriever in LangChain?",
            "Debug this TypeScript error: cannot read property of undefined.",
            "What is the difference between async/await and Promise chaining?",
            "Generate a REST API endpoint for user authentication.",
            "Explain what HNSW index is and how it works.",
        ],
    ),
    Route(
        name="legal_forensic",
        target=RouteTarget.LEGAL_AGENT,
        description="Legal analysis, contract review, compliance, and forensic reasoning.",
        examples=[
            "Analyze this clause for indemnification obligations.",
            "What are the GDPR compliance requirements for data retention?",
            "Does this contract contain a force majeure clause?",
            "Identify potential liability issues in this agreement.",
            "What does the arbitration clause mean for dispute resolution?",
            "Review this NDA for confidentiality breaches.",
        ],
    ),
    Route(
        name="calculation",
        target=RouteTarget.CALCULATOR,
        description="Pure mathematical calculations or unit conversions.",
        examples=[
            "What is 15% of 234,500?",
            "Convert 45 Celsius to Fahrenheit.",
            "Calculate compound interest at 7% over 10 years on $50,000.",
            "What is 2^32?",
            "Square root of 144?",
        ],
    ),
    Route(
        name="summarization",
        target=RouteTarget.SUMMARIZER,
        description="Summarize, condense, or extract key points from long documents.",
        examples=[
            "Summarize this 50-page report in 5 bullet points.",
            "Give me the key takeaways from this earnings call transcript.",
            "What are the main findings of this research paper?",
            "TL;DR this document.",
            "Extract the action items from this meeting transcript.",
        ],
    ),
    Route(
        name="conversational",
        target=RouteTarget.CONVERSATIONAL,
        description="Casual conversation, greetings, or very simple questions.",
        examples=[
            "Hello, how are you?",
            "What can you do?",
            "Who are you?",
            "Thanks!",
            "That's helpful.",
            "Can you help me?",
        ],
    ),
    Route(
        name="semantic_retrieval",
        target=RouteTarget.VECTOR_DB,
        description="Conceptual, research, or knowledge retrieval questions.",
        examples=[
            "What are the best practices for RAG pipelines?",
            "Explain how attention mechanisms work in transformers.",
            "What is semantic chunking and why is it important?",
            "How does HyDE improve retrieval accuracy?",
            "What is Reciprocal Rank Fusion?",
            "Describe the architecture of an agent swarm.",
        ],
    ),
]


class SemanticRouter:
    """
    Embedding-based intent classifier that routes queries to the correct
    pipeline target without invoking a full LLM.

    Flow:
    1. Build centroid embeddings for each route's example utterances.
    2. On incoming query: embed → cosine similarity vs each centroid.
    3. Highest score (if above threshold) wins; else fallback to HYBRID.
    """

    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = 0.45,
        routes: Optional[List[Route]] = None,
    ):
        self.threshold = confidence_threshold
        self._model_name = embed_model_name
        self._model: Optional["SentenceTransformer"] = None
        self.routes: List[Route] = routes if routes is not None else list(DEFAULT_ROUTES)
        self._built = False

    def _get_model(self) -> Optional["SentenceTransformer"]:
        if self._model is None and ST_AVAILABLE:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def build(self) -> "SemanticRouter":
        """Compute centroid embeddings for all route examples."""
        model = self._get_model()
        if model is None:
            return self

        for route in self.routes:
            embeddings = model.encode(route.examples, normalize_embeddings=True)
            route._centroid = np.mean(embeddings, axis=0)
            # Re-normalize centroid
            norm = np.linalg.norm(route._centroid)
            if norm > 0:
                route._centroid = route._centroid / norm

        self._built = True
        return self

    def route(self, query: str) -> RouterDecision:
        """
        Route a query to the most appropriate target.
        Returns a RouterDecision with confidence scores.
        """
        if not self._built:
            self.build()

        model = self._get_model()
        if model is None or not self.routes:
            return self._fallback(query, {})

        q_emb = model.encode([query], normalize_embeddings=True)[0]
        scores: Dict[str, float] = {}

        for route in self.routes:
            if route._centroid is not None:
                sim = float(np.dot(q_emb, route._centroid))
                scores[route.name] = sim

        if not scores:
            return self._fallback(query, scores)

        best_name = max(scores, key=lambda x: scores[x])
        best_score = scores[best_name]
        best_route = next(r for r in self.routes if r.name == best_name)

        if best_score < self.threshold:
            return self._fallback(query, scores)

        return RouterDecision(
            query=query,
            matched_route=best_name,
            target=best_route.target,
            confidence=best_score,
            all_scores=scores,
            fallback=False,
        )

    def _fallback(self, query: str, scores: Dict[str, float]) -> RouterDecision:
        return RouterDecision(
            query=query,
            matched_route="hybrid",
            target=RouteTarget.HYBRID,
            confidence=0.0,
            all_scores=scores,
            fallback=True,
        )

    def add_route(self, route: Route) -> None:
        """Add a custom route and rebuild centroids."""
        self.routes.append(route)
        self._built = False

    def get_routes_info(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.routes]


# ─── Quick smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    router = SemanticRouter()
    router.build()
    tests = [
        "What is the total revenue for this quarter?",
        "Write a function to sort a list in Python.",
        "Analyze this indemnification clause.",
        "What is 15% of 45,000?",
        "Hello!",
        "Explain how hybrid search with RRF works.",
    ]
    for q in tests:
        d = router.route(q)
        print(f"[{d.target.value:20s}] ({d.confidence:.3f}) → {q}")
