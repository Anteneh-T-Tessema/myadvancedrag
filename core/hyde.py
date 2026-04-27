"""
HyDE — Hypothetical Document Embeddings
Generates a hallucinated-but-structurally-correct answer to bridge the
vocabulary gap between short user queries and dense technical documents.
"""

from typing import Optional, Dict, Any
import time

try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


HYDE_SYSTEM_PROMPT = """You are a technical document synthesis engine.
Your task is to generate a dense, factual-sounding passage that would appear in an expert technical document or knowledge base, as if it were a direct answer to the query below.

Rules:
- Use domain-appropriate vocabulary, jargon, and syntax.
- Write 2–4 sentences. Be concise but information-dense.
- Do NOT say "I think" or hedge. Write as if from authoritative documentation.
- Even if you are uncertain, produce a plausible-sounding technical passage.
- Focus on structural patterns that would match relevant documents — accuracy to ground truth is secondary to retrieval signal quality.
"""

HYDE_USER_TEMPLATE = """Query: {query}

Generate the hypothetical document passage:"""


class HyDETransformer:
    """
    Query Transformation via Hypothetical Document Embeddings.

    Standard flow:
        raw_query → LLM generates hypothetical answer → embed hypothetical answer → search

    Advantage: Dense similarity between paragraph-length texts rather than
    query-to-document mismatch.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.3,
        max_tokens: int = 256,
        fallback_to_original: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_to_original = fallback_to_original

    def generate_hypothetical(self, query: str) -> Dict[str, Any]:
        """
        Generate a hypothetical document for the given query.

        Returns dict with:
            - original_query: str
            - hypothetical_doc: str (the generated passage)
            - model_used: str
            - latency_ms: float
            - success: bool
            - error: Optional[str]
        """
        if not OLLAMA_AVAILABLE:
            return self._error_response(query, "ollama package not installed")

        start = time.perf_counter()
        try:
            response = ollama_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": HYDE_USER_TEMPLATE.format(query=query),
                    },
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            latency = (time.perf_counter() - start) * 1000
            hypo_doc = response["message"]["content"].strip()
            return {
                "original_query": query,
                "hypothetical_doc": hypo_doc,
                "model_used": self.model,
                "latency_ms": round(latency, 2),
                "success": True,
                "error": None,
            }

        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            fallback = query if self.fallback_to_original else ""
            return {
                "original_query": query,
                "hypothetical_doc": fallback,
                "model_used": self.model,
                "latency_ms": round(latency, 2),
                "success": False,
                "error": str(exc),
            }

    def transform(self, query: str) -> str:
        """
        Returns the hypothetical document to use as the search vector.
        Falls back to original query if generation fails and fallback_to_original=True.
        """
        result = self.generate_hypothetical(query)
        if result["success"] and result["hypothetical_doc"]:
            return result["hypothetical_doc"]
        if self.fallback_to_original:
            return query
        raise RuntimeError(f"HyDE generation failed: {result['error']}")

    def list_local_models(self) -> list:
        """List available local Ollama models."""
        if not OLLAMA_AVAILABLE:
            return []
        try:
            models_response = ollama_client.list()
            return [m.model for m in models_response.models]
        except Exception:
            return []

    def _error_response(self, query: str, error: str) -> Dict[str, Any]:
        return {
            "original_query": query,
            "hypothetical_doc": query if self.fallback_to_original else "",
            "model_used": self.model,
            "latency_ms": 0.0,
            "success": False,
            "error": error,
        }
