"""
Advanced RAG Pipeline — Orchestrator
Wires together: Semantic Router → HyDE → Hybrid Search → Auto-Merge → LLM Generation
"""

import time
import uuid
from typing import List, Optional, Dict, Any

from core.router import SemanticRouter, RouteTarget
from core.hyde import HyDETransformer
from core.hybrid_search import HybridSearchEngine, SearchResult
from core.chunker import AdvancedChunker, Chunk

try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


GENERATION_SYSTEM_PROMPT = """You are an expert knowledge assistant operating inside an advanced RAG pipeline.
You will be provided with retrieved context chunks from a knowledge base. Use ONLY the provided context to answer the user's question.

Rules:
- Be precise, technical, and deterministic.
- If the context does not contain the answer, say so explicitly — do NOT hallucinate.
- Cite which piece of context supports your answer (reference chunk index).
- Structure your response clearly: direct answer first, supporting details second.
"""

GENERATION_USER_TEMPLATE = """Context Chunks:
{context}

---
User Question: {query}

Provide a precise, well-structured answer based solely on the context above:"""


class AdvancedRAGPipeline:
    """
    Production-Grade Advanced RAG Orchestrator.

    Pipeline stages (all optional / configurable):
    ┌─────────────────────────────────────────────────────────┐
    │  [1] Semantic Router  → Triage query intent             │
    │  [2] HyDE Transform   → Expand query to hypo-doc        │
    │  [3] Hybrid Search    → Dense + BM25 + RRF              │
    │  [4] Auto-Merge       → Child→Parent context expansion  │
    │  [5] LLM Generation   → Grounded answer via Ollama      │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.2",
        use_router: bool = True,
        use_hyde: bool = True,
        use_hybrid: bool = True,
        use_auto_merge: bool = True,
        top_k: int = 6,
        rrf_k: int = 60,
        hyde_temperature: float = 0.3,
        router_threshold: float = 0.45,
        chunking_strategy: str = "semantic",  # "semantic" | "parent_child" | "fixed"
    ):
        self.config = {
            "embed_model": embed_model,
            "llm_model": llm_model,
            "use_router": use_router,
            "use_hyde": use_hyde,
            "use_hybrid": use_hybrid,
            "use_auto_merge": use_auto_merge,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "hyde_temperature": hyde_temperature,
            "router_threshold": router_threshold,
            "chunking_strategy": chunking_strategy,
        }

        self.chunker = AdvancedChunker(embed_model_name=embed_model)
        self.search_engine = HybridSearchEngine(
            embed_model_name=embed_model,
            rrf_k=rrf_k,
            top_k=top_k,
        )
        self.router = SemanticRouter(
            embed_model_name=embed_model,
            confidence_threshold=router_threshold,
        )
        self.hyde = HyDETransformer(
            model=llm_model,
            temperature=hyde_temperature,
        )
        self._router_built = False
        self.documents: List[Dict[str, Any]] = []  # Track ingested docs

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def ingest(
        self,
        text: str,
        source: str = "unknown",
        doc_id: Optional[str] = None,
        extra_metadata: Optional[Dict] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document into the vector index.
        Returns ingestion stats.
        """
        doc_id = doc_id or str(uuid.uuid4())
        strat = strategy or self.config["chunking_strategy"]
        t0 = time.perf_counter()

        meta = extra_metadata or {}
        meta["source"] = source

        if strat == "semantic":
            chunks = self.chunker.semantic_chunk(text, doc_id=doc_id, source=source, extra_metadata=meta)
        elif strat == "parent_child":
            chunks = self.chunker.parent_child_chunk(text, doc_id=doc_id, source=source, extra_metadata=meta)
        else:
            chunks = self.chunker.fixed_chunk(text, doc_id=doc_id, source=source, extra_metadata=meta)

        chunk_dicts = [c.to_dict() for c in chunks]
        indexed = self.search_engine.add_chunks(chunk_dicts)

        elapsed = (time.perf_counter() - t0) * 1000
        doc_record = {
            "doc_id": doc_id,
            "source": source,
            "strategy": strat,
            "chunk_count": indexed,
            "token_count": sum(c.token_count for c in chunks),
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "latency_ms": round(elapsed, 2),
        }
        self.documents.append(doc_record)
        return doc_record

    # ─── Query ───────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        generate_answer: bool = True,
        metadata_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full Advanced RAG pipeline for a given query.
        Returns rich trace with all intermediate pipeline steps.
        """
        trace: Dict[str, Any] = {
            "query_id": str(uuid.uuid4()),
            "original_query": query,
            "pipeline_stages": [],
            "retrieved_chunks": [],
            "answer": None,
            "total_latency_ms": 0.0,
        }
        pipeline_start = time.perf_counter()

        # ── Stage 1: Semantic Routing ─────────────────────────────────────
        route_decision = None
        if self.config["use_router"]:
            if not self._router_built:
                self.router.build()
                self._router_built = True
            route_decision = self.router.route(query)
            trace["pipeline_stages"].append({
                "stage": "semantic_routing",
                "result": route_decision.to_dict(),
            })

            # For calculator / conversational, skip heavy retrieval
            if route_decision.target == RouteTarget.CALCULATOR:
                return self._handle_calculator(query, trace, pipeline_start)
            if route_decision.target == RouteTarget.CONVERSATIONAL:
                return self._handle_conversational(query, trace, pipeline_start)

        # ── Stage 2: HyDE Query Transformation ───────────────────────────
        search_query = query
        if self.config["use_hyde"]:
            hyde_result = self.hyde.generate_hypothetical(query)
            trace["pipeline_stages"].append({
                "stage": "hyde_transformation",
                "result": hyde_result,
            })
            if hyde_result["success"]:
                search_query = hyde_result["hypothetical_doc"]

        # ── Stage 3: Hybrid Search (Dense + BM25 + RRF) ──────────────────
        t_search = time.perf_counter()
        results: List[SearchResult] = self.search_engine.search(
            query=search_query,
            metadata_filter=metadata_filter,
            use_hybrid=self.config["use_hybrid"],
            auto_merge_parents=self.config["use_auto_merge"],
        )
        search_latency = (time.perf_counter() - t_search) * 1000

        trace["pipeline_stages"].append({
            "stage": "hybrid_search",
            "search_query_used": search_query[:200],
            "results_count": len(results),
            "latency_ms": round(search_latency, 2),
        })
        trace["retrieved_chunks"] = [r.to_dict() for r in results]

        # ── Stage 4: LLM Generation ───────────────────────────────────────
        if generate_answer and results:
            answer_result = self._generate_answer(query, results)
            trace["pipeline_stages"].append({
                "stage": "llm_generation",
                "result": answer_result,
            })
            trace["answer"] = answer_result.get("answer")

        trace["total_latency_ms"] = round(
            (time.perf_counter() - pipeline_start) * 1000, 2
        )
        return trace

    # ─── LLM Generation ──────────────────────────────────────────────────────

    def _generate_answer(
        self, query: str, chunks: List[SearchResult]
    ) -> Dict[str, Any]:
        if not OLLAMA_AVAILABLE:
            return {"answer": "[Ollama not available — install with: pip install ollama]", "success": False}

        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[{i+1}] (source: {chunk.metadata.get('source', 'unknown')}, "
                f"score: {chunk.rrf_score:.4f})\n{chunk.content}"
            )
        context_str = "\n\n".join(context_parts)

        t0 = time.perf_counter()
        try:
            response = ollama_client.chat(
                model=self.config["llm_model"],
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": GENERATION_USER_TEMPLATE.format(
                            context=context_str, query=query
                        ),
                    },
                ],
                options={"temperature": 0.1},
            )
            latency = (time.perf_counter() - t0) * 1000
            return {
                "answer": response["message"]["content"].strip(),
                "model": self.config["llm_model"],
                "latency_ms": round(latency, 2),
                "success": True,
                "context_chunks_used": len(chunks),
            }
        except Exception as exc:
            return {
                "answer": f"[Generation error: {exc}]",
                "success": False,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

    # ─── Special Route Handlers ───────────────────────────────────────────────

    def _handle_calculator(self, query: str, trace: dict, t0: float) -> dict:
        """Basic safe eval for arithmetic queries."""
        import re
        expr = re.sub(r"[^\d\s\+\-\*\/\(\)\.\%]", "", query)
        try:
            result = eval(compile(expr, "<string>", "eval"))
            answer = f"= {result}"
        except Exception:
            answer = "Unable to evaluate the expression. Please provide a valid arithmetic expression."
        trace["answer"] = answer
        trace["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return trace

    def _handle_conversational(self, query: str, trace: dict, t0: float) -> dict:
        """Lightweight LLM conversational response."""
        if OLLAMA_AVAILABLE:
            try:
                resp = ollama_client.chat(
                    model=self.config["llm_model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful, concise assistant."},
                        {"role": "user", "content": query},
                    ],
                    options={"temperature": 0.7, "num_predict": 150},
                )
                trace["answer"] = resp["message"]["content"].strip()
            except Exception as e:
                trace["answer"] = f"Hello! I'm your Advanced RAG assistant. ({e})"
        else:
            trace["answer"] = "Hello! I'm your Advanced RAG assistant. How can I help you today?"
        trace["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return trace

    # ─── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "index_stats": self.search_engine.get_stats(),
            "documents_ingested": len(self.documents),
            "documents": self.documents,
            "config": self.config,
        }

    def list_available_models(self) -> List[str]:
        return self.hyde.list_local_models()
