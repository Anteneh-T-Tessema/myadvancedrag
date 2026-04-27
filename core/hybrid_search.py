"""
Hybrid Search Engine — Dense Vector + BM25 Sparse Retrieval
with Reciprocal Rank Fusion (RRF) re-ranking.

Dense: sentence-transformers cosine similarity (in-memory FAISS-like)
Sparse: BM25 via rank_bm25
Fusion: RRF with configurable smoothing constant k
"""

import math
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


@dataclass
class SearchResult:
    """Unified search result after RRF fusion."""
    chunk_id: str
    content: str
    doc_id: str
    metadata: Dict[str, Any]
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    chunk_type: str = "unknown"
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "doc_id": self.doc_id,
            "metadata": self.metadata,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rrf_score": round(self.rrf_score, 6),
            "dense_score": round(self.dense_score, 6),
            "sparse_score": round(self.sparse_score, 6),
            "chunk_type": self.chunk_type,
            "parent_id": self.parent_id,
        }


class HybridSearchEngine:
    """
    Production Hybrid Retrieval with RRF.

    Architecture:
    ┌──────────┐     ┌──────────────────────┐     ┌────────────────────┐
    │  Query   │────▶│  Dense Vector Search │     │ BM25 Sparse Search │
    └──────────┘     └──────────┬───────────┘     └────────┬───────────┘
                                │  ranked list              │ ranked list
                                └──────────────┬────────────┘
                                               ▼
                                     ┌──────────────────┐
                                     │  RRF Fusion (k)  │
                                     └────────┬─────────┘
                                              ▼
                                     Unified ranked results
    """

    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        top_k: int = 10,
    ):
        self.rrf_k = rrf_k
        self.top_k = top_k
        self._model_name = embed_model_name
        self._model: Optional["SentenceTransformer"] = None

        # In-memory index: list of (chunk_dict, embedding_vector)
        self._index: List[Tuple[Dict[str, Any], np.ndarray]] = []
        self._bm25: Optional["BM25Okapi"] = None
        self._corpus_tokens: List[List[str]] = []
        self._corpus_chunks: List[Dict[str, Any]] = []

    # ─── Model ──────────────────────────────────────────────────────────────

    def _get_model(self) -> Optional["SentenceTransformer"]:
        if self._model is None and ST_AVAILABLE:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ─── Indexing ───────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and index chunks for both dense and sparse retrieval.
        chunks must have: chunk_id, content, doc_id, metadata, chunk_type.
        Returns number of chunks indexed.
        """
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        model = self._get_model()

        if model:
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            # Zero embeddings fallback (BM25 still works)
            embeddings = np.zeros((len(texts), 384))

        for chunk, emb in zip(chunks, embeddings):
            self._index.append((chunk, emb))

        # Rebuild BM25 corpus
        for chunk in chunks:
            tokens = chunk["content"].lower().split()
            self._corpus_tokens.append(tokens)
            self._corpus_chunks.append(chunk)

        if BM25_AVAILABLE and self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)

        return len(chunks)

    def clear(self):
        self._index = []
        self._bm25 = None
        self._corpus_tokens = []
        self._corpus_chunks = []

    @property
    def index_size(self) -> int:
        return len(self._index)

    # ─── Dense Search ────────────────────────────────────────────────────────

    def _dense_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float]]:
        """Returns list of (global_index, cosine_score) sorted by score desc."""
        if not self._index:
            return []

        model = self._get_model()
        if model is None:
            return []

        q_emb = model.encode([query], normalize_embeddings=True)[0]

        scores = []
        for idx, (chunk, emb) in enumerate(self._index):
            if metadata_filter and not self._match_filter(chunk["metadata"], metadata_filter):
                continue
            score = float(np.dot(q_emb, emb))
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ─── BM25 Sparse Search ──────────────────────────────────────────────────

    def _sparse_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float]]:
        """Returns list of (global_index, bm25_score) sorted by score desc."""
        if not self._bm25 or not self._corpus_chunks:
            return []

        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)

        results = []
        for corpus_idx, score in enumerate(bm25_scores):
            chunk = self._corpus_chunks[corpus_idx]
            if metadata_filter and not self._match_filter(chunk["metadata"], metadata_filter):
                continue
            # Map corpus_idx to global _index idx (they're built in same order)
            results.append((corpus_idx, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ─── RRF Fusion ──────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        top_k: int,
    ) -> List[Tuple[int, float, int, int]]:
        """
        Reciprocal Rank Fusion.
        RRF(d) = Σ 1 / (k + r(d))  for each ranker r

        Returns list of (global_idx, rrf_score, dense_rank, sparse_rank)
        """
        k = self.rrf_k
        dense_map = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}
        sparse_map = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_results)}

        all_ids = set(dense_map) | set(sparse_map)
        rrf_scores = {}
        for doc_id in all_ids:
            score = 0.0
            if doc_id in dense_map:
                score += 1.0 / (k + dense_map[doc_id])
            if doc_id in sparse_map:
                score += 1.0 / (k + sparse_map[doc_id])
            rrf_scores[doc_id] = score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (idx, score, dense_map.get(idx, -1), sparse_map.get(idx, -1))
            for idx, score in ranked
        ]

    # ─── Parent-Child Auto-Merge ──────────────────────────────────────────────

    def _auto_merge(
        self,
        results: List[Tuple[int, float, int, int]],
    ) -> List[Tuple[int, float, int, int]]:
        """
        If a retrieved chunk is a 'child', replace it with its parent if available.
        Deduplicates by parent_id.
        """
        seen_parents = set()
        merged = []
        for idx, rrf_score, dr, sr in results:
            chunk = self._index[idx][0]
            if chunk.get("chunk_type") == "child" and chunk.get("parent_id"):
                # Find parent in index
                parent_id = chunk["parent_id"]
                if parent_id in seen_parents:
                    continue
                parent_idx = next(
                    (i for i, (c, _) in enumerate(self._index) if c.get("chunk_id") == parent_id),
                    None,
                )
                if parent_idx is not None:
                    seen_parents.add(parent_id)
                    merged.append((parent_idx, rrf_score, dr, sr))
                    continue
            if chunk.get("chunk_id") not in seen_parents:
                seen_parents.add(chunk.get("chunk_id", idx))
                merged.append((idx, rrf_score, dr, sr))
        return merged

    # ─── Main Search API ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        auto_merge_parents: bool = True,
    ) -> List[SearchResult]:
        """
        Execute hybrid search and return ranked SearchResult list.

        Args:
            query: Natural language query
            top_k: Number of results (defaults to self.top_k)
            metadata_filter: Pre-filter on chunk metadata fields
            use_hybrid: If False, uses dense-only search
            auto_merge_parents: If True, child chunks are replaced by parent
        """
        k = top_k or self.top_k
        fetch_k = k * 3  # Over-fetch for fusion

        dense_results = self._dense_search(query, fetch_k, metadata_filter)

        dense_score_map = {idx: score for idx, score in dense_results}
        sparse_score_map: Dict[int, float] = {}

        if use_hybrid and self._bm25:
            sparse_results = self._sparse_search(query, fetch_k, metadata_filter)
            sparse_score_map = {idx: score for idx, score in sparse_results}
            fused = self._rrf_fuse(dense_results, sparse_results, k * 2)
        else:
            fused = [(idx, 1.0 / (self.rrf_k + rank + 1), rank + 1, -1)
                     for rank, (idx, _) in enumerate(dense_results[:k * 2])]

        if auto_merge_parents:
            fused = self._auto_merge(fused)

        final = fused[:k]
        output = []
        for idx, rrf_score, dense_rank, sparse_rank in final:
            chunk, _ = self._index[idx]
            output.append(SearchResult(
                chunk_id=chunk.get("chunk_id", str(uuid.uuid4())),
                content=chunk.get("content", ""),
                doc_id=chunk.get("doc_id", ""),
                metadata=chunk.get("metadata", {}),
                dense_rank=dense_rank if dense_rank > 0 else None,
                sparse_rank=sparse_rank if sparse_rank > 0 else None,
                rrf_score=rrf_score,
                dense_score=dense_score_map.get(idx, 0.0),
                sparse_score=sparse_score_map.get(idx, 0.0),
                chunk_type=chunk.get("chunk_type", "unknown"),
                parent_id=chunk.get("parent_id"),
            ))
        return output

    # ─── Utilities ───────────────────────────────────────────────────────────

    def _match_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Simple exact-match metadata filter."""
        for key, val in filter_dict.items():
            if metadata.get(key) != val:
                return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.index_size,
            "bm25_active": self._bm25 is not None,
            "embed_model": self._model_name,
            "rrf_k": self.rrf_k,
        }
