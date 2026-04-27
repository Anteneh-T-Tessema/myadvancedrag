"""
Advanced Chunking Strategies
Implements semantic chunking, parent-child (auto-merging) retrieval,
and metadata-tagged chunking for production-grade RAG pipelines.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class Chunk:
    """A processed document chunk with rich metadata."""
    chunk_id: str
    content: str
    chunk_type: str  # 'semantic', 'parent', 'child', 'fixed'
    doc_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "doc_id": self.doc_id,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }


class AdvancedChunker:
    """
    Production-grade chunker implementing three strategies:
    1. Semantic chunking (cosine similarity breakpoints)
    2. Parent-Child retrieval (auto-merging)
    3. Metadata-tagged fixed chunking (baseline)
    """

    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        child_chunk_size: int = 150,
        parent_chunk_size: int = 512,
        overlap: int = 50,
    ):
        self.similarity_threshold = similarity_threshold
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.overlap = overlap
        self._model = None
        self._model_name = embed_model_name

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ─── Utility ────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Rough token count via whitespace split."""
        return text.split()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _base_metadata(self, doc_id: str, source: str, strategy: str) -> Dict[str, Any]:
        return {
            "doc_id": doc_id,
            "source": source,
            "strategy": strategy,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        }

    # ─── Strategy 1: Semantic Chunking ──────────────────────────────────────

    def semantic_chunk(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source: str = "unknown",
        extra_metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        Embed sentences individually, detect topic-shift breakpoints via cosine
        similarity drops, and group sentences into semantically coherent chunks.
        """
        doc_id = doc_id or str(uuid.uuid4())
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return self._wrap_single(text, doc_id, source, "semantic", extra_metadata)

        model = self._get_model()
        if model is None:
            # Fallback to fixed chunking without a model
            return self.fixed_chunk(text, doc_id=doc_id, source=source)

        embeddings = model.encode(sentences, normalize_embeddings=True)

        # Detect breakpoints: sharp cosine drops signal topic changes
        breakpoints = []
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)

        # Group sentences into chunks
        chunks = []
        group_start = 0
        for bp in breakpoints:
            group = sentences[group_start:bp]
            if group:
                chunks.append(" ".join(group))
            group_start = bp
        # Tail group
        tail = sentences[group_start:]
        if tail:
            chunks.append(" ".join(tail))

        result = []
        for idx, content in enumerate(chunks):
            meta = self._base_metadata(doc_id, source, "semantic")
            meta.update({
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "similarity_threshold": self.similarity_threshold,
            })
            if extra_metadata:
                meta.update(extra_metadata)
            result.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                chunk_type="semantic",
                doc_id=doc_id,
                metadata=meta,
                token_count=len(self._tokenize(content)),
            ))

        return result

    # ─── Strategy 2: Parent-Child Retrieval ─────────────────────────────────

    def parent_child_chunk(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source: str = "unknown",
        extra_metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        Split into large parent chunks, then split each parent into fine-grained
        child chunks. Retrieval hits child; LLM receives parent context window.
        Returns flat list: parents first, then all children.
        """
        doc_id = doc_id or str(uuid.uuid4())
        words = text.split()
        all_chunks: List[Chunk] = []

        parent_chunks_text = self._sliding_window(words, self.parent_chunk_size, self.overlap)

        for p_idx, parent_text in enumerate(parent_chunks_text):
            parent_id = str(uuid.uuid4())
            p_meta = self._base_metadata(doc_id, source, "parent")
            p_meta.update({
                "chunk_index": p_idx,
                "total_parents": len(parent_chunks_text),
            })
            if extra_metadata:
                p_meta.update(extra_metadata)

            parent_chunk = Chunk(
                chunk_id=parent_id,
                content=parent_text,
                chunk_type="parent",
                doc_id=doc_id,
                metadata=p_meta,
                token_count=len(self._tokenize(parent_text)),
            )

            # Build child chunks from parent text
            child_words = parent_text.split()
            child_texts = self._sliding_window(child_words, self.child_chunk_size, self.overlap // 2)
            child_ids = []

            for c_idx, child_text in enumerate(child_texts):
                child_id = str(uuid.uuid4())
                c_meta = self._base_metadata(doc_id, source, "child")
                c_meta.update({
                    "parent_chunk_index": p_idx,
                    "child_index": c_idx,
                    "total_children": len(child_texts),
                    "parent_id": parent_id,
                })
                if extra_metadata:
                    c_meta.update(extra_metadata)

                child_chunk = Chunk(
                    chunk_id=child_id,
                    content=child_text,
                    chunk_type="child",
                    doc_id=doc_id,
                    parent_id=parent_id,
                    metadata=c_meta,
                    token_count=len(self._tokenize(child_text)),
                )
                child_ids.append(child_id)
                all_chunks.append(child_chunk)

            parent_chunk.children_ids = child_ids
            all_chunks.insert(len(all_chunks) - len(child_ids), parent_chunk)

        return all_chunks

    # ─── Strategy 3: Fixed + Metadata Tagged ────────────────────────────────

    def fixed_chunk(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source: str = "unknown",
        extra_metadata: Optional[Dict] = None,
        chunk_size: Optional[int] = None,
    ) -> List[Chunk]:
        """Fixed-size chunking with rich metadata injection."""
        doc_id = doc_id or str(uuid.uuid4())
        size = chunk_size or self.parent_chunk_size
        words = text.split()
        chunk_texts = self._sliding_window(words, size, self.overlap)

        result = []
        for idx, content in enumerate(chunk_texts):
            meta = self._base_metadata(doc_id, source, "fixed")
            meta.update({
                "chunk_index": idx,
                "total_chunks": len(chunk_texts),
                "chunk_size_target": size,
            })
            if extra_metadata:
                meta.update(extra_metadata)
            result.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                chunk_type="fixed",
                doc_id=doc_id,
                metadata=meta,
                token_count=len(self._tokenize(content)),
            ))
        return result

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _sliding_window(self, words: List[str], size: int, stride: int) -> List[str]:
        """Create overlapping text windows."""
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += size - stride
        return chunks

    def _wrap_single(
        self, text: str, doc_id: str, source: str, strategy: str, extra_metadata: Optional[Dict]
    ) -> List[Chunk]:
        meta = self._base_metadata(doc_id, source, strategy)
        if extra_metadata:
            meta.update(extra_metadata)
        return [Chunk(
            chunk_id=str(uuid.uuid4()),
            content=text,
            chunk_type=strategy,
            doc_id=doc_id,
            metadata=meta,
            token_count=len(self._tokenize(text)),
        )]
