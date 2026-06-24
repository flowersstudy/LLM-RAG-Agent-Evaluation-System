"""
Dense retriever using sentence-transformers embeddings with in-memory cosine similarity.

For production, swap with a vector DB. This implementation keeps Phase 1
self-contained with no external services.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.models import Document
from src.utils.logging import get_logger, get_tracer

logger = get_logger()

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class DenseRetriever:
    """Embedding-based retriever with in-memory document store."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)
        self._documents: List[Document] = []
        self._embeddings: np.ndarray | None = None

    def index(self, documents: List[Document]) -> None:
        """Index a list of documents for retrieval."""
        self._documents = documents
        if not documents:
            self._embeddings = np.array([])
            return
        texts = [doc.content for doc in documents]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)
        logger.info(f"Indexed {len(documents)} documents (dim={self._embeddings.shape[1]})")

    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve top-k documents by cosine similarity."""
        tracer = get_tracer()
        tracer.log("retrieval.start", query=query, top_k=top_k, corpus_size=len(self._documents))

        if self._embeddings is None or len(self._documents) == 0:
            tracer.log("retrieval.end", result_count=0)
            return []

        query_embedding = self._model.encode([query], normalize_embeddings=True)
        scores = np.dot(query_embedding, self._embeddings.T)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Document] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0.1:  # Low-quality threshold
                continue
            doc = self._documents[idx]
            results.append(Document(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=score,
            ))

        tracer.log("retrieval.end", result_count=len(results), top_scores=[d.score for d in results])
        return results

    @property
    def document_count(self) -> int:
        return len(self._documents)
