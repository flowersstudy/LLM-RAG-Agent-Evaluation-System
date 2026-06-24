"""
Dense retriever using sentence-transformers embeddings with in-memory cosine similarity.

Supports multiple embedding models. BGE series is the default — they use asymmetric
embeddings (query prefix vs plain document) which significantly improves retrieval quality.

BGE models tested:
- BAAI/bge-base-en-v1.5  (768 dims, recommended default)
- BAAI/bge-large-en-v1.5 (1024 dims, best quality)
- all-MiniLM-L6-v2       (384 dims, fast / lightweight baseline)
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.models import Document
from src.utils.logging import get_logger, get_tracer

logger = get_logger()

# BGE models use asymmetric instructions: a prefix for queries, raw text for docs.
# This is the key quality differentiator vs. symmetric models like MiniLM.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Model registry: name -> (query_prefix, description)
_MODELS = {
    "BAAI/bge-base-en-v1.5": (_BGE_QUERY_INSTRUCTION, "BGE-base v1.5 (768d)"),
    "BAAI/bge-large-en-v1.5": (_BGE_QUERY_INSTRUCTION, "BGE-large v1.5 (1024d)"),
    "all-MiniLM-L6-v2": ("", "MiniLM-L6 (384d, symmetric)"),
}

_DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


class DenseRetriever:
    """Embedding-based retriever with in-memory document store.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers model. BGE models recommended for retrieval.
        Use 'all-MiniLM-L6-v2' for a fast light-weight baseline.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._query_prefix, self._model_desc = _MODELS.get(
            model_name, ("", model_name)
        )
        logger.info(f"Loading embedding model: {model_name} ({self._model_desc})")
        self._model = SentenceTransformer(model_name)
        self._documents: List[Document] = []
        self._embeddings: np.ndarray | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int | None:
        if self._embeddings is not None:
            return self._embeddings.shape[1]
        return None

    def index(self, documents: List[Document]) -> None:
        """Index documents. Passages are encoded WITHOUT the query prefix."""
        self._documents = documents
        if not documents:
            self._embeddings = np.array([])
            return
        texts = [doc.content for doc in documents]
        self._embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(documents) > 100,
        )
        logger.info(
            f"Indexed {len(documents)} documents, dim={self._embeddings.shape[1]} "
            f"(model: {self._model_name})"
        )

    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve top-k documents by cosine similarity.

        BGE models: query is prefixed with the retrieval instruction.
        """
        tracer = get_tracer()
        tracer.log(
            "retrieval.start",
            query=query[:100],
            top_k=top_k,
            corpus_size=len(self._documents),
            model=self._model_name,
        )

        if self._embeddings is None or len(self._documents) == 0:
            tracer.log("retrieval.end", result_count=0)
            return []

        # Apply query prefix for asymmetric models (BGE)
        query_with_prefix = self._query_prefix + query
        query_embedding = self._model.encode(
            [query_with_prefix], normalize_embeddings=True
        )
        scores = np.dot(query_embedding, self._embeddings.T)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Document] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0.1:
                continue
            doc = self._documents[idx]
            results.append(Document(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=score,
            ))

        tracer.log(
            "retrieval.end",
            result_count=len(results),
            top_scores=[round(d.score or 0, 4) for d in results],
            top_ids=[d.id for d in results],
        )
        return results

    @property
    def document_count(self) -> int:
        return len(self._documents)
