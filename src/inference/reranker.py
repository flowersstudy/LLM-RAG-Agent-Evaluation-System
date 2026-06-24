"""
Cross-Encoder Reranker for two-stage retrieval.

Stage 1: Dense retriever (bi-encoder) retrieves candidate set (fast, approximate).
Stage 2: Reranker (cross-encoder) scores each (query, document) pair (slower, accurate).

The cross-encoder sees the full query-document interaction, catching semantic nuances
that bi-encoder cosine similarity misses. Typical precision improvement: +15-30%.

Default model: BAAI/bge-reranker-base (BGE series, optimized for retrieval reranking).
"""

from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from src.core.models import Document
from src.utils.logging import get_logger, get_tracer

logger = get_logger()

_DEFAULT_RERANKER = "BAAI/bge-reranker-base"


class Reranker:
    """Cross-encoder based reranker for two-stage retrieval.

    Parameters
    ----------
    model_name : str
        Cross-encoder model. BAAI/bge-reranker-base is the recommended default.
    """

    def __init__(self, model_name: str = _DEFAULT_RERANKER) -> None:
        self._model_name = model_name
        logger.info(f"Loading reranker model: {model_name}")
        self._model = CrossEncoder(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    async def rerank(
        self,
        query: str,
        candidates: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """Re-score candidates with cross-encoder and return top-k.

        Parameters
        ----------
        query : str
            The search query.
        candidates : List[Document]
            Documents to rerank (typically from dense retrieval).
        top_k : int
            Number of documents to return after reranking.

        Returns
        -------
        List[Document]
            Reranked documents with updated scores.
        """
        tracer = get_tracer()
        tracer.log(
            "reranker.start",
            model=self._model_name,
            candidate_count=len(candidates),
            top_k=top_k,
        )

        if not candidates:
            tracer.log("reranker.end", result_count=0)
            return []

        # Build (query, doc) pairs for cross-encoding
        pairs = [(query, doc.content) for doc in candidates]
        scores = self._model.predict(pairs, show_progress_bar=len(pairs) > 50)

        # Sort by cross-encoder score (descending) and take top_k
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Document] = []
        for score, doc in scored[:top_k]:
            results.append(Document(
                id=doc.id,
                content=doc.content,
                metadata={**doc.metadata, "rerank_score": float(score)},
                score=float(score),
            ))

        tracer.log(
            "reranker.end",
            result_count=len(results),
            top_scores=[round(d.score or 0, 4) for d in results],
            top_ids=[d.id for d in results],
        )
        return results


class TwoStageRetriever:
    """Two-stage retrieval: dense → rerank.

    This is the recommended retriever for production-quality RAG evaluation.
    Stage 1 retrieves a larger candidate set (e.g., top_k * 4).
    Stage 2 reranks with a cross-encoder to pick the top_k most relevant docs.
    """

    def __init__(
        self,
        dense_retriever,
        reranker: Reranker,
        first_stage_k_multiplier: int = 4,
        final_top_k: int = 5,
    ) -> None:
        self._dense = dense_retriever
        self._reranker = reranker
        self._first_stage_k = final_top_k * first_stage_k_multiplier
        self._final_top_k = final_top_k
        logger.info(
            f"Two-stage retriever: dense={dense_retriever.model_name}, "
            f"reranker={reranker.model_name}, "
            f"stage1_k={self._first_stage_k}, final_k={self._final_top_k}"
        )

    async def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        """Two-stage retrieve: dense → rerank."""
        final_k = top_k or self._final_top_k
        first_k = max(final_k * 4, self._first_stage_k)

        # Stage 1: fast dense retrieval
        candidates = await self._dense.retrieve(query, top_k=first_k)

        if len(candidates) <= final_k:
            return candidates

        # Stage 2: cross-encoder reranking
        return await self._reranker.rerank(query, candidates, top_k=final_k)

    def index(self, documents: List[Document]) -> None:
        """Index documents in the dense retriever. Reranker is stateless."""
        self._dense.index(documents)

    @property
    def model_name(self) -> str:
        return f"{self._dense.model_name} + {self._reranker.model_name}"

    @property
    def document_count(self) -> int:
        return self._dense.document_count
