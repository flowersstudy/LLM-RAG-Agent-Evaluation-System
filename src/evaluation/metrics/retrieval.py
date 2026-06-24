"""
Retrieval quality metrics: precision, recall, and NDCG.

These are reference-based metrics — they compare retrieved documents against
ground-truth relevance labels. No LLM judge needed.

Precision: of retrieved docs, how many are relevant?
Recall:    of all relevant docs, how many were retrieved?
NDCG:      position-weighted relevance (optional, for ranked retrieval eval)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.core.interfaces import Metric
from src.core.models import Document, Prediction, Task
from src.core.registry import register
from src.utils.logging import get_logger

logger = get_logger()


def _doc_ids(docs: List[Document]) -> set:
    return {d.id for d in docs}


@register("metric", "retrieval_precision")
class RetrievalPrecisionMetric(Metric):
    """Precision@k: fraction of retrieved docs that are relevant."""

    @property
    def name(self) -> str:
        return "retrieval_precision"

    @property
    def requires_ground_truth(self) -> bool:
        return True

    async def evaluate(self, task: Task, prediction: Prediction) -> Dict[str, Any]:
        retrieved_ids = _doc_ids(prediction.execution_trace.retrieved_docs)
        relevant_ids = _doc_ids(task.ground_truth_docs)

        if len(retrieved_ids) == 0:
            return {"score": 0.0, "rationale": "No documents retrieved.", "evidence": []}

        relevant_retrieved = retrieved_ids & relevant_ids
        precision = len(relevant_retrieved) / len(retrieved_ids)

        return {
            "score": round(precision, 4),
            "rationale": (
                f"{len(relevant_retrieved)} relevant out of {len(retrieved_ids)} retrieved "
                f"(precision={precision:.3f})"
            ),
            "evidence": sorted(relevant_retrieved),
        }


@register("metric", "retrieval_recall")
class RetrievalRecallMetric(Metric):
    """Recall@k: fraction of all relevant docs that were retrieved."""

    @property
    def name(self) -> str:
        return "retrieval_recall"

    @property
    def requires_ground_truth(self) -> bool:
        return True

    async def evaluate(self, task: Task, prediction: Prediction) -> Dict[str, Any]:
        retrieved_ids = _doc_ids(prediction.execution_trace.retrieved_docs)
        relevant_ids = _doc_ids(task.ground_truth_docs)

        if len(relevant_ids) == 0:
            return {"score": 1.0, "rationale": "No relevant docs defined — recall is vacuously 1.0.", "evidence": []}

        relevant_retrieved = retrieved_ids & relevant_ids
        recall = len(relevant_retrieved) / len(relevant_ids)

        return {
            "score": round(recall, 4),
            "rationale": (
                f"{len(relevant_retrieved)} retrieved out of {len(relevant_ids)} relevant "
                f"(recall={recall:.3f})"
            ),
            "evidence": sorted(relevant_retrieved),
        }


@register("metric", "answer_relevance")
class AnswerRelevanceMetric(Metric):
    """
    Measures whether the answer actually addresses the query.

    Uses LLM judge because relevance is semantic, not lexical.
    Even a faithful answer can be irrelevant (e.g., answering "what is X"
    when the user asked "how do I use X").
    """

    def __init__(self, judge) -> None:
        from src.evaluation.judge import StructuredJudge
        self._judge: StructuredJudge = judge

    @property
    def name(self) -> str:
        return "answer_relevance"

    @property
    def requires_ground_truth(self) -> bool:
        return False

    async def evaluate(self, task: Task, prediction: Prediction) -> Dict[str, Any]:
        rubric = """\
Evaluate whether the Answer is RELEVANT to the Query.

Definition: An answer is relevant if it directly addresses what the query
is asking. Consider: does it answer the right question? Is it complete?

Scoring:
- 1.0: Fully and directly addresses the query.
- 0.7-0.9: Addresses the query with minor omissions or tangents.
- 0.4-0.6: Partially addresses the query; significant parts missing or off-topic.
- 0.1-0.3: Mostly off-topic or only tangentially related.
- 0.0: Completely unrelated to the query."""

        result = await self._judge.judge(
            rubric=rubric,
            query=task.query,
            answer=prediction.answer,
        )

        return {
            "score": result["score"],
            "rationale": result.get("rationale", ""),
            "evidence": result.get("evidence", []),
        }
