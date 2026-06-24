"""
Faithfulness metric: measures whether every claim in the answer is supported
by the retrieved context. This is the #1 RAG safety metric — unfaithful
answers are hallucinations.

Uses structured LLM-judge rubric with NLI-inspired decomposition:
1. Break answer into atomic claims
2. For each claim, check if it is entailed by the retrieved context
3. Score = (entailed claims) / (total claims)
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.interfaces import Metric
from src.core.models import Prediction, Task
from src.core.registry import register
from src.evaluation.judge import StructuredJudge
from src.utils.logging import get_logger

logger = get_logger()

FAITHFULNESS_RUBRIC = """\
Evaluate whether the Answer is FAITHFUL to the Context.

Definition: An answer is faithful if EVERY factual claim it makes is directly
supported by the provided context. If the answer includes information not in
the context, it is unfaithful (hallucination).

Scoring:
- 1.0: Every claim is explicitly supported by the context. No extra claims.
- 0.7-0.9: Most claims supported; minor unsupported details.
- 0.4-0.6: Mixed — some claims supported, some unsupported.
- 0.1-0.3: Mostly unsupported claims; only tangential connection to context.
- 0.0: Answer is completely fabricated or contradicts the context.

For each piece of evidence, quote the specific sentence from the Context
that supports (or fails to support) a claim in the Answer."""


@register("metric", "faithfulness")
class FaithfulnessMetric(Metric):
    """Measures if the answer is supported by retrieved context."""

    def __init__(self, judge: StructuredJudge) -> None:
        self._judge = judge

    @property
    def name(self) -> str:
        return "faithfulness"

    @property
    def requires_ground_truth(self) -> bool:
        return False  # Only needs retrieved context, not ground truth

    async def evaluate(self, task: Task, prediction: Prediction) -> Dict[str, Any]:
        context = self._format_context(prediction.execution_trace.retrieved_docs)
        if not context:
            return {
                "score": 0.0,
                "rationale": "No context retrieved — cannot assess faithfulness.",
                "evidence": [],
            }

        result = await self._judge.judge(
            rubric=FAITHFULNESS_RUBRIC,
            query=task.query,
            answer=prediction.answer,
            context=context,
        )

        return {
            "score": result["score"],
            "rationale": result.get("rationale", ""),
            "evidence": result.get("evidence", []),
        }

    def _format_context(self, docs) -> str:
        parts = [f"[{d.id}] {d.content}" for d in docs]
        return "\n\n".join(parts)
