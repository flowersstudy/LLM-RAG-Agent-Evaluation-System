"""
Composite scorer: runs multiple metrics on a (task, prediction) pair
and produces a single EvaluationResult with all scores.
"""

from __future__ import annotations

from typing import Dict, List

from src.core.interfaces import Metric
from src.core.models import (
    EvaluationResult,
    FailureMode,
    FailureType,
    MetricScore,
    Prediction,
    Task,
)
from src.utils.logging import get_logger, get_tracer

logger = get_logger()

# Thresholds for automatic failure classification
# These can be overridden per experiment.
DEFAULT_FAILURE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "faithfulness": {"threshold": 0.5, "failure_type": FailureType.HALLUCINATION},
    "retrieval_precision": {"threshold": 0.3, "failure_type": FailureType.RETRIEVAL},
    "retrieval_recall": {"threshold": 0.3, "failure_type": FailureType.RETRIEVAL},
    "answer_relevance": {"threshold": 0.5, "failure_type": FailureType.REASONING},
}


class CompositeScorer:
    """Runs all configured metrics and produces a unified EvaluationResult."""

    def __init__(
        self,
        metrics: List[Metric],
        weights: Dict[str, float] | None = None,
        failure_thresholds: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        self._metrics = metrics
        self._weights = weights or {m.name: 1.0 for m in metrics}
        self._failure_thresholds = failure_thresholds or DEFAULT_FAILURE_THRESHOLDS

    async def evaluate(self, task: Task, prediction: Prediction) -> EvaluationResult:
        tracer = get_tracer()
        tracer.log("evaluation.start", task_id=task.id, model=prediction.model_id)

        scores: List[MetricScore] = []
        failure_modes: List[FailureMode] = []

        for metric in self._metrics:
            result = await metric.evaluate(task, prediction)
            ms = MetricScore(
                metric_name=metric.name,
                score=result["score"],
                rationale=result.get("rationale"),
                evidence=result.get("evidence", []),
            )
            scores.append(ms)

            # Auto-classify failures based on thresholds
            self._check_failure(ms, failure_modes)

        # Weighted aggregate
        total_weight = sum(self._weights.get(m.name, 1.0) for m in self._metrics)
        aggregate = (
            sum(
                s.score * self._weights.get(s.metric_name, 1.0)
                for s in scores
            )
            / total_weight
            if total_weight > 0
            else 0.0
        )

        tracer.log(
            "evaluation.end",
            task_id=task.id,
            aggregate_score=round(aggregate, 4),
            failure_count=len(failure_modes),
        )

        return EvaluationResult(
            task_id=task.id,
            model_id=prediction.model_id,
            scores=scores,
            failure_modes=failure_modes,
            aggregate_score=round(aggregate, 4),
        )

    def _check_failure(self, score: MetricScore, failures: List[FailureMode]) -> None:
        config = self._failure_thresholds.get(score.metric_name)
        if config is None:
            return
        if score.score < config["threshold"]:
            failures.append(FailureMode(
                type=config["failure_type"],
                severity=1.0 - score.score,
                description=f"{score.metric_name} = {score.score:.3f} (below threshold {config['threshold']})",
                evidence=score.rationale or f"Score: {score.score}",
            ))
