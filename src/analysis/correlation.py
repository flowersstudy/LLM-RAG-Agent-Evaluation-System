"""
Metric correlation analysis: discovers relationships between evaluation metrics.

Computes Pearson correlation matrices to answer questions like:
- Do faithful answers also tend to be relevant?
- Does tool selection accuracy predict task success?
"""

from __future__ import annotations

from typing import Dict, List

from src.core.models import EvaluationResult
from src.utils.logging import get_logger

logger = get_logger()


class MetricCorrelator:
    """Computes pairwise Pearson correlations between metrics."""

    def compute(
        self,
        results: List[EvaluationResult],
        metric_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-model correlation matrices.

        Returns:
            {model_id: {metric_a: {metric_b: correlation}}}
        """
        # Group by model
        by_model: Dict[str, List[EvaluationResult]] = {}
        for r in results:
            by_model.setdefault(r.model_id, []).append(r)

        per_model: Dict[str, Dict[str, Dict[str, float]]] = {}

        for model_id, model_results in by_model.items():
            # Build score vectors per metric
            vectors: Dict[str, List[float]] = {m: [] for m in metric_names}
            for r in model_results:
                for m in metric_names:
                    s = r.get_score(m)
                    vectors[m].append(s if s is not None else 0.0)

            # Pairwise Pearson
            matrix: Dict[str, Dict[str, float]] = {}
            for m1 in metric_names:
                matrix[m1] = {}
                for m2 in metric_names:
                    v1, v2 = vectors[m1], vectors[m2]
                    if len(v1) < 3:
                        matrix[m1][m2] = 0.0
                    else:
                        matrix[m1][m2] = round(self._pearson(v1, v2), 4)

            per_model[model_id] = matrix

        logger.info(f"Computed correlations across {len(metric_names)} metrics, {len(by_model)} models")
        return per_model

    def _pearson(self, x: List[float], y: List[float]) -> float:
        """Pearson correlation coefficient."""
        n = len(x)
        if n < 3:
            return 0.0
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        denom_x = sum((xi - mx) ** 2 for xi in x)
        denom_y = sum((yi - my) ** 2 for yi in y)
        if denom_x == 0 or denom_y == 0:
            return 0.0
        return num / ((denom_x * denom_y) ** 0.5)
