"""
Stratified analysis: breaks down scores by domain, difficulty, and task type.

Identifies per-model strengths and weaknesses across data slices.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.core.models import EvaluationResult, FailureType, Task
from src.utils.logging import get_logger

logger = get_logger()


class StratifiedAnalyzer:
    """Computes sliced performance breakdowns."""

    def analyze(
        self,
        results: List[EvaluationResult],
        tasks: List[Task],
        metric_names: List[str],
    ) -> Dict[str, Any]:
        """
        Break down scores by domain, difficulty, and task type.

        Returns structured dict with per-slice stats for each model.
        """
        task_map = {t.id: t for t in tasks}

        # Group results by model
        by_model: Dict[str, List[EvaluationResult]] = defaultdict(list)
        for r in results:
            by_model[r.model_id].append(r)

        breakdown: Dict[str, Any] = {
            "by_domain": {},
            "by_difficulty": {},
            "by_task_type": {},
            "model_weaknesses": {},
        }

        slice_fields = [
            ("by_domain", "domain"),
            ("by_difficulty", "difficulty"),
            ("by_task_type", lambda t: t.type.value if hasattr(t.type, "value") else str(t.type)),
        ]

        for slice_key, field in slice_fields:
            for model_id, model_results in by_model.items():
                breakdown[slice_key].setdefault(model_id, {})

                # Group by slice
                slices: Dict[str, List[float]] = defaultdict(list)
                for r in model_results:
                    task = task_map.get(r.task_id)
                    if task is None:
                        continue
                    if callable(field):
                        key = field(task)
                    else:
                        key = getattr(task, field, "unknown")
                    for m in metric_names:
                        s = r.get_score(m)
                        if s is not None:
                            slices.setdefault(key, []).append(s)

                for key, scores in slices.items():
                    if scores:
                        avg = sum(scores) / len(scores)
                        breakdown[slice_key][model_id][key] = {
                            "mean": round(avg, 4),
                            "count": len(scores),
                            "min": round(min(scores), 4),
                            "max": round(max(scores), 4),
                        }

        # Identify per-model weaknesses (slices with lowest mean)
        for model_id in by_model:
            weaknesses = []
            for slice_key in ["by_domain", "by_difficulty"]:
                model_slices = breakdown[slice_key].get(model_id, {})
                if model_slices:
                    worst = min(model_slices, key=lambda k: model_slices[k]["mean"])
                    weaknesses.append({
                        "slice": slice_key,
                        "key": worst,
                        "mean": model_slices[worst]["mean"],
                    })
            # Sort by mean ascending (worst first)
            weaknesses.sort(key=lambda w: w["mean"])
            breakdown["model_weaknesses"][model_id] = weaknesses[:3]

        logger.info(f"Breakdown computed: {len(by_model)} models, {len(tasks)} tasks")
        return breakdown
