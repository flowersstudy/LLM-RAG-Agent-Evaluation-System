"""
Report generator: produces human-readable Markdown reports from analysis data.

Generates a comprehensive report covering:
- Overall summary
- Per-domain/per-difficulty breakdowns
- Metric correlations
- Failure clusters
- Per-model weaknesses
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.core.models import AggregateReport, FailureCluster
from src.utils.logging import get_logger

logger = get_logger()


class ReportGenerator:
    """Generates Markdown reports from experiment analysis results."""

    def generate(
        self,
        report: AggregateReport,
        correlations: Dict[str, Any] | None = None,
        breakdown: Dict[str, Any] | None = None,
        output_dir: str | Path = "",
    ) -> str:
        """
        Generate a Markdown report string.

        If output_dir is provided, saves to {output_dir}/report.md.
        Returns the generated markdown string.
        """
        md = self._build(report, correlations, breakdown)

        if output_dir:
            output_path = Path(output_dir) / "report.md"
            output_path.write_text(md, encoding="utf-8")
            logger.info(f"Report saved to {output_path}")

        return md

    def _build(
        self,
        report: AggregateReport,
        correlations: Dict[str, Any] | None,
        breakdown: Dict[str, Any] | None,
    ) -> str:
        lines: List[str] = []

        # ── Header ───────────────────────────────────────
        lines.append(f"# Experiment Report — `{report.experiment_id}`")
        lines.append("")

        # ── Overall Summary ──────────────────────────────
        lines.append("## 1. Overall Summary")
        lines.append("")
        lines.append("| Metric | " + " | ".join(report.model_comparisons[0].model_ids if report.model_comparisons else ["N/A"]) + " |")
        lines.append("|---" * (1 + max(1, len(report.model_comparisons[0].model_ids if report.model_comparisons else []))) + "|")

        for c in report.model_comparisons:
            parts = [c.metric_name]
            for mid in c.model_ids:
                mean = c.mean_scores.get(mid, "N/A")
                std = c.std_scores.get(mid, "N/A")
                parts.append(f"{mean} ± {std}" if isinstance(mean, float) else "N/A")
            lines.append("| " + " | ".join(parts) + " |")
        lines.append("")

        # ── Win rates ────────────────────────────────────
        if report.model_comparisons and len(report.model_comparisons[0].win_rates) > 1:
            lines.append("### Win Rates")
            lines.append("")
            lines.append("| Model | " + " | ".join(report.model_comparisons[0].model_ids) + " |")
            lines.append("|---" * (1 + len(report.model_comparisons[0].model_ids)) + "|")
            for c in report.model_comparisons:
                parts = [c.metric_name]
                for mid in c.model_ids:
                    wr = c.win_rates.get(mid, 0.0)
                    parts.append(f"{wr:.1%}")
                lines.append("| " + " | ".join(parts) + " |")
            lines.append("")

        # ── Failure Distribution ─────────────────────────
        lines.append("## 2. Failure Distribution")
        lines.append("")
        if report.failure_distribution:
            lines.append("| Failure Type | Count |")
            lines.append("|---|---|")
            for ft, count in sorted(report.failure_distribution.items()):
                lines.append(f"| {ft} | {count} |")
        else:
            lines.append("No failures detected.")
        lines.append("")

        # Per-model failure rates
        if report.per_model_failure_rates:
            lines.append("### Per-Model Failure Rates")
            lines.append("")
            ft_types = sorted(set(ft for rates in report.per_model_failure_rates.values() for ft in rates))
            lines.append("| Model | " + " | ".join(ft_types) + " |")
            lines.append("|---" * (1 + len(ft_types)) + "|")
            for model_id, rates in report.per_model_failure_rates.items():
                parts = [model_id]
                for ft in ft_types:
                    rate = rates.get(ft, 0.0)
                    parts.append(f"{rate:.1%}")
                lines.append("| " + " | ".join(parts) + " |")
            lines.append("")

        # ── Domain Breakdown ─────────────────────────────
        if breakdown and "by_domain" in breakdown:
            lines.append("## 3. Domain Breakdown")
            lines.append("")
            for model_id, domains in breakdown["by_domain"].items():
                lines.append(f"### {model_id}")
                lines.append("")
                lines.append("| Domain | Mean Score | N | Min | Max |")
                lines.append("|---|---|---|---|---|")
                for domain, stats in sorted(domains.items()):
                    lines.append(f"| {domain} | {stats['mean']} | {stats['count']} | {stats['min']} | {stats['max']} |")
                lines.append("")

        # ── Difficulty Breakdown ─────────────────────────
        if breakdown and "by_difficulty" in breakdown:
            lines.append("## 4. Difficulty Breakdown")
            lines.append("")
            for model_id, diffs in breakdown["by_difficulty"].items():
                lines.append(f"### {model_id}")
                lines.append("")
                lines.append("| Difficulty | Mean Score | N | Min | Max |")
                lines.append("|---|---|---|---|---|")
                for diff, stats in sorted(diffs.items()):
                    lines.append(f"| {diff} | {stats['mean']} | {stats['count']} | {stats['min']} | {stats['max']} |")
                lines.append("")

        # ── Metric Correlations ──────────────────────────
        if correlations:
            lines.append("## 5. Metric Correlations")
            lines.append("")
            all_metrics = sorted(set(m for model_corr in correlations.values() for m in model_corr))
            for model_id, matrix in correlations.items():
                lines.append(f"### {model_id}")
                lines.append("")
                lines.append("| | " + " | ".join(all_metrics) + " |")
                lines.append("|---" * (1 + len(all_metrics)) + "|")
                for m1 in all_metrics:
                    parts = [m1]
                    for m2 in all_metrics:
                        v = matrix.get(m1, {}).get(m2, 0.0)
                        parts.append(f"{v:+.3f}")
                    lines.append("| " + " | ".join(parts) + " |")
                lines.append("")

            # Key findings
            lines.append("**Key relationships:**")
            for model_id, matrix in correlations.items():
                for m1 in all_metrics:
                    for m2 in all_metrics:
                        v = matrix.get(m1, {}).get(m2, 0.0)
                        if m1 < m2 and abs(v) > 0.3:
                            direction = "positive" if v > 0 else "negative"
                            lines.append(f"- `{m1}` ↔ `{m2}`: {direction} correlation ({v:+.3f}) ({model_id})")
            lines.append("")

        # ── Failure Clusters ─────────────────────────────
        if report.failure_clusters:
            lines.append("## 6. Failure Clusters")
            lines.append("")
            for cluster in report.failure_clusters:
                lines.append(f"### {cluster.label}")
                lines.append(f"- **Type**: {cluster.failure_type.value}")
                lines.append(f"- **Size**: {cluster.size} failures")
                examples = ", ".join(cluster.representative_examples[:5])
                lines.append(f"- **Examples**: {examples}")
                lines.append(f"- **Description**: {cluster.description[:300]}")
                lines.append("")

        # ── Model Weaknesses ─────────────────────────────
        if breakdown and "model_weaknesses" in breakdown:
            lines.append("## 7. Per-Model Weakness Diagnosis")
            lines.append("")
            for model_id, weaknesses in breakdown["model_weaknesses"].items():
                lines.append(f"### {model_id}")
                lines.append("")
                for w in weaknesses:
                    lines.append(f"- **{w['slice']}**: Weakest in `{w['key']}` (mean={w['mean']:.3f})")
                lines.append("")

        return "\n".join(lines)
