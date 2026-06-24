"""
Standalone analysis runner: re-analyzes existing experiment manifests.

Usage:
    python -m src.analysis.run_analysis experiments/deepseek_vs_kimi
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import AggregateReport, ExperimentManifest
from src.analysis.cluster import FailureClusterer
from src.analysis.correlation import MetricCorrelator
from src.analysis.breakdown import StratifiedAnalyzer
from src.analysis.reporter import ReportGenerator


def analyze_experiment(exp_dir: str) -> None:
    exp_path = Path(exp_dir)
    manifest_path = exp_path / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    manifest = ExperimentManifest(**data)
    metric_names = manifest.config.metrics

    # Rebuild aggregate report
    from src.pipeline.experiment import ExperimentRunner
    runner = ExperimentRunner()
    report = runner._aggregate(manifest)

    # Clustering
    clusterer = FailureClusterer()
    clusters = clusterer.cluster(manifest.results, manifest.tasks)
    report.failure_clusters = clusters

    # Correlation
    correlator = MetricCorrelator()
    correlations = correlator.compute(manifest.results, metric_names)

    # Breakdown
    analyzer = StratifiedAnalyzer()
    breakdown = analyzer.analyze(manifest.results, manifest.tasks, metric_names)

    # Generate report
    reporter = ReportGenerator()
    md = reporter.generate(report, correlations, breakdown, exp_path)
    print(f"Report generated: {exp_path / 'report.md'}")
    print(f"  Models: {manifest.config.models}")
    print(f"  Tasks: {len(manifest.results)}")
    print(f"  Clusters: {len(clusters)}")
    print(f"  Report size: {len(md)} chars")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.run_analysis <experiment_dir>")
        sys.exit(1)
    analyze_experiment(sys.argv[1])
