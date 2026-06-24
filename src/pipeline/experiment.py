"""
Experiment runner: orchestrates the full evaluation pipeline.

Flow: config → dataset → (for each model) → (for each task) → inference → evaluation → save
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.core.interfaces import LLMAdapter, Pipeline
from src.core.models import (
    AggregateReport,
    EvaluationResult,
    ExperimentConfig,
    ExperimentManifest,
    ModelComparison,
    Prediction,
    Task,
)
from src.core.registry import get_component
from src.dataset.loader import JSONDataset
from src.evaluation import CompositeScorer
from src.evaluation.judge import StructuredJudge
from src.evaluation.metrics.faithfulness import FaithfulnessMetric
from src.evaluation.metrics.retrieval import (
    AnswerRelevanceMetric,
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
)
from src.inference.adapters.anthropic import AnthropicAdapter
from src.inference.adapters.openai import OpenAIAdapter
from src.inference.rag import RAGPipeline
from src.inference.reranker import Reranker, TwoStageRetriever
from src.inference.retriever import DenseRetriever
from src.utils.logging import close_tracing, get_logger, init_tracing
from src.utils.reproducibility import seed_info, set_seed

logger = get_logger()

# Metric factory — maps metric name to constructor
_METRIC_REGISTRY: Dict[str, type] = {
    "faithfulness": FaithfulnessMetric,
    "retrieval_precision": RetrievalPrecisionMetric,
    "retrieval_recall": RetrievalRecallMetric,
    "answer_relevance": AnswerRelevanceMetric,
}


class ExperimentRunner:
    """Runs a complete evaluation experiment end to end."""

    async def run(self, config: ExperimentConfig) -> str:
        """
        Execute a full experiment.

        Returns the path to the experiment output directory.
        """
        set_seed(config.random_seed)

        # ── Setup output directory ─────────────────────────
        output_dir = Path("experiments") / config.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        init_tracing(output_dir / "trace.jsonl")

        logger.info(f"Experiment '{config.experiment_id}' starting. Output: {output_dir}")

        manifest = ExperimentManifest(config=config, status="running")

        # ── Save config snapshot ───────────────────────────
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, ensure_ascii=False, indent=2, default=str)

        try:
            # ── Load dataset ───────────────────────────────
            corpus_path = Path(config.dataset_path).parent / "corpus.json"
            dataset = JSONDataset(
                config.dataset_path,
                corpus_path=corpus_path if corpus_path.exists() else None,
            )
            tasks = list(dataset)
            if config.max_tasks:
                tasks = tasks[: config.max_tasks]
            manifest.tasks = tasks
            logger.info(f"Dataset loaded: {len(tasks)} tasks across {dataset.metadata.get('domains', [])}")

            # ── Build judge (shared across models) ─────────
            judge_llm = self._build_llm(config.llm_judge_model, config.judge_params)
            judge = StructuredJudge(judge_llm)

            # ── Build metrics ──────────────────────────────
            scorers: Dict[str, CompositeScorer] = {}
            metrics = self._build_metrics(judge, config.metrics)

            # ── Load corpus for retrieval ──────────────────
            corpus_path = Path(config.dataset_path).parent / "corpus.json"
            retriever = self._build_retriever(corpus_path, config)

            # ── Run for each model ─────────────────────────
            for model_id in config.models:
                logger.info(f"Evaluating model: {model_id}")
                model_params = config.model_params.get(model_id, {})
                llm = self._build_llm(model_id, model_params)
                pipeline = RAGPipeline(llm=llm, retriever=retriever)

                # Warm up metrics for this model
                scorers[model_id] = CompositeScorer(metrics=metrics)

                for task in tasks:
                    try:
                        prediction = await pipeline.run(task)
                        manifest.predictions.append(prediction)
                        result = await scorers[model_id].evaluate(task, prediction)
                        manifest.results.append(result)
                    except Exception as e:
                        logger.error(f"Task {task.id} failed for model {model_id}: {e}")
                        manifest.metadata.setdefault("failed_tasks", []).append(
                            {"task_id": task.id, "model_id": model_id, "error": str(e)}
                        )

            # ── Generate aggregate report ──────────────────
            manifest.completed_at = datetime.now()
            manifest.status = "completed"
            report = self._aggregate(manifest)
            manifest.metadata["aggregate_report"] = report.model_dump()

            # ── Save manifest ──────────────────────────────
            manifest_path = output_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest.model_dump(), f, ensure_ascii=False, indent=2, default=str)

            self._print_summary(report)

        except Exception:
            manifest.status = "failed"
            logger.exception("Experiment failed")
            raise
        finally:
            close_tracing()

        return str(output_dir)

    def _build_llm(self, model_id: str, params: Dict[str, Any]):
        """Build an LLM adapter with model-specific params.

        Provider is auto-detected from params.get('provider') or model name.
        api_key is resolved from:
        1. params dict (highest priority)
        2. environment variable: {PROVIDER}_API_KEY (e.g., ANTHROPIC_API_KEY)
        3. environment variable: OPENAI_API_KEY (fallback)
        """
        provider = params.get("provider", "openai")
        base_url = params.get("base_url")
        api_key = params.get("api_key")

        # Resolve api_key from environment if not in params
        if not api_key:
            env_prefix = params.get("env_prefix", provider.upper())
            api_key = (
                os.environ.get(f"{env_prefix}_API_KEY")
                or os.environ.get(f"{provider.upper()}_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

        if provider == "anthropic":
            return AnthropicAdapter(
                model=model_id,
                api_key=api_key,
                base_url=base_url,
            )
        else:
            return OpenAIAdapter(
                model=model_id,
                api_key=api_key,
                base_url=base_url,
            )

    def _build_metrics(self, judge: StructuredJudge, metric_names: List[str]) -> list:
        metrics = []
        for name in metric_names:
            cls = _METRIC_REGISTRY.get(name)
            if cls is None:
                logger.warning(f"Unknown metric: {name}, skipping")
                continue
            if cls in (FaithfulnessMetric, AnswerRelevanceMetric):
                metrics.append(cls(judge))
            else:
                metrics.append(cls())
        return metrics

    def _build_retriever(self, corpus_path: Path, config: ExperimentConfig):
        """Build retriever from config. Supports single-stage and two-stage (with reranker)."""
        from src.core.models import Document

        # Build dense retriever
        dense = DenseRetriever(model_name=config.retriever_model)

        # Index corpus if available
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                docs_raw = json.load(f)
            documents = [Document(**d) for d in docs_raw]
            dense.index(documents)
        else:
            logger.warning(f"Corpus not found at {corpus_path} — retriever will be empty")

        # Wrap in two-stage retriever if reranker is configured
        if config.reranker_model:
            reranker = Reranker(model_name=config.reranker_model)
            stage2 = TwoStageRetriever(
                dense_retriever=dense,
                reranker=reranker,
                final_top_k=config.retriever_top_k,
            )
            return stage2

        return dense

    def _aggregate(self, manifest: ExperimentManifest) -> AggregateReport:
        model_ids = manifest.config.models
        metric_names = manifest.config.metrics
        results = manifest.results

        comparisons: List[ModelComparison] = []
        failure_counts: Dict[str, int] = {}
        per_model_failures: Dict[str, Dict[str, float]] = {}

        for metric_name in metric_names:
            mean_scores: Dict[str, float] = {}
            std_scores: Dict[str, float] = {}
            for model_id in model_ids:
                model_results = [r for r in results if r.model_id == model_id]
                scores = [r.get_score(metric_name) for r in model_results]
                scores = [s for s in scores if s is not None]
                if scores:
                    mean_scores[model_id] = round(sum(scores) / len(scores), 4)
                    variance = sum((s - mean_scores[model_id]) ** 2 for s in scores) / len(scores)
                    std_scores[model_id] = round(variance ** 0.5, 4)

            # Simple pairwise win rate
            win_rates: Dict[str, float] = {}
            if len(model_ids) >= 2:
                for model_id in model_ids:
                    wins = 0
                    total = 0
                    model_results = {
                        r.task_id: r.get_score(metric_name)
                        for r in results
                        if r.model_id == model_id and r.get_score(metric_name) is not None
                    }
                    for other_id in model_ids:
                        if other_id == model_id:
                            continue
                        other_results = {
                            r.task_id: r.get_score(metric_name)
                            for r in results
                            if r.model_id == other_id and r.get_score(metric_name) is not None
                        }
                        common = set(model_results) & set(other_results)
                        for task_id in common:
                            if model_results[task_id] > other_results[task_id]:
                                wins += 1
                            total += 1
                    win_rates[model_id] = round(wins / total, 4) if total > 0 else 0.0

            comparisons.append(ModelComparison(
                model_ids=model_ids,
                metric_name=metric_name,
                mean_scores=mean_scores,
                std_scores=std_scores,
                win_rates=win_rates,
            ))

        # Failure distribution
        for r in results:
            for fm in r.failure_modes:
                ft = fm.type.value
                failure_counts[ft] = failure_counts.get(ft, 0) + 1

        # Per-model failure rates
        for model_id in model_ids:
            model_results = [r for r in results if r.model_id == model_id]
            model_failures: Dict[str, int] = {}
            for r in model_results:
                for fm in r.failure_modes:
                    ft = fm.type.value
                    model_failures[ft] = model_failures.get(ft, 0) + 1
            per_model_failures[model_id] = {
                ft: count / len(model_results) if model_results else 0
                for ft, count in model_failures.items()
            }

        return AggregateReport(
            experiment_id=manifest.config.experiment_id,
            model_comparisons=comparisons,
            failure_distribution=failure_counts,
            per_model_failure_rates=per_model_failures,
            summary=self._format_summary(comparisons, failure_counts, per_model_failures),
        )

    def _format_summary(
        self,
        comparisons: List[ModelComparison],
        failure_counts: Dict[str, int],
        per_model_failures: Dict[str, Dict[str, float]],
    ) -> str:
        lines = ["=== Experiment Summary ===", ""]
        for c in comparisons:
            lines.append(f"Metric: {c.metric_name}")
            for model_id in c.model_ids:
                mean = c.mean_scores.get(model_id, "N/A")
                std = c.std_scores.get(model_id, "N/A")
                lines.append(f"  {model_id}: mean={mean}, std={std}")
            lines.append("")
        lines.append("Failure distribution:")
        for ft, count in sorted(failure_counts.items()):
            lines.append(f"  {ft}: {count}")
        return "\n".join(lines)

    def _print_summary(self, report: AggregateReport) -> None:
        logger.info(report.summary)
