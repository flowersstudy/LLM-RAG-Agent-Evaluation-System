"""
Abstract base classes defining the extension contracts.

Every pluggable component (LLM adapter, metric, dataset loader, pipeline)
must implement its corresponding interface. The registry uses these ABCs
for dynamic discovery and validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Protocol

from .models import (
    EvaluationResult,
    ExecutionTrace,
    ExperimentConfig,
    Prediction,
    Task,
)


# ── Dataset ─────────────────────────────────────────────────────────

class Dataset(ABC):
    """Produces evaluation tasks. May load from disk or generate synthetically."""

    @abstractmethod
    def __iter__(self) -> Iterator[Task]:
        """Yield tasks one at a time (memory-efficient for large datasets)."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of tasks (may require loading the full dataset)."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Dataset-level metadata: source, size, domains, etc."""
        ...


# ── LLM Adapter ─────────────────────────────────────────────────────

class LLMAdapter(ABC):
    """
    Unified interface for LLM providers.

    Each adapter wraps one provider (OpenAI, Anthropic, local vLLM, etc.)
    and exposes a minimal surface: generate() for single-turn and
    generate_with_tools() for tool-calling.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this model (e.g., 'gpt-4', 'claude-sonnet-4-6')."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Single-turn text generation.

        Returns:
            (response_text, metadata) where metadata includes token usage, latency, model version.
        """
        ...

    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generation with tool calling support.

        Returns:
            (response_text, tool_calls, metadata) where tool_calls is a list of
            {name, arguments} dicts.
        """
        ...


# ── Retriever ───────────────────────────────────────────────────────

class Retriever(ABC):
    """Document retrieval interface for RAG pipelines."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """Return top-k documents (implementation-agnostic)."""
        ...


# ── Pipeline ────────────────────────────────────────────────────────

class Pipeline(ABC):
    """
    Executes a task and produces a prediction with full trace.

    Implementations: RAGPipeline, AgentPipeline.
    """

    @abstractmethod
    async def run(self, task: Task) -> Prediction:
        """Execute the pipeline on a task and return a prediction with trace."""
        ...


# ── Metric ──────────────────────────────────────────────────────────

class Metric(ABC):
    """A single evaluation metric."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric name (e.g., 'faithfulness', 'retrieval_precision')."""
        ...

    @property
    @abstractmethod
    def requires_ground_truth(self) -> bool:
        """Whether this metric needs ground_truth on the Task."""
        ...

    @abstractmethod
    async def evaluate(self, task: Task, prediction: Prediction) -> Dict[str, Any]:
        """
        Compute the metric.

        Returns a dict with at minimum:
            - score: float
            - rationale: Optional[str]
            - evidence: List[str]
        """
        ...


# ── Judge (LLM-as-Judge) ────────────────────────────────────────────

class Judge(ABC):
    """LLM-as-judge: uses an LLM to evaluate output quality with a rubric."""

    @abstractmethod
    async def judge(
        self,
        task: Task,
        prediction: Prediction,
        rubric: str,
    ) -> Dict[str, Any]:
        """
        Apply a structured rubric and return:
            - score: float
            - rationale: str
            - evidence: List[str]
        """
        ...


# ── Failure Classifier ──────────────────────────────────────────────

class FailureClassifier(ABC):
    """Classifies failures in a prediction given the task and trace."""

    @abstractmethod
    async def classify(
        self,
        task: Task,
        prediction: Prediction,
        eval_result: EvaluationResult,
    ) -> EvaluationResult:
        """
        Add failure classifications to an evaluation result.
        Returns the augmented EvaluationResult (with failure_modes populated).
        """
        ...


# ── Experiment Runner ───────────────────────────────────────────────

class ExperimentRunner(ABC):
    """Orchestrates a full experiment: dataset → inference → evaluation → analysis."""

    @abstractmethod
    async def run(self, config: ExperimentConfig) -> str:
        """
        Execute a full experiment.

        Returns:
            Path to the experiment output directory.
        """
        ...
