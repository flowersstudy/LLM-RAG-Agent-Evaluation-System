"""
Core data models for the LLM RAG/Agent evaluation system.

All models use Pydantic for validation, serialization, and schema export.
Immutability-by-default: models use frozen=True where post-construction
mutation would break reproducibility.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ───────────────────────────────────────────────────────────

class TaskType(str, Enum):
    RAG = "rag"
    AGENT = "agent"


class StepType(str, Enum):
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"


class FailureType(str, Enum):
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    HALLUCINATION = "hallucination"
    TOOL = "tool"


# ── Atomic Types ────────────────────────────────────────────────────

class Document(BaseModel):
    """A retrieved or ground-truth document chunk."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None  # Retriever relevance score, if applicable


class TokenUsage(BaseModel):
    """Token consumption for a single LLM call or aggregate."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: Optional[str] = None


class TraceStep(BaseModel):
    """A single step in an execution trace — the atomic unit of debugging."""
    step_type: StepType
    step_index: int
    input_repr: str = ""                     # Truncated string representation of input
    output_repr: str = ""                    # Truncated string representation of output
    full_input: Optional[Any] = None         # Full input (may be large; stored separately)
    full_output: Optional[Any] = None        # Full output
    latency_ms: float = 0.0
    token_usage: Optional[TokenUsage] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Task ────────────────────────────────────────────────────────────

class Task(BaseModel, frozen=True):
    """
    A single evaluation task. Immutable — tasks are the fixed reference
    point that predictions are compared against.
    """
    id: str
    type: TaskType
    query: str = Field(description="The input query or instruction for the system")
    ground_truth: str = Field(description="Expected answer or expected final state")
    ground_truth_docs: List[Document] = Field(
        default_factory=list,
        description="Relevant documents for retrieval evaluation (RAG tasks only)",
    )
    tools: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Available tool definitions (agent tasks only)",
    )
    expected_tool_sequence: Optional[List[str]] = Field(
        default=None,
        description="Expected tool names in order (agent tasks, optional)",
    )
    domain: str = Field(default="general", description="Knowledge domain for stratification")
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("ground_truth_docs")
    @classmethod
    def _validate_docs_for_rag(cls, v, info):
        if info.data.get("type") == TaskType.RAG and not v:
            raise ValueError("RAG tasks must have at least one ground_truth_doc")
        return v


# ── Prediction ──────────────────────────────────────────────────────

class ExecutionTrace(BaseModel):
    """Full execution trace — the 'black box recorder' for debugging."""
    task_id: str
    model_id: str
    steps: List[TraceStep] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    retrieved_docs: List[Document] = Field(default_factory=list)
    final_answer: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    """A model's complete output for a task, including the full trace."""
    task_id: str
    model_id: str
    answer: str
    execution_trace: ExecutionTrace
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Evaluation ──────────────────────────────────────────────────────

class FailureMode(BaseModel):
    """A classified failure with evidence anchored in the trace."""
    type: FailureType
    severity: float = Field(ge=0.0, le=1.0, description="0 = no impact, 1 = critical")
    description: str = Field(description="Human-readable explanation of the failure")
    evidence: str = Field(description="Quote or reference from trace steps supporting this classification")
    trace_step_indices: List[int] = Field(
        default_factory=list,
        description="Indices into ExecutionTrace.steps where this failure is visible",
    )


class MetricScore(BaseModel):
    """A single metric result with supporting details."""
    metric_name: str
    score: float
    rationale: Optional[str] = None          # LLM judge reasoning, if applicable
    evidence: List[str] = Field(default_factory=list)  # Supporting quotes
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Per-task, per-model evaluation output."""
    task_id: str
    model_id: str
    scores: List[MetricScore] = Field(default_factory=list)
    failure_modes: List[FailureMode] = Field(default_factory=list)
    aggregate_score: Optional[float] = Field(
        default=None,
        description="Weighted composite score, computed post-hoc",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_score(self, metric_name: str) -> Optional[float]:
        for s in self.scores:
            if s.metric_name == metric_name:
                return s.score
        return None


# ── Experiment ──────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """Snapshot of all parameters needed to reproduce an experiment."""
    experiment_id: str
    description: str = ""
    models: List[str] = Field(description="List of model IDs to evaluate")
    metrics: List[str] = Field(description="List of metric names to compute")
    dataset_path: str
    split: Optional[str] = None
    random_seed: int = 42
    llm_judge_model: str = "gpt-4"          # Model used for LLM-as-judge
    max_tasks: Optional[int] = None          # Limit for quick runs
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentManifest(BaseModel):
    """Top-level container for a complete experiment run."""
    config: ExperimentConfig
    tasks: List[Task] = Field(default_factory=list)
    predictions: List[Prediction] = Field(default_factory=list)
    results: List[EvaluationResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: Literal["running", "completed", "failed"] = "running"


# ── Analysis ────────────────────────────────────────────────────────

class FailureCluster(BaseModel):
    """A group of similar failures across tasks/models."""
    cluster_id: int
    label: str                                # Human-readable cluster name
    failure_type: FailureType
    size: int
    representative_examples: List[str] = Field(
        default_factory=list,
        description="Task IDs that best represent this cluster",
    )
    description: str = ""


class ModelComparison(BaseModel):
    """Side-by-side comparison of models across metrics."""
    model_ids: List[str]
    metric_name: str
    mean_scores: Dict[str, float] = Field(default_factory=dict)
    std_scores: Dict[str, float] = Field(default_factory=dict)
    win_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Pairwise win rate: model_a -> fraction of tasks where it outscores others",
    )
    significant: bool = False                 # Whether differences are statistically significant
    p_value: Optional[float] = None


class AggregateReport(BaseModel):
    """Full analysis output for an experiment."""
    experiment_id: str
    model_comparisons: List[ModelComparison] = Field(default_factory=list)
    failure_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="FailureType -> count across all models",
    )
    failure_clusters: List[FailureCluster] = Field(default_factory=list)
    per_model_failure_rates: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="model_id -> FailureType -> rate",
    )
    summary: str = ""
