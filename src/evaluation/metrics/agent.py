"""
Agent evaluation metrics for multi-step tool-calling workflows.

Metrics:
- TaskSuccessMetric: Did the agent achieve the goal? (LLM-judge)
- ToolSelectionAccuracyMetric: Were the correct tools used? (reference-based)
- ReasoningTraceCoherenceMetric: Is the reasoning logical? (LLM-judge)
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.interfaces import Metric
from src.core.models import Prediction, StepType, Task
from src.core.registry import register
from src.evaluation.judge import StructuredJudge
from src.utils.logging import get_logger

logger = get_logger()

TASK_SUCCESS_RUBRIC = """\
Evaluate whether the Agent's Answer successfully completes the task described in the Query.

Compare the Answer against the Ground Truth. Consider:
- Does the answer contain the same key facts as the ground truth?
- Is the answer complete (no missing critical information)?
- Is the answer factually consistent with the ground truth?

Scoring:
- 1.0: Answer is fully correct and complete. All key facts match ground truth.
- 0.7-0.9: Answer is mostly correct with minor omissions or imprecise values.
- 0.4-0.6: Answer has significant errors or omissions but captures the general idea.
- 0.1-0.3: Answer is mostly wrong but shows some attempt related to the query.
- 0.0: Answer is completely wrong, irrelevant, or missing."""

TOOL_SELECTION_RUBRIC = """\
Evaluate whether the Agent used the most appropriate tools for this task.

Consider:
- Were the tools used necessary and sufficient for the task?
- Could a more efficient sequence of tool calls achieve the same result?
- Were there any redundant or unnecessary tool calls?

Scoring:
- 1.0: All tool calls were necessary and sufficient. Optimal sequence.
- 0.7-0.9: Mostly correct tools but slightly suboptimal sequence or one extraneous call.
- 0.4-0.6: One missing or wrong tool, but generally on track.
- 0.1-0.3: Multiple wrong or missing tools.
- 0.0: Completely wrong tool selection."""

REASONING_COHERENCE_RUBRIC = """\
Evaluate whether the Agent's reasoning trace shows logical, coherent step-by-step thinking.

Review the reasoning steps in the execution trace. Consider:
- Does each step follow logically from the previous ones?
- Are there contradictions or circular reasoning?
- Is the final conclusion supported by the intermediate reasoning?
- Are tool results correctly interpreted?

Scoring:
- 1.0: Reasoning is perfectly logical, each step builds on prior ones, conclusion well-supported.
- 0.7-0.9: Reasoning is largely coherent with minor jumps or unclear transitions.
- 0.4-0.6: Some logical gaps or leaps but overall direction is correct.
- 0.1-0.3: Significant logical errors, contradictions, or misinterpretations.
- 0.0: Reasoning is incoherent, contradictory, or completely disconnected from the answer."""


@register("metric", "task_success")
class TaskSuccessMetric(Metric):
    """Measures whether the agent successfully completed the task."""

    def __init__(self, judge: StructuredJudge) -> None:
        self._judge = judge

    @property
    def name(self) -> str:
        return "task_success"

    @property
    def requires_ground_truth(self) -> bool:
        return True

    async def evaluate(self, task: Task, prediction: Prediction) -> Dict[str, Any]:
        result = await self._judge.judge(
            rubric=TASK_SUCCESS_RUBRIC,
            query=task.query,
            answer=prediction.answer,
            ground_truth=task.ground_truth,
        )
        return {
            "score": result["score"],
            "rationale": result.get("rationale", ""),
            "evidence": result.get("evidence", []),
        }


@register("metric", "tool_selection_accuracy")
class ToolSelectionAccuracyMetric(Metric):
    """
    Measures whether the agent selected the correct tools.

    Compares actual tool call names against the expected tool sequence.
    Uses sequence-alignment scoring: reward for correct tools in correct order,
    penalize wrong or missing tools.
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "tool_selection_accuracy"

    @property
    def requires_ground_truth(self) -> bool:
        return True  # Needs expected_tool_sequence

    async def evaluate(self, task: Task, prediction: Prediction) -> Dict[str, Any]:
        # Extract actual tool names from trace
        actual_tools = [
            step.metadata.get("tool_name", "")
            for step in prediction.execution_trace.steps
            if step.step_type == StepType.TOOL_CALL
        ]

        expected_sequence = task.expected_tool_sequence or []

        if not expected_sequence:
            return {
                "score": 1.0,
                "rationale": "No expected tool sequence defined — accuracy is vacuously 1.0.",
                "evidence": [],
            }

        if not actual_tools:
            return {
                "score": 0.0,
                "rationale": f"No tools were called, but expected: {expected_sequence}",
                "evidence": [],
            }

        # Sequence-aware scoring: longest common subsequence ratio
        lcs_len = self._lcs_length(expected_sequence, actual_tools)
        precision = lcs_len / len(actual_tools) if actual_tools else 0
        recall = lcs_len / len(expected_sequence) if expected_sequence else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "score": round(f1, 4),
            "rationale": (
                f"Expected: {expected_sequence}, Actual: {actual_tools}. "
                f"LCS={lcs_len}, Precision={precision:.3f}, Recall={recall:.3f}"
            ),
            "evidence": actual_tools,
        }

    def _lcs_length(self, a: list, b: list) -> int:
        """Longest common subsequence length (order-preserving)."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n]


@register("metric", "reasoning_trace_coherence")
class ReasoningTraceCoherenceMetric(Metric):
    """Measures whether the agent's reasoning trace is logically coherent."""

    def __init__(self, judge: StructuredJudge) -> None:
        self._judge = judge

    @property
    def name(self) -> str:
        return "reasoning_trace_coherence"

    @property
    def requires_ground_truth(self) -> bool:
        return False

    async def evaluate(self, task: Task, prediction: Prediction) -> Dict[str, Any]:
        # Format the trace for the judge
        trace_text = self._format_trace(prediction.execution_trace.steps)

        if not trace_text:
            return {
                "score": 0.0,
                "rationale": "No execution trace available.",
                "evidence": [],
            }

        result = await self._judge.judge(
            rubric=REASONING_COHERENCE_RUBRIC,
            query=f"Task: {task.query}\n\nExecution Trace:\n{trace_text}",
            answer=prediction.answer,
        )
        return {
            "score": result["score"],
            "rationale": result.get("rationale", ""),
            "evidence": result.get("evidence", []),
        }

    def _format_trace(self, steps) -> str:
        """Format trace steps into readable text for the judge."""
        lines = []
        for s in steps:
            if s.step_type == StepType.REASONING:
                lines.append(f"[Reasoning] {s.output_repr}")
            elif s.step_type == StepType.TOOL_CALL:
                lines.append(f"[Tool] {s.input_repr} → {s.output_repr}")
            elif s.step_type == StepType.GENERATION:
                lines.append(f"[Answer] {s.output_repr}")
        return "\n\n".join(lines) if lines else ""
