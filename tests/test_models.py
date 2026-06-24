"""Tests for core data models."""

import pytest
from src.core.models import (
    Document,
    ExecutionTrace,
    FailureMode,
    FailureType,
    MetricScore,
    Prediction,
    StepType,
    Task,
    TaskType,
    TokenUsage,
    TraceStep,
)


class TestTaskValidation:
    def test_rag_task_requires_docs(self):
        with pytest.raises(Exception):
            Task(
                id="test",
                type=TaskType.RAG,
                query="What is X?",
                ground_truth="X is Y",
                ground_truth_docs=[],  # Empty — should fail for RAG
            )

    def test_agent_task_requires_no_docs(self):
        task = Task(
            id="agent_1",
            type=TaskType.AGENT,
            query="Book a flight",
            ground_truth="Flight booked",
            tools=[{"name": "search_flights", "description": "..."}],
        )
        assert task.type == TaskType.AGENT

    def test_rag_task_with_docs_valid(self):
        doc = Document(id="d1", content="X is Y")
        task = Task(
            id="rag_1",
            type=TaskType.RAG,
            query="What is X?",
            ground_truth="X is Y",
            ground_truth_docs=[doc],
        )
        assert len(task.ground_truth_docs) == 1


class TestExecutionTrace:
    def test_trace_with_steps(self):
        step = TraceStep(
            step_type=StepType.GENERATION,
            step_index=0,
            input_repr="Hello",
            output_repr="World",
            latency_ms=100.0,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        trace = ExecutionTrace(
            task_id="t1",
            model_id="gpt-4o",
            steps=[step],
            total_latency_ms=100.0,
            total_token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert len(trace.steps) == 1
        assert trace.steps[0].token_usage.total_tokens == 15


class TestFailureMode:
    def test_severity_bounds(self):
        with pytest.raises(Exception):
            FailureMode(
                type=FailureType.HALLUCINATION,
                severity=1.5,  # Invalid
                description="Bad severity",
                evidence="...",
            )


class TestEvaluationResult:
    def test_get_score(self):
        result = type("_R", (), {
            "task_id": "t1",
            "model_id": "m1",
            "scores": [
                MetricScore(metric_name="faithfulness", score=0.8),
                MetricScore(metric_name="retrieval_precision", score=0.6),
            ],
            "failure_modes": [],
            "aggregate_score": 0.7,
        })()
        # Test get_score method on EvaluationResult — use the actual class
        from src.core.models import EvaluationResult
        er = EvaluationResult(
            task_id="t1",
            model_id="m1",
            scores=[
                MetricScore(metric_name="faithfulness", score=0.8),
                MetricScore(metric_name="retrieval_precision", score=0.6),
            ],
            aggregate_score=0.7,
        )
        assert er.get_score("faithfulness") == 0.8
        assert er.get_score("nonexistent") is None
