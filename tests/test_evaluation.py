"""Tests for evaluation metrics that don't require LLM calls."""

import pytest
from src.core.models import (
    Document,
    ExecutionTrace,
    Prediction,
    Task,
    TaskType,
    TokenUsage,
)
from src.evaluation.metrics.retrieval import (
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
)


@pytest.fixture
def rag_task():
    return Task(
        id="t1",
        type=TaskType.RAG,
        query="What is machine learning?",
        ground_truth="Machine learning is a subset of AI.",
        ground_truth_docs=[
            Document(id="d1", content="ML is a subset of AI."),
            Document(id="d2", content="Deep learning uses neural networks."),
            Document(id="d3", content="Supervised learning requires labeled data."),
        ],
        domain="ai",
        difficulty="easy",
    )


@pytest.fixture
def prediction_perfect():
    return Prediction(
        task_id="t1",
        model_id="test-model",
        answer="Machine learning is a subset of AI.",
        execution_trace=ExecutionTrace(
            task_id="t1",
            model_id="test-model",
            retrieved_docs=[
                Document(id="d1", content="ML is a subset of AI."),
                Document(id="d2", content="Deep learning uses neural networks."),
            ],
            total_token_usage=TokenUsage(),
        ),
    )


@pytest.fixture
def prediction_miss():
    return Prediction(
        task_id="t1",
        model_id="test-model",
        answer="I don't know.",
        execution_trace=ExecutionTrace(
            task_id="t1",
            model_id="test-model",
            retrieved_docs=[
                Document(id="d99", content="Unrelated content."),
                Document(id="d100", content="Also unrelated."),
            ],
            total_token_usage=TokenUsage(),
        ),
    )


class TestRetrievalPrecision:
    @pytest.mark.asyncio
    async def test_perfect_precision(self, rag_task, prediction_perfect):
        metric = RetrievalPrecisionMetric()
        result = await metric.evaluate(rag_task, prediction_perfect)
        assert result["score"] == 0.5  # 1 of 2 retrieved is relevant

    @pytest.mark.asyncio
    async def test_zero_precision(self, rag_task, prediction_miss):
        metric = RetrievalPrecisionMetric()
        result = await metric.evaluate(rag_task, prediction_miss)
        assert result["score"] == 0.0


class TestRetrievalRecall:
    @pytest.mark.asyncio
    async def test_partial_recall(self, rag_task, prediction_perfect):
        metric = RetrievalRecallMetric()
        result = await metric.evaluate(rag_task, prediction_perfect)
        # 1 relevant doc retrieved out of 3 relevant docs total
        assert result["score"] == 1 / 3

    @pytest.mark.asyncio
    async def test_zero_recall(self, rag_task, prediction_miss):
        metric = RetrievalRecallMetric()
        result = await metric.evaluate(rag_task, prediction_miss)
        assert result["score"] == 0.0
