"""
RAG pipeline: retrieve → format context → generate answer.

Captures every step in ExecutionTrace for downstream analysis.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.core.interfaces import LLMAdapter, Pipeline
from src.core.models import (
    Document,
    ExecutionTrace,
    Prediction,
    StepType,
    Task,
    TokenUsage,
    TraceStep,
)
from src.core.registry import register
from src.inference.retriever import DenseRetriever
from src.utils.logging import get_logger, get_tracer

logger = get_logger()

RAG_SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly.
Do not make up facts that are not in the context.

Context:
{context}"""


@register("pipeline", "rag")
class RAGPipeline(Pipeline):
    """Standard RAG pipeline: retrieve → format → generate."""

    def __init__(
        self,
        llm: LLMAdapter,
        retriever: DenseRetriever,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._top_k = top_k
        self._system_prompt_template = system_prompt or RAG_SYSTEM_PROMPT

    async def run(self, task: Task) -> Prediction:
        tracer = get_tracer()
        tracer.log("rag_pipeline.start", task_id=task.id, model=self._llm.model_id)

        steps: List[TraceStep] = []
        total_tokens = TokenUsage()
        pipeline_start = time.perf_counter()

        # ── Step 1: Retrieve ─────────────────────────────────
        retrieve_start = time.perf_counter()
        retrieved_docs = await self._retriever.retrieve(task.query, top_k=self._top_k)
        retrieve_latency = (time.perf_counter() - retrieve_start) * 1000

        steps.append(TraceStep(
            step_type=StepType.RETRIEVAL,
            step_index=0,
            input_repr=f"query: {task.query[:200]}",
            output_repr=f"{len(retrieved_docs)} docs retrieved",
            latency_ms=retrieve_latency,
        ))

        # ── Step 2: Format context ────────────────────────────
        context = self._format_context(retrieved_docs)
        system_prompt = self._system_prompt_template.format(context=context)

        # ── Step 3: Generate answer ───────────────────────────
        gen_start = time.perf_counter()
        answer, gen_metadata = await self._llm.generate(
            prompt=task.query,
            system_prompt=system_prompt,
        )
        gen_latency = (time.perf_counter() - gen_start) * 1000

        step_tokens = TokenUsage(
            prompt_tokens=gen_metadata.get("prompt_tokens", 0),
            completion_tokens=gen_metadata.get("completion_tokens", 0),
            total_tokens=gen_metadata.get("total_tokens", 0),
        )
        total_tokens = TokenUsage(
            prompt_tokens=total_tokens.prompt_tokens + step_tokens.prompt_tokens,
            completion_tokens=total_tokens.completion_tokens + step_tokens.completion_tokens,
            total_tokens=total_tokens.total_tokens + step_tokens.total_tokens,
        )

        steps.append(TraceStep(
            step_type=StepType.GENERATION,
            step_index=1,
            input_repr=f"query: {task.query[:200]}\ncontext: {len(context)} chars",
            output_repr=answer[:500],
            latency_ms=gen_latency,
            token_usage=step_tokens,
        ))

        total_latency = (time.perf_counter() - pipeline_start) * 1000

        trace = ExecutionTrace(
            task_id=task.id,
            model_id=self._llm.model_id,
            steps=steps,
            total_latency_ms=total_latency,
            total_token_usage=total_tokens,
            retrieved_docs=retrieved_docs,
            final_answer=answer,
        )

        tracer.log(
            "rag_pipeline.end",
            task_id=task.id,
            latency_ms=total_latency,
            retrieved_count=len(retrieved_docs),
            answer_len=len(answer),
        )

        return Prediction(
            task_id=task.id,
            model_id=self._llm.model_id,
            answer=answer,
            execution_trace=trace,
        )

    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a single context string."""
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[Document {i}] {doc.content}")
        return "\n\n".join(parts)
