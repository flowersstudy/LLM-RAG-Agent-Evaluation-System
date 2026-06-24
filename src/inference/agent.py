"""
Agent pipeline: ReAct-pattern execution for tool-calling agent workflows.

Flow: reasoning → tool_call → observe → repeat → final_answer

Each turn is captured in the ExecutionTrace as REASONING + TOOL_CALL steps,
enabling downstream metric computation and failure analysis.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from src.core.interfaces import LLMAdapter, Pipeline
from src.core.models import (
    ExecutionTrace,
    Prediction,
    StepType,
    Task,
    TokenUsage,
    TraceStep,
)
from src.core.registry import register
from src.inference.tools import MockToolExecutor
from src.utils.logging import get_logger, get_tracer

logger = get_logger()

AGENT_SYSTEM_PROMPT = """\
You are a research assistant agent. You have access to tools that help you
answer questions. Think step by step:

1. Analyze what you need to know to answer the query
2. Choose the appropriate tool(s) to gather information
3. Use the tool results to construct your answer
4. When you have enough information, provide a final answer

Important rules:
- Always reason about what you need before calling a tool
- If a tool returns no results, try a different approach
- When you can answer the query, output your final answer directly (no tool call)
- Be precise — use exact values from tool outputs in your answer"""


@register("pipeline", "agent")
class AgentPipeline(Pipeline):
    """ReAct-pattern agent execution pipeline.

    Executes a task by looping: reasoning → tool call → observation,
    until a final answer is produced or max_steps is reached.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        max_steps: int = 10,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._max_steps = max_steps
        self._system_prompt = system_prompt or AGENT_SYSTEM_PROMPT
        self._tool_executor = MockToolExecutor()

    async def run(self, task: Task) -> Prediction:
        tracer = get_tracer()
        tracer.log("agent_pipeline.start", task_id=task.id, model=self._llm.model_id)

        steps: List[TraceStep] = []
        total_tokens = TokenUsage()
        pipeline_start = time.perf_counter()
        step_index = 0

        # Build tools from task
        tools = task.tools if task.tools else []

        # Messages list for the conversation
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task.query},
        ]

        final_answer = None

        for turn in range(self._max_steps):
            # ── LLM call with tools ──────────────────────────
            llm_start = time.perf_counter()
            content, tool_calls, llm_meta = await self._llm.generate_with_tools(
                prompt="",  # Not used when we pass messages via kwargs
                tools=tools,
                messages=messages,
            )
            llm_latency = (time.perf_counter() - llm_start) * 1000

            step_tokens = TokenUsage(
                prompt_tokens=llm_meta.get("prompt_tokens", 0),
                completion_tokens=llm_meta.get("completion_tokens", 0),
                total_tokens=llm_meta.get("total_tokens", 0),
            )
            total_tokens.prompt_tokens += step_tokens.prompt_tokens
            total_tokens.completion_tokens += step_tokens.completion_tokens
            total_tokens.total_tokens += step_tokens.total_tokens

            # ── Record reasoning step ───────────────────────
            reasoning_content = content or "(tool call only)"
            steps.append(TraceStep(
                step_type=StepType.REASONING,
                step_index=step_index,
                input_repr=f"turn {turn + 1}: {task.query[:120]}",
                output_repr=reasoning_content[:500],
                latency_ms=llm_latency,
                token_usage=step_tokens,
            ))
            step_index += 1

            # ── No tool calls → final answer ────────────────
            if not tool_calls:
                final_answer = content
                break

            # ── Build assistant message with tool calls ────
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": f"call_{turn}_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", "{}") if isinstance(tc.get("arguments"), str) else json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
            messages.append(assistant_msg)

            # ── Execute each tool call ──────────────────────
            for i, tc in enumerate(tool_calls):
                tool_name = tc.get("name", "unknown")
                tool_args = self._parse_args(tc.get("arguments", "{}"))
                tool_start = time.perf_counter()
                tool_result = self._tool_executor.execute(tool_name, tool_args)
                tool_latency = (time.perf_counter() - tool_start) * 1000

                steps.append(TraceStep(
                    step_type=StepType.TOOL_CALL,
                    step_index=step_index,
                    input_repr=f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})",
                    output_repr=tool_result[:500],
                    latency_ms=tool_latency,
                    metadata={"tool_name": tool_name, "tool_args": tool_args},
                ))
                step_index += 1

                # Add tool response with matching tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{turn}_{i}",
                    "content": tool_result,
                })

        # ── If loop exhausted without final answer ──────────────
        if final_answer is None:
            final_answer = content or "(Agent did not produce a final answer)"

        total_latency = (time.perf_counter() - pipeline_start) * 1000

        trace = ExecutionTrace(
            task_id=task.id,
            model_id=self._llm.model_id,
            steps=steps,
            total_latency_ms=total_latency,
            total_token_usage=total_tokens,
            retrieved_docs=[],
            final_answer=final_answer,
            metadata={"max_steps": self._max_steps, "turns": turn + 1},
        )

        tracer.log(
            "agent_pipeline.end",
            task_id=task.id,
            latency_ms=total_latency,
            steps=len(steps),
            turns=turn + 1,
            answer_len=len(final_answer),
        )

        return Prediction(
            task_id=task.id,
            model_id=self._llm.model_id,
            answer=final_answer,
            execution_trace=trace,
        )

    def _parse_args(self, arguments: str | Dict[str, Any]) -> Dict[str, Any]:
        """Parse tool arguments from string or dict."""
        if isinstance(arguments, dict):
            return arguments
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool arguments: {arguments[:200]}")
            return {}
