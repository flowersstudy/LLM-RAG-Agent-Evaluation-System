"""
OpenAI adapter implementing the LLMAdapter interface.

Supports: GPT-4, GPT-4o, GPT-3.5-turbo, and OpenAI-compatible endpoints.
Each generate() call is traced via the structured logger.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from src.core.interfaces import LLMAdapter
from src.core.registry import register
from src.utils.logging import get_logger, get_tracer

logger = get_logger()


@register("llm_adapter", "openai")
class OpenAIAdapter(LLMAdapter):
    """LLMAdapter for OpenAI models."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def model_id(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tracer = get_tracer()
        tracer.log("llm_call.start", model=self._model, provider="openai", prompt_len=len(prompt))

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        elapsed = time.perf_counter() - start

        content = response.choices[0].message.content or ""
        usage = response.usage

        metadata: Dict[str, Any] = {
            "model": response.model,
            "latency_ms": elapsed * 1000,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "finish_reason": response.choices[0].finish_reason,
        }

        tracer.log(
            "llm_call.end",
            model=self._model,
            latency_ms=metadata["latency_ms"],
            tokens=metadata["total_tokens"],
        )

        return content, metadata

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tracer = get_tracer()
        tracer.log(
            "llm_tool_call.start",
            model=self._model,
            provider="openai",
            tool_count=len(tools),
        )

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        elapsed = time.perf_counter() - start

        choice = response.choices[0]
        content = choice.message.content or ""

        tool_calls_raw = getattr(choice.message, "tool_calls", None) or []
        tool_calls = [
            {"name": tc.function.name, "arguments": tc.function.arguments}
            for tc in tool_calls_raw
        ]

        usage = response.usage
        metadata: Dict[str, Any] = {
            "model": response.model,
            "latency_ms": elapsed * 1000,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "tool_calls_count": len(tool_calls),
        }

        tracer.log("llm_tool_call.end", **metadata)
        return content, tool_calls, metadata
