"""
Anthropic adapter implementing the LLMAdapter interface.

Supports: Claude Opus 4, Sonnet 4, Haiku 4.5, and earlier Claude 3.x models.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from src.core.interfaces import LLMAdapter
from src.core.registry import register
from src.utils.logging import get_logger, get_tracer

logger = get_logger()


@register("llm_adapter", "anthropic")
class AnthropicAdapter(LLMAdapter):
    """LLMAdapter for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    @property
    def model_id(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        tracer = get_tracer()
        tracer.log(
            "llm_call.start",
            model=self._model,
            provider="anthropic",
            prompt_len=len(prompt),
        )

        start = time.perf_counter()
        response = await self._client.messages.create(
            model=self._model,
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        elapsed = time.perf_counter() - start

        content = response.content[0].text if response.content else ""
        usage = response.usage

        metadata: Dict[str, Any] = {
            "model": response.model,
            "latency_ms": elapsed * 1000,
            "prompt_tokens": usage.input_tokens if usage else 0,
            "completion_tokens": usage.output_tokens if usage else 0,
            "total_tokens": (usage.input_tokens + usage.output_tokens) if usage else 0,
            "stop_reason": response.stop_reason,
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
        tracer = get_tracer()
        tracer.log(
            "llm_tool_call.start",
            model=self._model,
            provider="anthropic",
            tool_count=len(tools),
        )

        # Convert OpenAI-style tool format to Anthropic format
        anthropic_tools = self._convert_tools(tools)

        start = time.perf_counter()
        response = await self._client.messages.create(
            model=self._model,
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            tools=anthropic_tools,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        elapsed = time.perf_counter() - start

        content = ""
        tool_calls: List[Dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "arguments": block.input,
                })

        usage = response.usage
        metadata: Dict[str, Any] = {
            "model": response.model,
            "latency_ms": elapsed * 1000,
            "prompt_tokens": usage.input_tokens if usage else 0,
            "completion_tokens": usage.output_tokens if usage else 0,
            "total_tokens": (usage.input_tokens + usage.output_tokens) if usage else 0,
            "tool_calls_count": len(tool_calls),
        }

        tracer.log("llm_tool_call.end", **metadata)
        return content, tool_calls, metadata

    def _convert_tools(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        converted = []
        for tool in openai_tools:
            if "function" in tool:
                func = tool["function"]
                converted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            else:
                converted.append(tool)
        return converted
