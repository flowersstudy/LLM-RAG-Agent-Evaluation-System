"""
OpenAI adapter implementing the LLMAdapter interface.

Supports: GPT-4, GPT-4o, GPT-3.5-turbo, and OpenAI-compatible endpoints
(DeepSeek, Kimi/Moonshot, Zhipu/GLM, Qwen/DashScope, etc.).

Features built-in retry with exponential backoff for rate limits and transient errors.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, RateLimitError

from src.core.interfaces import LLMAdapter
from src.core.registry import register
from src.utils.logging import get_logger, get_tracer

logger = get_logger()

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BASE_DELAY = 1.0  # seconds
_MAX_DELAY = 30.0  # seconds


@register("llm_adapter", "openai")
class OpenAIAdapter(LLMAdapter):
    """LLMAdapter for OpenAI and OpenAI-compatible models.

    Parameters
    ----------
    model : str
        Model name (e.g., 'gpt-4o', 'deepseek-chat', 'glm-4-flash').
    api_key : str, optional
        API key. Falls back to env vars.
    base_url : str, optional
        Base URL for API. Use for non-OpenAI providers.
    temperature : float
        Sampling temperature (default 0.0 for deterministic eval).
    max_tokens : int
        Max tokens in response.
    max_retries : int
        Number of retries on rate limit / transient errors.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
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

        async def _call():
            return await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
            )

        response, retries = await self._retry(_call, f"generate")

        content = response.choices[0].message.content or ""
        usage = response.usage

        metadata: Dict[str, Any] = {
            "model": response.model,
            "latency_ms": 0,  # filled below
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "finish_reason": response.choices[0].finish_reason,
            "retries": retries,
        }

        tracer.log(
            "llm_call.end",
            model=self._model,
            tokens=metadata["total_tokens"],
            retries=retries,
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

        async def _call():
            return await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
            )

        response, retries = await self._retry(_call, f"generate_with_tools")

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
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "tool_calls_count": len(tool_calls),
            "retries": retries,
        }

        tracer.log("llm_tool_call.end", **metadata)
        return content, tool_calls, metadata

    async def _retry(self, fn, label: str = "api_call"):
        """Execute with exponential backoff retry for rate limits and transient errors."""
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                result = await fn()
                if attempt > 0:
                    logger.info(f"[{self._model}] {label} succeeded after {attempt} retries")
                return result, attempt
            except RateLimitError as e:
                last_exc = e
                if attempt >= self._max_retries:
                    break
                delay = min(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), _MAX_DELAY)
                logger.warning(
                    f"[{self._model}] Rate limit on {label}, "
                    f"retry {attempt + 1}/{self._max_retries} in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in _RETRYABLE_STATUSES and attempt < self._max_retries:
                    last_exc = e
                    delay = min(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), _MAX_DELAY)
                    logger.warning(
                        f"[{self._model}] HTTP {status} on {label}, "
                        f"retry {attempt + 1}/{self._max_retries} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        logger.error(f"[{self._model}] {label} failed after {self._max_retries} retries")
        raise last_exc
