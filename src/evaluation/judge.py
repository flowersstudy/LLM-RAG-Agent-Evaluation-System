"""
LLM-as-Judge: uses an LLM to evaluate outputs against structured rubrics.

Key design choice: every rubric requires (1) a numerical score, (2) a rationale,
and (3) evidence quotes from the source material. This constrains the judge
and produces auditable results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.interfaces import LLMAdapter
from src.core.registry import register
from src.utils.logging import get_logger, get_tracer

logger = get_logger()

JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge. Your task is to score an AI system's output
against a specific rubric. You must be objective, consistent, and evidence-based.

Always respond in JSON format:
{
  "score": <float between 0 and 1>,
  "rationale": "<brief explanation of why this score was assigned>",
  "evidence": ["<quote from source material supporting your judgment>"]
}"""


@register("judge", "structured")
class StructuredJudge:
    """LLM-based judge that evaluates against structured rubrics."""

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    async def judge(
        self,
        rubric: str,
        query: str,
        answer: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply a rubric to an answer.

        Args:
            rubric: The scoring rubric (criteria, score descriptions).
            query: The original user query.
            answer: The system's answer to evaluate.
            context: Retrieved context (for faithfulness checks).
            ground_truth: Reference answer (for relevance checks).

        Returns:
            Dict with score, rationale, evidence.
        """
        prompt_parts = [
            f"## Rubric\n{rubric}",
            f"## Query\n{query}",
            f"## Answer to Evaluate\n{answer}",
        ]
        if context:
            prompt_parts.append(f"## Context\n{context}")
        if ground_truth:
            prompt_parts.append(f"## Ground Truth\n{ground_truth}")

        prompt = "\n\n".join(prompt_parts)

        tracer = get_tracer()
        tracer.log("judge.start", rubric=rubric[:100])

        response_text, metadata = await self._llm.generate(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )

        result = self._parse_response(response_text)
        tracer.log("judge.end", score=result.get("score"))
        return result

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from judge response, with fallback for malformed output."""
        import json
        import re

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: return a default with the raw text as rationale
        logger.warning(f"Judge response not valid JSON, using fallback. Raw: {text[:200]}")
        return {
            "score": 0.0,
            "rationale": f"Failed to parse judge output: {text[:300]}",
            "evidence": [],
        }
