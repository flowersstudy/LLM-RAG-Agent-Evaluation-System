"""
Synthetic QA dataset generation from a document corpus.

Given a set of documents, generates (query, answer, relevant_docs) triples
using an LLM. This enables creating evaluation datasets from any text corpus
without manual annotation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.core.interfaces import LLMAdapter
from src.core.models import Document, Task
from src.utils.logging import get_logger, get_tracer

logger = get_logger()

QA_GENERATION_PROMPT = """\
You are generating evaluation data for a RAG system. Given a document,
create a question-answer pair that tests retrieval and reasoning.

Rules:
- The answer MUST be derivable from the document content.
- The question should be specific enough that the document is clearly relevant.
- Include questions of varying difficulty: simple lookup, multi-hop, inferential.
- Do NOT copy phrases verbatim from the document — rephrase.

Document:
{document}

Generate a JSON object:
{{
  "query": "<question>",
  "answer": "<ground truth answer>",
  "difficulty": "easy|medium|hard"
}}"""


class SyntheticQAGenerator:
    """Generates QA pairs from a document corpus using an LLM."""

    def __init__(
        self,
        llm: LLMAdapter,
        seed: int = 42,
    ) -> None:
        self._llm = llm
        random.seed(seed)

    async def generate(
        self,
        documents: List[Document],
        num_tasks: int = 50,
        output_path: Path | None = None,
    ) -> List[Task]:
        """
        Generate QA tasks from a document pool.

        Documents are sampled, then an LLM generates questions for each.
        """
        tracer = get_tracer()
        tracer.log("synthetic_qa.start", num_docs=len(documents), num_tasks=num_tasks)

        tasks: List[Task] = []
        docs_pool = documents.copy()

        for i in range(num_tasks):
            doc = random.choice(docs_pool)
            prompt = QA_GENERATION_PROMPT.format(document=doc.content[:3000])

            response, _ = await self._llm.generate(prompt)

            try:
                parsed = self._parse_response(response)
            except Exception:
                logger.warning(f"Failed to parse QA pair {i}, skipping. Response: {response[:200]}")
                continue

            task = Task(
                id=f"synthetic_{i:04d}",
                type="rag",
                query=parsed["query"],
                ground_truth=parsed["answer"],
                ground_truth_docs=[doc],
                domain=doc.metadata.get("domain", "general"),
                difficulty=parsed.get("difficulty", "medium"),
                metadata={"source": "synthetic", "source_doc_id": doc.id},
            )
            tasks.append(task)

        tracer.log("synthetic_qa.end", generated=len(tasks))

        if output_path:
            self._save(tasks, output_path)

        return tasks

    def _parse_response(self, text: str) -> Dict[str, Any]:
        import re
        # Extract JSON block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)

    def _save(self, tasks: List[Task], path: Path) -> None:
        data = {
            "metadata": {
                "source": "synthetic_qa_generation",
                "task_count": len(tasks),
            },
            "tasks": [
                {
                    "id": t.id,
                    "type": t.type.value,
                    "query": t.query,
                    "ground_truth": t.ground_truth,
                    "ground_truth_docs": [
                        {"id": d.id, "content": d.content, "metadata": d.metadata}
                        for d in t.ground_truth_docs
                    ],
                    "domain": t.domain,
                    "difficulty": t.difficulty,
                    "metadata": t.metadata,
                }
                for t in tasks
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(tasks)} synthetic tasks to {path}")
