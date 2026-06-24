"""
Dataset loader: reads tasks from JSON files on disk.

Format: a JSON object with "tasks" array + "metadata" dict, or a flat JSON array.
Each task can reference documents by ID (relevant_doc_ids) or inline them (ground_truth_docs).
When IDs are used, a corpus_path or documents list must be provided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.core.interfaces import Dataset
from src.core.models import Document, Task
from src.core.registry import register
from src.utils.logging import get_logger

logger = get_logger()


@register("dataset", "json")
class JSONDataset(Dataset):
    """Loads evaluation tasks from a JSON file.

    Tasks can reference ground-truth documents in two ways:
    1. Inline: each task has a "ground_truth_docs" field with full Document objects
    2. By ID: each task has "relevant_doc_ids" referencing a corpus file at corpus_path
    """

    def __init__(
        self,
        tasks_path: str | Path,
        corpus_path: Optional[str | Path] = None,
        documents: Optional[List[Document]] = None,
    ) -> None:
        self._tasks_path = Path(tasks_path)
        self._documents = documents or []
        # Auto-load corpus if provided
        if corpus_path:
            self._documents = self._load_corpus(Path(corpus_path))
        self._tasks: List[Task] = []
        self._metadata: Dict[str, Any] = {}
        self._load()

    def _load_corpus(self, path: Path) -> List[Document]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        docs = [Document(**d) for d in raw]
        logger.info(f"Loaded {len(docs)} documents from corpus at {path}")
        return docs

    def _load(self) -> None:
        with open(self._tasks_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            self._metadata = raw.get("metadata", {})
            items = raw.get("tasks", raw.get("data", []))
        else:
            items = raw

        doc_map = {d.id: d for d in self._documents}
        skipped_ids = 0

        for item in items:
            docs: List[Document] = []
            if item.get("ground_truth_docs"):
                docs = [Document(**d) for d in item["ground_truth_docs"]]
            elif item.get("relevant_doc_ids"):
                for did in item["relevant_doc_ids"]:
                    if did in doc_map:
                        docs.append(doc_map[did])
                    else:
                        skipped_ids += 1
                        logger.warning(f"Document '{did}' referenced by task '{item['id']}' not found in corpus")

            task = Task(
                id=item["id"],
                type=item.get("type", "rag"),
                query=item["query"],
                ground_truth=item["ground_truth"],
                ground_truth_docs=docs,
                tools=item.get("tools", []),
                expected_tool_sequence=item.get("expected_tool_sequence"),
                domain=item.get("domain", "general"),
                difficulty=item.get("difficulty", "medium"),
                metadata=item.get("metadata", {}),
            )
            self._tasks.append(task)

        if skipped_ids:
            logger.warning(f"{skipped_ids} document IDs could not be resolved")
        logger.info(f"Loaded {len(self._tasks)} tasks from {self._tasks_path}")

    def __iter__(self) -> Iterator[Task]:
        yield from self._tasks

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "source": str(self._tasks_path),
            "task_count": len(self._tasks),
            "domains": list(set(t.domain for t in self._tasks)),
            **self._metadata,
        }
