"""
Dataset loader: reads tasks from JSON files on disk.

Format: a JSON array of task objects matching the Task schema.
Documents referenced in tasks are stored alongside the task file
as a documents/ subdirectory or inline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.core.interfaces import Dataset
from src.core.models import Document, Task
from src.core.registry import register
from src.utils.logging import get_logger

logger = get_logger()


@register("dataset", "json")
class JSONDataset(Dataset):
    """Loads evaluation tasks from a JSON file."""

    def __init__(
        self,
        tasks_path: str | Path,
        documents: List[Document] | None = None,
    ) -> None:
        self._tasks_path = Path(tasks_path)
        self._documents = documents or []
        self._tasks: List[Task] = []
        self._metadata: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        with open(self._tasks_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            self._metadata = raw.get("metadata", {})
            items = raw.get("tasks", raw.get("data", []))
        else:
            items = raw

        for item in items:
            docs = []
            if item.get("ground_truth_docs"):
                docs = [Document(**d) for d in item["ground_truth_docs"]]
            elif item.get("relevant_doc_ids") and self._documents:
                doc_map = {d.id: d for d in self._documents}
                docs = [doc_map[did] for did in item["relevant_doc_ids"] if did in doc_map]

            task = Task(
                id=item["id"],
                type=item.get("type", "rag"),
                query=item["query"],
                ground_truth=item["ground_truth"],
                ground_truth_docs=docs,
                domain=item.get("domain", "general"),
                difficulty=item.get("difficulty", "medium"),
                metadata=item.get("metadata", {}),
            )
            self._tasks.append(task)

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
