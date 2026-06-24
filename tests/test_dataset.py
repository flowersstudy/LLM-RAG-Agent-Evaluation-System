"""Tests for dataset loading."""

import json
import tempfile
from pathlib import Path

from src.dataset.loader import JSONDataset


def test_load_json_dataset():
    data = {
        "tasks": [
            {
                "id": "t1",
                "type": "rag",
                "query": "What is AI?",
                "ground_truth": "AI is artificial intelligence",
                "ground_truth_docs": [
                    {"id": "d1", "content": "AI refers to artificial intelligence.", "metadata": {}}
                ],
                "domain": "ai",
                "difficulty": "easy",
            },
            {
                "id": "t2",
                "type": "rag",
                "query": "What is ML?",
                "ground_truth": "ML is machine learning",
                "ground_truth_docs": [
                    {"id": "d2", "content": "ML refers to machine learning.", "metadata": {}}
                ],
                "domain": "ai",
                "difficulty": "easy",
            },
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name

    try:
        dataset = JSONDataset(tmp_path)
        assert len(dataset) == 2
        assert dataset.metadata["task_count"] == 2
        tasks = list(dataset)
        assert tasks[0].id == "t1"
        assert tasks[1].domain == "ai"
    finally:
        Path(tmp_path).unlink()
