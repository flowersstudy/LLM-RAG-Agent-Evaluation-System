"""
Experiment configuration: YAML-based config parsing and validation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.core.models import ExperimentConfig
from src.utils.logging import get_logger

logger = get_logger()


def parse_config(path: str | Path) -> ExperimentConfig:
    """Parse an experiment configuration from YAML or JSON."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    return ExperimentConfig(
        experiment_id=data.get(
            "experiment_id",
            f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        description=data.get("description", ""),
        models=data["models"],
        metrics=data["metrics"],
        dataset_path=data["dataset_path"],
        split=data.get("split"),
        random_seed=data.get("random_seed", 42),
        llm_judge_model=data.get("llm_judge_model", "gpt-4o"),
        max_tasks=data.get("max_tasks"),
        metadata=data.get("metadata", {}),
    )
