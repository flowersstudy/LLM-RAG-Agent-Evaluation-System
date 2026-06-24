"""
Entry point for running experiments.

Usage:
    python -m src.pipeline.run --config configs/rag_eval_example.yaml
    python -m src.pipeline.run --config configs/rag_eval_example.yaml --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.config import parse_config
from src.pipeline.experiment import ExperimentRunner


async def main(config_path: str, dry_run: bool = False) -> None:
    config = parse_config(config_path)
    print(f"Experiment: {config.experiment_id}")
    print(f"Models: {config.models}")
    print(f"Metrics: {config.metrics}")
    print(f"Dataset: {config.dataset_path}")

    if dry_run:
        print("\n[DRY RUN] Config validated. Skipping execution.")
        return

    runner = ExperimentRunner()
    output_dir = await runner.run(config)
    print(f"\nExperiment complete. Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an LLM evaluation experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML/JSON")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    args = parser.parse_args()
    asyncio.run(main(args.config, args.dry_run))
