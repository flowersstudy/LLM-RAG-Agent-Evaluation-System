"""
Reproducibility utilities.

Every source of randomness in the system must be seeded and logged.
This module provides a single entry point for seed management and
a context manager that snapshots random state so experiments can
be replayed exactly.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np

# Sentinel for "no seed was set" — distinct from seed=0 which is valid
_UNSET = object()
_global_seed: int | None = None
_seed_sources: Dict[str, int] = {}


def set_seed(seed: int, source: str = "manual") -> None:
    """Set the global random seed and record the source."""
    global _global_seed
    _global_seed = seed
    _seed_sources[source] = seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_seed() -> int | None:
    """Return the current global seed, or None if not set."""
    return _global_seed


def seed_info() -> Dict[str, Any]:
    """Return all seed information for logging."""
    return {
        "global_seed": _global_seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "sources": dict(_seed_sources),
    }


@contextmanager
def seeded(seed: int):
    """
    Context manager that temporarily sets a seed and restores previous state.

    Usage:
        with seeded(42):
            # RNG operations here use seed 42
            ...
    """
    import random as _random
    import numpy as _np

    prev_random = _random.getstate()
    prev_numpy = _np.random.get_state()
    prev_hashseed = os.environ.get("PYTHONHASHSEED")

    try:
        _random.seed(seed)
        _np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        yield
    finally:
        _random.setstate(prev_random)
        _np.random.set_state(prev_numpy)
        if prev_hashseed is not None:
            os.environ["PYTHONHASHSEED"] = prev_hashseed
        else:
            os.environ.pop("PYTHONHASHSEED", None)
