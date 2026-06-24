"""
Structured logging for all evaluation runs.

Every intermediate step — LLM calls, retriever calls, metric computations,
failure classifications — is logged as structured JSON lines. This is
the audit trail that makes experiments debuggable and reproducible.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Standard Python logger for human-readable output
_logger = logging.getLogger("llm_eval")
_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
_logger.addHandler(_handler)


class TraceLogger:
    """
    JSON-lines logger that writes structured trace events.

    Each line is a self-contained JSON object with: timestamp, event_type,
    and event-specific fields. This format is queryable with jq, duckdb, etc.
    """

    def __init__(self, output_path: Optional[Path] = None) -> None:
        self.output_path = output_path
        self._file = None
        self._event_count = 0

    def open(self) -> None:
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.output_path, "w", encoding="utf-8")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def log(self, event_type: str, **fields: Any) -> None:
        """Write a structured event to the trace log."""
        event: Dict[str, Any] = {
            "timestamp": time.time(),
            "event_type": event_type,
            **fields,
        }
        line = json.dumps(event, ensure_ascii=False, default=str)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()
        self._event_count += 1

    @property
    def event_count(self) -> int:
        return self._event_count


# Module-level convenience functions
_trace_logger: Optional[TraceLogger] = None


def init_tracing(output_path: Optional[Path] = None) -> TraceLogger:
    """Initialize global trace logging. Call once per experiment."""
    global _trace_logger
    _trace_logger = TraceLogger(output_path)
    _trace_logger.open()
    return _trace_logger


def get_tracer() -> TraceLogger:
    """Get the current trace logger. Raises if not initialized."""
    if _trace_logger is None:
        raise RuntimeError("Trace logging not initialized. Call init_tracing() first.")
    return _trace_logger


def log_event(event_type: str, **fields: Any) -> None:
    """Convenience: log an event to the global tracer."""
    get_tracer().log(event_type, **fields)


def close_tracing() -> None:
    """Close global trace logging."""
    global _trace_logger
    if _trace_logger:
        _trace_logger.close()
        _trace_logger = None


def get_logger() -> logging.Logger:
    return _logger


@contextmanager
def latency_log(event_type: str, **context: Any):
    """Context manager that logs start, end, and duration for an operation."""
    start = time.perf_counter()
    log_event(f"{event_type}.start", **context)
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        log_event(f"{event_type}.error", duration_ms=elapsed * 1000, error=str(exc), **context)
        raise
    else:
        elapsed = time.perf_counter() - start
        log_event(f"{event_type}.end", duration_ms=elapsed * 1000, **context)
