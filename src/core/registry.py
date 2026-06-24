"""
Plugin registry for discoverable components.

Each component type (LLM adapter, metric, pipeline) has its own namespace.
Components register themselves via decorator or explicit call.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type, TypeVar

T = TypeVar("T")

_registry: Dict[str, Dict[str, Any]] = {
    "llm_adapter": {},
    "metric": {},
    "pipeline": {},
    "dataset": {},
    "judge": {},
    "failure_classifier": {},
}


def register(namespace: str, name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator: register a class under a namespace and name."""

    def decorator(cls: Type[T]) -> Type[T]:
        _registry.setdefault(namespace, {})[name] = cls
        return cls

    return decorator


def get_component(namespace: str, name: str) -> Any:
    """Retrieve a registered component by namespace and name."""
    if namespace not in _registry:
        raise KeyError(f"Unknown namespace: {namespace}")
    if name not in _registry[namespace]:
        available = list(_registry[namespace].keys())
        raise KeyError(f"No component '{name}' in namespace '{namespace}'. Available: {available}")
    return _registry[namespace][name]


def list_components(namespace: str) -> Dict[str, Any]:
    """List all registered components in a namespace."""
    return dict(_registry.get(namespace, {}))


def list_namespaces() -> list[str]:
    """List all registered namespaces."""
    return list(_registry.keys())
