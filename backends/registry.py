"""Backend registry: name to constructor factory."""
from __future__ import annotations

from collections.abc import Sequence

from backends.base import Backend
from backends.classical import ClassicalBackend

_REGISTRY: dict[str, type[Backend]] = {
    "classical": ClassicalBackend,
}


def available() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(_REGISTRY)


def get_backend(name: str, corpus: Sequence[str]) -> Backend:
    """Build a backend instance by name. Raises ValueError on unknown names."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend '{name}'; available: {available()}")
    return _REGISTRY[name](corpus)
