"""Pluggable similarity backends."""
from backends.base import Backend, BackendMatchResult
from backends.registry import available, get_backend

__all__ = ["Backend", "BackendMatchResult", "available", "get_backend"]
