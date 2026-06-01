"""Backend factory registry.

Backends register themselves under a string name; consumers ask for a configured
:class:`hat.core.protocols.LanguageModel` via :func:`create`. Keeps the API and
loop oblivious to which inference engine is in use."""

from __future__ import annotations

from collections.abc import Callable

from hat.abstract import LanguageModel

_REGISTRY: dict[str, Callable[..., LanguageModel]] = {}


def register(name: str) -> Callable[[Callable[..., LanguageModel]], Callable[..., LanguageModel]]:
    def deco(factory: Callable[..., LanguageModel]) -> Callable[..., LanguageModel]:
        _REGISTRY[name] = factory
        return factory

    return deco


def create(name: str, **kwargs) -> LanguageModel:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
