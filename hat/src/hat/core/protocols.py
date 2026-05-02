"""Pluggable seams as ``typing.Protocol`` interfaces.

Anything heavy (an HF model, a vector store, a trainer) is named here and
imported by reference. Concrete implementations live in
``hat.models``, ``hat.memory``, and friends.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from .schemas import (
    Interaction,
    MemoryTrace,
    ReplayBatch,
    ReplayExample,
    ScoreSignals,
    SWSObjective,
    SWSStats,
    WriteDecision,
)


# ---------- model backends ------------------------------------------------


@runtime_checkable
class LanguageModel(Protocol):
    name: str

    def generate(self, prompt: str, *, context: str | None = None, **kwargs) -> str: ...
    def token_logprobs(self, prompt: str, response: str) -> list[float]: ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# ---------- hippocampus pipeline -----------------------------------------


@runtime_checkable
class Abstractor(Protocol):
    def __call__(self, interaction: Interaction) -> MemoryTrace: ...


@runtime_checkable
class UncertaintyEstimator(Protocol):
    def __call__(self, interaction: Interaction) -> float: ...


@runtime_checkable
class FeedbackExtractor(Protocol):
    def __call__(self, interaction: Interaction) -> float: ...


@runtime_checkable
class NoveltyEstimator(Protocol):
    def __call__(self, trace: MemoryTrace) -> float: ...


@runtime_checkable
class WritePolicy(Protocol):
    threshold: float

    def score(self, trace: MemoryTrace, signals: ScoreSignals) -> float: ...
    def decide(self, trace: MemoryTrace, signals: ScoreSignals) -> WriteDecision: ...


@runtime_checkable
class ReplayBuilder(Protocol):
    def __call__(self, trace: MemoryTrace) -> Iterable[ReplayExample]: ...


# ---------- oracle / trainer ---------------------------------------------


@runtime_checkable
class OracleClient(Protocol):
    name: str

    def consult(self, interaction: Interaction) -> str: ...


@runtime_checkable
class Trainer(Protocol):
    def fit(self, batch: ReplayBatch, objective: SWSObjective) -> SWSStats: ...


__all__ = [
    "LanguageModel",
    "Embedder",
    "Abstractor",
    "UncertaintyEstimator",
    "FeedbackExtractor",
    "NoveltyEstimator",
    "WritePolicy",
    "ReplayBuilder",
    "OracleClient",
    "Trainer",
]
