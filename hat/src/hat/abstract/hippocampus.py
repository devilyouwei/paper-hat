"""Hippocampus interfaces (paper §3.4).

The hippocampus is responsible for selective memory consolidation:
abstraction → scoring → selection → replay, with optional dedup-based
CREATE/REVISE routing. Concrete implementations live in
:mod:`hat.core.hippocampus`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .schemas import Interaction, MemoryTrace, ReplayExample, ScoreSignals, WriteDecision


class Abstractor(ABC):
    """Maps a raw :class:`Interaction` to zero or more :class:`MemoryTrace`s.

    Mirrors paper Eq. ``abstraction``: ``m = H_abs(c, x, y, f)``. A turn
    can yield multiple traces when the user packs several independent
    knowledge points into a single utterance.

    An empty list signals DROP (no knowledge point worth storing).
    """

    @abstractmethod
    def __call__(self, interaction: Interaction) -> list[MemoryTrace]: ...


class UncertaintyEstimator(ABC):
    """Returns ``U(x) ∈ [0, 1]`` for an interaction (paper §3.4.2)."""

    @abstractmethod
    def __call__(self, interaction: Interaction) -> float: ...


class WritePolicy(ABC):
    """Projects ``ScoreSignals`` to a scalar and emits a :class:`WriteDecision`.

    The Neocortex requires the resulting decision to accept a write.
    """

    @property
    @abstractmethod
    def threshold(self) -> float: ...

    @abstractmethod
    def score(self, trace: MemoryTrace, signals: ScoreSignals) -> float: ...

    def decide(self, trace: MemoryTrace, signals: ScoreSignals) -> WriteDecision:
        s = self.score(trace, signals)
        return WriteDecision(
            trace_id=trace.id,
            score=s,
            threshold=self.threshold,
            signals=signals,
            accepted=s >= self.threshold,
        )


class ReplayBuilder(ABC):
    """Convert a retained trace into one or more training examples (paper §3.4.3)."""

    @abstractmethod
    def __call__(self, trace: MemoryTrace) -> Iterable[ReplayExample]: ...


# ---- dedup ---------------------------------------------------------------

DedupDecision = Literal["create", "revise"]


@dataclass(frozen=True)
class DedupResult:
    """Outcome of a single dedup routing call."""

    decision: DedupDecision
    matched_trace_id: str | None
    similarity: float


class Deduper(ABC):
    """Geometric router: decide CREATE vs REVISE for a candidate trace.

    Concrete implementations look up nearest neighbours in a vector
    index (see :class:`hat.abstract.neocortex.VectorIndex`) and route
    above a similarity threshold. The decision, the matched id, and any
    cached embedding are stamped onto ``trace.metadata.extras`` so the
    loop can hand them to the index without re-embedding.
    """

    threshold: float

    @abstractmethod
    def route(self, trace: MemoryTrace) -> DedupResult: ...


__all__ = [
    "Abstractor",
    "DedupDecision",
    "DedupResult",
    "Deduper",
    "ReplayBuilder",
    "UncertaintyEstimator",
    "WritePolicy",
]
