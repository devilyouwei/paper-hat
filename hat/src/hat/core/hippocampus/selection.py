from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import MemoryTrace, ScoreSignals, WriteDecision


class WritePolicy(ABC):
    """Base class for selection policies.

    A write policy projects ``ScoreSignals`` to a scalar and emits a
    :class:`WriteDecision` that the Neocortex requires to accept a write.
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


class UncertaintyGatePolicy(WritePolicy):
    """Single-signal policy: ``score(m) = U(m)``.

    A trace is written only when the cortex's uncertainty on the original
    response meets ``threshold`` — i.e. the model was unsure enough that the
    turn is worth remembering. All other signals are ignored.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def score(self, trace: MemoryTrace, signals: ScoreSignals) -> float:
        return float(signals.uncertainty)
