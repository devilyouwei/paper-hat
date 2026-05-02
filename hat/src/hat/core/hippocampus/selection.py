from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import MemoryTrace, ScoreSignals, WriteDecision


class WritePolicy(ABC):
    """Base class for selection policies (paper §3.4.2).

    A write policy fuses ``ScoreSignals`` into a scalar and emits a
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
            accepted=s > self.threshold,
        )


class LinearWritePolicy(WritePolicy):
    """``score(m) = αU(m) + βF(m) + γN(m)`` — paper Eq. ``selection_score``."""

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        threshold: float = 0.3,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def score(self, trace: MemoryTrace, signals: ScoreSignals) -> float:
        return (
            self.alpha * signals.uncertainty
            + self.beta * signals.feedback
            + self.gamma * signals.novelty
        )
