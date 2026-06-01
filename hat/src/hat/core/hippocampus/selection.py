from __future__ import annotations

from hat.abstract.hippocampus import WritePolicy
from hat.abstract.schemas import MemoryTrace, ScoreSignals


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
