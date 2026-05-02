from __future__ import annotations

from abc import ABC, abstractmethod

from ...schemas import Interaction


class UncertaintyEstimator(ABC):
    """Returns ``U(x) ∈ [0, 1]`` for an interaction.

    Real implementations: predictive entropy (paper Eq. ``uncertainty``),
    token-level confidence, self-consistency variance, sampled disagreement.
    """

    @abstractmethod
    def __call__(self, interaction: Interaction) -> float: ...


class ConstantUncertainty(UncertaintyEstimator):
    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def __call__(self, interaction: Interaction) -> float:
        return self.value
