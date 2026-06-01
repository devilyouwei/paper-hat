"""Slow-wave-sleep trainer interface (paper §3.7)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import ReplayBatch, SWSObjective, SWSStats


class SWSTrainer(ABC):
    """Performs ``θ ← θ - η ∇L_SWS`` with replay batches sampled from the
    Neocortex (paper §3.7).

    Concrete implementations live in :mod:`hat.core.sws` and own the
    parameter-efficient fine-tuning details (LoRA, Fisher-info for EWC, ...).
    """

    @abstractmethod
    def fit(self, batch: ReplayBatch, objective: SWSObjective) -> SWSStats: ...


__all__ = ["SWSTrainer"]
