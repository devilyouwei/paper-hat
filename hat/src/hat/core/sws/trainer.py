from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..schemas import ReplayBatch, SWSObjective, SWSStats


class SWSTrainer(ABC):
    """Performs ``θ ← θ - η ∇L_SWS`` with replay batches sampled from the
    Neocortex (paper §3.7).

    Concrete implementations live in ``hat.models.training`` and own the
    parameter-efficient fine-tuning details (LoRA, Fisher-info for EWC, …).
    """

    @abstractmethod
    def fit(self, batch: ReplayBatch, objective: SWSObjective) -> SWSStats: ...


class DryRunTrainer(SWSTrainer):
    """No-op trainer used by ``hat sleep --dry-run`` and plumbing tests."""

    def fit(self, batch: ReplayBatch, objective: SWSObjective) -> SWSStats:
        t0 = time.perf_counter()
        return SWSStats(
            cycle=batch.cycle,
            n_replayed=len(batch.examples),
            loss_sup=0.0,
            duration_seconds=time.perf_counter() - t0,
        )
