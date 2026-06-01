from __future__ import annotations

import time

from hat.abstract.schemas import ReplayBatch, SWSObjective, SWSStats
from hat.abstract.sws import SWSTrainer


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
