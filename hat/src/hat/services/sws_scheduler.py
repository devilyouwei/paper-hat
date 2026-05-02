"""Triggers SWS cycles based on interaction count (paper §3.8)."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.loop import WakeSleepLoop
from ..core.schemas import SWSObjective, SWSStats


@dataclass
class SWSScheduler:
    loop: WakeSleepLoop
    interval: int = 500  # paper default N
    _seen: int = 0
    _cycle: int = 0

    def on_interaction(self) -> SWSStats | None:
        self._seen += 1
        if self._seen >= self.interval:
            self._seen = 0
            self._cycle += 1
            return self.loop.sleep_step(cycle=self._cycle, objective=SWSObjective())
        return None
