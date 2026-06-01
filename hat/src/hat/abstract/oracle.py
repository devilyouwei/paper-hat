"""Oracle interface (paper §3.5).

External teacher consulted on demand when the Cortex is uncertain.
Concrete implementations live in :mod:`hat.core.oracle`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import Interaction


class Oracle(ABC):
    """External teacher consulted on demand (paper §3.5).

    The ``WakeSleepLoop`` only queries the oracle when ``U(x) > τ_O`` —
    paper Eq. ``oracle_query``.
    """

    name: str = "oracle"

    @abstractmethod
    def consult(self, interaction: Interaction) -> str: ...


__all__ = ["Oracle"]
