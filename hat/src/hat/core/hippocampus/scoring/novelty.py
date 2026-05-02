from __future__ import annotations

from abc import ABC, abstractmethod

from ...schemas import MemoryTrace


class NoveltyEstimator(ABC):
    """``N(m) = 1 - max_{m' ∈ N} cos(e(m), e(m'))``  — paper Eq. ``novelty``.

    The default :class:`AlwaysNovel` is a stand-in. A real implementation pulls
    embeddings from :mod:`hat.memory.embeddings` and queries the Neocortex.
    """

    @abstractmethod
    def __call__(self, trace: MemoryTrace) -> float: ...


class AlwaysNovel(NoveltyEstimator):
    """Returns ``1.0`` so every trace passes the novelty channel during dev."""

    def __call__(self, trace: MemoryTrace) -> float:
        return 1.0
