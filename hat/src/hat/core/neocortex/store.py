from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from ..schemas import MemoryTrace, WriteDecision


class NeocortexWriteError(RuntimeError):
    """Raised when a caller tries to write without a valid ``WriteDecision``."""


class NeocortexStore(ABC):
    """Long-term curated memory (paper §3.6).

    Only the Hippocampus Agent may write here. The interface enforces this by
    requiring an *accepted* :class:`WriteDecision` whose ``trace_id`` matches
    the trace being written. This is the type-level boundary between raw chat
    history and training data — see ADR-002.
    """

    def write(self, trace: MemoryTrace, decision: WriteDecision) -> None:
        if decision is None or decision.trace_id != trace.id:
            raise NeocortexWriteError("WriteDecision missing or trace_id mismatch")
        if not decision.accepted:
            raise NeocortexWriteError("WriteDecision was not accepted")
        self._persist(trace, decision)

    @abstractmethod
    def _persist(self, trace: MemoryTrace, decision: WriteDecision) -> None: ...

    @abstractmethod
    def __iter__(self) -> Iterator[MemoryTrace]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def sample(self, k: int) -> Iterable[MemoryTrace]:
        """Priority sample by score (paper Algorithm)."""


class InMemoryNeocortex(NeocortexStore):
    """Reference store backed by Python lists. For tests and the smoke path."""

    def __init__(self) -> None:
        self._traces: list[MemoryTrace] = []
        self._scores: list[float] = []

    def _persist(self, trace: MemoryTrace, decision: WriteDecision) -> None:
        self._traces.append(trace)
        self._scores.append(decision.score)

    def __iter__(self) -> Iterator[MemoryTrace]:
        return iter(self._traces)

    def __len__(self) -> int:
        return len(self._traces)

    def sample(self, k: int) -> Iterable[MemoryTrace]:
        if not self._traces:
            return []
        idxs = heapq.nlargest(
            min(k, len(self._traces)),
            range(len(self._traces)),
            key=lambda i: self._scores[i],
        )
        return [self._traces[i] for i in idxs]
