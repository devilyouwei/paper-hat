"""Neocortex interfaces (paper §3.6).

Long-term curated memory: stores accepted :class:`MemoryTrace`s and
provides the vector index used for dedup-based CREATE/REVISE routing.
Concrete implementations live in :mod:`hat.core.neocortex`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .schemas import MemoryTrace, WriteDecision


class NeocortexWriteError(RuntimeError):
    """Raised when a caller tries to write without a valid ``WriteDecision``."""


class NeocortexStore(ABC):
    """Long-term curated memory (paper §3.6).

    Only the Hippocampus Agent may write here. The interface enforces
    this by requiring an *accepted* :class:`WriteDecision` whose
    ``trace_id`` matches the trace being written. This is the type-level
    boundary between raw chat history and training data — see ADR-002.
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

    # -- session-aware extensions (optional for backends to override) --

    def entries_by_session(self, session_id: str) -> list[MemoryTrace]:
        """Return traces previously written for a given session.

        Default implementation walks ``__iter__`` and filters by
        ``trace.session_id``. Backends with indexes should override.
        """
        return [t for t in self if getattr(t, "session_id", None) == session_id]

    def revise(
        self,
        trace_id: str,
        *,
        query: str | None = None,
        target_response: str | None = None,
        rationale: str | None = None,
        append_interaction_id: str | None = None,
        push_history_entry: dict | None = None,
    ) -> MemoryTrace | None:
        """Mutate an existing trace in place (REVISE path).

        Backends that do not support in-place edits should override and raise.
        """
        raise NotImplementedError("revise() not supported by this backend")


# ---- vector index --------------------------------------------------------


@dataclass(slots=True)
class Match:
    trace_id: str
    similarity: float


class VectorIndex(ABC):
    """``(trace_id, vec)`` table used for dedup nearest-neighbour lookup."""

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def has(self, trace_id: str) -> bool: ...

    @abstractmethod
    def append(self, trace_id: str, vec: Sequence[float]) -> None: ...

    @abstractmethod
    def update(self, trace_id: str, vec: Sequence[float]) -> bool: ...

    @abstractmethod
    def remove(self, trace_id: str) -> bool: ...

    @abstractmethod
    def top1(
        self, vec: Sequence[float], *, exclude: str | None = None
    ) -> Match | None: ...


# ---- embedder ------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to fixed-dim float vectors. Vectors must be L2-normalised."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...


__all__ = [
    "Embedder",
    "Match",
    "NeocortexStore",
    "NeocortexWriteError",
    "VectorIndex",
]
