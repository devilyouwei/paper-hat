from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..schemas import MemoryTrace, ReplayExample


class ReplayBuilder(ABC):
    """Convert a retained trace into one or more training examples (paper §3.4.3)."""

    @abstractmethod
    def __call__(self, trace: MemoryTrace) -> Iterable[ReplayExample]: ...


class SupervisedReplayBuilder(ReplayBuilder):
    """Default: emit a single ``(query, target_response)`` pair per trace."""

    def __call__(self, trace: MemoryTrace) -> Iterable[ReplayExample]:
        target = trace.target_response or trace.cortex_response
        if not target:
            return ()
        return (
            ReplayExample(
                input=trace.query,
                target=target,
                source_trace_id=trace.id,
                is_oracle="oracle" in trace.metadata.source,
            ),
        )
