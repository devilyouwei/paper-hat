from __future__ import annotations

from collections.abc import Iterable

from hat.abstract.hippocampus import ReplayBuilder
from hat.abstract.schemas import MemoryTrace, ReplayExample


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
