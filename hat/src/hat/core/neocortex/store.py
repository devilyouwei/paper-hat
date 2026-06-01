from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator

from hat.abstract.neocortex import NeocortexStore, NeocortexWriteError
from hat.abstract.schemas import MemoryTrace, WriteDecision


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
        for i, tr in enumerate(self._traces):
            if tr.id != trace_id:
                continue
            data = tr.model_dump()
            if query is not None:
                data["query"] = query
            if target_response is not None:
                data["target_response"] = target_response
            if rationale is not None:
                data["rationale"] = rationale
            if append_interaction_id and append_interaction_id not in data.get(
                "interaction_ids", []
            ):
                data.setdefault("interaction_ids", []).append(append_interaction_id)
            if push_history_entry is not None:
                extras = data["metadata"].setdefault("extras", {})
                history = extras.setdefault("history", [])
                history.append(push_history_entry)
            self._traces[i] = MemoryTrace.model_validate(data)
            return self._traces[i]
        return None
