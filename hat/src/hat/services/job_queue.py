"""In-process job queue placeholder. Swap for Redis/RQ when needed."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Any


class InProcQueue:
    def __init__(self) -> None:
        self._q: deque[Any] = deque()

    def push(self, job: Any) -> None:
        self._q.append(job)

    def pop(self) -> Any | None:
        return self._q.popleft() if self._q else None

    def __iter__(self) -> Iterator[Any]:
        while self._q:
            yield self._q.popleft()

    def __len__(self) -> int:
        return len(self._q)
