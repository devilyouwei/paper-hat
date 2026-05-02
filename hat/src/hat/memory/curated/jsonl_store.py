from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ...core.neocortex.store import NeocortexStore
from ...core.schemas import MemoryTrace, WriteDecision


class JsonlNeocortex(NeocortexStore):
    """Reference Neocortex backed by a single JSONL file. Dev / tests only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _persist(self, trace: MemoryTrace, decision: WriteDecision) -> None:
        record = {"trace": trace.model_dump(mode="json"), "score": decision.score}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _load(self) -> list[tuple[MemoryTrace, float]]:
        if not self.path.exists():
            return []
        out: list[tuple[MemoryTrace, float]] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                out.append((MemoryTrace.model_validate(rec["trace"]), float(rec["score"])))
        return out

    def __iter__(self) -> Iterator[MemoryTrace]:
        for tr, _ in self._load():
            yield tr

    def __len__(self) -> int:
        return len(self._load())

    def sample(self, k: int) -> Iterable[MemoryTrace]:
        rows = self._load()
        rows.sort(key=lambda x: x[1], reverse=True)
        return [tr for tr, _ in rows[:k]]
