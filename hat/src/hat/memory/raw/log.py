from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from ...core.schemas import Interaction


class RawInteractionLog(ABC):
    """Append-only log of raw user interactions.

    Strictly separated from the Neocortex: only the Hippocampus Agent reads here
    to produce traces. No training pipeline touches raw logs directly.
    """

    @abstractmethod
    def append(self, interaction: Interaction) -> None: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Interaction]: ...


class JsonlRawLog(RawInteractionLog):
    """JSONL-backed reference implementation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, interaction: Interaction) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(interaction.model_dump_json() + "\n")

    def __iter__(self) -> Iterator[Interaction]:
        if not self.path.exists():
            return iter(())

        def gen() -> Iterator[Interaction]:
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield Interaction.model_validate_json(line)

        return gen()
