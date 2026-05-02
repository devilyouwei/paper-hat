from __future__ import annotations

from ..schemas import Interaction
from .base import Cortex


class NoopCortex(Cortex):
    """Identity-like Cortex used by tests and the bare end-to-end smoke path."""

    name = "noop"

    def generate(self, query: str, *, context: str | None = None) -> str:
        return f"[noop] {query}"

    def uncertainty(self, interaction: Interaction) -> float:
        return 0.5
