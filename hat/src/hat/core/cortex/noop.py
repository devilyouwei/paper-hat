from __future__ import annotations

from collections.abc import Sequence

from hat.abstract.schemas import Interaction
from hat.abstract.cortex import Cortex


class NoopCortex(Cortex):
    """Identity-like Cortex used by tests and the bare end-to-end smoke path."""

    name = "noop"

    def generate(self, query: str, *, context: str | None = None) -> str:
        return f"[noop] {query}"

    def chat(self, messages: Sequence[dict[str, str]], **_kwargs) -> str:
        last = next(
            (m for m in reversed(list(messages)) if m.get("role") == "user"),
            None,
        )
        return self.generate(last.get("content", "") if last else "")

    def stream_chat(self, messages: Sequence[dict[str, str]], **kwargs):
        text = self.chat(messages, **kwargs)
        # emit roughly one word per chunk so the UI still gets to render
        # incremental updates against the noop backend.
        for tok in text.split(" "):
            yield tok + " "

    def uncertainty(self, interaction: Interaction) -> float:
        return 0.5
