from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Interaction, MemoryTrace, TraceMetadata


class Abstractor(ABC):
    """Maps a raw :class:`Interaction` to a compact :class:`MemoryTrace`.

    Mirrors paper Eq. ``abstraction``: ``m = H_abs(c, x, y, f)``. Real
    implementations call a small summarization model or prompt the Cortex itself
    under an instruction template.
    """

    @abstractmethod
    def __call__(self, interaction: Interaction) -> MemoryTrace: ...


class IdentityAbstractor(Abstractor):
    """Default: copy fields verbatim. Useful for tests; replace in production."""

    def __call__(self, interaction: Interaction) -> MemoryTrace:
        return MemoryTrace(
            interaction_id=interaction.id,
            query=interaction.query,
            cortex_response=interaction.response,
            target_response=interaction.user_correction or interaction.response,
            rationale=None,
            metadata=TraceMetadata(source=interaction.source),
        )
