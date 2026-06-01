"""No-op oracle — returns the cortex response unchanged. Used for tests."""

from __future__ import annotations

from hat.abstract.oracle import Oracle
from hat.abstract.schemas import Interaction


class NoopOracle(Oracle):
    name = "noop-oracle"

    def consult(self, interaction: Interaction) -> str:
        return interaction.response or ""


__all__ = ["NoopOracle"]
