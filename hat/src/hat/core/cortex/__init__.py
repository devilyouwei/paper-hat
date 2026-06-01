"""Cortex — the online interaction model (paper §3.3)."""

from hat.abstract.cortex import Cortex
from .noop import NoopCortex

__all__ = ["Cortex", "NoopCortex"]
