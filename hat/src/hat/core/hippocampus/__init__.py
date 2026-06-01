r"""Hippocampus Agent — selective memory consolidation (paper §3.4)."""

from hat.abstract.hippocampus import (
    Abstractor,
    DedupResult,
    Deduper,
    ReplayBuilder,
    UncertaintyEstimator,
    WritePolicy,
)

from .abstraction import IdentityAbstractor, LLMAbstractor
from .dedup import EmbeddingDeduper
from .replay import SupervisedReplayBuilder
from .selection import UncertaintyGatePolicy

__all__ = [
    "Abstractor",
    "DedupResult",
    "Deduper",
    "EmbeddingDeduper",
    "IdentityAbstractor",
    "LLMAbstractor",
    "ReplayBuilder",
    "SupervisedReplayBuilder",
    "UncertaintyEstimator",
    "UncertaintyGatePolicy",
    "WritePolicy",
]
