r"""Hippocampus Agent — selective memory consolidation (paper §3.4).

Three composable stages:

* :mod:`abstraction` — compress raw interactions into ``MemoryTrace``\ s.
* :mod:`selection`  — score traces and emit a ``WriteDecision``.
* :mod:`replay`     — convert retained traces into training-ready ``ReplayExample``\ s.

Per-signal scorers (currently uncertainty only) live under :mod:`scoring`.
"""

from .abstraction import Abstractor, IdentityAbstractor, LLMAbstractor
from .dedup import DedupResult, EmbeddingDeduper
from .replay import ReplayBuilder, SupervisedReplayBuilder
from .selection import UncertaintyGatePolicy, WritePolicy

__all__ = [
    "Abstractor",
    "IdentityAbstractor",
    "LLMAbstractor",
    "DedupResult",
    "EmbeddingDeduper",
    "WritePolicy",
    "UncertaintyGatePolicy",
    "ReplayBuilder",
    "SupervisedReplayBuilder",
]
