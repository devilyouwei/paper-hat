"""All ABCs, ``typing.Protocol`` interfaces, and shared pydantic schemas.

Concrete implementations live in :mod:`hat.core` (algorithms, storage,
backends, lifecycle) and inherit from the bases defined here.

This package is the single source of truth for every pluggable seam in
HAT. Front-ends only need ``hat.abstract`` plus ``hat.core.runtime`` —
they should never reach into implementation modules directly.
"""

from .cortex import Cortex, LanguageModel
from .hippocampus import (
    Abstractor,
    DedupDecision,
    DedupResult,
    Deduper,
    ReplayBuilder,
    UncertaintyEstimator,
    WritePolicy,
)
from .neocortex import (
    Embedder,
    Match,
    NeocortexStore,
    NeocortexWriteError,
    VectorIndex,
)
from .oracle import Oracle
from .schemas import (
    Interaction,
    MemoryTrace,
    ReplayBatch,
    ReplayExample,
    ScoreSignals,
    Session,
    SWSObjective,
    SWSStats,
    TraceMetadata,
    WriteDecision,
)
from .sessions import RawInteractionLog, SessionStore, SessionStoreError
from .sws import SWSTrainer

__all__ = [
    # cortex
    "Cortex",
    "LanguageModel",
    # hippocampus
    "Abstractor",
    "DedupDecision",
    "DedupResult",
    "Deduper",
    "ReplayBuilder",
    "UncertaintyEstimator",
    "WritePolicy",
    # neocortex
    "Embedder",
    "Match",
    "NeocortexStore",
    "NeocortexWriteError",
    "VectorIndex",
    # oracle
    "Oracle",
    # schemas
    "Interaction",
    "MemoryTrace",
    "ReplayBatch",
    "ReplayExample",
    "ScoreSignals",
    "Session",
    "SWSObjective",
    "SWSStats",
    "TraceMetadata",
    "WriteDecision",
    # sessions
    "RawInteractionLog",
    "SessionStore",
    "SessionStoreError",
    # sws
    "SWSTrainer",
]
