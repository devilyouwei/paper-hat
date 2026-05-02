"""Pydantic schemas for data flowing through the wake–sleep loop.

These types are the *single source of truth* for what an interaction, a memory
trace, a write decision, and a replay batch look like. Every protocol in
``hat.core.protocols`` references them. Storage backends serialize them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex


class Interaction(BaseModel):
    """Raw interaction tuple ``(c, x, y, f)`` from paper §3.1."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    context: str | None = None
    query: str
    response: str | None = None
    feedback: float | None = None  # 0/1 binary or graded score
    user_correction: str | None = None
    timestamp: datetime = Field(default_factory=_now)
    source: str = "user"


class ScoreSignals(BaseModel):
    """Triplet ``(U, F, N)`` used by the linear write policy."""

    uncertainty: float = 0.0
    feedback: float = 0.0
    novelty: float = 0.0


class TraceMetadata(BaseModel):
    timestamp: datetime = Field(default_factory=_now)
    source: str = "user"
    signals: ScoreSignals = Field(default_factory=ScoreSignals)
    extras: dict = Field(default_factory=dict)


class MemoryTrace(BaseModel):
    """Compressed episode written to the Neocortex.

    Mirrors paper Eq. ``memory_trace`` ``(q, a_cortex, a_target, r, ξ)``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    interaction_id: str
    query: str
    cortex_response: str | None = None
    target_response: str | None = None
    rationale: str | None = None
    metadata: TraceMetadata = Field(default_factory=TraceMetadata)


class WriteDecision(BaseModel):
    """Authority token required to write a trace into the Neocortex.

    Storage backends refuse writes whose ``WriteDecision.accepted`` is False or
    whose ``trace_id`` does not match the trace being written. This makes the
    raw → curated transition explicit and unforgeable.
    """

    trace_id: str
    score: float
    threshold: float
    signals: ScoreSignals
    accepted: bool


class ReplayExample(BaseModel):
    """Training example produced by the Replay Builder (paper §3.4.3)."""

    input: str
    target: str
    weight: float = 1.0
    source_trace_id: str
    is_oracle: bool = False


class ReplayBatch(BaseModel):
    examples: list[ReplayExample]
    cycle: int = 0


class SWSObjective(BaseModel):
    """λ-weighted SWS objective from paper Eq. ``sws_loss``."""

    lambda_kd: float = 0.0
    lambda_stab: float = 0.0
    learning_rate: float = 2e-5
    epochs: int = 3


class SWSStats(BaseModel):
    cycle: int
    n_replayed: int
    loss_sup: float = 0.0
    loss_kd: float = 0.0
    loss_stab: float = 0.0
    duration_seconds: float = 0.0


__all__ = [
    "Interaction",
    "ScoreSignals",
    "TraceMetadata",
    "MemoryTrace",
    "WriteDecision",
    "ReplayExample",
    "ReplayBatch",
    "SWSObjective",
    "SWSStats",
]
