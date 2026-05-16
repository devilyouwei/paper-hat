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
    """Raw interaction tuple ``(c, x, y)`` from paper §3.1.

    Feedback / corrections are no longer separate fields — corrections appear
    as the *next* user turn in the same session and are detected by the
    abstractor's router prompt when comparing against prior session traces.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    session_id: str | None = None
    context: str | None = None
    query: str
    response: str | None = None
    timestamp: datetime = Field(default_factory=_now)
    source: str = "user"
    # Optional per-turn HAT annotations populated after ``wake_step`` runs.
    # Persisted with the raw session log so the UI can re-render the
    # uncertainty / route badges when a session is reopened. Keys in use:
    # ``uncertainty`` (float), ``decision`` (created|revised|skipped|dropped|
    # rejected), ``trace_id`` (str|None), ``novelty`` (float|None),
    # ``user_signal`` (float|None), ``reason`` (str|None).
    hat: dict | None = None


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
    session_id: str | None = None
    interaction_ids: list[str] = Field(default_factory=list)
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
