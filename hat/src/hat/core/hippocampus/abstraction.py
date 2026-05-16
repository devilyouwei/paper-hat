"""Trace abstraction: ``m = H_abs(c, x, y, f)`` (paper Eq. ``abstraction``).

The default :class:`IdentityAbstractor` copies fields verbatim — useful for
tests and for backends without a callable Cortex. The production path is
:class:`LLMAbstractor`, which asks the Cortex itself, in a **single LLM
call**, to read the latest turn against the session's existing traces and
decide CREATE / REVISE / DROP plus emit the canonical ``(query, target)``
pair to store.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from ..schemas import Interaction, MemoryTrace, TraceMetadata
from .scoring.llm_judge import call_judge, load_prompt, render


class Abstractor(ABC):
    """Maps a raw :class:`Interaction` to a compact :class:`MemoryTrace`.

    Mirrors paper Eq. ``abstraction``: ``m = H_abs(c, x, y, f)``. Real
    implementations call a small summarization model or prompt the Cortex
    itself under an instruction template.

    Implementations may additionally accept ``prior_traces`` (a list of dicts
    summarising existing traces for the current session) so they can route
    the interaction to either a CREATE, REVISE, or DROP path. REVISE intent
    is signalled via ``trace.metadata.extras["revise_of"] = <trace_id>``;
    DROP is signalled by returning ``None``.
    """

    @abstractmethod
    def __call__(
        self,
        interaction: Interaction,
        *,
        prior_traces: list[dict] | None = None,
    ) -> MemoryTrace | None: ...


class IdentityAbstractor(Abstractor):
    """Default: copy fields verbatim. Useful for tests; replace in production."""

    def __call__(
        self,
        interaction: Interaction,
        *,
        prior_traces: list[dict] | None = None,
    ) -> MemoryTrace:
        return MemoryTrace(
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            interaction_ids=[interaction.id],
            query=interaction.query,
            cortex_response=interaction.response,
            target_response=interaction.response,
            rationale=None,
            metadata=TraceMetadata(source=interaction.source),
        )


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from a possibly-chatty model output."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _stash_signals(trace: MemoryTrace, data: dict) -> None:
    """Persist the model's own novelty / user_signal / rationale judgement.

    Numeric scores are mirrored into ``metadata.signals`` (bridging the
    legacy 3-channel ``ScoreSignals`` shape so on-disk rows are intuitive)
    and the full record is also kept in ``extras`` for the UI.
    """
    extras = trace.metadata.extras
    for k_src, k_dst in (
        ("novelty", "route_novelty"),
        ("user_signal", "route_user_signal"),
        ("rationale", "route_reason"),
    ):
        v = data.get(k_src)
        if v is not None:
            extras[k_dst] = v
    sig = trace.metadata.signals
    nov = data.get("novelty")
    if isinstance(nov, (int, float)):
        sig.novelty = float(nov)
    usig = data.get("user_signal")
    if isinstance(usig, (int, float)):
        sig.feedback = float(usig)


def _is_nonempty_str(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


class LLMAbstractor(Abstractor):
    """Single-call abstractor.

    Asks the Cortex *once*, given the session's existing traces and the new
    turn, to emit a JSON envelope with the routing decision plus the
    canonical ``(query, target)`` pair to store. On any parse failure we
    fall back to :class:`IdentityAbstractor` so the wake step stays robust
    on tiny / drifted models.
    """

    def __init__(self, cortex, *, max_tokens: int = 384) -> None:
        self.cortex = cortex
        self.max_tokens = max_tokens
        self._template = load_prompt("abstraction")
        self._fallback = IdentityAbstractor()

    @staticmethod
    def _split_system_user(template: str) -> tuple[str, str]:
        marker = "## Input"
        if marker in template:
            system, body = template.split(marker, 1)
            return system.strip(), (marker + body).strip()
        return template.strip(), ""

    def _ask_llm(
        self,
        interaction: Interaction,
        prior_traces: list[dict],
    ) -> dict | None:
        system, user = self._split_system_user(self._template)
        rendered = render(
            user,
            context=(interaction.context or "").strip() or "(none)",
            prior_traces_json=json.dumps(prior_traces, ensure_ascii=False, default=str),
            query=interaction.query or "",
            response=interaction.response or "",
        )
        raw = call_judge(
            self.cortex, system=system, user=rendered,
            max_tokens=self.max_tokens,
        )
        return _extract_json(raw)

    def __call__(
        self,
        interaction: Interaction,
        *,
        prior_traces: list[dict] | None = None,
    ) -> MemoryTrace | None:
        priors = prior_traces or []
        data = self._ask_llm(interaction, priors)

        # Hard fallback when the model produced unparseable output: keep the
        # turn rather than silently drop it so the loop stays useful.
        if not isinstance(data, dict):
            return self._fallback(interaction)

        decision = str(data.get("decision") or "").strip().upper()
        if decision == "DROP":
            return None

        # Validate REVISE target id; fall back to CREATE if the model pointed
        # at a non-existent trace.
        prior: dict | None = None
        if decision == "REVISE":
            tid = data.get("trace_id")
            if _is_nonempty_str(tid):
                prior = next(
                    (t for t in priors if t.get("trace_id") == tid), None
                )
            if prior is None:
                decision = "CREATE"

        # Pull and sanity-check the Q/A pair. Reject one-word / fragment
        # targets — they pollute the training set.
        q_raw = data.get("query")
        t_raw = data.get("target")
        query = q_raw if _is_nonempty_str(q_raw) else interaction.query
        target = t_raw if _is_nonempty_str(t_raw) else interaction.response
        if not _is_nonempty_str(target) or len(target.strip()) < 4:
            # Treat malformed output the same as a parse failure rather than
            # writing a broken row.
            return self._fallback(interaction)

        extras: dict = {}
        if decision == "REVISE" and prior is not None:
            extras["revise_of"] = prior.get("trace_id")

        trace = MemoryTrace(
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            interaction_ids=[interaction.id],
            query=query,
            cortex_response=interaction.response,
            target_response=target,
            rationale=data.get("rationale") or None,
            metadata=TraceMetadata(source=interaction.source, extras=extras),
        )
        _stash_signals(trace, data)
        return trace
