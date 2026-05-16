"""Trace abstraction: ``m = H_abs(c, x, y, f)`` (paper Eq. ``abstraction``).

The default :class:`IdentityAbstractor` copies fields verbatim — useful for
tests and for backends without a callable Cortex. The production path is
:class:`LLMAbstractor`, which asks the Cortex to compress the interaction
into a JSON ``{summary, target, rationale}`` triple.
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
    implementations call a small summarization model or prompt the Cortex itself
    under an instruction template.

    Implementations may additionally accept ``prior_traces`` (a list of dicts
    summarising existing traces for the current session) and route the
    interaction to either a CREATE or REVISE path. The base contract returns a
    :class:`MemoryTrace`; REVISE intent is signalled via
    ``trace.metadata.extras["revise_of"] = <trace_id>`` so the caller can
    rewrite the existing entry in place.
    """

    @abstractmethod
    def __call__(
        self,
        interaction: Interaction,
        *,
        prior_traces: list[dict] | None = None,
    ) -> MemoryTrace: ...


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


def _stash_route_meta(trace: MemoryTrace, route_meta: dict | None) -> None:
    """Persist the router's novelty / user_signal / reason into ``extras``.

    These are the model's *own* judgements about why the trace was kept;
    they replace the old fixed-weight ``ScoreSignals`` channels and are
    surfaced in the UI for inspection. We keep only the well-known keys to
    avoid leaking arbitrary router output.
    """
    if not route_meta:
        return
    extras = trace.metadata.extras
    for k in ("novelty", "user_signal", "reason"):
        v = route_meta.get(k)
        if v is not None:
            extras[f"route_{k}"] = v


class LLMAbstractor(Abstractor):
    """Use the Cortex itself to summarise the turn into a memory trace.

    Two-step routing when ``prior_traces`` is supplied:

    1. Ask the model to route the turn into one of CREATE / REVISE / DROP.
       The router itself judges novelty (is the answer new to the model?)
       and the strength of the user signal (is the user teaching/correcting
       the model?) — fixed numeric weights are not used.
    2. Depending on (1) either run the summary prompt (CREATE), the revise
       prompt (REVISE) and tag the resulting trace with
       ``metadata.extras["revise_of"] = <prior_trace_id>``, or return
       ``None`` (DROP) so the wake step writes nothing.

    On any failure we fall back to the identity abstraction so the wake step
    is robust to prompt drift on small models.
    """

    def __init__(self, cortex, *, max_tokens: int = 256) -> None:
        self.cortex = cortex
        self.max_tokens = max_tokens
        self._template = load_prompt("abstraction")
        self._router_template = load_prompt("abstraction_router")
        self._revise_template = load_prompt("abstraction_revise")
        self._fallback = IdentityAbstractor()

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _split_system_user(template: str) -> tuple[str, str]:
        marker = "## Input"
        if marker in template:
            system, body = template.split(marker, 1)
            return system.strip(), (marker + body).strip()
        return template.strip(), ""

    def _route(
        self,
        interaction: Interaction,
        prior_traces: list[dict],
    ) -> tuple[str, str | None, dict]:
        """Return ``(decision, prior_trace_id, raw_data)``.

        ``decision`` is one of CREATE / REVISE / DROP. Falls back to CREATE
        on parse errors so the loop stays robust. ``raw_data`` carries any
        side fields (``novelty``, ``user_signal``, ``reason``) the router
        emitted so we can stash them in the trace's ``extras`` for later
        inspection.
        """
        prior_json = json.dumps(prior_traces, ensure_ascii=False, default=str)
        system, user = self._split_system_user(self._router_template)
        rendered = render(
            user,
            prior_traces_json=prior_json,
            query=interaction.query or "",
            response=interaction.response or "",
        )
        raw = call_judge(
            self.cortex, system=system, user=rendered,
            max_tokens=min(192, self.max_tokens),
        )
        data = _extract_json(raw) or {}
        decision = str(data.get("decision") or "").strip().upper()
        if decision not in ("CREATE", "REVISE", "DROP"):
            return "CREATE", None, data
        if decision == "REVISE":
            trace_id = data.get("trace_id")
            valid_ids = {t.get("trace_id") for t in prior_traces}
            if not trace_id or trace_id not in valid_ids:
                # Router asked for REVISE but pointed at a non-existent id;
                # safer to CREATE than to silently drop user-tagged content.
                return "CREATE", None, data
            return "REVISE", str(trace_id), data
        return decision, None, data

    def _summarise_create(self, interaction: Interaction) -> MemoryTrace | None:
        system, user = self._split_system_user(self._template)
        rendered = render(
            user,
            query=interaction.query or "",
            response=interaction.response or "",
        )
        raw = call_judge(
            self.cortex, system=system, user=rendered,
            max_tokens=self.max_tokens,
        )
        data = _extract_json(raw)
        if not data:
            return None
        target = data.get("target") or interaction.response
        return MemoryTrace(
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            interaction_ids=[interaction.id],
            query=interaction.query,
            cortex_response=interaction.response,
            target_response=target,
            rationale=(data.get("rationale") or data.get("summary") or None),
            metadata=TraceMetadata(source=interaction.source),
        )

    def _summarise_revise(
        self,
        interaction: Interaction,
        prior_trace: dict,
    ) -> MemoryTrace | None:
        system, user = self._split_system_user(self._revise_template)
        rendered = render(
            user,
            prior_trace_json=json.dumps(prior_trace, ensure_ascii=False, default=str),
            query=interaction.query or "",
            response=interaction.response or "",
        )
        raw = call_judge(
            self.cortex, system=system, user=rendered,
            max_tokens=self.max_tokens,
        )
        data = _extract_json(raw)
        if not data:
            return None
        target = (
            data.get("target")
            or interaction.response
            or prior_trace.get("target")
        )
        # The revise prompt also asks for a consolidated ``query`` so the
        # stored Q/A pair stays coherent after we overwrite ``target``. Fall
        # back to the prior query (then the new query) only when the model
        # forgot to supply one.
        query = (
            data.get("query")
            or prior_trace.get("query")
            or interaction.query
        )
        extras: dict = {"revise_of": prior_trace.get("trace_id")}
        return MemoryTrace(
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            interaction_ids=[interaction.id],
            query=query,
            cortex_response=interaction.response,
            target_response=target,
            rationale=(data.get("rationale") or data.get("summary") or None),
            metadata=TraceMetadata(source=interaction.source, extras=extras),
        )

    # -- public entrypoint -----------------------------------------------

    def __call__(
        self,
        interaction: Interaction,
        *,
        prior_traces: list[dict] | None = None,
    ) -> MemoryTrace | None:
        # No prior context → single-step CREATE path (legacy behaviour).
        if not prior_traces:
            trace = self._summarise_create(interaction)
            return trace if trace is not None else self._fallback(interaction)

        decision, prior_trace_id, route_meta = self._route(interaction, prior_traces)
        if decision == "DROP":
            # The router judged the turn not worth remembering; honour it.
            return None
        if decision == "REVISE" and prior_trace_id is not None:
            prior = next(
                (t for t in prior_traces if t.get("trace_id") == prior_trace_id),
                None,
            )
            if prior is not None:
                revised = self._summarise_revise(interaction, prior)
                if revised is not None:
                    _stash_route_meta(revised, route_meta)
                    return revised
        trace = self._summarise_create(interaction)
        if trace is None:
            trace = self._fallback(interaction)
        _stash_route_meta(trace, route_meta)
        return trace
