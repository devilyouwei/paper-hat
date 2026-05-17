"""Trace abstraction: ``m = H_abs(c, x, y, f)`` (paper Eq. ``abstraction``).

The default :class:`IdentityAbstractor` copies fields verbatim — useful for
tests and for backends without a callable Cortex. The production path is
:class:`LLMAbstractor`, a **two-step workflow**:

1. **Triage** — given only the current ``(query, response)`` and a short
   context, decide whether the turn carries a knowledge point worth
   remembering at all. No prior_traces are loaded; trivial small-talk
   gets dropped here without ever touching the routing step.
2. **Route** — only invoked when triage says *keep*. Receives the
   session's existing traces plus the new turn, decides CREATE vs
   REVISE, and emits the canonical ``(query, target)`` pair to store.
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
    """Two-step abstractor (paper Eq. ``abstraction``).

    The wake step is split into two LLM calls so that each call has a
    single, narrow responsibility and a small prompt:

    1. **Triage** — given ONLY the current ``(query, response)`` (and a
       short context), decide whether the turn carries a knowledge point
       worth remembering. No prior traces are consulted here; the model
       is just answering "keep or drop?". Cheap; small output budget.
    2. **Route** — only invoked when triage says *keep*. Receives the
       session's existing traces plus the new turn, and emits the routing
       decision (CREATE vs REVISE) together with the canonical
       ``(query, target)`` pair to store.

    Splitting these concerns keeps the route prompt focused (it no longer
    has to also argue about novelty) and prevents wasting tokens on
    pleasantries — those are dropped at step 1 without ever loading the
    prior_traces context.

    On any parse failure we fall back to :class:`IdentityAbstractor` —
    but only when ``priors`` is empty. If priors exist we must DROP
    instead, because copying the raw user utterance (often a meta
    correction like "错误，X 其实是 Y") into ``query`` would poison the
    training set.
    """

    def __init__(
        self,
        cortex,
        *,
        max_tokens_triage: int = 192,
        max_tokens_route: int = 512,
        context_char_budget: int = 1200,
    ) -> None:
        self.cortex = cortex
        self.max_tokens_triage = max_tokens_triage
        self.max_tokens_route = max_tokens_route
        # Cap the rendered ``{context}`` so a long prior turn doesn't blow
        # the prompt budget and cause the JSON output to get truncated
        # (which silently routes the turn through the IdentityAbstractor
        # fallback and forces an unwanted CREATE).
        self.context_char_budget = context_char_budget
        self._triage_template = load_prompt("abstraction_triage")
        self._route_template = load_prompt("abstraction_route")
        self._fallback = IdentityAbstractor()

    @staticmethod
    def _split_system_user(template: str) -> tuple[str, str]:
        marker = "## Input"
        if marker in template:
            system, body = template.split(marker, 1)
            return system.strip(), (marker + body).strip()
        return template.strip(), ""

    def _truncate_context(self, interaction: Interaction) -> str:
        ctx = (interaction.context or "").strip() or "(none)"
        if self.context_char_budget and len(ctx) > self.context_char_budget:
            # Keep the tail — the most recent turn is the most relevant
            # for both triage and routing on the current exchange.
            ctx = "…\n" + ctx[-self.context_char_budget :]
        return ctx

    def _triage(self, interaction: Interaction) -> dict | None:
        system, user = self._split_system_user(self._triage_template)
        rendered = render(
            user,
            context=self._truncate_context(interaction),
            query=interaction.query or "",
            response=interaction.response or "",
        )
        raw = call_judge(
            self.cortex, system=system, user=rendered,
            max_tokens=self.max_tokens_triage,
        )
        return _extract_json(raw)

    def _route(
        self, interaction: Interaction, prior_traces: list[dict]
    ) -> dict | None:
        system, user = self._split_system_user(self._route_template)
        rendered = render(
            user,
            context=self._truncate_context(interaction),
            prior_traces_json=json.dumps(prior_traces, ensure_ascii=False, default=str),
            query=interaction.query or "",
            response=interaction.response or "",
        )
        raw = call_judge(
            self.cortex, system=system, user=rendered,
            max_tokens=self.max_tokens_route,
        )
        return _extract_json(raw)

    def _fallback_or_drop(
        self, interaction: Interaction, priors: list[dict]
    ) -> MemoryTrace | None:
        """Handle an unparseable / malformed LLM response.

        A trace's ``query`` is supposed to be the *canonical applied form*
        of the lesson, produced by the abstractor LLM. When the LLM call
        fails we have two bad options:

        * Identity-copy the raw interaction. Safe only when there are no
          priors — there is nothing to confuse with and the raw Q/A
          *is* the knowledge point.
        * Drop the turn. Required whenever ``priors`` is non-empty: this
          turn is almost certainly a correction / clarification of an
          existing trace, and copying the user's meta-instruction
          (e.g. "错误，黄有为是...") verbatim into ``query`` would poison
          the dataset. Better to lose this row than to forge one.
        """
        if priors:
            return None
        trace = self._fallback(interaction)
        if trace is not None:
            trace.metadata.extras["abstractor_fallback"] = True
        return trace

    def __call__(
        self,
        interaction: Interaction,
        *,
        prior_traces: list[dict] | None = None,
    ) -> MemoryTrace | None:
        priors = prior_traces or []

        # ---- Step 1: triage --------------------------------------------
        triage = self._triage(interaction)
        # An explicit `keep: false` is the only path to DROP. A parse
        # failure here errs toward keeping the turn (routing will then
        # apply its own malformed-output policy via _fallback_or_drop).
        if isinstance(triage, dict) and triage.get("keep") is False:
            return None

        # ---- Step 2: route ---------------------------------------------
        data = self._route(interaction, priors)
        if not isinstance(data, dict):
            return self._fallback_or_drop(interaction, priors)

        decision = str(data.get("decision") or "").strip().upper()

        # Validate REVISE target id; fall back to CREATE if the model
        # pointed at a non-existent trace.
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
            return self._fallback_or_drop(interaction, priors)

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
        # Triage owns novelty / user_signal; route owns rationale. Merge
        # so _stash_signals sees a single dict with both halves populated.
        merged = dict(triage or {})
        merged.update(data)
        _stash_signals(trace, merged)
        return trace
