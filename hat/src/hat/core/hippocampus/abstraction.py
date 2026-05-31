"""Trace abstraction: ``m = H_abs(c, x, y, f)`` (paper Eq. ``abstraction``).

The default :class:`IdentityAbstractor` copies fields verbatim — useful
for tests and for backends without a callable Cortex. The production
path is :class:`LLMAbstractor`, a **two-step workflow** whose only job
now is to *extract knowledge points* from the current turn:

1. **Triage** — given only the current ``(query, response)`` and a
   short context, decide whether the turn carries a knowledge point
   worth remembering at all. Trivial small-talk gets dropped here.
2. **Extract** — only invoked when triage says *keep*. Emits one or
   more canonical ``(query, target)`` pairs from the current turn. A
   single user/assistant exchange may surface multiple independent
   knowledge points (multi-fact statements, etc.).

Routing the resulting traces to CREATE or REVISE (i.e. deciding whether
to overwrite an existing memory or append a new one) is **no longer the
abstractor's responsibility** — that is decided downstream by an
embedding similarity check against the curated index. See
:mod:`hat.core.hippocampus.dedup`.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable

from ...utils.logging import get_logger, truncate
from ..schemas import Interaction, MemoryTrace, TraceMetadata
from .scoring.llm_judge import call_judge, load_prompt, render

log = get_logger(__name__)


class Abstractor(ABC):
    """Maps a raw :class:`Interaction` to zero or more :class:`MemoryTrace`s.

    Mirrors paper Eq. ``abstraction``: ``m = H_abs(c, x, y, f)``. A turn
    can yield multiple traces when the user packs several independent
    knowledge points into a single utterance ("我三十岁，住在北京…").

    An empty list signals DROP (no knowledge point worth storing).
    """

    @abstractmethod
    def __call__(self, interaction: Interaction) -> list[MemoryTrace]: ...


class IdentityAbstractor(Abstractor):
    """Default: copy fields verbatim. Useful for tests; replace in production."""

    def __call__(self, interaction: Interaction) -> list[MemoryTrace]:
        trace = MemoryTrace(
            interaction_id=interaction.id,
            session_id=interaction.session_id,
            interaction_ids=[interaction.id],
            query=interaction.query,
            cortex_response=interaction.response,
            target_response=interaction.response,
            rationale=None,
            metadata=TraceMetadata(source=interaction.source),
        )
        return [trace]


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


def _is_nonempty_str(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


class LLMAbstractor(Abstractor):
    """Two-step abstractor (paper Eq. ``abstraction``).

    Step 1 (*triage*) decides keep-or-drop using only the current turn —
    cheap, small token budget, no prior context required. Step 2
    (*extract*) emits one or more canonical ``(query, target)`` pairs.

    On any parse failure we fall back to :class:`IdentityAbstractor` —
    safe because the dataset poison risk that used to live here (copying
    a meta-correction utterance verbatim into ``query`` while routing it
    onto a prior trace) no longer exists: routing has moved out of the
    abstractor and the downstream dedup step decides CREATE vs REVISE
    purely from the canonical query embedding.
    """

    def __init__(
        self,
        cortex,
        *,
        max_tokens_triage: int = 192,
        max_tokens_extract: int = 768,
        context_char_budget: int = 1200,
    ) -> None:
        self.cortex = cortex
        self.max_tokens_triage = max_tokens_triage
        self.max_tokens_extract = max_tokens_extract
        # Cap the rendered ``{context}`` so a long prior turn doesn't
        # blow the prompt budget and cause the JSON output to get
        # truncated (which silently routes the turn through the
        # IdentityAbstractor fallback).
        self.context_char_budget = context_char_budget
        self._triage_template = load_prompt("abstraction_triage")
        self._extract_template = load_prompt("abstraction_extract")
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
            ctx = "…\n" + ctx[-self.context_char_budget :]
        return ctx

    def _triage(
        self,
        interaction: Interaction,
        emit: Callable[[str, dict], None] | None = None,
    ) -> dict | None:
        system, user = self._split_system_user(self._triage_template)
        rendered = render(
            user,
            context=self._truncate_context(interaction),
            query=interaction.query or "",
            response=interaction.response or "",
        )
        log.info(
            "abstractor.triage.start iid={} sid={} query_chars={} response_chars={}",
            interaction.id, interaction.session_id,
            len(interaction.query or ""), len(interaction.response or ""),
        )
        if emit is not None:
            emit(
                "triage_start",
                {
                    "interaction_id": interaction.id,
                    "session_id": interaction.session_id,
                },
            )
        raw = call_judge(
            self.cortex, system=system, user=rendered,
            max_tokens=self.max_tokens_triage,
        )
        result = _extract_json(raw)
        log.info(
            "abstractor.triage.done iid={} keep={} parsed={} raw_chars={}",
            interaction.id,
            (result.get("keep") if isinstance(result, dict) else None),
            isinstance(result, dict),
            len(raw or ""),
        )
        if emit is not None:
            keep = result.get("keep") if isinstance(result, dict) else None
            reason = (
                (result.get("reason") or result.get("rationale"))
                if isinstance(result, dict) else None
            )
            emit(
                "triage_done",
                {
                    "interaction_id": interaction.id,
                    "session_id": interaction.session_id,
                    "keep": keep,
                    "reason": reason,
                },
            )
        return result

    def _extract(
        self,
        interaction: Interaction,
        emit: Callable[[str, dict], None] | None = None,
    ) -> dict | None:
        system, user = self._split_system_user(self._extract_template)
        rendered = render(
            user,
            context=self._truncate_context(interaction),
            query=interaction.query or "",
            response=interaction.response or "",
        )
        log.info(
            "abstractor.extract.start iid={} sid={}",
            interaction.id, interaction.session_id,
        )
        if emit is not None:
            emit(
                "extract_start",
                {
                    "interaction_id": interaction.id,
                    "session_id": interaction.session_id,
                },
            )
        raw = call_judge(
            self.cortex, system=system, user=rendered,
            max_tokens=self.max_tokens_extract,
        )
        result = _extract_json(raw)
        if isinstance(result, dict):
            kps = result.get("knowledge_points") or []
            log.info(
                "abstractor.extract.done iid={} n_kps={}",
                interaction.id, len(kps) if isinstance(kps, list) else 0,
            )
        else:
            log.warning(
                "abstractor.extract.unparseable iid={} raw='{}'",
                interaction.id, truncate(raw or "", limit=400),
            )
        if emit is not None:
            parsed = isinstance(result, dict)
            kps = (
                result.get("knowledge_points")
                if parsed and isinstance(result.get("knowledge_points"), list)
                else []
            )
            emit(
                "extract_done",
                {
                    "interaction_id": interaction.id,
                    "session_id": interaction.session_id,
                    "parsed": parsed,
                    "n_kps": len(kps),
                },
            )
        return result

    def _fallback_or_drop(self, interaction: Interaction) -> list[MemoryTrace]:
        """Handle an unparseable / malformed extract response.

        With routing removed from the abstractor there is no longer a
        dataset-poison risk in identity-copying the raw turn — the
        downstream dedup step decides whether this becomes a CREATE or
        a REVISE based on the canonical query embedding, not on the
        question text we forge here. So fall back to the identity
        abstractor and tag the trace for diagnostic purposes.
        """
        traces = self._fallback(interaction)
        for tr in traces:
            tr.metadata.extras["abstractor_fallback"] = True
        log.warning(
            "abstractor.fallback identity iid={} reason=unparseable",
            interaction.id,
        )
        return traces

    def __call__(
        self,
        interaction: Interaction,
        *,
        event_sink: Callable[[str, dict], None] | None = None,
    ) -> list[MemoryTrace]:
        log.info(
            "abstractor.call iid={} sid={}",
            interaction.id, interaction.session_id,
        )

        # ---- Step 1: triage --------------------------------------------
        triage = self._triage(interaction, emit=event_sink)
        if isinstance(triage, dict) and triage.get("keep") is False:
            log.info(
                "abstractor.dropped iid={} stage=triage reason={}",
                interaction.id, triage.get("reason") or triage.get("rationale"),
            )
            return []

        # ---- Step 2: extract knowledge points --------------------------
        data = self._extract(interaction, emit=event_sink)
        if not isinstance(data, dict):
            return self._fallback_or_drop(interaction)
        kps = data.get("knowledge_points")
        if not isinstance(kps, list):
            return self._fallback_or_drop(interaction)
        if not kps:
            log.info(
                "abstractor.dropped iid={} stage=extract reason=empty_kps",
                interaction.id,
            )
            return []

        triage_reason = (
            triage.get("reason") or triage.get("rationale")
            if isinstance(triage, dict) else None
        )

        traces: list[MemoryTrace] = []
        for idx, kp in enumerate(kps):
            if not isinstance(kp, dict):
                continue
            q_raw = kp.get("query")
            t_raw = kp.get("target")
            query = q_raw if _is_nonempty_str(q_raw) else interaction.query
            target = t_raw if _is_nonempty_str(t_raw) else interaction.response

            if not _is_nonempty_str(target) or len(target.strip()) < 4:
                log.info(
                    "abstractor.kp_rejected iid={} idx={} target='{}'",
                    interaction.id, idx, truncate(target or "", limit=80),
                )
                continue

            extras: dict = {
                "kp_index": idx,
                "kp_count": len(kps),
            }
            if triage_reason:
                extras["triage_reason"] = triage_reason
            kp_rationale = kp.get("rationale")
            if _is_nonempty_str(kp_rationale):
                extras["extract_rationale"] = kp_rationale

            traces.append(
                MemoryTrace(
                    interaction_id=interaction.id,
                    session_id=interaction.session_id,
                    interaction_ids=[interaction.id],
                    query=query,
                    cortex_response=interaction.response,
                    target_response=target,
                    rationale=kp_rationale or None,
                    metadata=TraceMetadata(
                        source=interaction.source, extras=extras
                    ),
                )
            )

        if not traces:
            log.info(
                "abstractor.dropped iid={} stage=extract reason=all_kps_rejected",
                interaction.id,
            )
        return traces
