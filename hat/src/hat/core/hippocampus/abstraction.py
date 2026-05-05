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
    """

    @abstractmethod
    def __call__(self, interaction: Interaction) -> MemoryTrace: ...


class IdentityAbstractor(Abstractor):
    """Default: copy fields verbatim. Useful for tests; replace in production."""

    def __call__(self, interaction: Interaction) -> MemoryTrace:
        # Stash the user-supplied correction in ``extras`` so downstream
        # scorers (notably the novelty judge) can read user-side input
        # without touching the model's own response.
        extras: dict = {}
        if interaction.user_correction:
            extras["user_correction"] = interaction.user_correction
        return MemoryTrace(
            interaction_id=interaction.id,
            query=interaction.query,
            cortex_response=interaction.response,
            target_response=interaction.user_correction or interaction.response,
            rationale=None,
            metadata=TraceMetadata(source=interaction.source, extras=extras),
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


class LLMAbstractor(Abstractor):
    """Use the Cortex itself to summarise the turn into a memory trace.

    On any failure we fall back to the identity abstraction so the wake step
    is robust to prompt drift on small models.
    """

    def __init__(self, cortex, *, max_tokens: int = 256) -> None:
        self.cortex = cortex
        self.max_tokens = max_tokens
        self._template = load_prompt("abstraction")
        self._fallback = IdentityAbstractor()

    def __call__(self, interaction: Interaction) -> MemoryTrace:
        marker = "## Input"
        if marker in self._template:
            system, body = self._template.split(marker, 1)
            user = marker + body
        else:
            system, user = self._template, ""
        rendered = render(
            user,
            query=interaction.query or "",
            response=interaction.response or "",
            correction=interaction.user_correction or "",
        )
        raw = call_judge(
            self.cortex, system=system.strip(), user=rendered.strip(),
            max_tokens=self.max_tokens,
        )
        data = _extract_json(raw)
        if not data:
            return self._fallback(interaction)
        target = data.get("target") or interaction.user_correction or interaction.response
        # Same extras handoff as IdentityAbstractor — keep user-side input
        # available to scorers that mustn't see the model's response.
        extras: dict = {}
        if interaction.user_correction:
            extras["user_correction"] = interaction.user_correction
        return MemoryTrace(
            interaction_id=interaction.id,
            query=interaction.query,
            cortex_response=interaction.response,
            target_response=target,
            rationale=(data.get("rationale") or data.get("summary") or None),
            metadata=TraceMetadata(source=interaction.source, extras=extras),
        )
