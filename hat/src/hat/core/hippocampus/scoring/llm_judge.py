"""Shared utilities for LLM-as-judge scoring.

The novelty / feedback estimators and the abstractor all share the same
plumbing: load a prompt template from ``hippocampus/prompts/<name>.md``,
substitute fields, call ``cortex.chat`` with conservative generation
settings, strip ``<think>`` blocks, and parse the result.

Score parsing is intentionally permissive: we accept ``"0.7"``, ``"score:
0.7"``, ``"7/10"``, ``"70%"``, etc., so a slightly chatty model still
yields a usable scalar. Anything we cannot parse falls back to ``fallback``.
"""

from __future__ import annotations

import re
from pathlib import Path

from ....utils.logging import format_messages, format_text_block, get_logger

log = get_logger(__name__)

# Prompt files live next to the scoring package. Using ``__file__`` keeps
# them packaged together with the source — no run-time path config needed.
_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Cache prompt files in-process. These are tiny and read-mostly; reloading
# every call would dominate latency on small models.
_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Load ``hippocampus/prompts/<name>.md`` (cached)."""
    if name not in _PROMPT_CACHE:
        path = _PROMPT_DIR / f"{name}.md"
        _PROMPT_CACHE[name] = path.read_text(encoding="utf-8")
    return _PROMPT_CACHE[name]


# Regex strategies, applied in order. Each pattern's first capturing group is
# converted to a float and clipped to [0, 1]. ``%`` and ``X/Y`` forms are
# normalised before clipping.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_NUM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)(?![\d.])")
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_FRAC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    lower = text.lower()
    if "<think>" in lower and "</think>" not in lower:
        idx = lower.rfind("<think>")
        text = text[:idx]
    return text.strip()


def parse_score(text: str, *, fallback: float = 0.0) -> float:
    """Extract the first numeric score in ``text`` and clip to ``[0, 1]``.

    Tries (in order): bare decimal in [0,1] → percentage → ``a/b`` fraction →
    bare decimal divided by 10 (Likert-style) → ``fallback``.
    """
    text = _strip_think(text)
    if not text:
        return fallback

    # 1) bare decimal already in [0, 1]
    for m in _NUM_RE.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v

    # 2) percentage
    m = _PCT_RE.search(text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1)) / 100.0))
        except ValueError:
            pass

    # 3) fraction "a/b"
    m = _FRAC_RE.search(text)
    if m:
        try:
            num, den = float(m.group(1)), float(m.group(2))
            if den > 0:
                return max(0.0, min(1.0, num / den))
        except ValueError:
            pass

    # 4) Likert 0-10 → 0-1
    m = _NUM_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
        except ValueError:
            return fallback
        if 0.0 <= v <= 10.0:
            return max(0.0, min(1.0, v / 10.0))

    return fallback


def render(template: str, **fields: str) -> str:
    """Format ``template`` with ``{field}`` placeholders, tolerating missing keys.

    ``str.format`` would raise on stray braces in the prompt body; we use a
    conservative regex substitution instead.
    """
    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(fields.get(key, ""))

    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", _sub, template)


def call_judge(
    cortex,
    *,
    system: str,
    user: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> str:
    """Call ``cortex.chat`` (or ``generate``) for a judge prompt.

    Returns the raw model output (with ``<think>`` blocks left intact — the
    parser strips them). Returns an empty string on failure so callers can
    fall back to a default score without raising.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    log.debug(
        "llm_judge.call max_tokens={} temperature={} system_chars={} user_chars={}",
        max_tokens, temperature, len(system or ""), len(user or ""),
    )
    log.debug("llm_judge.prompt\n{}", format_messages(messages, title="llm_judge"))
    try:
        out = cortex.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            # Disable Qwen3-style thinking so the parser sees the answer
            # directly. Backends that don't recognise the kwarg ignore it.
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as e:
        log.warning("llm_judge.call failed: {}: {}", type(e).__name__, e)
        return ""
    log.debug("llm_judge.response\n{}", format_text_block(out or "", title="raw output"))
    return out
