"""Project-wide logging utilities.

Thin wrapper around `loguru <https://github.com/Delgan/loguru>`_ that

* configures a single stderr sink with a consistent format (timestamp,
  level, source location, message) on first use,
* exposes :func:`get_logger` for per-module loggers — call sites get a
  ``logger`` bound with ``module=__name__`` so messages can be filtered or
  grouped by origin without touching loguru directly,
* is fully idempotent: importing this module from many places does not
  duplicate sinks or messages.

Typical usage::

    from hat.utils.logging import get_logger

    log = get_logger(__name__)
    log.info("loaded model {}", path)
    log.exception("oops")  # logs traceback at ERROR level
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger

# Format pieces:
#   {time:YYYY-MM-DD HH:mm:ss.SSS}  — wall-clock, millisecond precision
#   {level:<7}                       — INFO/DEBUG/WARNING/... padded
#   {extra[module]}                  — logical module bound via get_logger
#   {name}:{function}:{line}         — actual call site (file stack)
#   {message}                        — formatted message
_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:<7}</level> | "
    "<cyan>{extra[module]}</cyan> | "
    "<dim>{name}:{function}:{line}</dim> - "
    "<level>{message}</level>"
)

_FILE_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {extra[module]} | "
    "{name}:{function}:{line} - {message}"
)

_configured = False


def setup(
    level: str | None = None,
    *,
    log_file: str | os.PathLike[str] | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure loguru once. Subsequent calls are no-ops.

    :param level: minimum level for the stderr sink. Falls back to the
        ``HAT_LOG_LEVEL`` env var, then ``INFO``.
    :param log_file: optional file path. If given (or if ``HAT_LOG_FILE`` is
        set), all messages at DEBUG+ are also written there with rotation.
    """
    global _configured
    if _configured:
        return

    # ``.env`` is the canonical place users set ``HAT_LOG_LEVEL`` /
    # ``HAT_LOG_FILE``. Pydantic-Settings only loads it into ``Settings``
    # instances, not into ``os.environ``, so logging (which fires at
    # import time, before ``Settings`` is built) would otherwise miss it.
    # Walk up from cwd looking for the first ``.env`` and merge it in
    # without clobbering values already set in the real environment.
    try:
        from dotenv import load_dotenv

        for parent in (Path.cwd(), *Path.cwd().parents):
            candidate = parent / ".env"
            if candidate.is_file():
                load_dotenv(candidate, override=False)
                break
    except ImportError:  # python-dotenv missing — tolerate gracefully
        pass

    level = (level or os.environ.get("HAT_LOG_LEVEL") or "INFO").upper()
    logger.remove()
    # Bind a default ``module`` so records emitted via the raw ``logger``
    # (i.e. without going through ``get_logger``) still render correctly.
    logger.configure(extra={"module": "hat"})
    logger.add(
        sys.stderr,
        level=level,
        format=_FMT,
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )

    file_path = log_file or os.environ.get("HAT_LOG_FILE")
    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(path),
            level="DEBUG",
            format=_FILE_FMT,
            rotation=rotation,
            retention=retention,
            backtrace=True,
            diagnose=False,
            enqueue=True,
        )

    _configured = True


def get_logger(name: str | None = None, **extra: Any):
    """Return a logger bound with ``module=name`` (and any extra fields).

    Pass ``__name__`` from each module so messages carry their origin::

        log = get_logger(__name__)
    """
    if not _configured:
        setup()
    return logger.bind(module=name or "hat", **extra)


# --------------------------------------------------------------------------
# Conversation / payload pretty-printers
# --------------------------------------------------------------------------
# Used by every component that talks to an LLM (cortex chat, abstractor
# triage/route, oracle, …) so the exact prompt — with newlines preserved
# and role boundaries clearly marked — shows up in the logs at DEBUG.
# Wrap large fields in :func:`truncate` so a single multi-megabyte trace
# does not flood the terminal.


_ROLE_GLYPHS: dict[str, str] = {
    "system": "⚙",
    "user": "👤",
    "assistant": "🤖",
    "tool": "🔧",
    "function": "🔧",
}


def _coerce_message(msg: Any) -> tuple[str, str]:
    """Return ``(role, content)`` for either a dict or a pydantic-ish object."""
    if isinstance(msg, Mapping):
        role = str(msg.get("role") or "?")
        content = msg.get("content")
    else:
        role = str(getattr(msg, "role", "?") or "?")
        content = getattr(msg, "content", "")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = str(content)
    return role, content


def truncate(text: str, *, limit: int = 4000) -> str:
    """Clip ``text`` to ``limit`` chars with a trailing "+N more" marker.

    ``limit <= 0`` disables truncation. Newlines are preserved as-is.
    """
    if not text or limit <= 0 or len(text) <= limit:
        return text or ""
    return text[:limit] + f"\n… (+{len(text) - limit} more chars)"


def format_messages(
    messages: Sequence[Any] | Iterable[Any],
    *,
    title: str | None = None,
    max_chars_per_message: int = 2000,
) -> str:
    """Render a chat-completions-style message list as a multi-line block.

    Each message is printed with a role marker (``system``/``user``/
    ``assistant``/``tool``); content is indented and **newlines are
    preserved** so the exact prompt sent to the model is readable in the
    log. Per-message content longer than ``max_chars_per_message`` is
    truncated with a ``+N more`` marker (set to ``0`` to disable).

    Intended use::

        log.debug("cortex.chat\\n{}", format_messages(messages, title="cortex.chat"))
    """
    msgs = list(messages or [])
    header = f"┌─ {title} ({len(msgs)} msg{'s' if len(msgs) != 1 else ''})" if title \
        else f"┌─ messages ({len(msgs)})"
    lines: list[str] = [header]
    for idx, raw in enumerate(msgs):
        role, content = _coerce_message(raw)
        glyph = _ROLE_GLYPHS.get(role.lower(), "•")
        lines.append(f"├─ [{idx}] {glyph} {role}")
        body = truncate(content, limit=max_chars_per_message)
        if body == "":
            lines.append("│  (empty)")
        else:
            for ln in body.splitlines() or [""]:
                lines.append(f"│  {ln}")
    lines.append("└─")
    return "\n".join(lines)


def format_text_block(text: str, *, title: str | None = None, max_chars: int = 4000) -> str:
    """Render a single multi-line string inside a labelled box.

    Useful for logging a raw LLM response (newlines preserved, optionally
    clipped). Pair with :func:`format_messages` for request/response logs.
    """
    body = truncate(text or "", limit=max_chars)
    header = f"┌─ {title}" if title else "┌─"
    lines = [header]
    if body == "":
        lines.append("│  (empty)")
    else:
        for ln in body.splitlines() or [""]:
            lines.append(f"│  {ln}")
    lines.append("└─")
    return "\n".join(lines)


__all__ = [
    "get_logger",
    "logger",
    "setup",
    "format_messages",
    "format_text_block",
    "truncate",
]
