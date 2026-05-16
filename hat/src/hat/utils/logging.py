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


__all__ = ["get_logger", "logger", "setup"]
