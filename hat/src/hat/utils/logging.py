from __future__ import annotations

import sys

from loguru import logger

_configured = False


def setup(level: str = "INFO") -> None:
    """Configure loguru once. Idempotent."""
    global _configured
    if _configured:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    _configured = True


__all__ = ["logger", "setup"]
