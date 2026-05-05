"""Oracle — on-demand external teacher (paper §3.5)."""

from .base import NoopOracle, Oracle
from .cost_guard import CostGuard, OracleQuotaExceeded
from .openai_compat import OpenAICompatibleOracle

__all__ = [
    "Oracle",
    "NoopOracle",
    "OpenAICompatibleOracle",
    "CostGuard",
    "OracleQuotaExceeded",
]
