"""Oracle — on-demand external teacher (paper §3.5)."""

from hat.abstract.oracle import Oracle
from .cost_guard import CostGuard, OracleQuotaExceeded
from .noop import NoopOracle
from .openai_compat import OpenAICompatibleOracle

__all__ = [
    "Oracle",
    "NoopOracle",
    "OpenAICompatibleOracle",
    "CostGuard",
    "OracleQuotaExceeded",
]
