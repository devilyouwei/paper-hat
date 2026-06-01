"""Scoring signals consumed by the write policy.

Only ``uncertainty`` is used to gate trace creation. Corrections surface
through the natural multi-turn conversation and are detected by the
abstractor's router prompt.
"""

from hat.abstract.hippocampus import UncertaintyEstimator

from .uncertainty import ConstantUncertainty, LogprobUncertainty

__all__ = [
    "UncertaintyEstimator",
    "ConstantUncertainty",
    "LogprobUncertainty",
]
