"""Scoring signals consumed by the write policy.

Only ``uncertainty`` is currently used to gate trace creation. The feedback /
novelty channels were removed; corrections now surface through the natural
multi-turn conversation and are detected by the abstractor's router prompt.
"""

from .uncertainty import ConstantUncertainty, LogprobUncertainty, UncertaintyEstimator

__all__ = [
    "UncertaintyEstimator",
    "ConstantUncertainty",
    "LogprobUncertainty",
]
