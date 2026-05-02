"""Per-signal scorers used by the linear write policy.

Each scorer returns a scalar in roughly ``[0, 1]``. Bring your own implementation
for production use; defaults here are minimal placeholders so the loop runs."""

from .feedback import BinaryFeedback, FeedbackExtractor
from .novelty import AlwaysNovel, NoveltyEstimator
from .uncertainty import ConstantUncertainty, UncertaintyEstimator

__all__ = [
    "UncertaintyEstimator",
    "ConstantUncertainty",
    "FeedbackExtractor",
    "BinaryFeedback",
    "NoveltyEstimator",
    "AlwaysNovel",
]
