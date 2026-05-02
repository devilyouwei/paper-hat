from __future__ import annotations

from abc import ABC, abstractmethod

from ...schemas import Interaction


class FeedbackExtractor(ABC):
    """Maps explicit / implicit user supervision to a scalar."""

    @abstractmethod
    def __call__(self, interaction: Interaction) -> float: ...


class BinaryFeedback(FeedbackExtractor):
    """``1.0`` if a correction or explicit feedback exists, else ``0.0``.

    A correction always wins over a numeric ``feedback`` field so verified
    errors (paper §3.4.2 "Feedback") are guaranteed to be consolidated.
    """

    def __call__(self, interaction: Interaction) -> float:
        if interaction.user_correction:
            return 1.0
        if interaction.feedback is not None:
            return float(interaction.feedback)
        return 0.0
