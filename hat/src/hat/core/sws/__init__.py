"""SWS Trainer — slow-wave-sleep parameter updates (paper §3.7)."""

from hat.abstract.sws import SWSTrainer

from .trainer import DryRunTrainer

__all__ = ["DryRunTrainer", "SWSTrainer"]
