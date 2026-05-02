"""Dependency container.

Wires concrete defaults into the loop. Controllers depend only on Protocols/ABCs,
so swapping backends never touches HTTP code."""

from __future__ import annotations

from functools import lru_cache

from ..config.settings import get_settings
from ..core.cortex.base import Cortex
from ..core.cortex.noop import NoopCortex
from ..core.hippocampus import (
    IdentityAbstractor,
    LinearWritePolicy,
    SupervisedReplayBuilder,
)
from ..core.hippocampus.scoring import (
    AlwaysNovel,
    BinaryFeedback,
    ConstantUncertainty,
)
from ..core.loop import WakeSleepLoop
from ..core.neocortex.store import InMemoryNeocortex
from ..core.sws.trainer import DryRunTrainer
from ..memory.raw.log import JsonlRawLog


@lru_cache
def get_cortex() -> Cortex:
    s = get_settings()
    if s.cortex_backend == "noop":
        return NoopCortex()
    if s.cortex_backend == "hf":
        # Lazy import: torch/transformers only required when actually used.
        from ..core.cortex.hf_cortex import HFCortex
        from ..models.backends import hf as _hf  # registers backend

        lm = _hf.build_hf_model(
            model_path=s.hf_model_path,
            device=s.hf_device,
            dtype=s.hf_dtype,
            max_new_tokens=s.hf_max_new_tokens,
            temperature=s.hf_temperature,
        )
        return HFCortex(lm)
    if s.cortex_backend == "mlx":
        # Apple Silicon native; lazy import so non-mac installs are fine.
        from ..core.cortex.mlx_cortex import MLXCortex
        from ..models.backends import mlx as _mlx  # registers backend

        lm = _mlx.build_mlx_model(
            model_path=s.mlx_model_path,
            max_tokens=s.mlx_max_tokens,
            temperature=s.mlx_temperature,
        )
        return MLXCortex(lm)
    raise ValueError(f"unknown HAT_CORTEX_BACKEND={s.cortex_backend!r}")


@lru_cache
def get_loop() -> WakeSleepLoop:
    s = get_settings()
    return WakeSleepLoop(
        cortex=get_cortex(),
        abstractor=IdentityAbstractor(),
        uncertainty=ConstantUncertainty(0.5),
        feedback=BinaryFeedback(),
        novelty=AlwaysNovel(),
        write_policy=LinearWritePolicy(s.alpha, s.beta, s.gamma, s.write_threshold),
        replay_builder=SupervisedReplayBuilder(),
        neocortex=InMemoryNeocortex(),
        trainer=DryRunTrainer(),
        oracle=None,
        oracle_threshold=s.oracle_threshold,
    )


@lru_cache
def get_raw_log() -> JsonlRawLog:
    return JsonlRawLog(get_settings().raw_log_path)
