"""Dependency container.

Wires concrete defaults into the loop. Controllers depend only on Protocols/ABCs,
so swapping backends never touches HTTP code.

Active-model selection is delegated to :class:`hat.models.manager.ModelManager`
so the UI / management API can hot-swap the Cortex without restarting the
server. ``get_cortex()`` returns the manager's active model when one has been
selected, otherwise the env-driven bootstrap Cortex (noop / hf / mlx)."""

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
from ..core.sws.trainer import DryRunTrainer
from ..memory.curated.jsonl_store import JsonlNeocortex
from ..memory.raw.log import SessionRawLog
from ..memory.raw.sessions import JsonlSessionStore
from ..models.manager import get_manager


def _bootstrap_cortex() -> Cortex:
    """Build the initial Cortex.

    For ``hf``/``mlx`` we auto-activate the first installed catalog entry
    under ``model/<backend>/``. If nothing is installed, fall back to the
    noop Cortex — the user must download a model from the UI / ``/api/models``
    before chat works. There is no env-driven model path any more; weights
    live under ``model/<backend>/<id>/`` by convention."""
    s = get_settings()
    if s.cortex_backend in {"mlx", "hf"}:
        mgr = get_manager()
        for entry in mgr.list_models(s.cortex_backend):
            if entry["installed"]:
                return mgr.set_active(s.cortex_backend, entry["id"])
        # nothing installed yet — degrade gracefully
        return NoopCortex()
    if s.cortex_backend == "noop":
        return NoopCortex()
    raise ValueError(f"unknown HAT_CORTEX_BACKEND={s.cortex_backend!r}")


@lru_cache
def _initial_cortex() -> Cortex:
    return _bootstrap_cortex()


def get_cortex() -> Cortex:
    """Active Cortex: manager override if set, else the env-driven bootstrap."""
    if get_manager().active() is not None:
        backend, model_id = get_manager()._active  # type: ignore[union-attr]
        return get_manager().load(backend, model_id)
    return _initial_cortex()


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
        neocortex=JsonlNeocortex(
            s.neocortex_path, traces_path=s.neocortex_traces_path
        ),
        trainer=DryRunTrainer(),
        oracle=None,
        oracle_threshold=s.oracle_threshold,
    )


def swap_active_cortex(backend: str, model_id: str) -> Cortex:
    """Set the manager's active model and update the loop in place.

    The loop holds a strong reference to the current cortex; we must drop it
    *before* the manager builds the new model, otherwise on CUDA the old
    weights stay resident during the new load and we OOM.
    """
    mgr = get_manager()
    active = mgr.active()
    if active and (active["backend"], active["id"]) != (backend, model_id):
        # Park the loop on a placeholder so it stops referencing the old cortex.
        get_loop().cortex = NoopCortex()
    cortex = mgr.set_active(backend, model_id)
    get_loop().cortex = cortex
    return cortex


def deactivate_cortex() -> int:
    """Unload every cached cortex and point the loop at the Noop fallback.

    Returns the number of evicted entries so callers can report back.
    """
    n = get_manager().unload_all()
    get_loop().cortex = NoopCortex()
    # Drop the cached bootstrap so the next chat doesn't silently reload it.
    _initial_cortex.cache_clear()
    return n


@lru_cache
def get_session_store() -> JsonlSessionStore:
    """Process-wide session store (one ``runs/raw/`` tree)."""
    return JsonlSessionStore(get_settings().raw_root)


@lru_cache
def get_raw_log() -> SessionRawLog:
    """Backwards-compatible :class:`RawInteractionLog` view of the session
    store. Callers that don't know about sessions land in the synthetic
    ``default`` session; the chat controller passes a real session id."""
    return SessionRawLog(get_session_store(), session_id=None)
