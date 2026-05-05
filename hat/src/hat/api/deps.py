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
    LLMAbstractor,
    LinearWritePolicy,
    SupervisedReplayBuilder,
)
from ..core.hippocampus.scoring import (
    AlwaysNovel,
    BinaryFeedback,
    ConstantUncertainty,
    LLMFeedbackJudge,
    LLMNoveltyJudge,
    LogprobUncertainty,
)
from ..core.loop import WakeSleepLoop
from ..core.oracle import (
    CostGuard,
    OpenAICompatibleOracle,
    Oracle,
)
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
    cortex = get_cortex()
    return WakeSleepLoop(
        cortex=cortex,
        abstractor=_make_abstractor(cortex),
        uncertainty=_make_uncertainty(cortex),
        feedback=_make_feedback(cortex),
        novelty=_make_novelty(cortex),
        write_policy=LinearWritePolicy(s.alpha, s.beta, s.gamma, s.write_threshold),
        replay_builder=SupervisedReplayBuilder(),
        neocortex=JsonlNeocortex(
            s.neocortex_path, traces_path=s.neocortex_traces_path
        ),
        trainer=DryRunTrainer(),
        oracle=_make_oracle(),
        oracle_threshold=s.oracle_threshold,
    )


# ---- hippocampus component factories ----------------------------------
#
# The wake/sleep loop is constructed once and reused across requests, but the
# active Cortex can be hot-swapped via the management API. ``swap_active_cortex``
# below rebuilds the LLM-backed scorers/abstractor in place so the loop's
# uncertainty/novelty/feedback estimators always point at the current model.
# Stub implementations are used when the cortex is the noop fallback so we
# don't burn forward passes on placeholder text.


def _is_noop(cortex: Cortex) -> bool:
    return isinstance(cortex, NoopCortex)


def _make_abstractor(cortex: Cortex):
    return IdentityAbstractor() if _is_noop(cortex) else LLMAbstractor(cortex)


def _make_uncertainty(cortex: Cortex):
    if _is_noop(cortex):
        return ConstantUncertainty(0.5)
    return LogprobUncertainty(cortex)


def _make_feedback(cortex: Cortex):
    if _is_noop(cortex):
        return BinaryFeedback()
    return LLMFeedbackJudge(cortex)


def _make_novelty(cortex: Cortex):
    if _is_noop(cortex):
        return AlwaysNovel()
    return LLMNoveltyJudge(cortex)


# ---- oracle factory ---------------------------------------------------
#
# Oracle is opt-in via ``HAT_ORACLE_ENABLED``. When disabled (the default)
# the loop runs purely local. When enabled and an API key is present, we
# build an OpenAI-compatible client wrapped in a process-wide CostGuard so
# rate and daily-budget limits are uniform across requests.

_oracle_cost_guard: CostGuard | None = None


def _make_oracle() -> Oracle | None:
    s = get_settings()
    if not s.oracle_enabled:
        return None
    # No key, no oracle (local servers can ignore keys, but defaulting to
    # OpenAI without one would just fail every request).
    if s.oracle_base_url.startswith("https://api.openai.com") and not s.oracle_api_key:
        return None

    global _oracle_cost_guard
    if _oracle_cost_guard is None:
        _oracle_cost_guard = CostGuard(
            rps=s.oracle_rps,
            daily_calls=s.oracle_daily_calls,
            audit_path=s.oracle_audit_path,
        )
    return OpenAICompatibleOracle(
        base_url=s.oracle_base_url,
        model=s.oracle_model,
        api_key=s.oracle_api_key,
        cost_guard=_oracle_cost_guard,
    )


def _refresh_hippocampus(loop: WakeSleepLoop, cortex: Cortex) -> None:
    """Re-bind the hippocampus scorers/abstractor to ``cortex`` in place.

    Called after a model swap so logprob uncertainty / LLM judges talk to
    the new Cortex instead of the previous one.
    """
    loop.abstractor = _make_abstractor(cortex)
    loop.uncertainty = _make_uncertainty(cortex)
    loop.feedback = _make_feedback(cortex)
    loop.novelty = _make_novelty(cortex)


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
    _refresh_hippocampus(get_loop(), cortex)
    return cortex


def deactivate_cortex() -> int:
    """Unload every cached cortex and point the loop at the Noop fallback.

    Returns the number of evicted entries so callers can report back.
    """
    n = get_manager().unload_all()
    fallback = NoopCortex()
    get_loop().cortex = fallback
    _refresh_hippocampus(get_loop(), fallback)
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
