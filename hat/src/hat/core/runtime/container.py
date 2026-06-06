"""Service container.

Owns the long-lived singletons used by every controller: the wake/sleep
loop, session store, raw interaction log. Wires concrete defaults into
the loop so HTTP code never instantiates protocols directly.

Active-model selection is delegated to :class:`hat.models.manager.ModelManager`
so the management API can hot-swap the Cortex without restarting the
server. ``get_cortex()`` returns the manager's active model when one has
been selected, otherwise the env-driven bootstrap Cortex (noop / hf /
mlx).
"""

from __future__ import annotations

from functools import lru_cache

from hat.config.settings import get_settings
from hat.abstract.cortex import Cortex
from hat.core.cortex.noop import NoopCortex
from hat.core.hippocampus import (
    EmbeddingDeduper,
    IdentityAbstractor,
    LLMAbstractor,
    SupervisedReplayBuilder,
    UncertaintyGatePolicy,
)
from hat.core.hippocampus.scoring import (
    ConstantUncertainty,
    LogprobUncertainty,
)
from hat.core.loop import WakeSleepLoop
from hat.core.oracle import (
    CostGuard,
    OpenAICompatibleOracle,
    Oracle,
)
from hat.core.sws.trainer import DryRunTrainer
from hat.core.neocortex.jsonl_store import JsonlNeocortex
from hat.core.neocortex.vector_index import NpzVectorIndex
from hat.core.neocortex.embeddings.managed import Embedder, ManagedEmbedder
from hat.core.sessions.raw_log import SessionRawLog
from hat.core.sessions.store import JsonlSessionStore
from hat.core.lifecycle.embedding_manager import get_embedding_manager
from hat.core.lifecycle.manager import get_manager


def _bootstrap_cortex() -> Cortex:
    """Build the initial Cortex.

    For ``hf``/``mlx``/``cloud`` we auto-activate the first installed catalog
    entry under ``model/<backend>/``. Cloud entries are always "installed"
    (they call a remote API), so building the cortex makes no network call.
    If nothing is installed, fall back to the noop Cortex — the user must
    download a model from the UI / ``/api/models`` before chat works. There is
    no env-driven model path any more; weights live under
    ``model/<backend>/<id>/`` by convention."""
    s = get_settings()
    if s.cortex_backend in {"mlx", "hf", "cloud"}:
        mgr = get_manager()
        for entry in mgr.list_models(s.cortex_backend):
            if entry["installed"]:
                return mgr.set_active(s.cortex_backend, entry["id"])
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
    neocortex = JsonlNeocortex(
        s.neocortex_path, traces_path=s.neocortex_traces_path
    )
    return WakeSleepLoop(
        cortex=cortex,
        abstractor=_make_abstractor(cortex, neocortex=neocortex),
        uncertainty=_make_uncertainty(cortex),
        write_policy=UncertaintyGatePolicy(s.write_threshold),
        replay_builder=SupervisedReplayBuilder(),
        neocortex=neocortex,
        trainer=DryRunTrainer(),
        oracle=_make_oracle(),
        oracle_threshold=s.oracle_threshold,
        deduper=_make_deduper(),
        embed_tag=_active_embed_tag(),
    )


# ---- hippocampus component factories ----------------------------------


def _is_noop(cortex: Cortex) -> bool:
    return isinstance(cortex, NoopCortex)


def _make_abstractor(cortex: Cortex, *, neocortex=None):
    if _is_noop(cortex):
        return IdentityAbstractor()
    return LLMAbstractor(cortex, neocortex=neocortex)


def _make_uncertainty(cortex: Cortex):
    if _is_noop(cortex):
        return ConstantUncertainty(0.5)
    return LogprobUncertainty(cortex)


@lru_cache
def _get_vector_index_for(backend: str, model_id: str) -> NpzVectorIndex:
    """Per-(backend, id) NPZ index for managed embedders.

    Cached so repeat ``_make_deduper()`` calls under the same active
    embedder reuse the same in-memory ``NpzVectorIndex`` (and therefore
    its lock).
    """
    from hat.config.settings import embed_index_path_for

    return NpzVectorIndex(embed_index_path_for(backend, model_id))


def _active_embedder() -> tuple[Embedder, NpzVectorIndex, str] | None:
    """Resolve (embedder, index, tag) for the wake-step deduper.

    Returns ``None`` when no managed embedder has been activated via
    ``/api/embedding-models/active``; the loop's deduper is then ``None``
    and dedup-driven REVISE routing is skipped for the turn.
    """
    mgr = get_embedding_manager()
    active = mgr.active()
    if active is None:
        return None
    backend, model_id = active["backend"], active["id"]
    emb = mgr.load(backend, model_id)
    idx = _get_vector_index_for(backend, model_id)
    tag = emb.tag if isinstance(emb, ManagedEmbedder) else f"{backend}/{model_id}"
    return emb, idx, tag


def _make_deduper() -> EmbeddingDeduper | None:
    s = get_settings()
    if not s.dedup_enabled:
        return None
    pair = _active_embedder()
    if pair is None:
        return None
    embedder, index, _tag = pair
    return EmbeddingDeduper(
        embedder=embedder,
        index=index,
        threshold=s.dedup_threshold,
    )


def _active_embed_tag() -> str | None:
    """Tag for ``metadata.extras.embed_model`` on accepted traces, or
    ``None`` when no managed embedder is active (in which case rows are
    not stamped)."""
    pair = _active_embedder()
    return pair[2] if pair is not None else None


# ---- oracle factory ---------------------------------------------------

_oracle_cost_guard: CostGuard | None = None


def _make_oracle() -> Oracle | None:
    s = get_settings()
    if not s.oracle_enabled:
        return None
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
    """Re-bind the hippocampus abstractor / uncertainty to ``cortex`` in
    place. Called after a model swap so the LLM-backed components talk to
    the new Cortex instead of the previous one."""
    loop.abstractor = _make_abstractor(cortex, neocortex=loop.neocortex)
    loop.uncertainty = _make_uncertainty(cortex)


def swap_active_cortex(backend: str, model_id: str) -> Cortex:
    """Set the manager's active model and update the loop in place.

    The loop holds a strong reference to the current cortex; we must drop
    it *before* the manager builds the new model, otherwise on CUDA the
    old weights stay resident during the new load and we OOM.
    """
    mgr = get_manager()
    active = mgr.active()
    if active and (active["backend"], active["id"]) != (backend, model_id):
        get_loop().cortex = NoopCortex()
    cortex = mgr.set_active(backend, model_id)
    get_loop().cortex = cortex
    _refresh_hippocampus(get_loop(), cortex)
    return cortex


def deactivate_cortex() -> int:
    """Unload every cached cortex and point the loop at the Noop fallback."""
    fallback = NoopCortex()
    loop = get_loop()
    loop.cortex = fallback
    _refresh_hippocampus(loop, fallback)
    _initial_cortex.cache_clear()
    return get_manager().unload_all()


# ---- embedder swap ----------------------------------------------------


def swap_active_embedder(backend: str, model_id: str) -> Embedder:
    """Activate a managed embedder and rebuild the loop's deduper."""
    mgr = get_embedding_manager()
    emb = mgr.set_active(backend, model_id)
    _rebuild_deduper()
    return emb


def deactivate_embedder() -> int:
    """Unload all managed embedders; loop's deduper drops to ``None``."""
    n = get_embedding_manager().unload_all()
    _rebuild_deduper()
    return n


def _rebuild_deduper() -> None:
    """Replace ``loop.deduper`` with one bound to the current active pair."""
    loop = get_loop()
    loop.deduper = _make_deduper()
    loop.embed_tag = _active_embed_tag()


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


def prior_traces_for_session(session_id: str, *, limit: int = 8) -> list:
    """Return up to ``limit`` most recent :class:`MemoryTrace` for a session.

    Returns an empty list if the active neocortex backend doesn't support
    session lookup, so the caller can simply pass the result through to
    ``wake_step(prior_traces=...)`` without further checks.
    """
    if not session_id:
        return []
    store = get_loop().neocortex
    fetch = getattr(store, "entries_by_session", None)
    if fetch is None:
        return []
    rows = fetch(session_id)
    if limit and len(rows) > limit:
        rows = rows[-limit:]
    from hat.core.neocortex.jsonl_store import _sft_to_trace
    out: list = []
    for r in rows:
        try:
            trace, _ = _sft_to_trace(r)
        except (ValueError, TypeError, KeyError):
            continue
        out.append(trace)
    return out
