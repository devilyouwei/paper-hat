from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config.settings import get_settings
from .controllers import chat as chat_controller
from .controllers import embedding_models as embedding_models_controller
from .controllers import models as models_controller
from .controllers import neocortex as neocortex_controller
from .controllers import openai as openai_controller
from .controllers import sessions as sessions_controller

_STATIC_DIR = Path(__file__).resolve().parents[1] / "ui"


def _dedup_policy(s) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Build the ``dedup`` block for ``/api/policy``.

    Includes the active managed embedder (if any) and the per-model NPZ
    path so the UI can show which embedding store rows are landing in.
    Without an active embedder, dedup is effectively disabled even when
    ``dedup_enabled`` is true.
    """
    from ..config.settings import embed_index_path_for  # noqa: PLC0415
    from hat.core.lifecycle.embedding_manager import get_embedding_manager  # noqa: PLC0415

    active = get_embedding_manager().active()
    if active is not None:
        active_payload: dict[str, str] | None = {
            "backend": active["backend"], "id": active["id"],
        }
        index_path: str | None = str(
            embed_index_path_for(active["backend"], active["id"])
        )
    else:
        active_payload = None
        index_path = None
    return {
        "enabled": s.dedup_enabled,
        "threshold": s.dedup_threshold,
        "active_embedder": active_payload,
        "index_path": index_path,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="HAT", version="0.0.1")
    app.include_router(chat_controller.router, prefix="/chat", tags=["chat"])
    app.include_router(openai_controller.router, prefix="/v1", tags=["openai"])
    app.include_router(
        models_controller.router, prefix="/api/models", tags=["models"]
    )
    app.include_router(
        embedding_models_controller.router,
        prefix="/api/embedding-models",
        tags=["embedding-models"],
    )
    app.include_router(
        sessions_controller.router, prefix="/api/sessions", tags=["sessions"]
    )
    app.include_router(
        neocortex_controller.router, prefix="/api/neocortex", tags=["neocortex"]
    )

    # Static frontends. The new Vue SPA lives in ``ui/dist`` (built by
    # ``make ui-build``); the legacy vanilla UI remains under ``/ui``.
    _DIST_DIR = _STATIC_DIR / "dist"
    if _DIST_DIR.is_dir():
        app.mount(
            "/ui/dist",
            StaticFiles(directory=str(_DIST_DIR)),
            name="ui-dist",
        )
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR)), name="ui")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        s = get_settings()
        return {
            "status": "ok",
            "cortex_backend": s.cortex_backend,
            "model_root": str(s.model_root),
        }

    @app.get("/api/policy")
    def policy() -> dict[str, object]:
        """Expose the live hippocampus selection policy and oracle config.

        The Memory tab consumes this to render the scoring rule
        ``score = U`` together with the gate threshold currently in use
        and the oracle trigger settings. Read-only — there is no setter;
        tweak via env (``HAT_*``) and restart.
        """
        s = get_settings()
        return {
            "write_policy": {
                "kind": "uncertainty_gate",
                "threshold": s.write_threshold,
            },
            "oracle": {
                "enabled": s.oracle_enabled,
                "model": s.oracle_model if s.oracle_enabled else None,
                "threshold": s.oracle_threshold,
                "rps": s.oracle_rps,
                "daily_calls": s.oracle_daily_calls,
            },
            "dedup": _dedup_policy(s),
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        # Browsers request /favicon.ico by default; serve the bundled .ico
        # straight from the static directory.
        return FileResponse(_STATIC_DIR / "favicon.ico", media_type="image/x-icon")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        spa_index = _DIST_DIR / "index.html"
        if spa_index.is_file():
            return FileResponse(spa_index, media_type="text/html")
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/legacy", include_in_schema=False)
    def legacy_index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    return app


app = create_app()
