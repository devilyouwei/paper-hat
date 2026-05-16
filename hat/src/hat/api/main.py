from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config.settings import get_settings
from .routers import chat as chat_router
from .routers import models as models_router
from .routers import neocortex as neocortex_router
from .routers import openai_compat as openai_router
from .routers import sessions as sessions_router

_STATIC_DIR = Path(__file__).resolve().parents[1] / "ui"


def create_app() -> FastAPI:
    app = FastAPI(title="HAT", version="0.0.1")
    app.include_router(chat_router.router, prefix="/chat", tags=["chat"])
    app.include_router(openai_router.router, prefix="/v1", tags=["openai"])
    app.include_router(models_router.router, prefix="/api/models", tags=["models"])
    app.include_router(
        sessions_router.router, prefix="/api/sessions", tags=["sessions"]
    )
    app.include_router(
        neocortex_router.router, prefix="/api/neocortex", tags=["neocortex"]
    )

    # Static frontend (vanilla HTML/CSS/JS), served from the same origin so
    # ``make serve`` is the only command needed.
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR)), name="ui",)

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

        The Memory tab consumes this to render the scoring formula
        ``score = αU + βF + γN`` with the actual coefficients and the
        threshold currently in use, plus the oracle trigger settings.
        Read-only — there is no setter; tweak via env (``HAT_*``) and
        restart.
        """
        s = get_settings()
        return {
            "write_policy": {
                "kind": "linear",
                "alpha": s.alpha,
                "beta": s.beta,
                "gamma": s.gamma,
                "threshold": s.write_threshold,
                "feedback_bypass": True,  # F=1 force-accepts (paper §3.4.2)
            },
            "oracle": {
                "enabled": s.oracle_enabled,
                "model": s.oracle_model if s.oracle_enabled else None,
                "threshold": s.oracle_threshold,
                "rps": s.oracle_rps,
                "daily_calls": s.oracle_daily_calls,
            },
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        # Browsers request /favicon.ico by default; serve the bundled .ico
        # straight from the static directory.
        return FileResponse(_STATIC_DIR / "favicon.ico", media_type="image/x-icon")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    return app


app = create_app()
