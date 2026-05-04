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

# Project-root assets (logo, etc.) — repo layout is hat/<asset> at the same
# level as src/. Resolve once so the path survives reloads.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOGO_PATH = _PROJECT_ROOT / "logo.png"
_STATIC_DIR = Path(__file__).resolve().parents[1] / "ui" / "static"


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
    app.mount(
        "/ui/static",
        StaticFiles(directory=str(_STATIC_DIR)),
        name="ui-static",
    )

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        s = get_settings()
        return {
            "status": "ok",
            "cortex_backend": s.cortex_backend,
            "model_root": str(s.model_root),
        }

    @app.get("/logo.png", include_in_schema=False)
    def logo() -> FileResponse:
        return FileResponse(_LOGO_PATH, media_type="image/png")

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
