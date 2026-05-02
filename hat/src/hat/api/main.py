from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ..config.settings import get_settings
from .routers import chat as chat_router
from .routers import openai_compat as openai_router

# Project-root assets (logo, etc.) — repo layout is hat/<asset> at the same
# level as src/. Resolve once so the path survives reloads.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOGO_PATH = _PROJECT_ROOT / "logo.png"


def create_app() -> FastAPI:
    app = FastAPI(title="HAT", version="0.0.1")
    app.include_router(chat_router.router, prefix="/chat", tags=["chat"])
    app.include_router(openai_router.router, prefix="/v1", tags=["openai"])

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        s = get_settings()
        return {
            "status": "ok",
            "cortex_backend": s.cortex_backend,
            "mlx_model_path": s.mlx_model_path,
            "hf_model_path": s.hf_model_path,
        }

    @app.get("/logo.png", include_in_schema=False)
    def logo() -> FileResponse:
        return FileResponse(_LOGO_PATH, media_type="image/png")

    return app


app = create_app()
