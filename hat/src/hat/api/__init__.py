"""FastAPI surface. Routers → controllers → injected protocols."""

from .main import app, create_app

__all__ = ["app", "create_app"]
