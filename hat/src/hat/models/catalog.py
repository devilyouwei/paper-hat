"""Model catalog loading.

The catalog is a list of ``CatalogEntry`` objects describing models the user
*can* download. Defaults ship as package data under
``hat/models/catalogs/<backend>.yaml`` and can be overridden per project by
placing a YAML with the same shape at ``<HAT_MODEL_ROOT>/<backend>/catalog.yaml``.

This module is import-light: no torch / mlx imports.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel

from ..config.settings import get_settings

SUPPORTED_BACKENDS: tuple[str, ...] = ("mlx", "hf")
SUPPORTED_EMBED_BACKENDS: tuple[str, ...] = ("mlx_embed", "hf_embed")
ALL_SUPPORTED_BACKENDS: tuple[str, ...] = SUPPORTED_BACKENDS + SUPPORTED_EMBED_BACKENDS


class CatalogEntry(BaseModel):
    id: str
    repo_id: str
    display: str
    size_gb: float | None = None
    notes: str | None = None


def _override_path(backend: str) -> Path:
    return get_settings().model_root / backend / "catalog.yaml"


def _read_yaml(text: str) -> list[CatalogEntry]:
    raw = yaml.safe_load(text) or []
    return [CatalogEntry(**e) for e in raw]


def load_catalog(backend: str) -> list[CatalogEntry]:
    """Return the catalog for ``backend``. Override file wins over defaults."""
    override = _override_path(backend)
    if override.exists():
        return _read_yaml(override.read_text(encoding="utf-8"))
    try:
        text = (
            resources.files("hat.models.catalogs")
            .joinpath(f"{backend}.yaml")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return []
    return _read_yaml(text)


__all__ = [
    "CatalogEntry",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_EMBED_BACKENDS",
    "ALL_SUPPORTED_BACKENDS",
    "load_catalog",
]
