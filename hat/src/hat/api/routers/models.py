"""Model management endpoints.

* ``GET    /api/models?backend=`` — catalog + installed status
* ``POST   /api/models/download``  — snapshot_download a catalog entry
* ``POST   /api/models/active``    — load + set as the loop's active Cortex
* ``GET    /api/models/active``    — current active model
* ``DELETE /api/models/active``    — unload all models, free memory
* ``DELETE /api/models/{backend}/{id}`` — remove installed weights from disk
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...models.manager import ModelManagerError, get_manager
from ..deps import deactivate_cortex, swap_active_cortex
from ..schemas.models import (
    ActiveModel,
    CatalogItem,
    ModelActionRequest,
    ModelDownloadResponse,
    ModelListResponse,
)

router = APIRouter()


@router.get("", response_model=ModelListResponse)
def list_models(backend: str = Query(default="mlx")) -> ModelListResponse:
    try:
        items = get_manager().list_models(backend)
    except ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    return ModelListResponse(
        backend=backend,  # type: ignore[arg-type]
        items=[CatalogItem(**i) for i in items],
    )


@router.post("/download", response_model=ModelDownloadResponse)
def download_model(req: ModelActionRequest) -> ModelDownloadResponse:
    try:
        path = get_manager().download(req.backend, req.id)
    except ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # pragma: no cover - network errors surface here
        raise HTTPException(500, f"download failed: {e}") from e
    return ModelDownloadResponse(
        backend=req.backend, id=req.id, local_dir=str(path)
    )


@router.post("/active", response_model=ActiveModel)
def set_active(req: ModelActionRequest) -> ActiveModel:
    try:
        cortex = swap_active_cortex(req.backend, req.id)
    except ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # backend load errors (OOM, missing extras, ...)
        # Without this, FastAPI converts the exception to a bare 500 with no
        # body, and the UI shows nothing useful. Surface the type + message
        # so the toast can tell the user *why* activation failed.
        raise HTTPException(
            500, f"activate failed: {type(e).__name__}: {e}"
        ) from e
    return ActiveModel(
        backend=req.backend, id=req.id, name=getattr(cortex, "name", None)
    )


@router.get("/active", response_model=ActiveModel | None)
def get_active() -> ActiveModel | None:
    a = get_manager().active()
    if a is None:
        return None
    return ActiveModel(backend=a["backend"], id=a["id"])


@router.delete("/active")
def deactivate() -> dict:
    """Unload the current model (and any other cached weights), freeing
    GPU/Metal memory. The loop falls back to the Noop cortex until a new
    model is activated."""
    n = deactivate_cortex()
    return {"unloaded": n}


@router.delete("/{backend}/{model_id:path}")
def delete_model(backend: str, model_id: str) -> dict:
    """Remove an installed model's weights from disk. Refuses if it's
    currently the active model."""
    try:
        removed = get_manager().delete(backend, model_id)
    except ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    return {"backend": backend, "id": model_id, "removed": removed}
