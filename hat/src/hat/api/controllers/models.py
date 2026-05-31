"""Model management endpoints (mounted at ``/api/models``)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..schemas.models import (
    ActiveModel,
    CatalogItem,
    ModelActionRequest,
    ModelDownloadResponse,
    ModelListResponse,
)
from ..services import models as svc

router = APIRouter()


@router.get("", response_model=ModelListResponse)
def list_models(backend: str = Query(default="mlx")) -> ModelListResponse:
    try:
        items = svc.list_models(backend)
    except svc.ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    return ModelListResponse(
        backend=backend,  # type: ignore[arg-type]
        items=[CatalogItem(**i) for i in items],
    )


@router.post("/download", response_model=ModelDownloadResponse)
def download_model(req: ModelActionRequest) -> ModelDownloadResponse:
    try:
        path = svc.download(req.backend, req.id)
    except svc.ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # pragma: no cover - network errors surface here
        raise HTTPException(500, f"download failed: {e}") from e
    return ModelDownloadResponse(backend=req.backend, id=req.id, local_dir=path)


@router.get("/download/stream")
def download_stream(
    backend: str = Query(...), id: str = Query(...)
) -> StreamingResponse:
    try:
        gen = svc.download_stream(backend, id)
    except svc.DownloadAlreadyRunningError as e:
        raise HTTPException(409, str(e)) from e
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/download/cancel")
def cancel_download(req: ModelActionRequest) -> dict:
    try:
        svc.cancel_download(req.backend, req.id)
    except svc.NoActiveDownloadError as e:
        raise HTTPException(404, str(e)) from e
    return {"backend": req.backend, "id": req.id, "cancelled": True}


@router.post("/active", response_model=ActiveModel)
def set_active(req: ModelActionRequest) -> ActiveModel:
    try:
        cortex = svc.activate(req.backend, req.id)
    except svc.ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # backend load errors (OOM, missing extras, ...)
        raise HTTPException(
            500, f"activate failed: {type(e).__name__}: {e}"
        ) from e
    return ActiveModel(
        backend=req.backend, id=req.id, name=getattr(cortex, "name", None)
    )


@router.get("/active", response_model=ActiveModel | None)
def get_active() -> ActiveModel | None:
    a = svc.get_active()
    if a is None:
        return None
    return ActiveModel(backend=a["backend"], id=a["id"])


@router.delete("/active")
def deactivate() -> dict:
    return {"unloaded": svc.deactivate()}


@router.delete("/{backend}/{model_id:path}")
def delete_model(backend: str, model_id: str) -> dict:
    try:
        removed = svc.delete(backend, model_id)
    except svc.ModelManagerError as e:
        raise HTTPException(400, str(e)) from e
    return {"backend": backend, "id": model_id, "removed": removed}
