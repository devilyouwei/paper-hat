"""Runtime manager for installed / active models.

* Resolves catalog entry → local directory under ``model/<backend>/<id>/``.
* Downloads from HuggingFace via ``huggingface_hub.snapshot_download``.
* Caches loaded Cortex instances so switching back to a previously-loaded
  model is instant.
* Owns the *active* (backend, id) pointer the WakeSleepLoop reads through.

The manager is process-local. If you scale to a separate sleep worker, it
keeps its own cache there — that is intentional, since each process has its
own GPU/Metal context.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from ..config.settings import get_settings
from ..core.cortex.base import Cortex
from .catalog import SUPPORTED_BACKENDS, CatalogEntry, load_catalog


class ModelManagerError(RuntimeError):
    pass


class ModelManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._cache: dict[tuple[str, str], Cortex] = {}
        self._active: tuple[str, str] | None = None

    # ---------- paths ----------------------------------------------------

    def model_dir(self, backend: str, model_id: str) -> Path:
        return get_settings().model_root / backend / model_id

    def is_installed(self, backend: str, model_id: str) -> bool:
        d = self.model_dir(backend, model_id)
        if not d.is_dir():
            return False
        # Treat a directory containing at least one weight-shaped file as
        # installed. Avoids importing huggingface_hub just to introspect.
        for child in d.iterdir():
            if child.suffix in {".safetensors", ".bin", ".gguf"}:
                return True
        return False

    # ---------- listing --------------------------------------------------

    def list_models(self, backend: str) -> list[dict]:
        if backend not in SUPPORTED_BACKENDS:
            raise ModelManagerError(f"unsupported backend {backend!r}")
        out: list[dict] = []
        for entry in load_catalog(backend):
            out.append(
                {
                    **entry.model_dump(),
                    "backend": backend,
                    "installed": self.is_installed(backend, entry.id),
                    "local_dir": str(self.model_dir(backend, entry.id)),
                }
            )
        return out

    def _entry(self, backend: str, model_id: str) -> CatalogEntry:
        for e in load_catalog(backend):
            if e.id == model_id:
                return e
        raise ModelManagerError(
            f"unknown model id {model_id!r} for backend {backend!r}"
        )

    # ---------- download -------------------------------------------------

    def download(self, backend: str, model_id: str) -> Path:
        entry = self._entry(backend, model_id)
        dst = self.model_dir(backend, model_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # pragma: no cover
            raise ModelManagerError(
                "huggingface_hub is required for downloads"
            ) from e
        # Pin the HF blob cache inside the project's model root so weights
        # are not duplicated under ~/.cache/huggingface. ``local_dir`` keeps
        # the snapshot layout flat under model/<backend>/<id>/.
        cache_root = get_settings().model_root / ".hf-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=entry.repo_id,
            local_dir=str(dst),
            cache_dir=str(cache_root),
        )
        return dst

    # ---------- load + activate -----------------------------------------

    def _build_cortex(self, backend: str, path: str) -> Cortex:
        s = get_settings()
        if backend == "mlx":
            from ..core.cortex.mlx_cortex import MLXCortex
            from .backends.mlx import build_mlx_model

            return MLXCortex(build_mlx_model(path))
        if backend == "hf":
            from ..core.cortex.hf_cortex import HFCortex
            from .backends.hf import build_hf_model

            return HFCortex(
                build_hf_model(path, device=s.hf_device, dtype=s.hf_dtype)
            )
        raise ModelManagerError(f"backend {backend!r} cannot host a model")

    def load(self, backend: str, model_id: str) -> Cortex:
        key = (backend, model_id)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if not self.is_installed(backend, model_id):
                raise ModelManagerError(
                    f"model {model_id!r} not installed; download it first"
                )
            cortex = self._build_cortex(backend, str(self.model_dir(backend, model_id)))
            self._cache[key] = cortex
            return cortex

    def set_active(self, backend: str, model_id: str) -> Cortex:
        cortex = self.load(backend, model_id)
        with self._lock:
            self._active = (backend, model_id)
        return cortex

    def active(self) -> dict | None:
        if self._active is None:
            return None
        backend, model_id = self._active
        return {"backend": backend, "id": model_id}


_manager: ModelManager | None = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


__all__ = ["ModelManager", "ModelManagerError", "get_manager"]
