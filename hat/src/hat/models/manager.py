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

from collections.abc import Iterator
from pathlib import Path
import shutil
import threading
from threading import Lock

from ..config.settings import get_settings
from ..core.cortex.base import Cortex
from ..utils.logging import get_logger
from .catalog import SUPPORTED_BACKENDS, CatalogEntry, load_catalog

log = get_logger(__name__)


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
        log.info(
            "downloading model backend={} id={} repo={} -> {}",
            backend, model_id, entry.repo_id, dst,
        )
        try:
            snapshot_download(
                repo_id=entry.repo_id,
                local_dir=str(dst),
                cache_dir=str(cache_root),
            )
        except Exception:
            log.exception(
                "download failed backend={} id={} repo={}",
                backend, model_id, entry.repo_id,
            )
            raise
        log.info("download complete backend={} id={}", backend, model_id)
        return dst

    # ---------- streaming download (with progress + cancel) -------------

    def download_streaming(
        self,
        backend: str,
        model_id: str,
        cancel_event: threading.Event,
        *,
        poll_interval: float = 0.5,
    ) -> Iterator[dict]:
        """Yield progress events while downloading a catalog entry.

        Events have a ``stage`` field:

        * ``start``     — total file / byte counts known
        * ``progress``  — per-poll snapshot (``bytes_done``, ``files_done``)
        * ``done``      — local_dir on disk
        * ``cancelled`` — caller set ``cancel_event``; partial dir removed
        * ``error``     — terminal error (``message`` field)

        Cancellation is *best effort*: the worker thread cannot interrupt a
        single in-flight file download cleanly, so we wait for the current
        file to finish (or fail) before tearing down the destination
        directory. The next file is never started once cancellation is
        observed between files.
        """
        try:
            entry = self._entry(backend, model_id)
        except ModelManagerError as e:
            yield {"stage": "error", "message": str(e)}
            return

        dst = self.model_dir(backend, model_id)
        dst.mkdir(parents=True, exist_ok=True)
        cache_root = get_settings().model_root / ".hf-cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as e:  # pragma: no cover
            yield {"stage": "error", "message": f"huggingface_hub missing: {e}"}
            return

        log.info(
            "streaming download backend={} id={} repo={}",
            backend, model_id, entry.repo_id,
        )

        # Resolve the list of repo files + sizes up front so we can render
        # a meaningful progress bar instead of an indeterminate spinner.
        try:
            info = HfApi().model_info(entry.repo_id, files_metadata=True)
        except Exception as e:
            log.exception("model_info failed repo={}", entry.repo_id)
            yield {"stage": "error", "message": f"model_info failed: {e}"}
            return

        siblings = list(getattr(info, "siblings", None) or [])
        files: list[tuple[str, int]] = [
            (s.rfilename, int(getattr(s, "size", 0) or 0)) for s in siblings
        ]
        files_total = len(files)
        bytes_total = sum(sz for _, sz in files)
        # Fallback when the API omits file sizes: use the catalog's
        # advertised size_gb so the bar at least has a denominator.
        if bytes_total <= 0 and entry.size_gb:
            bytes_total = int(entry.size_gb * (1024**3))

        yield {
            "stage": "start",
            "backend": backend,
            "id": model_id,
            "repo_id": entry.repo_id,
            "files_total": files_total,
            "bytes_total": bytes_total,
            "local_dir": str(dst),
        }

        bytes_done = 0
        files_done = 0

        for rel, sz in files:
            if cancel_event.is_set():
                break

            target_path = dst / rel
            # If a previous run already fetched this file, skip the worker.
            if target_path.exists() and sz and target_path.stat().st_size >= sz:
                bytes_done += sz
                files_done += 1
                yield {
                    "stage": "progress",
                    "file": rel,
                    "files_done": files_done,
                    "files_total": files_total,
                    "bytes_done": bytes_done,
                    "bytes_total": bytes_total,
                }
                continue

            result: dict = {"err": None}

            def _worker(rel=rel) -> None:
                try:
                    hf_hub_download(
                        repo_id=entry.repo_id,
                        filename=rel,
                        local_dir=str(dst),
                        cache_dir=str(cache_root),
                    )
                except Exception as e:  # noqa: BLE001 - reported via stream
                    result["err"] = e

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            while t.is_alive():
                # Poll partial file size on disk so very large shards show
                # within-file progress, not just per-file ticks.
                cur = (
                    target_path.stat().st_size
                    if target_path.exists()
                    else 0
                )
                yield {
                    "stage": "progress",
                    "file": rel,
                    "files_done": files_done,
                    "files_total": files_total,
                    "bytes_done": bytes_done + cur,
                    "bytes_total": bytes_total,
                }
                if cancel_event.is_set():
                    # Let the in-flight file settle to a consistent state
                    # before we rmtree() the directory — yanking the file
                    # out from under the worker can leave HF's atomic-write
                    # tempfile dangling.
                    break
                t.join(timeout=poll_interval)

            if cancel_event.is_set():
                t.join()  # wait out the in-flight write
                shutil.rmtree(dst, ignore_errors=True)
                log.info(
                    "download cancelled backend={} id={}", backend, model_id
                )
                yield {
                    "stage": "cancelled",
                    "files_done": files_done,
                    "files_total": files_total,
                    "bytes_done": bytes_done,
                    "bytes_total": bytes_total,
                }
                return

            if result["err"] is not None:
                log.exception(
                    "download error backend={} id={} file={}",
                    backend, model_id, rel,
                )
                yield {
                    "stage": "error",
                    "file": rel,
                    "message": f"{type(result['err']).__name__}: {result['err']}",
                }
                return

            bytes_done += sz or (
                target_path.stat().st_size if target_path.exists() else 0
            )
            files_done += 1
            yield {
                "stage": "progress",
                "file": rel,
                "files_done": files_done,
                "files_total": files_total,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
            }

        # Loop exit without an explicit cancel branch: either everything
        # downloaded or the very first iteration saw ``cancel_event`` set
        # before any file was touched. Handle the latter explicitly.
        if cancel_event.is_set():
            shutil.rmtree(dst, ignore_errors=True)
            log.info("download cancelled backend={} id={}", backend, model_id)
            yield {
                "stage": "cancelled",
                "files_done": files_done,
                "files_total": files_total,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
            }
            return

        log.info("download complete backend={} id={}", backend, model_id)
        yield {
            "stage": "done",
            "backend": backend,
            "id": model_id,
            "local_dir": str(dst),
            "files_total": files_total,
            "bytes_total": bytes_total,
        }

    # ---------- load + activate -----------------------------------------

    def _build_cortex(self, backend: str, path: str) -> Cortex:
        s = get_settings()
        log.info("building cortex backend={} path={}", backend, path)
        if backend == "mlx":
            from ..core.cortex.mlx_cortex import MLXCortex
            from .backends.mlx import build_mlx_model

            return MLXCortex(build_mlx_model(path))
        if backend == "hf":
            from ..core.cortex.hf_cortex import HFCortex
            from .backends.hf import build_hf_model

            return HFCortex(
                build_hf_model(
                    path,
                    device=s.hf_device,
                    dtype=s.hf_dtype,
                    offload=s.hf_offload,
                    max_gpu_gb=s.hf_max_gpu_gb,
                    max_cpu_gb=s.hf_max_cpu_gb,
                    offload_dir=str(s.hf_offload_dir),
                    load_in_4bit=s.hf_load_in_4bit,
                )
            )
        raise ModelManagerError(f"backend {backend!r} cannot host a model")

    def load(self, backend: str, model_id: str) -> Cortex:
        key = (backend, model_id)
        with self._lock:
            if key in self._cache:
                log.debug("model cache hit backend={} id={}", backend, model_id)
                return self._cache[key]
            if not self.is_installed(backend, model_id):
                log.error(
                    "load aborted: model not installed backend={} id={}",
                    backend, model_id,
                )
                raise ModelManagerError(
                    f"model {model_id!r} not installed; download it first"
                )
            log.info("loading model backend={} id={}", backend, model_id)
            try:
                cortex = self._build_cortex(
                    backend, str(self.model_dir(backend, model_id))
                )
            except Exception:
                log.exception(
                    "model load failed backend={} id={}", backend, model_id
                )
                raise
            self._cache[key] = cortex
            log.info("model loaded backend={} id={}", backend, model_id)
            return cortex

    def _release_cortex(self, cortex: Cortex) -> None:
        """Best-effort teardown so the previous model frees GPU/Metal memory.

        The Cortex / LM wrappers keep a reference to a HF or MLX model + a
        tokenizer; dropping the wrapper alone is not enough on CUDA/MPS where
        the allocator caches blocks. We null out the heavy attrs and ask the
        framework to release its cache.
        """
        lm = getattr(cortex, "lm", None)
        for obj in (lm, cortex):
            for attr in ("model", "tokenizer", "_model", "_tokenizer"):
                if hasattr(obj, attr):
                    try:
                        setattr(obj, attr, None)
                    except Exception:
                        pass
        import gc

        gc.collect()
        try:  # CUDA / MPS allocator
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
                if callable(empty):
                    empty()
        except ImportError:
            pass
        try:  # MLX (Metal) allocator
            import mlx.core as mx

            clear = getattr(mx, "clear_cache", None) or getattr(
                getattr(mx, "metal", None), "clear_cache", None
            )
            if callable(clear):
                clear()
        except ImportError:
            pass

    def unload(self, backend: str, model_id: str) -> bool:
        """Drop a cached cortex and free its memory. Returns True if removed."""
        key = (backend, model_id)
        with self._lock:
            cortex = self._cache.pop(key, None)
            if self._active == key:
                self._active = None
        if cortex is None:
            log.debug("unload no-op (not cached) backend={} id={}", backend, model_id)
            return False
        log.info("unloading model backend={} id={}", backend, model_id)
        self._release_cortex(cortex)
        log.info("model unloaded backend={} id={}", backend, model_id)
        return True

    def unload_all(self) -> int:
        """Drop every cached cortex and clear the active pointer."""
        with self._lock:
            evicted = list(self._cache.values())
            self._cache.clear()
            self._active = None
        if evicted:
            log.info("unloading all models count={}", len(evicted))
        for old in evicted:
            self._release_cortex(old)
        return len(evicted)

    def delete(self, backend: str, model_id: str) -> bool:
        """Remove an installed model's weights from disk. Refuses to delete
        the currently-active model. Returns True if files were removed."""
        if backend not in SUPPORTED_BACKENDS:
            raise ModelManagerError(f"unsupported backend {backend!r}")
        with self._lock:
            if self._active == (backend, model_id):
                raise ModelManagerError(
                    "cannot delete the active model; unload it first"
                )
            # Drop any cached (but not active) cortex for this id too.
            cortex = self._cache.pop((backend, model_id), None)
        if cortex is not None:
            self._release_cortex(cortex)
        d = self.model_dir(backend, model_id)
        if not d.exists():
            log.debug("delete no-op (not on disk) backend={} id={}", backend, model_id)
            return False
        log.warning("deleting model weights backend={} id={} dir={}", backend, model_id, d)
        shutil.rmtree(d, ignore_errors=False)
        return True

    def set_active(self, backend: str, model_id: str) -> Cortex:
        new_key = (backend, model_id)

        # Fast path: already cached — just flip the pointer and evict siblings.
        with self._lock:
            if new_key in self._cache:
                cortex = self._cache[new_key]
                stale = [k for k in self._cache if k != new_key]
                evicted = [self._cache.pop(k) for k in stale]
                self._active = new_key
            else:
                cortex = None
                evicted = list(self._cache.values())
                self._cache.clear()
                self._active = None
        if evicted:
            log.info("evicting {} cached model(s) before activating {}/{}", len(evicted), backend, model_id)
            for old in evicted:
                self._release_cortex(old)
        if cortex is not None:
            log.info("active model set (cached) backend={} id={}", backend, model_id)
            return cortex

        # Slow path: build the new cortex *after* freeing the old one so the
        # GPU/Metal allocator only has to host one model at a time.
        if not self.is_installed(backend, model_id):
            raise ModelManagerError(
                f"model {model_id!r} not installed; download it first"
            )
        log.info("activating model backend={} id={}", backend, model_id)
        try:
            cortex = self._build_cortex(
                backend, str(self.model_dir(backend, model_id))
            )
        except Exception:
            log.exception("activation failed backend={} id={}", backend, model_id)
            raise
        with self._lock:
            self._cache[new_key] = cortex
            self._active = new_key
        log.info("active model set backend={} id={}", backend, model_id)
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
