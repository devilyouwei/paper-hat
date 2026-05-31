"""Runtime manager for installed / active *embedding* models.

Mirror of :class:`hat.models.manager.ModelManager` but for the embedding
side of the pipeline. Owns:

* a ``(backend, id) -> Embedder`` cache so swapping back to a previously
  loaded embedder is instant;
* the *active* ``(backend, id)`` pointer the wake-step deduper reads
  through;
* the same SSE download mechanism backed by ``huggingface_hub``.

Models live on disk under the same ``model/<backend>/<id>/`` layout as
LLM checkpoints; the backend names are ``mlx_embed`` and ``hf_embed`` so
they don't collide with the Cortex catalogs.
"""

from __future__ import annotations

import gc
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from threading import Lock

from ..config.settings import get_settings
from ..memory.embeddings import Embedder
from ..utils.logging import get_logger
from .catalog import SUPPORTED_EMBED_BACKENDS, CatalogEntry, load_catalog

log = get_logger(__name__)


class EmbeddingManagerError(RuntimeError):
    pass


class EmbeddingManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._cache: dict[tuple[str, str], Embedder] = {}
        self._active: tuple[str, str] | None = None

    # ---------- paths ----------------------------------------------------

    def model_dir(self, backend: str, model_id: str) -> Path:
        return get_settings().model_root / backend / model_id

    def is_installed(self, backend: str, model_id: str) -> bool:
        d = self.model_dir(backend, model_id)
        if not d.is_dir():
            return False
        for child in d.iterdir():
            if child.suffix in {".safetensors", ".bin", ".gguf", ".npz"}:
                return True
        return False

    # ---------- listing --------------------------------------------------

    def list_models(self, backend: str) -> list[dict]:
        if backend not in SUPPORTED_EMBED_BACKENDS:
            raise EmbeddingManagerError(f"unsupported embed backend {backend!r}")
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
        raise EmbeddingManagerError(
            f"unknown embed model id {model_id!r} for backend {backend!r}"
        )

    # ---------- download (blocking + streaming) -------------------------

    def download(self, backend: str, model_id: str) -> Path:
        entry = self._entry(backend, model_id)
        dst = self.model_dir(backend, model_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # pragma: no cover
            raise EmbeddingManagerError(
                "huggingface_hub is required for downloads"
            ) from e
        cache_root = get_settings().model_root / ".hf-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        log.info(
            "[embed] downloading backend={} id={} repo={} -> {}",
            backend, model_id, entry.repo_id, dst,
        )
        snapshot_download(
            repo_id=entry.repo_id,
            local_dir=str(dst),
            cache_dir=str(cache_root),
        )
        log.info("[embed] download complete backend={} id={}", backend, model_id)
        return dst

    def download_streaming(
        self,
        backend: str,
        model_id: str,
        cancel_event: threading.Event,
        *,
        poll_interval: float = 0.5,
    ) -> Iterator[dict]:
        """SSE-shaped iterator. Yields ``stage`` events identical to the
        LLM ``ModelManager`` so the UI can reuse its progress renderer."""
        try:
            entry = self._entry(backend, model_id)
        except EmbeddingManagerError as e:
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

        try:
            info = HfApi().model_info(entry.repo_id, files_metadata=True)
        except Exception as e:  # noqa: BLE001
            yield {"stage": "error", "message": f"model_info failed: {e}"}
            return

        siblings = list(getattr(info, "siblings", None) or [])
        files: list[tuple[str, int]] = [
            (s.rfilename, int(getattr(s, "size", 0) or 0)) for s in siblings
        ]
        files_total = len(files)
        bytes_total = sum(sz for _, sz in files)
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
                except Exception as e:  # noqa: BLE001
                    result["err"] = e

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            while t.is_alive():
                cur = target_path.stat().st_size if target_path.exists() else 0
                yield {
                    "stage": "progress",
                    "file": rel,
                    "files_done": files_done,
                    "files_total": files_total,
                    "bytes_done": bytes_done + cur,
                    "bytes_total": bytes_total,
                }
                if cancel_event.is_set():
                    break
                t.join(timeout=poll_interval)

            if cancel_event.is_set():
                t.join()
                shutil.rmtree(dst, ignore_errors=True)
                yield {
                    "stage": "cancelled",
                    "files_done": files_done,
                    "files_total": files_total,
                    "bytes_done": bytes_done,
                    "bytes_total": bytes_total,
                }
                return
            if result["err"] is not None:
                yield {
                    "stage": "error",
                    "file": rel,
                    "message": f"{type(result['err']).__name__}: {result['err']}",
                }
                return
            bytes_done += sz or (target_path.stat().st_size if target_path.exists() else 0)
            files_done += 1
            yield {
                "stage": "progress",
                "file": rel,
                "files_done": files_done,
                "files_total": files_total,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
            }

        if cancel_event.is_set():
            shutil.rmtree(dst, ignore_errors=True)
            yield {
                "stage": "cancelled",
                "files_done": files_done,
                "files_total": files_total,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
            }
            return

        log.info("[embed] download complete backend={} id={}", backend, model_id)
        yield {
            "stage": "done",
            "backend": backend,
            "id": model_id,
            "local_dir": str(dst),
            "files_total": files_total,
            "bytes_total": bytes_total,
        }

    # ---------- load + activate -----------------------------------------

    def _build_embedder(self, backend: str, path: str) -> Embedder:
        s = get_settings()
        log.info("[embed] building embedder backend={} path={}", backend, path)
        from ..memory.embeddings import ManagedEmbedder

        if backend == "mlx_embed":
            from .backends.mlx_embed import build_mlx_embed_model

            inner = build_mlx_embed_model(path, device=s.embed_device)
            return ManagedEmbedder(inner, backend=backend, model_id=Path(path).name)
        if backend == "hf_embed":
            from .backends.hf_embed import build_hf_embed_model

            inner = build_hf_embed_model(path, device=s.embed_device)
            return ManagedEmbedder(inner, backend=backend, model_id=Path(path).name)
        raise EmbeddingManagerError(
            f"backend {backend!r} cannot host an embedding model"
        )

    def load(self, backend: str, model_id: str) -> Embedder:
        key = (backend, model_id)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if not self.is_installed(backend, model_id):
                raise EmbeddingManagerError(
                    f"embed model {model_id!r} not installed; download it first"
                )
            try:
                emb = self._build_embedder(
                    backend, str(self.model_dir(backend, model_id))
                )
            except Exception:
                log.exception(
                    "[embed] load failed backend={} id={}", backend, model_id
                )
                raise
            self._cache[key] = emb
            return emb

    def _release(self, emb: Embedder) -> None:
        inner = getattr(emb, "inner", None)
        for obj in (inner, emb):
            for attr in ("model", "tokenizer", "_model", "_tokenizer"):
                if hasattr(obj, attr):
                    try:
                        setattr(obj, attr, None)
                    except Exception:
                        pass
        gc.collect()
        try:  # MLX (Metal) allocator
            import mlx.core as mx  # type: ignore[import-not-found]

            clear = getattr(mx, "clear_cache", None) or getattr(
                getattr(mx, "metal", None), "clear_cache", None
            )
            if callable(clear):
                clear()
        except ImportError:
            pass

    def unload(self, backend: str, model_id: str) -> bool:
        key = (backend, model_id)
        with self._lock:
            emb = self._cache.pop(key, None)
            if self._active == key:
                self._active = None
        if emb is None:
            return False
        self._release(emb)
        log.info("[embed] unloaded backend={} id={}", backend, model_id)
        return True

    def unload_all(self) -> int:
        with self._lock:
            evicted = list(self._cache.values())
            self._cache.clear()
            self._active = None
        for old in evicted:
            self._release(old)
        return len(evicted)

    def delete(self, backend: str, model_id: str) -> bool:
        if backend not in SUPPORTED_EMBED_BACKENDS:
            raise EmbeddingManagerError(f"unsupported embed backend {backend!r}")
        with self._lock:
            if self._active == (backend, model_id):
                raise EmbeddingManagerError(
                    "cannot delete the active embed model; unload it first"
                )
            cached = self._cache.pop((backend, model_id), None)
        if cached is not None:
            self._release(cached)
        d = self.model_dir(backend, model_id)
        if not d.exists():
            return False
        log.warning(
            "[embed] deleting weights backend={} id={} dir={}", backend, model_id, d,
        )
        shutil.rmtree(d, ignore_errors=False)
        return True

    def set_active(self, backend: str, model_id: str) -> Embedder:
        new_key = (backend, model_id)
        with self._lock:
            if new_key in self._cache:
                emb = self._cache[new_key]
                stale = [k for k in self._cache if k != new_key]
                evicted = [self._cache.pop(k) for k in stale]
                self._active = new_key
            else:
                emb = None
                evicted = list(self._cache.values())
                self._cache.clear()
                self._active = None
        for old in evicted:
            self._release(old)
        if emb is not None:
            log.info(
                "[embed] active set (cached) backend={} id={}", backend, model_id,
            )
            return emb
        if not self.is_installed(backend, model_id):
            raise EmbeddingManagerError(
                f"embed model {model_id!r} not installed; download it first"
            )
        emb = self._build_embedder(backend, str(self.model_dir(backend, model_id)))
        with self._lock:
            self._cache[new_key] = emb
            self._active = new_key
        log.info("[embed] active set backend={} id={}", backend, model_id)
        return emb

    def active(self) -> dict | None:
        if self._active is None:
            return None
        backend, model_id = self._active
        return {"backend": backend, "id": model_id}


_manager: EmbeddingManager | None = None


def get_embedding_manager() -> EmbeddingManager:
    global _manager
    if _manager is None:
        _manager = EmbeddingManager()
    return _manager


__all__ = [
    "EmbeddingManager",
    "EmbeddingManagerError",
    "get_embedding_manager",
]
