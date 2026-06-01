"""Runtime settings.

Single ``Settings`` object pulled from environment (``HAT_*``) and ``.env``.
Add YAML overlays in a later iteration if/when experiment matrices grow."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HAT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "dev"
    log_level: str = "INFO"

    # Selection: single-signal uncertainty gate. A trace is written only when
    # the cortex's uncertainty on its own response meets this threshold.
    write_threshold: float = 0.3
    oracle_threshold: float = 0.7

    # Storage
    # Raw chat history is now organised per-session under ``raw_root``:
    #   <raw_root>/index.json            session metadata list
    #   <raw_root>/sessions/<id>.jsonl   per-session interaction log
    raw_root: Path = Path("runs/raw")
    # Curated Neocortex memory. ``neocortex_path`` is the SFT-format file
    # consumed by the SWS trainer; ``neocortex_traces_path`` keeps the full
    # trace records (including signals/scores) for inspection.
    neocortex_path: Path = Path("runs/neocortex/train.jsonl")
    neocortex_traces_path: Path = Path("runs/neocortex/traces.jsonl")
    # Legacy single-file raw log (read-only fallback for old data).
    raw_log_path: Path = Path("runs/raw_log.jsonl")
    model_root: Path = Path("model")

    # Cortex backend selection. Concrete model paths are NOT in env any
    # more — they're discovered at runtime under ``model/<backend>/<id>/``
    # via the catalog and ``ModelManager``. Use the UI / ``/api/models``
    # to download and activate.
    cortex_backend: str = "noop"  # noop | hf | mlx

    # HF hardware preferences (constructor-time, can't change per request).
    hf_device: str = "auto"  # auto | cpu | cuda | mps
    hf_dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    # When the chosen device is a single GPU and the model doesn't fit, set
    # ``hf_offload=true`` to spill layers to CPU RAM (and optionally disk).
    # Implemented via accelerate's ``device_map="auto"`` + ``max_memory``
    # budget. ``hf_max_gpu_gb`` caps GPU usage (None = no cap, fill the
    # card); ``hf_max_cpu_gb`` caps host RAM usage; if both are set and the
    # weights still don't fit, accelerate spills the rest to
    # ``hf_offload_dir`` on disk.
    hf_offload: bool = False
    hf_max_gpu_gb: float | None = None
    hf_max_cpu_gb: float | None = None
    hf_offload_dir: Path = Path("runs/hf-offload")
    # 4-bit quantization (bitsandbytes). Cuts VRAM ~3-4× with minor quality
    # loss. Requires ``bitsandbytes`` and a CUDA GPU.
    hf_load_in_4bit: bool = False

    # Per-request generation defaults (used when the client doesn't supply
    # ``temperature`` / ``max_tokens`` in the chat-completions request).
    default_temperature: float = 0.7
    default_max_tokens: int = 512

    # LLM-as-judge (abstractor / uncertainty) prompt context.
    # Cap how many of the current session's prior messages are flattened
    # into ``Interaction.context`` so the judge prompt stays bounded; the
    # tail (most-recent ``N``) is kept. Set to 0 to disable the cap.
    judge_history_max_messages: int = 20
    # When true, the abstractor prepends the most recent neocortex-saved
    # ``query`` for the current session to the judge context, so the LLM
    # can see what's already been remembered and avoid duplicate triage
    # decisions.
    judge_include_recent_neocortex: bool = True

    # Oracle (OpenAI-compatible HTTP)
    oracle_base_url: str = "https://api.openai.com/v1"
    oracle_model: str = "gpt-4o-mini"
    oracle_api_key: str | None = None
    # Oracle is opt-in: set ``HAT_ORACLE_ENABLED=true`` to consult the
    # external teacher when the cortex is unsure. Without this flag the
    # wake step is purely local.
    oracle_enabled: bool = False
    # Cost / rate limits. ``oracle_rps`` caps the throughput of consult
    # calls (sliding-window); ``oracle_daily_calls`` is a hard budget per
    # UTC day. Set either to 0 to disable that limiter.
    oracle_rps: float = 0.5
    oracle_daily_calls: int = 200
    # Where to append a JSONL audit log of every consult call. Useful for
    # post-hoc cost reconciliation. ``None`` disables the audit trail.
    oracle_audit_path: Path = Path("runs/oracle/audit.jsonl")

    # UI -> server
    ui_base_url: str = "http://127.0.0.1:8000/v1"
    ui_model: str = "hat-cortex"

    # Embedding-based deduplication for curated memory. Each accepted
    # trace's canonical query is embedded with the active managed
    # embedder and compared against a per-model NPZ side-index; cosine
    # similarity above ``dedup_threshold`` routes the new knowledge
    # point to a REVISE (overwrite the matched trace's target) instead
    # of a CREATE. Vector storage layout is
    # ``<embed_index_root>/<backend>__<id>.npz``; pick the active model
    # via ``/api/embedding-models/active``.
    dedup_enabled: bool = True
    dedup_threshold: float = 0.82
    embed_device: str = "auto"  # auto | cpu | cuda | mps
    embed_backend: str = "mlx_embed"  # mlx_embed | hf_embed
    embed_id: str | None = None
    embed_index_root: Path = Path("runs/neocortex/embeddings")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Memoised settings accessor; safe to call from anywhere."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def embed_index_path_for(backend: str, model_id: str) -> Path:
    """Per-(backend, id) NPZ path under ``embed_index_root``.

    Layout: ``<embed_index_root>/<backend>__<id>.npz``. The double
    underscore separator avoids collisions when a model id contains
    slashes; we still ``replace('/', '_')`` defensively.
    """
    s = get_settings()
    safe_id = model_id.replace("/", "_")
    return s.embed_index_root / f"{backend}__{safe_id}.npz"
