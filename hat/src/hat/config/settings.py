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

    # Selection (αU + βF + γN) — paper §3.4.2 / §4 defaults
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    write_threshold: float = 0.3
    oracle_threshold: float = 0.7

    # Storage
    raw_log_path: Path = Path("runs/raw_log.jsonl")
    neocortex_path: Path = Path("runs/neocortex.jsonl")
    model_root: Path = Path("model")

    # Cortex backend selection. Concrete model paths are NOT in env any
    # more — they're discovered at runtime under ``model/<backend>/<id>/``
    # via the catalog and ``ModelManager``. Use the UI / ``/api/models``
    # to download and activate.
    cortex_backend: str = "noop"  # noop | hf | mlx

    # HF hardware preferences (constructor-time, can't change per request).
    hf_device: str = "auto"  # auto | cpu | cuda | mps
    hf_dtype: str = "auto"  # auto | float16 | bfloat16 | float32

    # Per-request generation defaults (used when the client doesn't supply
    # ``temperature`` / ``max_tokens`` in the chat-completions request).
    default_temperature: float = 0.7
    default_max_tokens: int = 512

    # Oracle (OpenAI-compatible HTTP)
    oracle_base_url: str = "https://api.openai.com/v1"
    oracle_model: str = "gpt-4o-mini"
    oracle_api_key: str | None = None

    # UI -> server
    ui_base_url: str = "http://127.0.0.1:8000/v1"
    ui_model: str = "hat-cortex"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Memoised settings accessor; safe to call from anywhere."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
