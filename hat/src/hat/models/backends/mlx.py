"""MLX (Apple Silicon) backend.

Uses `mlx-lm <https://github.com/ml-explore/mlx-lm>`_ — Apple's official
Metal-native LLM runtime. Runs comfortably on an 8 GB M1 with 4-bit quantized
models (e.g. ``mlx-community/Qwen2.5-1.5B-Instruct-4bit`` ≈ 1 GB on disk,
~1.5 GB resident).

Install with::

    uv sync --extra mlx

Implements :class:`hat.core.protocols.LanguageModel` and exposes a
``chat(messages)`` method on top of the tokenizer's chat template so multi-turn
conversations route through the model's native role formatting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..registry import register


class MLXLanguageModel:
    """Thin wrapper around `mlx_lm.load` / `mlx_lm.generate`."""

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        try:
            from mlx_lm import load
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "MLX backend requires extra deps: `uv sync --extra mlx` "
                "(Apple Silicon only)"
            ) from e

        self.name = f"mlx:{model_path}"
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature

        # `load` accepts a local directory or a HF repo id; the latter is fetched
        # on first use and cached under ~/.cache/huggingface.
        self.model, self.tokenizer = load(model_path)

    # -- LanguageModel protocol -------------------------------------------

    def generate(self, prompt: str, *, context: str | None = None, **kwargs: Any) -> str:
        messages: list[dict[str, str]] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def token_logprobs(self, prompt: str, response: str) -> list[float]:
        # Placeholder — see HF backend note. mlx-lm exposes per-token logprobs
        # via the streaming `generate_step` API; wire that up when the
        # uncertainty estimator needs it.
        return []

    # -- chat-template path ------------------------------------------------

    def chat(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))

        # `mlx-lm.generate` returns the decoded continuation only.
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        return text.strip()


@register("mlx")
def build_mlx_model(
    model_path: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **_: Any,
) -> MLXLanguageModel:
    return MLXLanguageModel(
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )


__all__ = ["MLXLanguageModel", "build_mlx_model"]
