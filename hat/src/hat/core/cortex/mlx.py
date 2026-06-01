"""MLX (Apple Silicon) backend.

Uses `mlx-lm <https://github.com/ml-explore/mlx-lm>`_ — Apple's official
Metal-native LLM runtime. Runs comfortably on an 8 GB M1 with 4-bit quantized
models (e.g. ``mlx-community/Qwen2.5-0.5B-Instruct-4bit`` ≈ 0.3 GB on disk,
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

from hat.utils.logging import get_logger
from hat.core.cortex.registry import register

log = get_logger(__name__)


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
        log.info("[mlx] loading model path={} max_tokens={} temp={}", model_path, max_tokens, temperature)
        try:
            self.model, self.tokenizer = load(model_path)
        except Exception:
            log.exception("[mlx] failed to load model path={}", model_path)
            raise
        log.info("[mlx] model loaded path={}", model_path)

        # Same problem as the HF backend: many ChatML-style instruct models
        # use ``<|im_end|>`` (Qwen, Yi) or ``<|eot_id|>`` (Llama-3) as the
        # turn terminator, which is *not* the tokenizer's ``eos_token``.
        # Without registering them, ``mlx_lm.generate`` happily continues past
        # the stop token and small / 4-bit models then loop on themselves.
        # ``TokenizerWrapper.add_eos_token`` is idempotent and safe.
        for tok in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tid = None
            if tid is not None and tid >= 0:
                try:
                    self.tokenizer.add_eos_token(tok)
                except Exception:
                    # Older mlx-lm releases may not accept unknown tokens;
                    # ignore and rely on the default eos.
                    pass

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

    def _render_prompt(
        self, messages: Sequence[dict[str, str]], **template_kwargs: Any
    ) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    @staticmethod
    def _split_template_kwargs(kwargs: dict) -> dict:
        """Pop chat-template-only kwargs (e.g. ``enable_thinking``)."""
        out: dict = {}
        if "enable_thinking" in kwargs:
            out["enable_thinking"] = bool(kwargs.pop("enable_thinking"))
        # OpenAI-style nested form
        nested = kwargs.pop("chat_template_kwargs", None)
        if isinstance(nested, dict):
            out.update(nested)
        return out

    def chat(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        template_kwargs = self._split_template_kwargs(kwargs)
        prompt = self._render_prompt(messages, **template_kwargs)

        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))
        # 1.05 matches Qwen2.5's recommended sampling hyperparameters; the
        # main reason this is here at all is the wider context window below
        # — mlx-lm's default ``repetition_context_size=20`` is too short for
        # multi-turn chat, so the penalty effectively does nothing once the
        # prompt grows past a few hundred tokens. 256 covers any plausible
        # recent suffix without changing decoding behaviour for short inputs.
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.05))
        repetition_context_size = int(
            kwargs.get("repetition_context_size", 256)
        )

        log.debug(
            "[mlx] chat request msgs={} max_tokens={} temp={}",
            len(messages), max_tokens, temperature,
        )
        try:
            text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature),
                logits_processors=make_logits_processors(
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=repetition_context_size,
                ),
                verbose=False,
            )
        except Exception:
            log.exception("[mlx] chat generation failed")
            raise
        return text.strip()

    def stream_chat(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ):
        """Yield decoded text chunks as the model generates them."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        template_kwargs = self._split_template_kwargs(kwargs)
        prompt = self._render_prompt(messages, **template_kwargs)

        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.05))
        repetition_context_size = int(
            kwargs.get("repetition_context_size", 256)
        )

        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            logits_processors=make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            ),
        ):
            text = getattr(resp, "text", None)
            if text:
                yield text


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


# ---------------------------------------------------------------------------
# Cortex adapter
# ---------------------------------------------------------------------------

from hat.abstract.cortex import Cortex
from hat.abstract.schemas import Interaction


class MLXCortex(Cortex):
    """Wraps an :class:`MLXLanguageModel`. Same shape as :class:`HFCortex`."""

    def __init__(self, lm: MLXLanguageModel, name: str | None = None) -> None:
        self.lm = lm
        self.name = name or getattr(lm, "name", "mlx-cortex")

    def generate(self, query: str, *, context: str | None = None, **kwargs: Any) -> str:
        return self.lm.generate(query, context=context, **kwargs)

    def chat(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        return self.lm.chat(messages, **kwargs)

    def stream_chat(self, messages: Sequence[dict[str, str]], **kwargs: Any):
        yield from self.lm.stream_chat(messages, **kwargs)

    def uncertainty(self, interaction: Interaction) -> float:
        # TODO: predictive entropy from mlx-lm `generate_step` logprobs.
        return 0.5


__all__ = ["MLXLanguageModel", "build_mlx_model", "MLXCortex"]
