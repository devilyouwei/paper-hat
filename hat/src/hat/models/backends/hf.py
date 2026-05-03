"""HuggingFace Transformers backend.

Loads a chat-tuned causal LM (e.g. Qwen2.5-Instruct) from a local path and
exposes a small ``chat(messages)`` API on top of the model's chat template.
Implements :class:`hat.core.protocols.LanguageModel`.

Install with::

    uv sync --extra hf
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..registry import register


def _resolve_dtype(name: str):
    import torch

    table = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return table.get(name, "auto")


def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFLanguageModel:
    """Thin wrapper around `AutoModelForCausalLM` with chat-template support."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "HF backend requires extra deps: `uv sync --extra hf`"
            ) from e

        self.name = f"hf:{model_path}"
        self.model_path = model_path
        self.device = _resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=_resolve_dtype(dtype),
            device_map=self.device if self.device != "mps" else None,
            trust_remote_code=True,
        )
        if self.device == "mps":
            self.model = self.model.to("mps")
        self.model.eval()

    # -- LanguageModel protocol -------------------------------------------

    def generate(self, prompt: str, *, context: str | None = None, **kwargs: Any) -> str:
        messages: list[dict[str, str]] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def token_logprobs(self, prompt: str, response: str) -> list[float]:
        # Minimal placeholder: return empty list. A faithful implementation
        # runs a forward pass with `labels` and gathers per-token log-softmax.
        return []

    # -- chat-template path ------------------------------------------------

    def chat(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        max_new_tokens = int(
            kwargs.get("max_new_tokens", kwargs.get("max_tokens", self.max_new_tokens))
        )
        temperature = float(kwargs.get("temperature", self.temperature))
        do_sample = temperature > 0.0

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@register("hf")
def build_hf_model(
    model_path: str,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    **_: Any,
) -> HFLanguageModel:
    return HFLanguageModel(
        model_path=model_path,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


__all__ = ["HFLanguageModel", "build_hf_model"]

