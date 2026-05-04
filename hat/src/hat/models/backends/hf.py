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
    """Thin wrapper around `AutoModelForCausalLM` with chat-template support.

    When ``offload=True``, the model is loaded with accelerate's
    ``device_map="auto"`` and a ``max_memory`` budget so that layers which
    don't fit on the GPU spill to CPU RAM (and, if needed, to disk under
    ``offload_dir``). When ``load_in_4bit=True``, weights are quantised via
    bitsandbytes (CUDA only).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        *,
        offload: bool = False,
        max_gpu_gb: float | None = None,
        max_cpu_gb: float | None = None,
        offload_dir: str | None = None,
        load_in_4bit: bool = False,
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

        load_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(dtype),
            "trust_remote_code": True,
        }

        # Optional 4-bit quantisation via bitsandbytes (CUDA only).
        if load_in_4bit:
            try:
                import torch
                from transformers import BitsAndBytesConfig
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "hf_load_in_4bit=True requires `bitsandbytes`"
                ) from e
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            # bnb manages dtype; drop the user's torch_dtype to avoid clash.
            load_kwargs.pop("torch_dtype", None)

        if offload:
            # Accelerate-driven sharding: GPU first, then CPU RAM, then disk.
            from pathlib import Path as _P

            max_memory: dict[str | int, str] = {}
            if self.device in ("cuda", "mps"):
                if max_gpu_gb is not None:
                    max_memory[0] = f"{max_gpu_gb:.2f}GiB"
            if max_cpu_gb is not None:
                max_memory["cpu"] = f"{max_cpu_gb:.2f}GiB"
            load_kwargs["device_map"] = "auto"
            if max_memory:
                load_kwargs["max_memory"] = max_memory
            if offload_dir:
                _P(offload_dir).mkdir(parents=True, exist_ok=True)
                load_kwargs["offload_folder"] = offload_dir
        else:
            # Single-device path (legacy behaviour).
            load_kwargs["device_map"] = (
                self.device if self.device != "mps" else None
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if not offload and self.device == "mps":
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

    @staticmethod
    def _split_template_kwargs(kwargs: dict) -> dict:
        out: dict = {}
        if "enable_thinking" in kwargs:
            out["enable_thinking"] = bool(kwargs.pop("enable_thinking"))
        nested = kwargs.pop("chat_template_kwargs", None)
        if isinstance(nested, dict):
            out.update(nested)
        return out

    def _prepare_inputs(
        self, messages: Sequence[dict[str, str]], template_kwargs: dict
    ):
        text = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def chat(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        import torch

        template_kwargs = self._split_template_kwargs(kwargs)
        inputs = self._prepare_inputs(messages, template_kwargs)

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

    def stream_chat(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ):
        """Yield decoded text chunks via :class:`TextIteratorStreamer`."""
        from threading import Thread

        from transformers import TextIteratorStreamer

        template_kwargs = self._split_template_kwargs(kwargs)
        inputs = self._prepare_inputs(messages, template_kwargs)

        max_new_tokens = int(
            kwargs.get("max_new_tokens", kwargs.get("max_tokens", self.max_new_tokens))
        )
        temperature = float(kwargs.get("temperature", self.temperature))
        do_sample = temperature > 0.0

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join()


@register("hf")
def build_hf_model(
    model_path: str,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    *,
    offload: bool = False,
    max_gpu_gb: float | None = None,
    max_cpu_gb: float | None = None,
    offload_dir: str | None = None,
    load_in_4bit: bool = False,
    **_: Any,
) -> HFLanguageModel:
    return HFLanguageModel(
        model_path=model_path,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        offload=offload,
        max_gpu_gb=max_gpu_gb,
        max_cpu_gb=max_cpu_gb,
        offload_dir=offload_dir,
        load_in_4bit=load_in_4bit,
    )


__all__ = ["HFLanguageModel", "build_hf_model"]

