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
import warnings

from ...utils.logging import get_logger
from ..registry import register

log = get_logger(__name__)


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
        log.info(
            "[hf] loading model path={} device={} dtype={} offload={} 4bit={}",
            model_path, self.device, dtype, offload, load_in_4bit,
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

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except RuntimeError as e:
            # Safety net: if the caller forgot to enable offload and a large
            # checkpoint OOMs on CUDA, retry once with accelerate offload to
            # spill layers into host RAM / disk instead of failing activation.
            is_cuda_oom = (
                self.device == "cuda"
                and not offload
                and (
                    type(e).__name__ == "OutOfMemoryError"
                    or "cuda out of memory" in str(e).lower()
                )
            )
            if not is_cuda_oom:
                raise
            from pathlib import Path as _P

            retry_kwargs = dict(load_kwargs)
            retry_kwargs["device_map"] = "auto"
            max_memory: dict[str | int, str] = {}
            if max_gpu_gb is not None:
                max_memory[0] = f"{max_gpu_gb:.2f}GiB"
            if max_cpu_gb is not None:
                max_memory["cpu"] = f"{max_cpu_gb:.2f}GiB"
            if max_memory:
                retry_kwargs["max_memory"] = max_memory
            if offload_dir:
                _P(offload_dir).mkdir(parents=True, exist_ok=True)
                retry_kwargs["offload_folder"] = offload_dir
            warnings.warn(
                "CUDA OOM during HF model load; retrying with offload "
                "(device_map='auto'). Set HAT_HF_OFFLOAD=true to make this "
                "the default behaviour.",
                RuntimeWarning,
            )
            log.warning(
                "[hf] CUDA OOM during load path={}; retrying with offload",
                model_path,
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **retry_kwargs)
        if not offload and self.device == "mps":
            self.model = self.model.to("mps")
        self.model.eval()
        log.info("[hf] model loaded path={} device={}", model_path, self.device)

        # Chat-template stop tokens. Many instruct models (Qwen2.5, Llama-3,
        # ChatML-style families) use a turn terminator like ``<|im_end|>`` or
        # ``<|eot_id|>`` that is *different* from ``tokenizer.eos_token``
        # (``<|endoftext|>``). If we only pass ``eos_token_id=eos_token_id``
        # to ``generate``, the model emits the chat terminator, the runtime
        # does not recognise it as a stop, and generation runs to
        # ``max_new_tokens``. Collect every plausible terminator id once.
        self._stop_token_ids = self._collect_stop_token_ids()

    def _collect_stop_token_ids(self) -> list[int]:
        """Resolve every chat-template stop token the tokenizer knows about.

        Includes the canonical ``eos_token_id`` plus common chat-turn
        terminators (``<|im_end|>``, ``<|eot_id|>``, ``<|end|>``) when the
        tokenizer has them in its vocab. Returns a de-duplicated list.
        """
        ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            ids.append(int(self.tokenizer.eos_token_id))
        for tok in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>"):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tid = None
            # ``convert_tokens_to_ids`` returns ``unk_token_id`` for unknown
            # tokens on some tokenizers; filter those out.
            if (
                tid is not None
                and tid >= 0
                and tid != self.tokenizer.unk_token_id
                and tid not in ids
            ):
                ids.append(int(tid))
        return ids

    # -- LanguageModel protocol -------------------------------------------

    def generate(self, prompt: str, *, context: str | None = None, **kwargs: Any) -> str:
        messages: list[dict[str, str]] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def token_logprobs(self, prompt: str, response: str) -> list[float]:
        """Return per-token log-probabilities of ``response`` conditioned on
        ``prompt`` (no chat template applied — caller is expected to pass an
        already-rendered string, or use :meth:`chat_logprobs`).

        Implementation: tokenise ``prompt`` and ``prompt+response`` separately,
        run a single forward pass over the concatenation, then gather
        ``log_softmax`` at each response position against the actual response
        token id. Returns an empty list if the response tokenises to nothing.
        """
        import torch
        import torch.nn.functional as F

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        full_ids = self.tokenizer(prompt + response, return_tensors="pt").input_ids
        # Response slice in the concatenated sequence.
        start = prompt_ids.shape[-1]
        if full_ids.shape[-1] <= start:
            return []
        full_ids = full_ids.to(self.model.device)
        with torch.no_grad():
            logits = self.model(full_ids).logits  # (1, T, V)
        # Predicting token t uses logits at position t-1.
        target_ids = full_ids[0, start:]
        pred_logits = logits[0, start - 1 : -1, :]
        if pred_logits.shape[0] == 0 or target_ids.shape[0] == 0:
            return []
        logp = F.log_softmax(pred_logits.float(), dim=-1)
        gathered = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return gathered.detach().cpu().tolist()

    def chat_logprobs(
        self, messages: Sequence[dict[str, str]], response: str
    ) -> list[float]:
        """Like :meth:`token_logprobs` but applies the chat template to
        ``messages`` first so the prompt mirrors what :meth:`chat` actually
        feeds the model. Used by the uncertainty estimator."""
        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.token_logprobs(prompt, response)

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
        # Mild repetition penalty as a safety net for tiny / 4-bit instruct
        # checkpoints. 1.05 matches Qwen2.5's recommended sampling hyper-
        # parameters; higher values (1.1+) start to noticeably hurt code
        # generation and lists. Override per-request via kwargs if needed.
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.05))

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Pass *all* known stop ids so chat-template terminators (‘<|im_end|>’,
        # ‘<|eot_id|>’, …) actually end the turn — without this, instruct
        # models can run past the turn boundary on top of which they were
        # trained and the runtime hits ``max_new_tokens`` instead.
        if self._stop_token_ids:
            gen_kwargs["eos_token_id"] = self._stop_token_ids
        if do_sample:
            gen_kwargs.setdefault("top_p", float(kwargs.get("top_p", 0.9)))

        log.debug(
            "[hf] chat request msgs={} max_new_tokens={} temp={} do_sample={}",
            len(messages), max_new_tokens, temperature, do_sample,
        )
        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
        except Exception:
            log.exception("[hf] chat generation failed")
            raise

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

        repetition_penalty = float(kwargs.get("repetition_penalty", 1.05))
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs: dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        if self._stop_token_ids:
            gen_kwargs["eos_token_id"] = self._stop_token_ids
        if do_sample:
            gen_kwargs.setdefault("top_p", float(kwargs.get("top_p", 0.9)))
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

