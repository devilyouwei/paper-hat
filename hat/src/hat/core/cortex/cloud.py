"""Cloud (remote API) LLM backend.

Talks to any OpenAI-compatible endpoint implementing the
``/v1/chat/completions`` shape: OpenAI, Azure OpenAI, vLLM, Ollama (with
``OLLAMA_OPENAI=1``), Groq, Together, etc. Unlike the MLX and HF backends this
runs no local weights, so it is *platform-independent* — usable on Apple
Silicon, CUDA hosts, or plain CPU boxes alike.

The HTTP layer mirrors :class:`hat.core.oracle.openai_compat.OpenAICompatibleOracle`
(stdlib ``urllib`` only, no extra deps). The remote model name, API root and
API key are supplied by the catalog entry / settings via ``build_cloud_model``.

Implements :class:`hat.abstract.cortex.LanguageModel` and exposes
``chat`` / ``stream_chat`` so multi-turn conversations route through the
provider's native role formatting.
"""

from __future__ import annotations

import contextlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

from hat.utils.logging import get_logger

log = get_logger(__name__)


class CloudError(RuntimeError):
    pass


class CloudLanguageModel:
    """OpenAI-compatible chat-completions client exposing the LM API."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"cloud:{model}"
        log.info(
            "[cloud] init model={} url={} key={}",
            model, self.base_url, "set" if api_key else "none",
        )

    # -- HTTP helpers ------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - network
            detail = ""
            with contextlib.suppress(Exception):
                detail = e.read().decode("utf-8", "replace")
            log.error("[cloud] http {} {}: {}", e.code, url, detail[:400])
            raise CloudError(f"cloud request failed: {e.code} {e.reason}") from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            log.error("[cloud] request error {}: {}", type(e).__name__, e)
            raise CloudError(f"cloud request failed: {e}") from e

    # -- LanguageModel protocol -------------------------------------------

    def generate(
        self, prompt: str, *, context: str | None = None, **kwargs: Any
    ) -> str:
        messages: list[dict[str, str]] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def token_logprobs(self, prompt: str, response: str) -> list[float]:
        """Best-effort per-token logprobs for ``response`` given ``prompt``.

        OpenAI-compatible chat completions can return per-token logprobs when
        ``logprobs=true`` is requested, but they describe the *model's own*
        generation, not an arbitrary ``response`` string. We therefore ask the
        model to continue from ``prompt`` and return the logprobs of what it
        produced; consumers (the uncertainty estimator) only need a logprob
        distribution to score confidence, not an exact teacher-forced match.

        Returns ``[]`` when the provider doesn't expose logprobs — callers
        fall back to a neutral uncertainty.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            data = self._post(
                "chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "logprobs": True,
                },
            )
        except CloudError:
            return []
        return _extract_logprobs(data)

    # -- chat path ---------------------------------------------------------

    def chat(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
        }
        log.debug("[cloud] chat msgs={} model={}", len(messages), self.model)
        data = self._post("chat/completions", payload)
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise CloudError(f"unexpected chat response: {e}") from e

    def stream_chat(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> Iterator[str]:
        """Yield decoded text chunks via the SSE streaming API.

        Falls back to a single-shot :meth:`chat` if the endpoint doesn't
        support streaming (any error mid-stream surfaces the buffered text).
        """
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "stream": True,
        }
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"]
                        text = delta.get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    if text:
                        yield text
        except (urllib.error.URLError, TimeoutError) as e:
            log.warning("[cloud] stream failed, falling back to chat: {}", e)
            yield self.chat(messages, **kwargs)


def _extract_logprobs(data: dict) -> list[float]:
    """Pull per-token logprobs from an OpenAI-style chat completion."""
    try:
        content = data["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return []
    out: list[float] = []
    for tok in content or []:
        lp = tok.get("logprob") if isinstance(tok, dict) else None
        if isinstance(lp, (int, float)):
            out.append(float(lp))
    return out


def build_cloud_model(
    model: str,
    *,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    api_key_env: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: float = 60.0,
    **_: Any,
) -> CloudLanguageModel:
    if api_key is None and api_key_env:
        api_key = os.environ.get(api_key_env)
    return CloudLanguageModel(
        model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )


__all__ = ["CloudLanguageModel", "CloudError", "build_cloud_model"]


# ---------------------------------------------------------------------------
# Cortex adapter
# ---------------------------------------------------------------------------

import math

from hat.abstract.cortex import Cortex
from hat.abstract.schemas import Interaction


class CloudCortex(Cortex):
    """Wraps a :class:`CloudLanguageModel`. Same shape as MLX/HF cortexes."""

    def __init__(self, lm: CloudLanguageModel, name: str | None = None) -> None:
        self.lm = lm
        self.name = name or getattr(lm, "name", "cloud-cortex")

    def generate(self, query: str, *, context: str | None = None, **kwargs: Any) -> str:
        return self.lm.generate(query, context=context, **kwargs)

    def chat(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        return self.lm.chat(messages, **kwargs)

    def stream_chat(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        yield from self.lm.stream_chat(messages, **kwargs)

    def uncertainty(self, interaction: Interaction) -> float:
        """Predictive uncertainty from API logprobs when available.

        Uses mean per-token probability (``exp(mean logprob)``) as a
        confidence proxy and returns ``1 - confidence``. Falls back to the
        neutral ``0.5`` (matching the MLX backend) when the provider does not
        expose logprobs.
        """
        logprobs = self.lm.token_logprobs(interaction.query, interaction.response or "")
        if not logprobs:
            return 0.5
        mean_lp = sum(logprobs) / len(logprobs)
        confidence = math.exp(mean_lp)  # in (0, 1]
        return max(0.0, min(1.0, 1.0 - confidence))


__all__ = ["CloudLanguageModel", "CloudError", "build_cloud_model", "CloudCortex"]
