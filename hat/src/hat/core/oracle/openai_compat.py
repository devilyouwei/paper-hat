"""OpenAI-compatible HTTP oracle client.

Talks to any endpoint that implements the ``/v1/chat/completions`` shape:
OpenAI, Anthropic-via-proxy, vLLM, Ollama (with ``OLLAMA_OPENAI=1``), Groq,
Together, etc. The endpoint and model name are pulled from settings; the
API key is optional (some local servers don't require one).

Calls go through a :class:`CostGuard` so rate and daily-budget limits are
enforced uniformly across backends. Network and HTTP errors are swallowed
into an empty string — the wake step interprets that as "no correction
available" and proceeds without oracle augmentation, never crashing the
chat loop.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..schemas import Interaction
from .base import Oracle
from .cost_guard import CostGuard, OracleQuotaExceeded
from .prompts import ORACLE_SYSTEM


class OpenAICompatibleOracle(Oracle):
    """Oracle that calls an OpenAI ``chat/completions`` endpoint.

    The ``base_url`` should point at the API root (without ``/chat/completions``).
    For OpenAI proper that's ``https://api.openai.com/v1``; for Anthropic via
    a compatibility proxy or vLLM/Ollama, supply the matching URL.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        cost_guard: CostGuard | None = None,
        timeout: float = 30.0,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.cost_guard = cost_guard
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = f"oracle:{model}"

    def consult(self, interaction: Interaction) -> str:
        # Cost / rate gate. Failures here mean "skip oracle this turn"
        # rather than "abort": we'd rather miss an augmentation than break
        # the wake loop.
        if self.cost_guard is not None:
            try:
                self.cost_guard.acquire(reason="wake_step")
            except OracleQuotaExceeded:
                return ""

        messages = [
            {"role": "system", "content": ORACLE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"User query:\n{interaction.query}\n\n"
                    f"Smaller model's response:\n{interaction.response or ''}\n\n"
                    "Provide the corrected, ground-truth reply now."
                ),
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return ""

        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            return ""
