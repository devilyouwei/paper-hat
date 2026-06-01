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

from ...utils.logging import format_messages, format_text_block, get_logger, truncate
from hat.abstract.schemas import Interaction
from hat.abstract.oracle import Oracle
from .cost_guard import CostGuard, OracleQuotaExceeded
from .prompts import ORACLE_SYSTEM

log = get_logger(__name__)


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
            except OracleQuotaExceeded as e:
                log.warning(
                    "oracle.quota_exceeded iid={} name={} : {}",
                    interaction.id, self.name, e,
                )
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
        log.info(
            "oracle.consult iid={} name={} model={} url={}",
            interaction.id, self.name, self.model, self.base_url,
        )
        log.debug(
            "oracle.consult.prompt\n{}",
            format_messages(messages, title="oracle.consult"),
        )
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
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            log.warning(
                "oracle.consult.failed iid={} {}: {}",
                interaction.id, type(e).__name__, e,
            )
            return ""

        try:
            reply = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            log.warning(
                "oracle.consult.bad_response iid={} {}: {} raw='{}'",
                interaction.id, type(e).__name__, e,
                truncate(json.dumps(data, ensure_ascii=False), limit=400),
            )
            return ""
        log.debug(
            "oracle.consult.reply iid={} chars={}\n{}",
            interaction.id, len(reply),
            format_text_block(reply, title="oracle reply"),
        )
        return reply
