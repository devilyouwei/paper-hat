# ADR-003 — Model backend as a Protocol

## Status
Accepted.

## Context
Reviewers and operators want to run HAT against different inference engines: HuggingFace Transformers for full control, vLLM for throughput, Ollama for local development. Hard-coding any one of them couples the algorithmic core to a specific dependency.

## Decision
Define ``hat.core.protocols.LanguageModel`` as a ``typing.Protocol`` with two methods:

* ``generate(prompt, *, context=None, **kwargs) -> str``
* ``token_logprobs(prompt, response) -> list[float]``  (used by the predictive-entropy uncertainty estimator)

Backends live under ``hat.models.backends`` and are registered with ``hat.models.registry``. Each backend lazy-imports its heavy dependency, so the package imports cleanly with only the base extras installed.

## Consequences
* Adding a backend is one new module + one ``@register("name")`` decorator.
* Tests use a no-op ``Cortex`` that does not implement ``LanguageModel`` directly; ``Cortex`` and ``LanguageModel`` are decoupled because the Cortex *uses* a backend rather than *being* one.
* The Oracle (paper §3.5) is a separate ``OracleClient`` Protocol; today's reference implementation talks OpenAI-compatible HTTP, but the seam is identical.
