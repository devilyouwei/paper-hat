# `core/oracle/` — on-demand external teacher (paper §3.5)

The Oracle is consulted only when the Cortex is unsure (`U(x) > τ_O`),
and only when explicitly enabled via `HAT_ORACLE_ENABLED=true`. Its
reply replaces the Cortex's response as the supervision target for the
abstracted trace.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Re-exports `Oracle`, `NoopOracle`, `OpenAICompatibleOracle`, `CostGuard`, `OracleQuotaExceeded`. |
| `base.py` | `Oracle` ABC (`consult(interaction) -> str`) + `NoopOracle` echo for tests. |
| `openai_compat.py` | `OpenAICompatibleOracle` — HTTP client for any `/v1/chat/completions` endpoint (OpenAI, vLLM, Ollama with `OLLAMA_OPENAI=1`, Groq, Together, …). Network errors degrade silently to an empty correction. |
| `cost_guard.py` | `CostGuard` — sliding-window rate limit + per-day budget cap, with a JSONL audit log. Raises `OracleQuotaExceeded` when a limit is hit. |
| `prompts.py` | `ORACLE_SYSTEM` — the teacher system prompt. Constant string (wording matters for downstream parsing), not a separate `.md`. |
