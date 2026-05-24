# `core/hippocampus/scoring/` — selection signals

Per-signal scorers consumed by `WritePolicy`. Currently only the
uncertainty channel is wired in; the feedback / novelty channels were
removed in favour of letting the session-aware abstractor detect
corrections from the natural multi-turn conversation.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Re-exports `UncertaintyEstimator`, `ConstantUncertainty`, `LogprobUncertainty`. |
| `uncertainty.py` | Paper §3.4.2 uncertainty signal. `ConstantUncertainty` is a fixed-value placeholder for the no-op loop; `LogprobUncertainty` computes `U(x) = 1 − exp(mean_t log p(y_t | x, y_<t))` against the active Cortex's `chat_logprobs`. |
| `llm_judge.py` | Shared plumbing for any LLM-as-judge step (abstractor triage/route, future scorers): prompt loading + caching from [`../prompts/`](../prompts/), `<think>`-stripping, permissive numeric parsing (`"0.7"`, `"7/10"`, `"70%"`, …) with a fallback. |
