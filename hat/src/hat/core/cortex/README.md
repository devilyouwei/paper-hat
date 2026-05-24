# `core/cortex/` — online interaction model (paper §3.3)

The Cortex is the "wake-time" learner: it answers user queries and
exposes an uncertainty signal that the Hippocampus uses to gate trace
consolidation.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Re-exports `Cortex`, `NoopCortex`. |
| `base.py` | `Cortex` ABC. Required surface: `generate(query, *, context)`, `chat(messages, **kw)`, `stream_chat(messages, **kw)`, `uncertainty(interaction)`. |
| `noop.py` | `NoopCortex` — env-free echo Cortex. Used by tests and as the bootstrap before a model is activated. |
| `hf_cortex.py` | `HFCortex` — wraps an `HFLanguageModel` from [`../../models/backends/hf.py`](../../models/backends/hf.py). Forwards `enable_thinking` and full message lists into the HF chat template; uncertainty uses per-token log-probs via `chat_logprobs`. |
| `mlx_cortex.py` | `MLXCortex` — wraps an `MLXLanguageModel`. Same shape as `HFCortex` for Apple Silicon. |

Concrete cortices are built by `ModelManager` (see
[`../../models/README.md`](../../models/README.md)); the API layer never
instantiates them directly.
