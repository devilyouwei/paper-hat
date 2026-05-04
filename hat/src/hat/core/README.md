# `core/` — paper algorithms

Pure Python. No FastAPI, no torch, no I/O. Every heavy dependency is a Protocol
or ABC defined here and implemented elsewhere (`hat.models`, `hat.memory`, …).

| File / package | Paper section |
| --- | --- |
| `schemas.py` | §3.1, §3.4, §3.7 data types |
| `protocols.py` | All pluggable seams |
| `cortex/` | §3.3 — `base.Cortex`, `noop.NoopCortex`, `mlx_cortex.MLXCortex`, `hf_cortex.HFCortex` |
| `hippocampus/` | §3.4 (abstraction, selection, replay) + scoring channels |
| `neocortex/` | §3.6 — write-token contract enforced here |
| `oracle/` | §3.5 |
| `sws/` | §3.7 |
| `loop.py` | §3.8 — wake/sleep orchestrator |

## Cortex contract

`Cortex` (in `cortex/base.py`) defines the canonical surface:

| Method | Used by |
| --- | --- |
| `generate(query, *, context=None, **kw)` | HAT-native `/chat` |
| `chat(messages, **kw)` | OpenAI-compatible non-streaming path |
| `stream_chat(messages, **kw)` *(generator)* | OpenAI-compatible SSE path |
| `uncertainty(interaction)` | Hippocampus selection (αU + βF + γN) |

The two model-backed cortices (`MLXCortex`, `HFCortex`) delegate to a
`LanguageModel` from `hat.models.backends`; `NoopCortex` is the env-free
fallback used when no model is active and by tests.

`stream_chat` and `chat` both pop `enable_thinking` (or a nested
`chat_template_kwargs`) and forward it to `tokenizer.apply_chat_template`,
so think-mode toggling lives at the backend layer.
