# `models/backends/` — `LanguageModel` implementations

Concrete inference backends that satisfy the `LanguageModel` protocol in
[`../../core/protocols.py`](../../core/protocols.py). Each backend
lazy-imports its heavy dependencies, so importing this package never
forces torch / mlx into memory.

## Files

| File | Backend name | Extra | Notes |
| --- | --- | --- | --- |
| `__init__.py` | — | — | Package marker. |
| `hf.py` | `hf` | `uv sync --extra hf` | HuggingFace Transformers. Streaming via `TextIteratorStreamer` on a worker thread. Supports Accelerate `device_map="auto"` + `max_memory` offload (auto-retry once on CUDA OOM), bitsandbytes 4-bit, and `chat_logprobs(messages, response)` used by `LogprobUncertainty`. |
| `mlx.py` | `mlx` | `uv sync --extra mlx` | Apple Metal via `mlx-lm`. Streaming via `mlx_lm.stream_generate`. |

Both backends implement `generate()`, `chat(messages, **kw)`,
`stream_chat(messages, **kw)`, and `token_logprobs()`, and forward
`enable_thinking` / `chat_template_kwargs` verbatim to
`tokenizer.apply_chat_template` for Qwen3-style models. They also collect
every plausible chat-turn terminator (`<|im_end|>`, `<|eot_id|>`,
`<|end|>`, …) at load time so chat turns actually stop without hitting
`max_new_tokens`.

Backends register themselves with `@register("name")` from
[`../registry.py`](../registry.py); the manager then discovers them via
`create("name", ...)`.
