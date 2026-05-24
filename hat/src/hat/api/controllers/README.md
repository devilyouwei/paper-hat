# `api/controllers/` — pure-Python orchestration

Controllers turn validated request DTOs into calls against the core
protocols (`WakeSleepLoop`, `RawInteractionLog`, `JsonlSessionStore`, …).
No FastAPI imports here — routers are responsible for HTTP concerns.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker. |
| `chat.py` | `ChatController` — runs one wake step and appends to the raw log. Returns the HAT-native `ChatResponse`. Never touches the Neocortex directly. |
| `openai_compat.py` | `OpenAIChatController` — implements `/v1/chat/completions`. Forwards full message lists into the Cortex's chat template, runs the wake step on the last user turn, persists each turn to the session store, emits SSE chunks with `hat_consolidated` / `hat_trace_id` / `hat_session_id` extras, and auto-titles brand-new sessions. |

Streaming is handled in `OpenAIChatController.handle_stream(req)`, a
generator that yields `data: {chunk}\n\n` lines and runs consolidation
**after** the stream completes on the accumulated text.
