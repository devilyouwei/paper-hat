# `api/routers/` — FastAPI path adapters

Thin shells that bind URL paths to controllers. Routers are the **only**
layer allowed to import from `fastapi`; controllers and below stay pure
Python so they're trivially testable and swappable.

## Files

| File | Mounted at | Purpose |
| --- | --- | --- |
| `__init__.py` | — | Package marker. |
| `chat.py` | `/chat` | HAT-native chat shape (`POST /chat`). Delegates to `controllers.chat.ChatController`. |
| `openai_compat.py` | `/v1` | OpenAI-compatible surface: `GET /v1/models`, `POST /v1/chat/completions` (non-streaming + SSE streaming, with `extra_body.chat_template_kwargs` / `session_id` honoured). |
| `models.py` | `/api/models` | Model lifecycle: list catalog, `snapshot_download` (blocking + SSE progress + cancel), set/get/clear active Cortex, delete weights. |
| `sessions.py` | `/api/sessions` | Chat-session CRUD (list, create, get, rename, delete). |
| `neocortex.py` | `/api/neocortex` | Curated-memory browser/editor — list / get / patch / delete entries. Writes still flow through the wake/sleep loop's `WriteDecision` contract (ADR-002). |

Every dependency (loop, raw log, session store, model manager) is injected
via `Depends(get_*)` from [`../deps.py`](../deps.py), so routers never
construct backends directly.
