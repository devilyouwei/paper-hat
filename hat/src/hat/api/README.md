# `api/` — FastAPI surface

```
api/
├── main.py            # create_app() + /healthz + /logo.png
├── deps.py            # singleton container: settings → manager → loop → log
├── routers/           # thin path → controller adapters (FastAPI types)
│   ├── chat.py        # POST /chat              (HAT-native shape)
│   ├── openai_compat.py  # GET /v1/models, POST /v1/chat/completions
│   └── models.py      # /api/models{,/download,/active}
├── controllers/       # pure-Python orchestration; no FastAPI imports
│   ├── chat.py
│   └── openai_compat.py
└── schemas/           # Pydantic request/response models
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/healthz` | liveness + active backend |
| `POST` | `/chat` | HAT-native chat shape |
| `GET`  | `/v1/models` | OpenAI-compatible model card |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat. Honours `stream=true` (SSE), `extra_body.chat_template_kwargs` (e.g. `enable_thinking` for Qwen3-style models) and `extra_body.session_id` (or top-level `session_id`). The non-streaming response and the SSE chunks both carry `hat_consolidated` / `hat_trace_id` / `hat_session_id` extras |
| `GET`  | `/api/models?backend=` | catalog with installed status |
| `POST` | `/api/models/download` | snapshot_download into `model/<backend>/<id>/` |
| `POST` | `/api/models/active` | activate a catalog entry (evicts the previous Cortex first) |
| `GET`  | `/api/models/active` | currently active `(backend, id)` |
| `DELETE` | `/api/models/active` | unload every cached Cortex, free GPU/Metal memory, fall back to Noop |
| `GET` | `/api/sessions` | list chat sessions (newest first) |
| `POST` | `/api/sessions` | create a new (empty) session, optional title |
| `GET` | `/api/sessions/{id}` | session metadata + full message log |
| `PATCH` | `/api/sessions/{id}` | rename a session |
| `DELETE` | `/api/sessions/{id}` | delete a session and its log |
| `GET` | `/api/neocortex` | list all curated memory entries |
| `GET` | `/api/neocortex/{trace_id}` | fetch a single curated entry |
| `PATCH` | `/api/neocortex/{trace_id}` | edit query / response / score |
| `DELETE` | `/api/neocortex/{trace_id}` | drop the entry from the SFT file and trace sidecar |

## Layering

Routers depend on FastAPI; controllers depend only on Protocols/ABCs from
`hat.core` and `hat.memory`. The web UI under `/` (vanilla HTML/CSS/JS,
mounted from [`src/hat/ui/static/`](../ui/static/)) calls the same REST
surface, so
behaviour is identical between the two front-ends.

`deps.py` is the single dependency container. `swap_active_cortex` /
`deactivate_cortex` are the only helpers that mutate the loop's `cortex`
reference; both go through `ModelManager` so allocator caches are released
correctly (see ADR-004).

## Streaming controller

`OpenAIChatController.handle_stream(req)` is a generator that yields
`data: {chunk_json}\n\n` lines (an opening role chunk, one chunk per text
delta, a closing chunk with `finish_reason="stop"` plus the HAT extras, then
`data: [DONE]\n\n`). It runs `loop.wake_step` **after** the stream completes
on the accumulated text, so consolidation still happens on streamed turns.
