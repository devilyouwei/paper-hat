# `api/schemas/` — Pydantic DTOs

Request / response models used by the routers. These are HTTP-layer
shapes; the canonical domain types live in
[`../../core/schemas.py`](../../core/schemas.py).

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker. |
| `chat.py` | `ChatRequest` / `ChatResponse` for the HAT-native `/chat` endpoint. |
| `openai.py` | OpenAI Chat-Completions compatible DTOs (`ChatCompletionRequest`, `ChatCompletionResponse`, streaming chunk shapes). |
| `models.py` | `CatalogItem` and friends for `/api/models` — backend, repo id, install status, size. |
| `sessions.py` | Session list / detail / patch DTOs for `/api/sessions`. |
| `neocortex.py` | Curated-memory entry DTOs for `/api/neocortex` — wraps the on-disk SFT row with derived `query` / `response` projections so the UI can edit without parsing the `messages` list itself. |
