# `memory/raw/` — append-only chat history

Stores every wake-step interaction. **Strictly off-limits to the
training pipeline** — only the Hippocampus Agent reads here, to abstract
traces (ADR-002).

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker. |
| `sessions.py` | `JsonlSessionStore` + `Session` model. The production raw-log writer. Each session is a topic-scoped conversation (id, title, created/updated timestamps, message count) stored as one JSONL file under `<raw_root>/sessions/<id>.jsonl`, with `<raw_root>/index.json` holding metadata. Public methods: `list / latest / get / create / rename / delete / append / messages`. |
| `log.py` | `RawInteractionLog` ABC + `JsonlRawLog` (single-file backend, kept for tools and tests). Production code uses `SessionRawLog(store, session_id=...)` so the wake-step writer stays uniform across both backends. |

The session title is auto-summarised by the Cortex on the first turn
(`max_tokens=24`) so the UI sidebar shows topical labels rather than raw
queries.
