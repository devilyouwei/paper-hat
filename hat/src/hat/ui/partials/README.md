# `ui/partials/` — HTML partials

Per-tab HTML fragments included by `../index.html`. Plain markup —
behaviour lives in [`../js/`](../js/README.md) and styling in
[`../css/`](../css/README.md).

## Files

| File | Tab | Purpose |
| --- | --- | --- |
| `chat.html` | Chat | Session sidebar, model controls, streaming chatbot, generation-settings disclosure (temperature / max-tokens / **Enable thinking** / **Show thinking process**). |
| `models.html` | Models | Catalog browser per backend with *Download* and *Use as active* buttons. |
| `memory.html` | Memory | Curated-memory (Neocortex) table with inline edit / delete. |
