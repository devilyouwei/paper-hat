# `ui/js/` — frontend JavaScript

Vanilla ES modules. No framework, no bundler — files are loaded directly
by the browser from the static mount.

## Files

| File | Purpose |
| --- | --- |
| `main.js` | Boot entrypoint. Caches first-load fetches (`/healthz`, active model) and wires top-level tab switching. |
| `util.js` | Tiny DOM/string helpers: `$`/`$$`, `escapeHtml`, `renderBubbleHtml`, `toast`. |
| `api.js` | Same-origin HTTP wrapper (`jget`, `jpost`, `jpatch`, `jdelete`) with consistent error handling. |
| `chat.js` | Chat tab. Manages the session sidebar, streams `chat.completion.chunk` SSE responses (parsed manually from `fetch().body.getReader()`), renders the `<think>…</think>` aside, shows the per-turn `U=…` badge, and forwards the active `session_id` so the server can persist turns. |
| `models.js` | Models tab. Catalog browser per backend with one-click *Download* / *Use* / *Unload*. |
| `memory.js` | Memory tab. Curated-memory (Neocortex) browser, inline edit, and delete via `/api/neocortex`. |
| `traces.js` | Trace-lifecycle right-hand panel. Renders the chronological timeline of `hat_trace_event` SSE chunks emitted by the wake pipeline. |
