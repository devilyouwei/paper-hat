# `ui/` — user interface

* `static/` — vanilla HTML/CSS/JS web app served by FastAPI at <http://127.0.0.1:8000/>.
  No build step, no JS framework. Talks to the same REST surface any external
  client would use.

The web UI is mounted automatically when the API server starts; there is no
separate UI process to launch. Run:

```sh
make serve
# open http://127.0.0.1:8000/
```

## Web app structure

Three tabs, all backed by REST:

### Chat tab
- **Sidebar (left)** — ChatGPT-style session list:
  - `+ New chat` button → `POST /api/sessions`.
  - Clickable list of past sessions (title + message count).
  - `Delete current` removes the active session and falls back to the latest one.
- **Main column** — model controls + chatbot:
  - Backend selector (`mlx` / `hf`) + active-model dropdown + **Active model** /
    **Unload** buttons. *Use* calls `POST /api/models/active`; the manager evicts
    the previous Cortex first to keep peak memory at one model. *Unload* calls
    `DELETE /api/models/active` and frees all GPU/Metal memory.
  - Streaming chatbot. Tokens are appended to the assistant bubble as they
    arrive over SSE (parsed manually from `fetch().body.getReader()`). The
    session id is forwarded in the request body, and the server's
    `hat_session_id` extra is captured into JS state so follow-up turns reuse
    the same session.
  - Every assistant bubble shows a small `U=…` badge with the cortex's
    uncertainty on its own response. Turns whose `U` falls below the gate
    threshold display `U=… · skipped` and never become a trace.
  - "Generation settings" disclosure: temperature, max-tokens, **Enable
    thinking** (forwarded as `chat_template_kwargs.enable_thinking`), and
    **Show thinking process** (UI-only filter that styles `<think>…</think>`
    blocks as a muted aside, or hides them entirely).

On first load the app auto-opens the most recent session, creating one if the
index is empty.

### Models tab
Catalog browser per backend with one-click *Download* and *Use as active*.
Sizes and installed-status badges come from `/api/models`.

### Memory tab
Curated-memory (Neocortex) browser. Lists every accepted entry in a plain
HTML table (id, score, query, response). *Edit* opens an inline editor for
query / response / score (`PATCH /api/neocortex/{trace_id}`); *Delete*
removes the entry from both the SFT file and the trace sidecar.

Writes still go exclusively through the wake/sleep loop's `WriteDecision`
contract (ADR-002); the Memory tab only edits or removes entries that were
already accepted, which is a manual-curation surface for the operator.

## Streaming + think filter

`renderThink()` in [static/app.js](static/app.js) is a small splitter that
turns a token buffer into a list of `{kind: "text"|"think", value}` parts.
The bubble is re-rendered on every chunk so the user sees tokens stream in,
and `<think>…</think>` blocks are either shown as a styled aside or dropped
entirely depending on the **Show thinking process** checkbox.

Because the bubble is HTML-escaped before insertion (`escapeHtml`), there is
no XSS surface even though the Cortex output is user-generated.

## Sessions

Sessions are persisted server-side under `runs/raw/sessions/<id>.jsonl` (see
[../memory/README.md](../memory/README.md)). The first turn of a brand-new
session triggers an auto-titling call against the active Cortex (`max_tokens=24`)
so the sidebar shows topical labels rather than raw queries.

The web app and any external OpenAI client share the exact same surface;
behaviour is identical.
