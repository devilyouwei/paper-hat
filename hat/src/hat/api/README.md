# `api/` — FastAPI surface

Routers (`routers/`) map paths to controllers (`controllers/`). Controllers
depend only on Protocols/ABCs from `hat.core` and `hat.memory`, not on FastAPI
types — this keeps them testable in isolation and reusable by the Gradio UI and
the CLI.

`deps.py` is the single dependency container; swap defaults there to switch
backends globally.
