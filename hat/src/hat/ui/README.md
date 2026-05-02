# `ui/` — user interfaces

* `gradio_app.py` — chat UI for end users. Reuses `ChatController`.
* `dashboard.py` — Streamlit operator dashboard for live signals and SWS stats.

Both UIs depend only on the API controllers, so behavior is identical to the
REST surface.
