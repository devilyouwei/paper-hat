"""Gradio chat UI.

Talks to a running HAT server over:

* ``/v1/chat/completions`` for chat (OpenAI-compatible)
* ``/api/models/*``        for catalog browsing, downloads, and active-model
                           switching

Run::

    make serve   # in one terminal
    make ui      # in another
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx

from ..config.settings import get_settings


def _server_root() -> str:
    """Strip the ``/v1`` suffix from ``HAT_UI_BASE_URL`` to get the host root."""
    return get_settings().ui_base_url.rstrip("/").removesuffix("/v1")


def _client():
    from openai import OpenAI

    s = get_settings()
    return OpenAI(base_url=s.ui_base_url, api_key="hat-local")


# ---------- model-management HTTP helpers ----------------------------------


def _api_list_models(backend: str) -> list[dict]:
    r = httpx.get(f"{_server_root()}/api/models", params={"backend": backend}, timeout=10)
    r.raise_for_status()
    return r.json()["items"]


def _api_active() -> dict | None:
    r = httpx.get(f"{_server_root()}/api/models/active", timeout=10)
    r.raise_for_status()
    return r.json()


def _api_download(backend: str, model_id: str) -> dict:
    # No timeout — downloads can be GB-sized and take minutes.
    r = httpx.post(
        f"{_server_root()}/api/models/download",
        json={"backend": backend, "id": model_id},
        timeout=None,
    )
    r.raise_for_status()
    return r.json()


def _api_set_active(backend: str, model_id: str) -> dict:
    r = httpx.post(
        f"{_server_root()}/api/models/active",
        json={"backend": backend, "id": model_id},
        timeout=None,
    )
    r.raise_for_status()
    return r.json()


def _label(item: dict) -> str:
    tag = "✓ installed" if item["installed"] else "↓ not installed"
    size = f", {item['size_gb']:.1f} GB" if item.get("size_gb") else ""
    return f"{item['display']} ({tag}{size})"


def _options(items: list[dict]) -> list[tuple[str, str]]:
    """Gradio Dropdown choices: list of (label, value) where value is the id."""
    return [(_label(i), i["id"]) for i in items]


def _installed_options(items: list[dict]) -> list[tuple[str, str]]:
    return [(i["display"], i["id"]) for i in items if i["installed"]]


# ---------- main UI --------------------------------------------------------


def build():  # pragma: no cover - UI
    import gradio as gr

    s = get_settings()
    client = _client()

    SYSTEM_PROMPT = (
        "You are HAT, a helpful assistant. Be concise and accurate."
    )

    def _flatten(content: object) -> str:
        """Gradio 6 may pass content as ``[{"type":"text","text":...}, ...]``;
        the OpenAI-compat endpoint expects plain strings."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(str(p.get("text") or p.get("content") or ""))
                else:
                    parts.append(str(p))
            return "".join(parts)
        return str(content) if content is not None else ""

    def respond(
        message: object,
        history: list[dict],
        correction: str,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[tuple[list[dict], str]]:
        history = list(history or [])
        history.append({"role": "user", "content": _flatten(message)})

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history:
            msgs.append({"role": h["role"], "content": _flatten(h.get("content"))})
        if correction:
            msgs.append(
                {"role": "system", "content": f"[user correction]: {correction}"}
            )

        try:
            resp = client.chat.completions.create(
                model=s.ui_model,
                messages=msgs,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            text = resp.choices[0].message.content or ""
            if getattr(resp, "hat_consolidated", None):
                text += f"\n\n_— consolidated as trace {getattr(resp, 'hat_trace_id', '?')}_"
        except Exception as e:
            text = f"[error] {type(e).__name__}: {e}"

        history.append({"role": "assistant", "content": text})
        yield history, ""

    # ---- model tab callbacks ---------------------------------------------

    def on_backend_change(backend: str):
        try:
            items = _api_list_models(backend)
        except Exception as e:
            return gr.update(choices=[], value=None), f"failed to load catalog: {e}"
        return gr.update(choices=_options(items), value=items[0]["id"] if items else None), ""

    def on_download(backend: str, model_id: str):
        if not model_id:
            return gr.update(), "select a model first"
        try:
            info = _api_download(backend, model_id)
            items = _api_list_models(backend)
            return (
                gr.update(choices=_options(items), value=model_id),
                f"downloaded {model_id} → {info['local_dir']}",
            )
        except Exception as e:
            return gr.update(), f"download failed: {e}"

    def on_use(backend: str, model_id: str):
        if not model_id:
            return "select a model first", gr.update(), gr.update()
        try:
            info = _api_set_active(backend, model_id)
            items_for_chat = _api_list_models(backend)
            chat_label = f"Active: {info['backend']}/{info['id']}"
            return (
                f"active model: {info['backend']}/{info['id']} ({info.get('name') or '?'})",
                gr.update(choices=_installed_options(items_for_chat), value=model_id),
                gr.update(value=chat_label),
            )
        except Exception as e:
            return f"failed to activate: {e}", gr.update(), gr.update()

    def on_chat_switch(backend: str, model_id: str):
        if not model_id:
            return gr.update(value="(no model selected)")
        try:
            info = _api_set_active(backend, model_id)
            return gr.update(value=f"Active: {info['backend']}/{info['id']}")
        except Exception as e:
            return gr.update(value=f"switch failed: {e}")

    def initial_chat_dropdown(backend: str):
        try:
            items = _api_list_models(backend)
        except Exception:
            return gr.update(choices=[], value=None)
        active = None
        try:
            a = _api_active()
            active = a["id"] if a else None
        except Exception:
            pass
        choices = _installed_options(items)
        return gr.update(choices=choices, value=active or (choices[0][1] if choices else None))

    def initial_active_label():
        try:
            a = _api_active()
        except Exception:
            return "Active: (server unreachable)"
        if not a:
            return f"Active: bootstrap ({s.cortex_backend})"
        return f"Active: {a['backend']}/{a['id']}"

    # ---- layout ----------------------------------------------------------

    with gr.Blocks(title="HAT — local chat") as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=0, min_width=120):
                logo_src = f"{_server_root()}/logo.png"
                gr.HTML(
                    f'<img src="{logo_src}" alt="HAT" '
                    f'style="width:96px;height:auto;display:block;margin:8px;" />'
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    f"# HAT — Hippocampus-Augmented Transformer\n"
                    f"Server: `{s.ui_base_url}`"
                )
                active_label = gr.Markdown(initial_active_label())

        with gr.Tabs():
            # -------- Chat tab --------
            with gr.Tab("Chat"):
                with gr.Row():
                    chat_backend = gr.Radio(
                        choices=["mlx", "hf"],
                        value=s.cortex_backend if s.cortex_backend in ("mlx", "hf") else "mlx",
                        label="Backend",
                        scale=1,
                    )
                    chat_model = gr.Dropdown(
                        choices=[], value=None, label="Active model (installed)", scale=3,
                    )
                    chat_use = gr.Button("Use this model", scale=1)
                chatbot = gr.Chatbot(label="Cortex", height=480)
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask anything…",
                        scale=4,
                        autofocus=True,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                correction = gr.Textbox(
                    label="Optional correction (feedback for the Hippocampus)",
                    placeholder="If the previous answer was wrong, write the correct answer here.",
                )
                with gr.Accordion("Generation settings", open=False):
                    with gr.Row():
                        temp_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.5,
                            step=0.05,
                            value=s.default_temperature,
                            label="Temperature",
                        )
                        max_tokens_slider = gr.Slider(
                            minimum=32,
                            maximum=4096,
                            step=32,
                            value=s.default_max_tokens,
                            label="Max tokens",
                        )
                clear = gr.Button("Clear conversation")

                send.click(
                    respond,
                    [msg, chatbot, correction, temp_slider, max_tokens_slider],
                    [chatbot, msg],
                )
                msg.submit(
                    respond,
                    [msg, chatbot, correction, temp_slider, max_tokens_slider],
                    [chatbot, msg],
                )
                clear.click(lambda: ([], ""), outputs=[chatbot, msg])

                chat_backend.change(initial_chat_dropdown, [chat_backend], [chat_model])
                chat_use.click(
                    on_chat_switch, [chat_backend, chat_model], [active_label]
                )

            # -------- Models tab --------
            with gr.Tab("Models"):
                gr.Markdown(
                    "Browse the catalog, download weights into "
                    "`model/<backend>/<id>/`, and pick which model the Cortex uses."
                )
                with gr.Row():
                    mgr_backend = gr.Radio(
                        choices=["mlx", "hf"], value="mlx", label="Backend", scale=1
                    )
                    mgr_model = gr.Dropdown(
                        choices=[], value=None, label="Catalog", scale=3
                    )
                with gr.Row():
                    btn_download = gr.Button("Download", variant="secondary")
                    btn_use = gr.Button("Use as active", variant="primary")
                mgr_status = gr.Textbox(label="Status", interactive=False)

                mgr_backend.change(
                    on_backend_change, [mgr_backend], [mgr_model, mgr_status]
                )
                btn_download.click(
                    on_download, [mgr_backend, mgr_model], [mgr_model, mgr_status]
                )
                btn_use.click(
                    on_use,
                    [mgr_backend, mgr_model],
                    [mgr_status, chat_model, active_label],
                )

        # Populate dropdowns once the UI is ready.
        demo.load(on_backend_change, [mgr_backend], [mgr_model, mgr_status])
        demo.load(initial_chat_dropdown, [chat_backend], [chat_model])

    return demo


def main() -> None:  # pragma: no cover
    build().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=__import__("gradio").themes.Soft(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
