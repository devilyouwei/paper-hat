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


def _api_deactivate() -> dict:
    r = httpx.delete(f"{_server_root()}/api/models/active", timeout=60)
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


# ---------- <think>...</think> streaming filter ----------------------------


class _ThinkFilter:
    """Stream-friendly filter for the Qwen3.5-style ``<think>...</think>``
    block. ``show_thinking=False`` drops the block entirely; ``True`` keeps the
    raw text untouched (the UI styles it later)."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, show_thinking: bool) -> None:
        self.show = bool(show_thinking)
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        if self.show:
            return chunk  # passthrough; UI gets raw stream
        self._buf += chunk
        out = ""
        while True:
            if self._in_think:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    # keep last (len(CLOSE)-1) chars in case CLOSE straddles
                    keep = len(self.CLOSE) - 1
                    if len(self._buf) > keep:
                        self._buf = self._buf[-keep:]
                    return out
                self._buf = self._buf[idx + len(self.CLOSE) :]
                self._in_think = False
                continue
            idx = self._buf.find(self.OPEN)
            if idx == -1:
                keep = len(self.OPEN) - 1
                if len(self._buf) > keep:
                    out += self._buf[:-keep]
                    self._buf = self._buf[-keep:]
                return out
            out += self._buf[:idx]
            self._buf = self._buf[idx + len(self.OPEN) :]
            self._in_think = True

    def flush(self) -> str:
        if self.show:
            tail, self._buf = self._buf, ""
            return tail
        if self._in_think:
            self._buf = ""
            return ""
        tail, self._buf = self._buf, ""
        return tail


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
        enable_thinking: bool,
        show_thinking: bool,
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

        # initial empty assistant slot — we mutate `.content` as chunks arrive.
        history.append({"role": "assistant", "content": ""})
        yield history, ""

        filt = _ThinkFilter(show_thinking=show_thinking)
        visible = ""

        try:
            stream = client.chat.completions.create(
                model=s.ui_model,
                messages=msgs,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                stream=True,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": bool(enable_thinking),
                    }
                },
            )
            consolidated_note = ""
            for ev in stream:
                if not ev.choices:
                    # final chunk may arrive without choices but with HAT extras
                    if getattr(ev, "hat_consolidated", None):
                        consolidated_note = (
                            f"\n\n_— consolidated as trace "
                            f"{getattr(ev, 'hat_trace_id', '?')}_"
                        )
                    continue
                delta = ev.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    out = filt.feed(piece)
                    if out:
                        visible += out
                        history[-1]["content"] = visible
                        yield history, ""
                # also harvest HAT extras if the SDK exposed them on the chunk
                if getattr(ev, "hat_consolidated", None):
                    consolidated_note = (
                        f"\n\n_— consolidated as trace "
                        f"{getattr(ev, 'hat_trace_id', '?')}_"
                    )
            tail = filt.flush()
            if tail:
                visible += tail
            if consolidated_note:
                visible += consolidated_note
            history[-1]["content"] = visible or "(no content)"
            yield history, ""
        except Exception as e:
            history[-1]["content"] = f"[error] {type(e).__name__}: {e}"
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

    def on_deactivate(backend: str):
        try:
            res = _api_deactivate()
        except Exception as e:
            return gr.update(value=f"deactivate failed: {e}"), gr.update()
        # Refresh the installed-models dropdown so the now-inactive entry no
        # longer appears as "selected".
        try:
            items = _api_list_models(backend)
            dd = gr.update(choices=_installed_options(items), value=None)
        except Exception:
            dd = gr.update(value=None)
        return (
            gr.update(value=f"Active: — (unloaded {res.get('unloaded', 0)} model(s))"),
            dd,
        )

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
                    chat_unload = gr.Button("Unload", scale=1)
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
                    with gr.Row():
                        enable_thinking_cb = gr.Checkbox(
                            value=False,
                            label="Enable thinking (Qwen3 <think>…)",
                            info="Forwarded as chat_template_kwargs.enable_thinking; only effective on think-capable models.",
                        )
                        show_thinking_cb = gr.Checkbox(
                            value=False,
                            label="Show thinking process",
                            info="If off, the <think>…</think> block is hidden from the chat view.",
                        )
                clear = gr.Button("Clear conversation")

                send.click(
                    respond,
                    [msg, chatbot, correction, temp_slider, max_tokens_slider,
                     enable_thinking_cb, show_thinking_cb],
                    [chatbot, msg],
                )
                msg.submit(
                    respond,
                    [msg, chatbot, correction, temp_slider, max_tokens_slider,
                     enable_thinking_cb, show_thinking_cb],
                    [chatbot, msg],
                )
                clear.click(lambda: ([], ""), outputs=[chatbot, msg])

                chat_backend.change(initial_chat_dropdown, [chat_backend], [chat_model])
                chat_use.click(
                    on_chat_switch, [chat_backend, chat_model], [active_label]
                )
                chat_unload.click(
                    on_deactivate, [chat_backend], [active_label, chat_model]
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
        server_name="0.0.0.0",
        server_port=7860,
        theme=__import__("gradio").themes.Soft(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
