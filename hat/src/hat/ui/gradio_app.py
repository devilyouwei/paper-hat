"""Gradio chat UI.

Talks to a running HAT server over its OpenAI-compatible endpoint
(``/v1/chat/completions``) so the same UI works with any OpenAI-compatible
backend (HAT, vLLM, Ollama, GPT-4o, …).

Run::

    make serve   # in one terminal
    make ui      # in another
"""

from __future__ import annotations

from collections.abc import Iterator

from ..config.settings import get_settings


def _client():
    from openai import OpenAI

    s = get_settings()
    return OpenAI(base_url=s.ui_base_url, api_key="hat-local")


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

    def respond(message: object, history: list[dict], correction: str) -> Iterator[tuple[list[dict], str]]:
        # Gradio's `messages` chat format: list[{"role","content"}]
        history = list(history or [])
        history.append({"role": "user", "content": _flatten(message)})

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history:
            msgs.append({"role": h["role"], "content": _flatten(h.get("content"))})
        if correction:
            # Tag the correction so the Hippocampus can see it when we wire
            # explicit feedback through later.
            msgs.append(
                {"role": "system", "content": f"[user correction]: {correction}"}
            )

        try:
            resp = client.chat.completions.create(
                model=s.ui_model,
                messages=msgs,
                temperature=s.hf_temperature,
            )
            text = resp.choices[0].message.content or ""
            extra = getattr(resp, "hat_consolidated", None)
            if extra:
                text += f"\n\n_— consolidated as trace {getattr(resp, 'hat_trace_id', '?')}_"
        except Exception as e:  # surface server errors in the UI
            text = f"[error] {type(e).__name__}: {e}"

        history.append({"role": "assistant", "content": text})
        yield history, ""

    with gr.Blocks(title="HAT — local chat") as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=0, min_width=120):
                # Served by the HAT FastAPI app at /logo.png; falls back to the
                # repo-relative file when the Gradio app is launched standalone.
                logo_src = f"{s.ui_base_url.rstrip('/').removesuffix('/v1')}/logo.png"
                gr.HTML(
                    f'<img src="{logo_src}" alt="HAT" '
                    f'style="width:96px;height:auto;display:block;margin:8px;" />'
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    f"# HAT — Hippocampus-Augmented Transformer\n"
                    f"Server: `{s.ui_base_url}` &nbsp;·&nbsp; "
                    f"Model: `{s.ui_model}` &nbsp;·&nbsp; "
                    f"Cortex: `{s.cortex_backend}`"
                )
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
            label="Optional correction (becomes feedback for the Hippocampus)",
            placeholder="If the previous answer was wrong, write the correct answer here.",
        )
        clear = gr.Button("Clear conversation")

        send.click(respond, [msg, chatbot, correction], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, correction], [chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])

    return demo


def main() -> None:  # pragma: no cover
    build().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=__import__("gradio").themes.Soft(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
