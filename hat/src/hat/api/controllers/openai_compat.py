"""Controller for OpenAI-compatible chat completions.

Strategy:

* The conversation prefix (everything except the last user message) is fed to
  the Cortex as a flat ``context`` string. If the Cortex has a native
  :meth:`chat` method (e.g. :class:`HFCortex`), the full message list is
  forwarded so the model's chat template handles role formatting natively.
* The last user message is then run through the wake step so the Hippocampus
  Agent can score and (selectively) consolidate the turn.
* A single response is returned in OpenAI ``chat.completion`` shape, or a
  generator of ``chat.completion.chunk`` SSE payloads when streaming.
* Each turn is appended to the caller-supplied session (``req.session_id``)
  via the session store. The first turn of a brand-new session triggers an
  asynchronous-style title summary (kept short) so the UI can label it.
"""

from __future__ import annotations

import uuid
import re
from collections.abc import Iterator
from dataclasses import dataclass

from ...core.loop import WakeSleepLoop
from ...core.schemas import Interaction
from ...memory.raw.log import RawInteractionLog
from ...memory.raw.sessions import (
    JsonlSessionStore,
    Session,
    SessionStoreError,
)
from ..schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)

def _split_messages(messages: list[ChatMessage]) -> tuple[list[ChatMessage], ChatMessage]:
    if not messages:
        raise ValueError("messages must be non-empty")
    last = messages[-1]
    if last.role != "user":
        raise ValueError("the final message must have role='user'")
    return messages[:-1], last


def _flatten_history(history: list[ChatMessage]) -> str | None:
    if not history:
        return None
    parts = [f"{m.role}: {m.content}" for m in history]
    return "\n".join(parts)


def _build_gen_kwargs(req: ChatCompletionRequest) -> dict:
    gen_kwargs: dict = {}
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        gen_kwargs["max_tokens"] = req.max_tokens
    if req.chat_template_kwargs:
        gen_kwargs["chat_template_kwargs"] = dict(req.chat_template_kwargs)
    return gen_kwargs


def _truncate_title(text: str, *, limit: int = 48) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "New chat"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (closed or unterminated) from a model
    response so that thinking-style models still produce a usable title."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    # Drop an unterminated leading <think> (some models forget the close tag
    # when truncated by max_tokens).
    lower = text.lower()
    if "<think>" in lower and "</think>" not in lower:
        idx = lower.rfind("<think>")
        text = text[:idx]
    return text.strip()


def _summarize_title(cortex, query: str) -> str:
    """Best-effort short-title generation. Falls back to the truncated query
    if the cortex is the noop one or if anything throws."""
    try:
        if hasattr(cortex, "chat") and callable(cortex.chat):
            prompt = (
                "Summarize the user's message into a short, plain-text chat "
                "title (max 8 words, no quotes, no punctuation at the end)."
            )
            text = cortex.chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=64,
                temperature=0.2,
                # Disable Qwen3-style <think> blocks for title generation.
                # Backends that don't understand the kwarg simply ignore it.
                chat_template_kwargs={"enable_thinking": False},
            )
            cleaned = _strip_think(text or "").strip().strip('"').strip("'")
            # First non-empty line only — titles should be a single line.
            if cleaned:
                cleaned = cleaned.splitlines()[0].strip().strip('"').strip("'")
            if cleaned and not cleaned.lower().startswith("[noop]"):
                return _truncate_title(cleaned)
    except Exception:
        pass
    return _truncate_title(query)


def _summarize_events(events: list[dict], trace) -> dict | None:
    """Distil the lifecycle event stream into the compact per-turn record
    that is persisted with the raw session log. Used so the UI can re-render
    the uncertainty / route badge when the session is reopened.

    Returns ``None`` when no useful signal was captured (e.g. the loop never
    ran). The shape matches what ``attachUncertaintyBadge`` consumes."""
    if not events:
        return None
    by_stage: dict[str, dict] = {}
    for ev in events:
        s = ev.get("stage")
        if isinstance(s, str):
            by_stage[s] = ev
    u_ev = by_stage.get("uncertainty") or {}
    uncertainty = u_ev.get("uncertainty")
    if uncertainty is None:
        sig = u_ev.get("signals") or {}
        uncertainty = sig.get("uncertainty")
    # Final decision in the lifecycle. Order matters: a turn can transition
    # uncertainty -> skipped, or uncertainty -> abstracting -> dropped, etc.
    for stage in ("revised", "created", "rejected", "dropped", "skipped"):
        if stage in by_stage:
            decision = stage
            break
    else:
        decision = None
    routed = by_stage.get("routed") or {}
    record: dict = {}
    if uncertainty is not None:
        record["uncertainty"] = float(uncertainty)
    if decision is not None:
        record["decision"] = decision
    if trace is not None and getattr(trace, "id", None):
        record["trace_id"] = trace.id
    for key in ("novelty", "user_signal", "reason"):
        val = routed.get(key)
        if val is not None:
            record[key] = val
    return record or None


@dataclass
class OpenAIChatController:
    loop: WakeSleepLoop
    raw_log: RawInteractionLog  # legacy fallback (single-file log)
    sessions: JsonlSessionStore | None = None

    # ----- session helpers ---------------------------------------------

    def _resolve_session(
        self, session_id: str | None, *, first_query: str
    ) -> Session | None:
        """Return the target session for this turn (or ``None`` if no session
        store is wired). May create the session if the caller passed an unknown
        id; auto-titles brand-new sessions on the first turn."""
        if self.sessions is None:
            return None
        if session_id:
            try:
                return self.sessions.get(session_id)
            except SessionStoreError:
                # Caller knows the id they want — honour it by creating
                # the session under that id is not supported, so fall back
                # to a fresh one.
                pass
        # No session id (or unknown one) — create a new one with a stub title
        # that the caller upgrades after generation completes.
        return self.sessions.create(title=_truncate_title(first_query))

    def _persist_turn(
        self, session: Session | None, interaction: Interaction
    ) -> None:
        if session is not None and self.sessions is not None:
            self.sessions.append(session.id, interaction)
            # First turn: refine the title using the cortex if we have a
            # better summary now that we've already paid for inference.
            if session.message_count <= 1 and session.title in {
                "New chat",
                _truncate_title(interaction.query),
            }:
                better = _summarize_title(self.loop.cortex, interaction.query)
                if better and better != session.title:
                    try:
                        self.sessions.rename(session.id, better)
                    except SessionStoreError:
                        pass
        else:
            self.raw_log.append(interaction)

    # ----- non-streaming -----------------------------------------------

    def handle(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        history, last = _split_messages(req.messages)
        gen_kwargs = _build_gen_kwargs(req)
        session = self._resolve_session(req.session_id, first_query=last.content)

        # Pull session-scoped prior traces so the abstractor can decide
        # CREATE vs REVISE. Done lazily to avoid a hard import cycle.
        prior_traces: list = []
        if session is not None:
            from ..deps import prior_traces_for_session
            prior_traces = prior_traces_for_session(session.id)

        events: list[dict] = []

        def _sink(stage: str, payload: dict) -> None:
            events.append({"stage": stage, **payload})

        cortex = self.loop.cortex
        response_text: str
        if hasattr(cortex, "chat") and callable(cortex.chat):
            response_text = cortex.chat(
                [m.model_dump() for m in req.messages], **gen_kwargs
            )
            interaction = Interaction(
                session_id=session.id if session else None,
                context=_flatten_history(history),
                query=last.content,
                response=response_text,
            )
            trace = self.loop.wake_step(
                interaction, prior_traces=prior_traces or None, event_sink=_sink,
            )
        else:
            interaction = Interaction(
                session_id=session.id if session else None,
                context=_flatten_history(history),
                query=last.content,
            )
            trace = self.loop.wake_step(
                interaction, prior_traces=prior_traces or None, event_sink=_sink,
            )
            response_text = interaction.response or ""

        interaction.hat = _summarize_events(events, trace)
        self._persist_turn(session, interaction)

        return ChatCompletionResponse(
            model=getattr(cortex, "name", req.model),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            hat_consolidated=trace is not None,
            hat_trace_id=trace.id if trace else None,
            hat_session_id=session.id if session else None,
            hat_trace_events=events or None,
        )

    # ------------------------------------------------------------------ stream

    def handle_stream(self, req: ChatCompletionRequest) -> Iterator[str]:
        """Yield SSE-formatted ``data: {...}\\n\\n`` strings (plus terminator).

        Falls back to a single-chunk emit if the active Cortex does not
        implement ``stream_chat``.
        """
        history, last = _split_messages(req.messages)
        gen_kwargs = _build_gen_kwargs(req)
        session = self._resolve_session(req.session_id, first_query=last.content)

        cortex = self.loop.cortex
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model_name = getattr(cortex, "name", req.model)

        def chunk(delta: ChatCompletionDelta, finish: str | None = None,
                  *, hat_consolidated: bool | None = None,
                  hat_trace_id: str | None = None,
                  hat_session_id: str | None = None,
                  hat_trace_event: dict | None = None) -> str:
            payload = ChatCompletionChunk(
                id=chat_id,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0, delta=delta, finish_reason=finish
                    )
                ],
                hat_consolidated=hat_consolidated,
                hat_trace_id=hat_trace_id,
                hat_session_id=hat_session_id,
                hat_trace_event=hat_trace_event,
            )
            return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"

        # Opening role chunk + session id (so the UI can pin newly-created
        # sessions before the stream completes).
        yield chunk(
            ChatCompletionDelta(role="assistant"),
            hat_session_id=session.id if session else None,
        )

        msgs = [m.model_dump() for m in req.messages]
        collected: list[str] = []

        if hasattr(cortex, "stream_chat") and callable(cortex.stream_chat):
            try:
                for piece in cortex.stream_chat(msgs, **gen_kwargs):
                    if not piece:
                        continue
                    collected.append(piece)
                    yield chunk(ChatCompletionDelta(content=piece))
            except Exception as e:  # pragma: no cover - runtime safety net
                yield chunk(
                    ChatCompletionDelta(content=f"\n[stream error] {type(e).__name__}: {e}")
                )
        else:
            # No streaming support — fall back to a single blocking chat() call.
            text = (
                cortex.chat(msgs, **gen_kwargs)
                if hasattr(cortex, "chat") and callable(cortex.chat)
                else cortex.generate(last.content, context=_flatten_history(history))
            )
            collected.append(text)
            yield chunk(ChatCompletionDelta(content=text))

        full = "".join(collected)
        interaction = Interaction(
            session_id=session.id if session else None,
            context=_flatten_history(history),
            query=last.content,
            response=full,
        )

        # Pull prior traces (session-scoped) so the abstractor can REVISE.
        prior_traces: list = []
        if session is not None:
            from ..deps import prior_traces_for_session
            prior_traces = prior_traces_for_session(session.id)

        events: list[dict] = []

        def _sink(stage: str, payload: dict) -> None:
            events.append({"stage": stage, **payload})

        trace = self.loop.wake_step(
            interaction, prior_traces=prior_traces or None, event_sink=_sink,
        )
        interaction.hat = _summarize_events(events, trace)
        self._persist_turn(session, interaction)

        # Forward each lifecycle event as its own chunk so the UI can render
        # trace creation/revision in real time alongside the response.
        for ev in events:
            yield chunk(ChatCompletionDelta(), hat_trace_event=ev)

        # Closing chunk + DONE marker.
        yield chunk(
            ChatCompletionDelta(),
            finish="stop",
            hat_consolidated=trace is not None,
            hat_trace_id=trace.id if trace else None,
            hat_session_id=session.id if session else None,
        )
        yield "data: [DONE]\n\n"
