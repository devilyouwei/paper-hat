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

import threading
import uuid
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

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


# --------------------------------------------------------------------------
# Per-session generation lock
# --------------------------------------------------------------------------
# A session may have at most one *in-flight* model generation at any time.
# When a new turn arrives while the previous turn is still streaming tokens,
# we (1) signal the prior turn to stop, (2) wait for it to persist whatever
# it had generated so far, and (3) then let the new turn proceed. The wake
# step (abstraction) is intentionally NOT covered by this lock — it runs
# after the lock has been released so a fast follow-up does not block on
# the (slower) memory-routing LLM call.

_GEN_REGISTRY_LOCK = threading.Lock()


@dataclass
class _GenSlot:
    """Handle on an in-flight generation for a given session.

    ``stop`` is set by a later turn that wants to preempt this one;
    ``done`` is set by this turn once it has finished persisting (whether
    completed normally or interrupted), so the next turn knows it is safe
    to start generating.
    """

    stop: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)


_GEN_LOCKS: dict[str, _GenSlot] = {}


def _acquire_gen_slot(session_id: str | None) -> _GenSlot:
    """Reserve the generation slot for ``session_id``.

    If a previous turn is still generating for the same session, signal it
    to stop and wait (briefly) for it to drain. Then atomically register
    ourselves as the active slot.
    """
    slot = _GenSlot()
    if not session_id:
        return slot
    with _GEN_REGISTRY_LOCK:
        prior = _GEN_LOCKS.get(session_id)
        _GEN_LOCKS[session_id] = slot
    if prior is not None and prior is not slot:
        prior.stop.set()
        # Bounded wait — never let a stuck prior turn deadlock the session.
        prior.done.wait(timeout=15.0)
    return slot


def _release_gen_slot(session_id: str | None, slot: _GenSlot) -> None:
    """Release the slot. Safe to call multiple times."""
    if session_id:
        with _GEN_REGISTRY_LOCK:
            # Only drop ourselves; a later turn may have already replaced
            # us in the registry.
            if _GEN_LOCKS.get(session_id) is slot:
                _GEN_LOCKS.pop(session_id, None)
    slot.done.set()


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

    def _update_hat(self, session: Session | None, hat: dict | None) -> None:
        """Best-effort rewrite of the last turn's ``hat`` metadata after the
        wake step finishes. We persist the turn *before* running the wake
        step so a fast follow-up message never observes a missing history
        line; this writes the badge data back into the same row once it's
        available."""
        if hat is None or session is None or self.sessions is None:
            return
        try:
            self.sessions.update_last_hat(session.id, hat)
        except Exception:
            # Non-fatal: badge data is best-effort, history is what matters.
            pass

    def _rebuild_history(
        self, session: Session | None, last: ChatMessage
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Return ``(history, full_messages)`` for this turn.

        When a session id is in play, the server is the **single source of
        truth** for the conversation: we ignore whatever ``messages[:-1]``
        the client sent and rebuild the prefix from the persisted session
        log. This kills two classes of bug:

        - Front-end races where switching sessions mid-stream causes the
          client to send messages tagged with the wrong ``session_id``.
        - "The model forgot what I just told it": every turn is guaranteed
          to see the persisted prior turns because ``_persist_turn`` writes
          before the (slow) wake step.

        Without a session store we fall back to the client-supplied list.
        """
        if session is None or self.sessions is None:
            # Stateless mode (legacy /chat). Trust the client.
            return [], []  # caller will use req.messages directly
        try:
            stored = self.sessions.messages(session.id)
        except Exception:
            return [], []
        history: list[ChatMessage] = []
        for it in stored:
            if it.query:
                history.append(ChatMessage(role="user", content=it.query))
            if it.response:
                history.append(ChatMessage(role="assistant", content=it.response))
        full = history + [last]
        return history, full

    # ----- non-streaming -----------------------------------------------

    def handle(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        _, last = _split_messages(req.messages)
        gen_kwargs = _build_gen_kwargs(req)
        session = self._resolve_session(req.session_id, first_query=last.content)

        # Authoritative history: rebuild from the session store. The client
        # is no longer expected to send the prior turns — ``runs/raw`` is
        # the single source of truth.
        rebuilt_history, rebuilt_full = self._rebuild_history(session, last)
        if rebuilt_full:
            history = rebuilt_history
            effective_messages = rebuilt_full
        else:
            history = req.messages[:-1]
            effective_messages = list(req.messages)

        session_key = session.id if session else None
        slot = _acquire_gen_slot(session_key)
        try:
            cortex = self.loop.cortex
            response_text = cortex.chat(
                [m.model_dump() for m in effective_messages], **gen_kwargs
            )
            interaction = Interaction(
                session_id=session.id if session else None,
                context=_flatten_history(history),
                query=last.content,
                response=response_text,
            )
            # Persist BEFORE wake_step / before releasing the lock so a
            # fast follow-up turn always sees this exchange.
            self._persist_turn(session, interaction)
        finally:
            _release_gen_slot(session_key, slot)

        # Wake step runs OUTSIDE the generation lock: a fast follow-up
        # turn can start generating immediately while this turn's
        # abstractor LLM call (slower) finishes in parallel.
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
        self._update_hat(session, interaction.hat)

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

        # Authoritative history rebuild (see ``_rebuild_history`` docstring).
        rebuilt_history, rebuilt_full = self._rebuild_history(session, last)
        if rebuilt_full:
            history = rebuilt_history
            effective_messages = rebuilt_full
        else:
            effective_messages = list(req.messages)

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

        msgs = [m.model_dump() for m in effective_messages]
        collected: list[str] = []
        interrupted = False

        session_key = session.id if session else None
        slot = _acquire_gen_slot(session_key)
        try:
            try:
                for piece in cortex.stream_chat(msgs, **gen_kwargs):
                    # Cooperative cancellation: a later turn on this
                    # session has asked us to stop. Persist what we
                    # have and exit cleanly.
                    if slot.stop.is_set():
                        interrupted = True
                        break
                    if not piece:
                        continue
                    collected.append(piece)
                    yield chunk(ChatCompletionDelta(content=piece))
            except Exception as e:  # pragma: no cover - runtime safety net
                yield chunk(
                    ChatCompletionDelta(
                        content=f"\n[stream error] {type(e).__name__}: {e}"
                    )
                )

            full = "".join(collected)
            interaction = Interaction(
                session_id=session.id if session else None,
                context=_flatten_history(history),
                query=last.content,
                response=full,
            )

            # Persist BEFORE releasing the lock so the next turn (which
            # is currently waiting on ``slot.done``) is guaranteed to see
            # this exchange when it rebuilds history from disk. For an
            # interrupted turn ``full`` is whatever tokens we had time to
            # collect — that is the "已生成的上下文" we promise to keep.
            self._persist_turn(session, interaction)
        finally:
            _release_gen_slot(session_key, slot)

        # If we were interrupted mid-generation, the next turn is already
        # queued and waiting. End the stream now; the abstractor decision
        # for this partial turn (if any) will be written back to disk
        # asynchronously and surface on the next session refresh.
        if interrupted:
            yield chunk(
                ChatCompletionDelta(),
                finish="stop",
                hat_session_id=session.id if session else None,
            )
            yield "data: [DONE]\n\n"
            self._run_wake_async(session, interaction)
            return

        # Wake step runs OUTSIDE the generation lock: a fast follow-up
        # turn does not have to wait for the (slower) abstractor LLM call.
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
        self._update_hat(session, interaction.hat)

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

    # ----- background wake (used for interrupted turns) -----------------

    def _run_wake_async(
        self, session: Session | None, interaction: Interaction
    ) -> None:
        """Run ``wake_step`` for an interrupted turn in a background
        thread so the user-facing SSE stream can close immediately.

        Used only for partial / interrupted turns where the client has
        already moved on to the next message. For completed turns we run
        wake_step inline so the lifecycle events still stream live.
        """
        def _run() -> None:
            try:
                prior_traces: list = []
                if session is not None:
                    from ..deps import prior_traces_for_session
                    prior_traces = prior_traces_for_session(session.id)
                events: list[dict] = []

                def _sink(stage: str, payload: dict) -> None:
                    events.append({"stage": stage, **payload})

                trace = self.loop.wake_step(
                    interaction,
                    prior_traces=prior_traces or None,
                    event_sink=_sink,
                )
                interaction.hat = _summarize_events(events, trace)
                self._update_hat(session, interaction.hat)
            except Exception:  # pragma: no cover - background safety net
                pass

        threading.Thread(
            target=_run, name="hat-wake-async", daemon=True
        ).start()
