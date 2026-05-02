from __future__ import annotations

from dataclasses import dataclass

from ...core.loop import WakeSleepLoop
from ...core.schemas import Interaction
from ...memory.raw.log import RawInteractionLog
from ..schemas.chat import ChatRequest, ChatResponse


@dataclass
class ChatController:
    """Run one wake step and append the interaction to the raw log.

    The Hippocampus is what (sometimes) promotes the interaction to a curated
    Neocortex trace; this controller is *not* allowed to touch the Neocortex.
    """

    loop: WakeSleepLoop
    raw_log: RawInteractionLog

    def handle(self, req: ChatRequest) -> ChatResponse:
        interaction = Interaction(
            context=req.context,
            query=req.query,
            feedback=req.feedback,
            user_correction=req.user_correction,
        )
        trace = self.loop.wake_step(interaction)
        self.raw_log.append(interaction)
        return ChatResponse(
            response=interaction.response or "",
            consolidated=trace is not None,
            trace_id=trace.id if trace else None,
            signals=trace.metadata.signals if trace else None,
        )
