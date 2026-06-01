import { ref } from "vue";
import { postSSE } from "@/api/sse";
import { useSessionsStore } from "@/stores/sessions";
import { useTracesStore } from "@/stores/traces";
import type { ChatCompletionDelta, ChatRequest, TraceEvent } from "@/api/types";

export interface ChatSettings {
  temperature: number;
  max_tokens: number;
  enable_thinking: boolean;
}

export function useChatStream() {
  const sessions = useSessionsStore();
  const traces = useTracesStore();

  const sending = ref(false);
  const liveResponse = ref("");
  const liveHat = ref<{ uncertainty?: number; decision?: string; reason?: string; stage?: string } | null>(null);
  let controller: AbortController | null = null;

  async function send(text: string, settings: ChatSettings) {
    if (!text.trim() || sending.value) return;
    sending.value = true;
    liveResponse.value = "";
    liveHat.value = null;

    sessions.appendUser(text);
    const turnSessionId = sessions.currentId;

    abort("new send");
    controller = new AbortController();

    const body: ChatRequest = {
      model: "hat-cortex",
      messages: [{ role: "user", content: text }],
      stream: true,
      temperature: settings.temperature,
      max_tokens: settings.max_tokens,
      chat_template_kwargs: { enable_thinking: settings.enable_thinking },
      ...(turnSessionId ? { session_id: turnSessionId } : {}),
    };

    try {
      await postSSE("/v1/chat/completions", body, {
        signal: controller.signal,
        onMessage(data) {
          if (data === "[DONE]") return;
          let obj: ChatCompletionDelta;
          try {
            obj = JSON.parse(data);
          } catch {
            return;
          }
          if (obj.hat_session_id) sessions.setSessionIdIfMissing(obj.hat_session_id);
          if (obj.hat_trace_event) handleTrace(obj.hat_trace_event);
          const delta = obj.choices?.[0]?.delta?.content || "";
          if (delta) {
            liveResponse.value += delta;
            sessions.setLastResponse(liveResponse.value);
          }
        },
      });
      if (sessions.currentId === turnSessionId && !controller.signal.aborted) {
        await sessions.refreshList(sessions.currentId);
      }
    } catch (e) {
      const name = (e as Error)?.name;
      if (name !== "AbortError") {
        const msg = `network error: ${(e as Error)?.message || e}`;
        liveResponse.value = liveResponse.value || msg;
        sessions.setLastResponse(liveResponse.value);
      }
    } finally {
      sending.value = false;
      controller = null;
    }
  }

  function handleTrace(ev: TraceEvent) {
    traces.append(ev);
    // Mirror chat.js attachUncertaintyBadge logic into liveHat on the last bubble.
    const stage = ev.stage;
    const u = typeof ev.uncertainty === "number" ? ev.uncertainty : liveHat.value?.uncertainty;
    const next = {
      uncertainty: u,
      decision:
        stage === "routed" && typeof ev.decision === "string"
          ? ev.decision.toUpperCase()
          : liveHat.value?.decision,
      reason: (ev.reason as string | undefined) ?? liveHat.value?.reason,
      stage: stage as string,
    };
    liveHat.value = next;
    sessions.setLastHat({
      uncertainty: next.uncertainty,
      decision:
        stage === "skipped" || stage === "dropped"
          ? stage
          : stage === "routed"
            ? ev.decision === "REVISE" || ev.decision === "revised"
              ? "revised"
              : "created"
            : undefined,
      reason: next.reason,
    });
  }

  function abort(reason = "abort") {
    if (controller) {
      try {
        controller.abort(reason);
      } catch {
        /* ignore */
      }
      controller = null;
    }
  }

  return { sending, liveResponse, liveHat, send, abort };
}
