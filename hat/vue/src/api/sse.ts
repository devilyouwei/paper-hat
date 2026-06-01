import { fetchEventSource, type EventSourceMessage } from "@microsoft/fetch-event-source";

export interface SSEHandlers {
  onMessage: (data: string) => void;
  onError?: (err: unknown) => void;
  onOpen?: () => void;
  onClose?: () => void;
  signal?: AbortSignal;
}

/** POST-based SSE (used for /v1/chat/completions). */
export async function postSSE(path: string, body: unknown, h: SSEHandlers): Promise<void> {
  await fetchEventSource(path, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal: h.signal,
    openWhenHidden: true,
    async onopen(res) {
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`SSE open failed: ${res.status} ${text}`);
      }
      h.onOpen?.();
    },
    onmessage(ev: EventSourceMessage) {
      if (!ev.data) return;
      h.onMessage(ev.data);
    },
    onerror(err) {
      h.onError?.(err);
      throw err;
    },
    onclose() {
      h.onClose?.();
    },
  });
}
