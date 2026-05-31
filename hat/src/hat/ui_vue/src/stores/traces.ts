import { defineStore } from "pinia";
import { ref } from "vue";
import { listNeocortex } from "@/api/neocortex";
import type { TraceEvent } from "@/api/types";

export interface TraceCard {
  key: string;        // stable id for in-place updates (e.g. triage:iid)
  stage: string;
  trace_id?: string;
  body: TraceEvent;
}

function eventKey(ev: TraceEvent): string {
  const iid = ev.interaction_id || "";
  if (ev.stage === "triage_start" || ev.stage === "triage_done") return `triage:${iid}`;
  if (ev.stage === "route_start" || ev.stage === "route_done") return `route:${iid}`;
  if (ev.stage === "uncertainty" || ev.stage === "skipped") return `uncertainty:${iid}`;
  if (ev.stage === "abstracting") return `abstracting:${iid}`;
  // Distinct cards: per stage + trace_id (fallback per stage + iid + random index)
  return `${ev.stage}:${ev.trace_id || iid || crypto.randomUUID()}`;
}

export const useTracesStore = defineStore("traces", () => {
  const cards = ref<TraceCard[]>([]);

  function append(ev: TraceEvent) {
    const key = eventKey(ev);
    const card: TraceCard = { key, stage: String(ev.stage || "event"), trace_id: ev.trace_id, body: ev };
    const idx = cards.value.findIndex((c) => c.key === key);
    if (idx >= 0) cards.value.splice(idx, 1, card);
    else cards.value.push(card);
  }

  function clear() {
    cards.value = [];
  }

  async function loadForSession(sessionId: string) {
    clear();
    try {
      const rows = await listNeocortex({ sessionId });
      for (const row of rows) {
        append({
          stage: "created",
          trace_id: row.trace_id,
          target_response: row.response,
          rationale:
            (row.metadata?.extras as Record<string, unknown> | undefined)?.["rationale"] as
              | string
              | undefined,
        });
      }
    } catch {
      // best-effort; panel stays empty
    }
  }

  return { cards, append, clear, loadForSession };
});
