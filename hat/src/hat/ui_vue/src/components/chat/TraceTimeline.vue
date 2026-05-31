<script setup lang="ts">
import { computed } from "vue";
import { NButton, NEmpty, NScrollbar, NTag } from "naive-ui";
import { Trash } from "@vicons/tabler";
import { NIcon } from "naive-ui";
import { useTracesStore, type TraceCard } from "@/stores/traces";

const traces = useTracesStore();

const STAGE_BADGE: Record<string, string> = {
  uncertainty: "U",
  abstracting: "ABS",
  triage_start: "TRIAGE",
  triage_done: "TRIAGE",
  extract_start: "EXTRACT",
  extract_done: "EXTRACT",
  extracted: "KP",
  dedup: "DEDUP",
  route_start: "ROUTE",
  route_done: "ROUTE",
  routed: "ROUTE",
  scored: "SCORE",
  created: "NEW",
  revised: "EDIT",
  rejected: "DROP",
  skipped: "SKIP",
  dropped: "DROP",
};

function classify(stage: string): "created" | "revised" | "rejected" | "pending" | "info" {
  if (stage === "created") return "created";
  if (stage === "revised" || stage === "dedup") return "revised";
  if (stage === "rejected" || stage === "dropped" || stage === "skipped") return "rejected";
  if (
    stage === "triage_start" ||
    stage === "route_start" ||
    stage === "extract_start" ||
    stage === "abstracting"
  )
    return "pending";
  return "info";
}

function shortId(id?: string): string {
  if (!id) return "";
  return id.length > 10 ? id.slice(0, 10) + "…" : id;
}

function badge(stage: string): string {
  return STAGE_BADGE[stage] || stage.slice(0, 4).toUpperCase();
}

function fmt(n: unknown, digits = 2): string {
  return typeof n === "number" ? n.toFixed(digits) : "—";
}

function bodyText(c: TraceCard): string {
  const ev = c.body;
  switch (c.stage) {
    case "uncertainty":
      return `U=${fmt(ev.uncertainty, 3)} (threshold ${fmt(ev.threshold)})`;
    case "skipped":
      return `gate skipped: U=${fmt(ev.uncertainty, 3)} < ${fmt(ev.threshold)}`;
    case "triage_start":
      return "running triage…";
    case "triage_done": {
      const verdict = ev.keep === false ? "drop" : ev.keep === true ? "keep" : "?";
      return `triage: ${verdict}${ev.reason ? ` · ${ev.reason}` : ""}`;
    }
    case "extract_start":
      return "extracting knowledge points…";
    case "extract_done": {
      const n = (ev.n_kps as number) ?? 0;
      return ev.parsed === false
        ? "extraction unparseable"
        : `extracted ${n} knowledge point${n === 1 ? "" : "s"}`;
    }
    case "extracted": {
      const n = (ev.n_kps as number) ?? 0;
      return `${n} knowledge point${n === 1 ? "" : "s"} extracted`;
    }
    case "dedup": {
      const dec = String(ev.decision || "?").toUpperCase();
      const sim = fmt(ev.similarity, 3);
      const thr = fmt(ev.threshold, 2);
      const matched = ev.matched_trace_id ? ` · matched ${shortId(String(ev.matched_trace_id))}` : "";
      return `${dec} · sim=${sim} (≥${thr})${matched}`;
    }
    case "route_start": {
      const n = (ev.n_priors as number) || 0;
      return `routing… (${n} prior${n === 1 ? "" : "s"})`;
    }
    case "route_done": {
      const dec = ev.decision ? String(ev.decision).toUpperCase() : ev.parsed ? "?" : "unparseable";
      return `route: ${dec}${ev.rationale ? ` · ${ev.rationale}` : ""}`;
    }
    case "routed":
      return `Decision: ${ev.decision || "?"}`;
    case "scored":
      return `score ${fmt(ev.score)} / ${fmt(ev.threshold)} → ${ev.accepted ? "accept" : "reject"}`;
    case "created":
    case "revised": {
      const t = String(ev.target_response || "").trim();
      return t.length > 200 ? t.slice(0, 200) + "…" : t;
    }
    case "rejected":
      return `below threshold (score ${fmt(ev.score)} < ${fmt(ev.threshold)})`;
    case "dropped":
      return "abstractor dropped this turn";
    case "abstracting": {
      const n = ((ev.prior_trace_ids as unknown[]) || []).length;
      return n ? `Considering ${n} prior trace${n === 1 ? "" : "s"}…` : "Compressing turn…";
    }
    default:
      return "";
  }
}

const list = computed(() => traces.cards);
</script>

<template>
  <aside class="trace-panel">
    <header class="head">
      <h3>Trace activity</h3>
      <NButton size="tiny" tertiary :disabled="!list.length" @click="traces.clear">
        <template #icon><NIcon><Trash /></NIcon></template>
        Clear
      </NButton>
    </header>

    <NScrollbar class="body">
      <NEmpty v-if="!list.length" description="No traces yet" size="small" class="empty" />
      <ul v-else class="timeline">
        <li
          v-for="c in list"
          :key="c.key"
          class="card"
          :class="`stage-${classify(c.stage)}`"
        >
          <div class="head-row">
            <NTag size="tiny" round class="badge">{{ badge(c.stage) }}</NTag>
            <span class="id hat-mono">{{ shortId(c.trace_id) }}</span>
          </div>
          <div class="body-text">{{ bodyText(c) }}</div>
          <div v-if="c.body.rationale && c.stage !== 'route_done'" class="rationale">
            {{ c.body.rationale }}
          </div>
        </li>
      </ul>
    </NScrollbar>
  </aside>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.trace-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--hat-panel);
  border: 1px solid var(--hat-border);
  border-radius: $radius;
  min-height: 0;
}

.head {
  flex: 0 0 auto;
  padding: $space-3 $space-4;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--hat-border);
  h3 {
    margin: 0;
    font-size: 12px;
    color: var(--hat-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
}

.body {
  flex: 1 1 auto;
  min-height: 0;
}

.empty {
  padding: $space-4;
}

.timeline {
  list-style: none;
  margin: 0;
  padding: $space-3;
  display: flex;
  flex-direction: column;
  gap: $space-2;
}

.card {
  border: 1px solid var(--hat-border);
  border-radius: $radius-sm;
  padding: $space-2 $space-3;
  background: var(--hat-panel-2);
  font-size: 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;

  &.stage-created {
    border-left: 3px solid var(--hat-ok);
  }
  &.stage-revised {
    border-left: 3px solid var(--hat-accent);
  }
  &.stage-rejected {
    border-left: 3px solid var(--hat-danger);
    opacity: 0.85;
  }
  &.stage-pending {
    border-left: 3px solid var(--hat-muted);
    font-style: italic;
  }
  &.stage-info {
    border-left: 3px solid var(--hat-border);
  }
}

.head-row {
  display: flex;
  align-items: center;
  gap: $space-2;
}

.id {
  color: var(--hat-muted);
  font-size: 11px;
}

.body-text {
  color: var(--hat-text);
  white-space: pre-wrap;
}

.rationale {
  color: var(--hat-muted);
  font-size: 11px;
  border-top: 1px dashed var(--hat-border);
  padding-top: 4px;
}
</style>
