<script setup lang="ts">
import { ref, watch } from "vue";
import { NSpin, NTimeline, NTimelineItem } from "naive-ui";
import type { NeocortexEntry, SessionDetail } from "@/api/types";
import { getSession } from "@/api/sessions";

const props = defineProps<{ entry: NeocortexEntry | null }>();

const detail = ref<SessionDetail | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);

const cache = new Map<string, SessionDetail>();

watch(
  () => props.entry?.session_id,
  async (sid) => {
    detail.value = null;
    error.value = null;
    if (!sid) return;
    if (cache.has(sid)) {
      detail.value = cache.get(sid)!;
      return;
    }
    loading.value = true;
    try {
      const d = await getSession(sid);
      cache.set(sid, d);
      detail.value = d;
    } catch (e) {
      error.value = (e as Error).message;
    } finally {
      loading.value = false;
    }
  },
  { immediate: true },
);
</script>

<template>
  <section class="source">
    <header class="head">
      <h4>Source</h4>
      <span class="hat-muted small">
        <template v-if="entry?.session_id">
          session <code class="hat-mono">{{ entry.session_id.slice(0, 8) }}</code>
          · {{ entry.interaction_ids?.length || 0 }} interaction(s)
        </template>
        <template v-else>no session linked</template>
      </span>
    </header>

    <p v-if="!entry?.session_id" class="hat-muted small">
      This trace has no session reference (older / oracle-seeded data).
    </p>

    <NSpin v-else-if="loading" size="small" class="spin" />

    <p v-else-if="error" class="hat-muted small">Failed to load session: {{ error }}</p>

    <template v-else-if="detail">
      <div class="title">
        <strong>{{ detail.session.title || "Untitled session" }}</strong>
        <span class="hat-muted small" v-if="detail.session.created_at">
          · created {{ detail.session.created_at.slice(0, 19).replace("T", " ") }}
        </span>
      </div>

      <NTimeline class="timeline">
        <NTimelineItem
          v-for="(m, i) in (detail.messages || []).filter((m) => entry?.interaction_ids?.includes(m.id))"
          :key="m.id"
          :title="`#${i + 1} · ${m.id.slice(0, 8)}`"
          :time="m.timestamp ? m.timestamp.slice(11, 19) : ''"
        >
          <div class="turn">
            <div class="line"><span class="role user">user</span>{{ m.query }}</div>
            <div class="line"><span class="role asst">assistant</span>{{ m.response }}</div>
          </div>
        </NTimelineItem>
      </NTimeline>
    </template>
  </section>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.source {
  display: flex;
  flex-direction: column;
  gap: $space-2;
}

.head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: $space-2;
  h4 {
    margin: 0;
    font-size: 12px;
    color: var(--hat-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
}

.small {
  font-size: 12px;
}

.title {
  margin: $space-2 0;
}

.timeline {
  margin-top: $space-2;
}

.turn {
  display: flex;
  flex-direction: column;
  gap: $space-2;
}

.line {
  display: flex;
  gap: $space-2;
  align-items: flex-start;
  font-size: 12.5px;
}

.role {
  font-family: $font-mono;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 999px;
  flex: 0 0 auto;
  letter-spacing: 0.04em;
  text-transform: uppercase;

  &.user {
    background: color-mix(in srgb, var(--hat-accent) 25%, transparent);
    color: var(--hat-accent);
  }
  &.asst {
    background: var(--hat-panel-2);
    color: var(--hat-muted);
  }
}

.spin {
  align-self: center;
  padding: $space-3;
}
</style>
