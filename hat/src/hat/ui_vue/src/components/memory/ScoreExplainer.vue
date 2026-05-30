<script setup lang="ts">
import { computed } from "vue";
import { NCollapse, NCollapseItem, NTag } from "naive-ui";
import { useAppStore } from "@/stores/app";

const app = useAppStore();

const policy = computed(() => app.policy?.write_policy);
const oracle = computed(() => app.policy?.oracle);
const threshold = computed(() => policy.value?.threshold ?? 0.3);
</script>

<template>
  <NCollapse class="explainer">
    <NCollapseItem title="How are entries scored?" name="score">
      <p class="hat-muted small">
        <code>score = U(m)</code> — the cortex's uncertainty on its own response. We write to
        long-term memory when <code>U ≥ {{ threshold.toFixed(2) }}</code>.
      </p>
      <ul class="signal-list">
        <li><b>U</b> — uncertainty: <code>1 − exp(mean log p)</code> over response tokens.</li>
        <li><b>N</b> — novelty score from the router (vs. nearest prior trace).</li>
        <li><b>S</b> — user signal: explicit teaching/correction detected.</li>
      </ul>

      <div v-if="oracle" class="oracle">
        <NTag :type="oracle.enabled ? 'success' : 'default'" size="small" round>
          Oracle {{ oracle.enabled ? "enabled" : "disabled" }}
        </NTag>
        <span v-if="oracle.enabled" class="hat-muted small">
          {{ oracle.model }} consulted when U &gt; {{ (oracle.threshold ?? 0).toFixed(2) }} ·
          limits: {{ oracle.rps ?? "?" }}/s · {{ oracle.daily_calls ?? "?" }}/day
        </span>
      </div>
    </NCollapseItem>
  </NCollapse>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.explainer {
  background: var(--hat-panel);
  border: 1px solid var(--hat-border);
  border-radius: $radius;
  padding: $space-2 $space-3;
}

.signal-list {
  margin: $space-2 0 0;
  padding-left: 1.2em;
  font-size: 12px;
  color: var(--hat-muted);
  li {
    margin-bottom: 2px;
  }
}

.oracle {
  margin-top: $space-3;
  display: flex;
  align-items: center;
  gap: $space-2;
}

.small {
  font-size: 12px;
}
</style>
