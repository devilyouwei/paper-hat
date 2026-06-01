<script setup lang="ts">
import { computed } from "vue";
import { NProgress, NTag } from "naive-ui";
import type { NeocortexEntry } from "@/api/types";
import { useAppStore } from "@/stores/app";

const props = defineProps<{ entry: NeocortexEntry }>();
const app = useAppStore();

const threshold = computed(() => app.policy?.write_policy?.threshold ?? 0.3);

function clamp01(n?: number) {
  if (typeof n !== "number" || Number.isNaN(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

const sig = computed(() => props.entry.signals || {});
const extras = computed(
  () => (props.entry.metadata?.extras || {}) as Record<string, unknown>,
);
const u = computed(() => clamp01(sig.value.uncertainty));
const accepted = computed(() => u.value >= threshold.value);
const oracleName = computed(
  () => (extras.value["oracle_name"] as string | undefined) || "oracle",
);
const isOracle = computed(() => !!extras.value["oracle"]);

const bars = computed(() => [
  { key: "U", value: u.value, hint: "uncertainty: 1 − exp(mean log p)" },
]);
</script>

<template>
  <section class="breakdown">
    <header class="head">
      <span class="formula hat-mono">score = U</span>
      <NTag :type="accepted ? 'success' : 'default'" size="small" round>
        {{ accepted ? `accepted (U ≥ ${threshold.toFixed(2)})` : `below threshold (< ${threshold.toFixed(2)})` }}
      </NTag>
    </header>

    <div v-for="b in bars" :key="b.key" class="row" :title="b.hint">
      <span class="lbl hat-mono">{{ b.key }}</span>
      <NProgress
        :percentage="b.value * 100"
        :height="8"
        :show-indicator="false"
        :status="b.key === 'U' && accepted ? 'success' : 'default'"
        class="bar"
      />
      <span class="num hat-mono">{{ b.value.toFixed(2) }}</span>
    </div>

    <p v-if="isOracle" class="hat-muted small">
      Augmented by <b>{{ oracleName }}</b> — the response above is the teacher's answer.
    </p>
  </section>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.breakdown {
  display: flex;
  flex-direction: column;
  gap: $space-2;
}

.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: $space-2;
}

.formula {
  color: var(--hat-muted);
  font-size: 12px;
}

.row {
  display: grid;
  grid-template-columns: 24px 1fr 48px;
  gap: $space-2;
  align-items: center;

  .lbl {
    color: var(--hat-muted);
    font-size: 12px;
  }
  .num {
    text-align: right;
    font-size: 12px;
  }
}

.small {
  font-size: 12px;
}
</style>
