<script setup lang="ts">
import { NSelect, NInput, NButton, type SelectOption } from "naive-ui";
import { Refresh } from "@vicons/tabler";
import { NIcon } from "naive-ui";

const props = defineProps<{
  backend: string;
  filter: "all" | "installed" | "missing";
  search: string;
}>();
const emit = defineEmits<{
  (e: "update:backend", v: string): void;
  (e: "update:filter", v: "all" | "installed" | "missing"): void;
  (e: "update:search", v: string): void;
  (e: "refresh"): void;
}>();

const backendOptions: SelectOption[] = [
  { label: "MLX (embed)", value: "mlx_embed" },
  { label: "HF (embed)", value: "hf_embed" },
  { label: "Cloud (embed)", value: "cloud_embed" },
];

const filterOptions: SelectOption[] = [
  { label: "All", value: "all" },
  { label: "Installed", value: "installed" },
  { label: "Missing", value: "missing" },
];
</script>

<template>
  <div class="toolbar">
    <div class="cell">
      <label>Backend</label>
      <NSelect
        :value="props.backend"
        :options="backendOptions"
        size="small"
        @update:value="(v: string) => emit('update:backend', v)"
      />
    </div>
    <div class="cell">
      <label>Status</label>
      <NSelect
        :value="props.filter"
        :options="filterOptions"
        size="small"
        @update:value="(v: 'all' | 'installed' | 'missing') => emit('update:filter', v)"
      />
    </div>
    <div class="cell grow">
      <label>Search</label>
      <NInput
        :value="props.search"
        placeholder="id, repo, display, notes…"
        size="small"
        clearable
        @update:value="(v: string) => emit('update:search', v)"
      />
    </div>
    <NButton size="small" tertiary @click="emit('refresh')">
      <template #icon><NIcon><Refresh /></NIcon></template>
      Refresh
    </NButton>
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.toolbar {
  display: flex;
  gap: $space-3;
  align-items: flex-end;
  padding: $space-3;
  background: var(--hat-panel);
  border: 1px solid var(--hat-border);
  border-radius: $radius;
  flex-wrap: wrap;
}

.cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 160px;
  label {
    font-size: 11px;
    color: var(--hat-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
}
.cell.grow {
  flex: 1 1 auto;
  min-width: 220px;
}
</style>
