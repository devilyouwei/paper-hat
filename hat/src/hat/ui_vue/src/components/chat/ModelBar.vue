<script setup lang="ts">
import { computed, onMounted, watch } from "vue";
import { NSelect, NButton, NTag, useMessage, type SelectOption } from "naive-ui";
import { useAppStore } from "@/stores/app";
import { useModelsStore } from "@/stores/models";

const app = useAppStore();
const models = useModelsStore();
const message = useMessage();

const backendOptions: SelectOption[] = [
  { label: "MLX", value: "mlx" },
  { label: "HF", value: "hf" },
];

const installedOptions = computed<SelectOption[]>(() =>
  models.items.filter((it) => it.installed).map((it) => ({ label: it.display, value: it.id })),
);

const selectedModel = computed({
  get: () => models.active?.backend === models.backend ? models.active?.id : undefined,
  set: (v: string | undefined) => {
    if (v) activate(v);
  },
});

async function refresh() {
  // Boot priority: active backend → health.cortex_backend → mlx
  const initial =
    models.active?.backend ||
    (app.health?.cortex_backend && app.health.cortex_backend !== "noop"
      ? app.health.cortex_backend
      : "mlx");
  models.backend = initial;
  await models.loadCatalog();
}

watch(() => models.backend, () => models.loadCatalog());

async function activate(id: string) {
  try {
    await models.activate(models.backend, id);
    app.setActive(models.active);
    message.success(`Activated ${models.backend}/${id}`);
  } catch (e) {
    message.error(`Activate failed: ${(e as Error).message}`);
  }
}

async function unload() {
  try {
    await models.unload();
    app.setActive(null);
    message.success("Model unloaded");
  } catch (e) {
    message.error(`Unload failed: ${(e as Error).message}`);
  }
}

onMounted(refresh);
</script>

<template>
  <div class="bar">
    <div class="cell">
      <label>Backend</label>
      <NSelect
        v-model:value="models.backend"
        :options="backendOptions"
        size="small"
      />
    </div>
    <div class="cell grow">
      <label>Model</label>
      <NSelect
        v-model:value="selectedModel"
        :options="installedOptions"
        placeholder="Pick an installed model…"
        size="small"
        clearable
        filterable
      />
    </div>
    <div class="actions">
      <NButton size="small" type="primary" :disabled="!selectedModel" @click="selectedModel && activate(selectedModel)">
        Use
      </NButton>
      <NButton size="small" tertiary :disabled="!models.active" @click="unload">Unload</NButton>
    </div>
    <NTag v-if="app.active" size="small" type="success" round class="active-tag">
      {{ app.active.backend }}/{{ app.active.id }}
    </NTag>
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.bar {
  display: flex;
  align-items: flex-end;
  gap: $space-3;
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
  min-width: 120px;
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

.actions {
  display: inline-flex;
  gap: $space-2;
  align-self: flex-end;
}

.active-tag {
  margin-left: auto;
}
</style>
