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

// Source of truth for "currently active model" is app.active (from /api/models/active,
// fetched during app.boot()). The select reflects it whenever the chosen backend
// matches the active model's backend.
const selectedModel = computed({
  get: () =>
    app.active && app.active.backend === models.backend ? app.active.id : undefined,
  set: (v: string | undefined) => {
    if (v) activate(v);
  },
});

function pickInitialBackend(): string {
  if (app.active?.backend) return app.active.backend;
  const cb = app.health?.cortex_backend;
  if (cb && cb !== "noop") return cb;
  return "mlx";
}

async function syncFromActive() {
  const target = pickInitialBackend();
  if (models.backend !== target) {
    models.backend = target;
    // The watch below will reload the catalog.
  } else if (!models.items.length) {
    await models.loadCatalog();
  }
}

watch(() => models.backend, () => models.loadCatalog());

// React to async arrival of /api/models/active (boot() resolves after mount).
watch(
  () => [app.booted, app.active?.backend, app.active?.id] as const,
  syncFromActive,
  { immediate: false },
);

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

onMounted(async () => {
  models.backend = pickInitialBackend();
  await models.loadCatalog();
});
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
      <NButton size="small" tertiary :disabled="!app.active" @click="unload">Unload</NButton>
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
