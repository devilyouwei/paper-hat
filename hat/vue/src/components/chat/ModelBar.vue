<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { NSelect, useMessage, type SelectOption } from "naive-ui";
import { useAppStore } from "@/stores/app";
import { useModelsStore } from "@/stores/models";
import { useEmbeddingModelsStore } from "@/stores/embeddingModels";

const app = useAppStore();
const models = useModelsStore();
const embeds = useEmbeddingModelsStore();
const message = useMessage();

const platformOptions: SelectOption[] = [
  { label: "MLX", value: "mlx" },
  { label: "HF", value: "hf" },
  { label: "Cloud", value: "cloud" },
];

const platform = ref<string>("mlx");
const busy = ref(false);

const EMBED_BACKEND: Record<string, string> = {
  mlx: "mlx_embed",
  hf: "hf_embed",
  cloud: "cloud_embed",
};

const embedBackend = computed(() => EMBED_BACKEND[platform.value] ?? "mlx_embed");

const llmOptions = computed<SelectOption[]>(() =>
  models.items
    .filter((it) => it.installed)
    .map((it) => ({ label: it.display, value: it.id })),
);

const embedOptions = computed<SelectOption[]>(() =>
  embeds.items
    .filter((it) => it.installed)
    .map((it) => ({ label: it.display, value: it.id })),
);

const selectedLlm = computed({
  get: () =>
    app.active && app.active.backend === platform.value ? app.active.id : null,
  set: (v: string | null) => {
    if (v) activateLlm(v);
    else unloadLlm();
  },
});

const selectedEmbed = computed({
  get: () =>
    embeds.active && embeds.active.backend === embedBackend.value
      ? embeds.active.id
      : null,
  set: (v: string | null) => {
    if (v) activateEmbed(v);
    else unloadEmbed();
  },
});

function pickInitialPlatform(): string {
  const known = ["mlx", "hf", "cloud"];
  if (app.active?.backend && known.includes(app.active.backend)) {
    return app.active.backend;
  }
  const cb = app.health?.cortex_backend;
  if (cb && known.includes(cb)) return cb;
  return "mlx";
}

async function reloadForPlatform() {
  busy.value = true;
  try {
    models.backend = platform.value;
    embeds.backend = embedBackend.value;
    await Promise.all([
      models.loadCatalog(platform.value),
      embeds.loadCatalog(embedBackend.value),
    ]);
  } finally {
    busy.value = false;
  }
}

watch(platform, reloadForPlatform);

// React to async arrival of /api/models/active (boot() resolves after mount).
watch(
  () => [app.booted, app.active?.backend, app.active?.id] as const,
  () => {
    const target = pickInitialPlatform();
    if (platform.value !== target) platform.value = target;
  },
);

async function activateLlm(id: string) {
  busy.value = true;
  try {
    await models.activate(platform.value, id);
    app.setActive(models.active);
    message.success(`Activated ${platform.value}/${id}`);
  } catch (e) {
    message.error(`Activate failed: ${(e as Error).message}`);
  } finally {
    busy.value = false;
  }
}

async function unloadLlm() {
  busy.value = true;
  try {
    await models.unload();
    app.setActive(null);
  } catch (e) {
    message.error(`Unload failed: ${(e as Error).message}`);
  } finally {
    busy.value = false;
  }
}

async function activateEmbed(id: string) {
  busy.value = true;
  try {
    await embeds.activate(embedBackend.value, id);
    message.success(`Embedder: ${embedBackend.value}/${id}`);
  } catch (e) {
    message.error(`Embedder activate failed: ${(e as Error).message}`);
  } finally {
    busy.value = false;
  }
}

async function unloadEmbed() {
  busy.value = true;
  try {
    await embeds.unload();
  } catch (e) {
    message.error(`Embedder unload failed: ${(e as Error).message}`);
  } finally {
    busy.value = false;
  }
}

onMounted(async () => {
  platform.value = pickInitialPlatform();
  await Promise.all([
    models.loadCatalog(platform.value),
    embeds.loadCatalog(embedBackend.value),
    embeds.refreshActive(),
  ]);
});
</script>

<template>
  <div class="bar">
    <div class="cell">
      <label>Platform</label>
      <NSelect
        v-model:value="platform"
        :options="platformOptions"
        :disabled="busy"
        size="small"
      />
    </div>
    <div class="cell grow">
      <label>Model</label>
      <NSelect
        v-model:value="selectedLlm"
        :options="llmOptions"
        :disabled="busy"
        :loading="busy"
        placeholder="Pick a model…"
        size="small"
        clearable
        filterable
      />
    </div>
    <div class="cell grow">
      <label>Embedder</label>
      <NSelect
        v-model:value="selectedEmbed"
        :options="embedOptions"
        :disabled="busy"
        :loading="busy"
        placeholder="Pick an embedder…"
        size="small"
        clearable
        filterable
      />
    </div>
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
</style>
