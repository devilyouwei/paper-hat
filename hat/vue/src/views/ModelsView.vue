<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { NEmpty, NScrollbar, NBadge } from "naive-ui";
import ModelsToolbar from "@/components/models/ModelsToolbar.vue";
import ModelCard from "@/components/models/ModelCard.vue";
import { useModelsStore } from "@/stores/models";
import { useAppStore } from "@/stores/app";

const models = useModelsStore();
const app = useAppStore();

const filter = ref<"all" | "installed" | "missing">("all");
const search = ref("");

const filtered = computed(() => {
  const q = search.value.trim().toLowerCase();
  return models.items.filter((it) => {
    if (filter.value === "installed" && !it.installed) return false;
    if (filter.value === "missing" && it.installed) return false;
    if (!q) return true;
    return [it.id, it.repo_id, it.display, it.notes || ""]
      .some((v) => v.toLowerCase().includes(q));
  });
});

const installedCount = computed(() => models.items.filter((it) => it.installed).length);

async function refresh() {
  await models.loadCatalog();
  await models.refreshActive();
  app.setActive(models.active);
}

watch(() => models.backend, refresh);

onMounted(refresh);

function onBackend(v: string) {
  models.backend = v;
}
</script>

<template>
  <div class="models-page">
    <header class="page-head">
      <div>
        <h2>Models</h2>
        <p class="hat-muted">
          <NBadge :value="installedCount" :max="999" type="success" />
          installed out of
          <NBadge :value="models.items.length" :max="999" />
          available
        </p>
      </div>
    </header>

    <ModelsToolbar
      :backend="models.backend"
      :filter="filter"
      :search="search"
      @update:backend="onBackend"
      @update:filter="(v) => (filter = v)"
      @update:search="(v) => (search = v)"
      @refresh="refresh"
    />

    <NScrollbar class="grid-wrap">
      <NEmpty v-if="!filtered.length" :description="models.loading ? 'Loading…' : 'No models match.'" class="empty" />
      <div v-else class="grid">
        <ModelCard v-for="it in filtered" :key="`${it.backend}/${it.id}`" :item="it" />
      </div>
    </NScrollbar>
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.models-page {
  display: flex;
  flex-direction: column;
  gap: $space-3;
  height: 100%;
  min-height: 0;
}

.page-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  h2 {
    margin: 0;
    font-size: 18px;
  }
  p {
    margin: 4px 0 0;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
}

.grid-wrap {
  flex: 1 1 auto;
  min-height: 0;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: $space-3;
  padding-bottom: $space-3;
}

.empty {
  padding: $space-6;
}
</style>
