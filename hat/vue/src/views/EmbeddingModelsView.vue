<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { NEmpty, NScrollbar, NBadge } from "naive-ui";
import EmbeddingModelsToolbar from "@/components/embedding-models/EmbeddingModelsToolbar.vue";
import EmbeddingModelCard from "@/components/embedding-models/EmbeddingModelCard.vue";
import { useEmbeddingModelsStore } from "@/stores/embeddingModels";

const store = useEmbeddingModelsStore();

const filter = ref<"all" | "installed" | "missing">("all");
const search = ref("");

const filtered = computed(() => {
  const q = search.value.trim().toLowerCase();
  return store.items.filter((it) => {
    if (filter.value === "installed" && !it.installed) return false;
    if (filter.value === "missing" && it.installed) return false;
    if (!q) return true;
    return [it.id, it.repo_id, it.display, it.notes || ""].some((v) =>
      v.toLowerCase().includes(q),
    );
  });
});

const installedCount = computed(
  () => store.items.filter((it) => it.installed).length,
);

async function refresh() {
  await store.loadCatalog();
  await store.refreshActive();
}

watch(() => store.backend, refresh);

onMounted(refresh);

function onBackend(v: string) {
  store.backend = v;
}
</script>

<template>
  <div class="models-page">
    <header class="page-head">
      <div>
        <p class="hat-muted">
          <NBadge :value="installedCount" :max="999" type="success" />
          installed out of
          <NBadge :value="store.items.length" :max="999" />
          available
        </p>
      </div>
    </header>

    <EmbeddingModelsToolbar
      :backend="store.backend"
      :filter="filter"
      :search="search"
      @update:backend="onBackend"
      @update:filter="(v) => (filter = v)"
      @update:search="(v) => (search = v)"
      @refresh="refresh"
    />

    <NScrollbar class="grid-wrap">
      <NEmpty
        v-if="!filtered.length"
        :description="store.loading ? 'Loading…' : 'No embedding models match.'"
        class="empty"
      />
      <div v-else class="grid">
        <EmbeddingModelCard
          v-for="it in filtered"
          :key="`${it.backend}/${it.id}`"
          :item="it"
        />
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
