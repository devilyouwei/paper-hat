<script setup lang="ts">
import { computed, h, onMounted, ref } from "vue";
import {
  NDataTable,
  NInput,
  NButton,
  NTag,
  NBadge,
  NSelect,
  type DataTableColumns,
  type SelectOption,
} from "naive-ui";
import { Refresh, Search } from "@vicons/tabler";
import { NIcon } from "naive-ui";
import { useMemoryStore } from "@/stores/memory";
import { useEmbeddingModelsStore } from "@/stores/embeddingModels";
import type { NeocortexEntry } from "@/api/types";
import ScoreExplainer from "@/components/memory/ScoreExplainer.vue";
import MemoryEditorDrawer from "@/components/memory/MemoryEditorDrawer.vue";

const mem = useMemoryStore();
const embeds = useEmbeddingModelsStore();
const editing = ref<NeocortexEntry | null>(null);
const drawerOpen = ref(false);

onMounted(async () => {
  await embeds.loadCatalog();
  await embeds.refreshActive();
  await mem.refresh();
});

const embedFilterOptions = computed<SelectOption[]>(() => {
  const opts: SelectOption[] = [{ label: "All embedders", value: "__all__" }];
  for (const it of embeds.items) {
    if (!it.installed) continue;
    const tag = `${it.backend}/${it.id}`;
    opts.push({ label: tag, value: tag });
  }
  return opts;
});

const embedFilterValue = computed({
  get: () => mem.embedFilter ?? "__all__",
  set: (v: string) => {
    mem.setEmbedFilter(v === "__all__" ? null : v);
  },
});

function fmt(n: unknown, d = 2): string {
  return typeof n === "number" ? n.toFixed(d) : "—";
}

function isOracle(row: NeocortexEntry): boolean {
  const extras = (row.metadata?.extras || {}) as Record<string, unknown>;
  if (extras["oracle"]) return true;
  const src = (row.metadata as { source?: unknown })?.source;
  return typeof src === "string" && src.toLowerCase().includes("oracle");
}

function openEditor(e: NeocortexEntry) {
  editing.value = e;
  drawerOpen.value = true;
}

const columns = computed<DataTableColumns<NeocortexEntry>>(() => [
  {
    title: "#",
    key: "_idx",
    width: 50,
    render: (_row, idx) => idx + 1,
  },
  {
    title: "Trace",
    key: "trace_id",
    width: 130,
    render: (row) =>
      h("div", { class: "hat-row" }, [
        h("code", { class: "hat-mono" }, row.trace_id.slice(0, 8)),
        isOracle(row)
          ? h(
              NTag,
              { type: "warning", size: "tiny", round: true },
              () => "oracle",
            )
          : null,
      ]),
  },
  {
    title: "Score",
    key: "score",
    width: 80,
    sorter: (a, b) => (a.score || 0) - (b.score || 0),
    render: (row) => h("strong", { class: "hat-mono" }, fmt(row.score)),
  },
  {
    title: "U",
    key: "u",
    width: 64,
    render: (row) =>
      h("span", { class: "hat-mono hat-muted" }, fmt(row.signals?.uncertainty)),
  },
  {
    title: "Query",
    key: "query",
    ellipsis: { tooltip: true },
    render: (row) => row.query || "",
  },
  {
    title: "Response",
    key: "response",
    ellipsis: { tooltip: true },
    render: (row) => row.response || "",
  },
  {
    title: "",
    key: "actions",
    width: 120,
    render: (row) =>
      h("div", { class: "hat-row" }, [
        h(
          NButton,
          { size: "tiny", tertiary: true, onClick: () => openEditor(row) },
          () => "Edit",
        ),
      ]),
  },
]);
</script>

<template>
  <div class="memory-page">
    <header class="page-head">
      <div>
        <p class="hat-muted small">
          <NBadge :value="mem.filtered.length" :max="9999" type="success" />
          shown of
          <NBadge :value="mem.entries.length" :max="9999" />
          curated traces
        </p>
      </div>
      <NButton
        tertiary
        size="small"
        :loading="mem.loading"
        @click="mem.refresh"
      >
        <template #icon
          ><NIcon><Refresh /></NIcon
        ></template>
        Refresh
      </NButton>
    </header>

    <ScoreExplainer />

    <div class="toolbar">
      <NSelect
        v-model:value="embedFilterValue"
        :options="embedFilterOptions"
        size="small"
        style="min-width: 220px"
      />
      <NInput
        :value="mem.query"
        placeholder="Search query / response / trace…"
        clearable
        size="small"
        class="grow"
        @update:value="(v: string) => (mem.query = v)"
      >
        <template #prefix
          ><NIcon><Search /></NIcon
        ></template>
      </NInput>
    </div>

    <div class="table-wrap">
      <NDataTable
        :columns="columns"
        :data="mem.filtered"
        :row-key="(r: NeocortexEntry) => r.trace_id"
        :loading="mem.loading"
        size="small"
        :pagination="{ pageSize: 25 }"
        flex-height
        striped
        class="mem-table"
      />
    </div>

    <MemoryEditorDrawer v-model:show="drawerOpen" :entry="editing" />
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.memory-page {
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
    display: flex;
    align-items: center;
    gap: 8px;
  }
}

.small {
  font-size: 12px;
}

.toolbar {
  display: flex;
  gap: $space-2;
  align-items: center;
}

.grow {
  flex: 1 1 auto;
}

.table-wrap {
  flex: 1 1 auto;
  min-height: 320px;
  background: var(--hat-panel);
  border: 1px solid var(--hat-border);
  border-radius: $radius;
  padding: $space-2;
  display: flex;
  flex-direction: column;
}

.mem-table {
  flex: 1 1 auto;
  min-height: 0;
}
</style>
