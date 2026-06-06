<script setup lang="ts">
import { computed } from "vue";
import { NCard, NTag, NButton, NProgress, useMessage, useDialog } from "naive-ui";
import { CloudDownload, Trash, PlayerPlay, X } from "@vicons/tabler";
import { NIcon } from "naive-ui";
import type { CatalogItem } from "@/api/types";
import { useModelsStore } from "@/stores/models";
import { useAppStore } from "@/stores/app";
import { formatBytes } from "@/utils/format";

const props = defineProps<{ item: CatalogItem }>();

const models = useModelsStore();
const app = useAppStore();
const message = useMessage();
const dialog = useDialog();

const dl = computed(() => models.downloadFor(props.item.backend, props.item.id));
const isActive = computed(
  () => app.active?.backend === props.item.backend && app.active?.id === props.item.id,
);
const isCloud = computed(
  () => props.item.backend === "cloud" || props.item.backend === "cloud_embed",
);
const downloading = computed(
  () => dl.value?.state === "pending" || dl.value?.state === "downloading",
);

const progressPct = computed(() => {
  const d = dl.value;
  if (!d || !d.bytes_total) return 0;
  return Math.min(100, Math.round((d.bytes_done / d.bytes_total) * 100));
});

const progressLabel = computed(() => {
  const d = dl.value;
  if (!d) return "";
  const b = `${formatBytes(d.bytes_done)} / ${formatBytes(d.bytes_total)}`;
  const f = d.files_total ? ` · ${d.files_done}/${d.files_total} files` : "";
  return `${b}${f}`;
});

async function activate() {
  try {
    await models.activate(props.item.backend, props.item.id);
    app.setActive(models.active);
    message.success(`Activated ${props.item.id}`);
  } catch (e) {
    message.error(`Activate failed: ${(e as Error).message}`);
  }
}

function download() {
  models.startDownload(props.item.backend, props.item.id);
}

async function cancel() {
  await models.cancelDownload(props.item.backend, props.item.id);
  message.info(`Cancelled ${props.item.id}`);
}

function remove() {
  dialog.warning({
    title: "Delete weights?",
    content: `Remove local files for “${props.item.display}”?`,
    positiveText: "Delete",
    negativeText: "Cancel",
    onPositiveClick: async () => {
      try {
        await models.remove(props.item.backend, props.item.id);
        message.success("Deleted");
      } catch (e) {
        message.error(`Delete failed: ${(e as Error).message}`);
      }
    },
  });
}
</script>

<template>
  <NCard
    class="model-card"
    :class="{ active: isActive }"
    size="small"
    :bordered="true"
  >
    <header class="head">
      <div class="title">
        <strong>{{ item.display }}</strong>
        <span class="id hat-mono hat-muted">{{ item.id }}</span>
      </div>
      <NTag
        v-if="isActive"
        type="success"
        size="small"
        round
      >
        ● Active
      </NTag>
      <NTag
        v-else-if="isCloud"
        type="warning"
        size="small"
        round
      >
        ☁ Cloud
      </NTag>
      <NTag
        v-else-if="item.installed"
        type="info"
        size="small"
        round
      >
        ✓ Installed
      </NTag>
      <NTag v-else size="small" round>↓ Missing</NTag>
    </header>

    <div class="meta">
      <span class="hat-muted hat-mono">{{ item.repo_id }}</span>
      <span v-if="item.size_gb" class="hat-muted">{{ item.size_gb.toFixed(1) }} GB</span>
    </div>

    <p v-if="item.notes" class="notes">{{ item.notes }}</p>

    <footer class="foot">
      <template v-if="downloading">
        <div class="progress">
          <NProgress
            type="line"
            :percentage="progressPct"
            :height="8"
            :show-indicator="false"
          />
          <span class="progress-text hat-muted">{{ progressLabel }}</span>
        </div>
        <NButton size="small" tertiary type="warning" @click="cancel">
          <template #icon><NIcon><X /></NIcon></template>
          Cancel
        </NButton>
      </template>
      <template v-else>
        <NButton
          v-if="item.installed && !isActive"
          size="small"
          type="primary"
          @click="activate"
        >
          <template #icon><NIcon><PlayerPlay /></NIcon></template>
          Use
        </NButton>
        <NButton
          v-else-if="!item.installed"
          size="small"
          type="primary"
          @click="download"
        >
          <template #icon><NIcon><CloudDownload /></NIcon></template>
          Download
        </NButton>
        <NButton
          v-if="item.installed && !isCloud"
          size="small"
          tertiary
          type="error"
          @click="remove"
        >
          <template #icon><NIcon><Trash /></NIcon></template>
          Delete
        </NButton>
      </template>
    </footer>
  </NCard>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.model-card {
  transition: border-color 0.15s ease, transform 0.15s ease;
  &.active {
    border-color: var(--hat-accent);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--hat-accent) 35%, transparent);
  }
  &:hover {
    transform: translateY(-1px);
  }
}

.head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: $space-2;
  margin-bottom: $space-2;
}

.title {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  flex: 1 1 auto;
  strong {
    font-size: 14px;
    color: var(--hat-text);
    line-height: 1.2;
    word-break: break-word;
  }
  .id {
    font-size: 11px;
  }
}

.meta {
  display: flex;
  align-items: center;
  gap: $space-3;
  font-size: 11px;
  margin-bottom: $space-2;
}

.notes {
  font-size: 12px;
  color: var(--hat-muted);
  margin: 0 0 $space-3;
  line-height: 1.5;
}

.foot {
  display: flex;
  gap: $space-2;
  align-items: center;
}

.progress {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  .progress-text {
    font-size: 11px;
  }
}
</style>
