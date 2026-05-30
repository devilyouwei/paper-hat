<script setup lang="ts">
import { ref, watch } from "vue";
import { NDrawer, NDrawerContent, NInput, NButton, NDivider, useMessage, useDialog } from "naive-ui";
import type { NeocortexEntry } from "@/api/types";
import { useMemoryStore } from "@/stores/memory";
import ScoreBreakdown from "./ScoreBreakdown.vue";
import SourceTrace from "./SourceTrace.vue";

const props = defineProps<{ show: boolean; entry: NeocortexEntry | null }>();
const emit = defineEmits<{ (e: "update:show", v: boolean): void }>();

const mem = useMemoryStore();
const message = useMessage();
const dialog = useDialog();

const query = ref("");
const response = ref("");

watch(
  () => props.entry,
  (e) => {
    query.value = e?.query || "";
    response.value = e?.response || "";
  },
  { immediate: true },
);

async function save() {
  if (!props.entry) return;
  try {
    await mem.save(props.entry.trace_id, { query: query.value, response: response.value });
    message.success(`Saved ${props.entry.trace_id.slice(0, 8)}…`);
    emit("update:show", false);
  } catch (e) {
    message.error(`Save failed: ${(e as Error).message}`);
  }
}

function remove() {
  if (!props.entry) return;
  const id = props.entry.trace_id;
  dialog.warning({
    title: "Delete this trace?",
    content: `Trace ${id.slice(0, 8)}… will be permanently removed.`,
    positiveText: "Delete",
    negativeText: "Cancel",
    onPositiveClick: async () => {
      try {
        await mem.remove(id);
        message.success("Deleted");
        emit("update:show", false);
      } catch (e) {
        message.error(`Delete failed: ${(e as Error).message}`);
      }
    },
  });
}
</script>

<template>
  <NDrawer
    :show="show"
    :width="640"
    placement="right"
    @update:show="(v) => emit('update:show', v)"
  >
    <NDrawerContent closable>
      <template #header>
        <span class="hat-mono small">trace {{ entry?.trace_id?.slice(0, 12) }}…</span>
      </template>

      <div v-if="entry" class="editor">
        <label class="field">
          <span class="lbl">Query</span>
          <NInput
            v-model:value="query"
            type="textarea"
            :autosize="{ minRows: 2, maxRows: 6 }"
          />
        </label>
        <label class="field">
          <span class="lbl">Response</span>
          <NInput
            v-model:value="response"
            type="textarea"
            :autosize="{ minRows: 6, maxRows: 16 }"
          />
        </label>

        <NDivider />

        <ScoreBreakdown :entry="entry" />

        <NDivider />

        <SourceTrace :entry="entry" />
      </div>

      <template #footer>
        <div class="footer">
          <NButton type="error" tertiary @click="remove">Delete</NButton>
          <div class="space" />
          <NButton @click="emit('update:show', false)">Cancel</NButton>
          <NButton type="primary" @click="save">Save</NButton>
        </div>
      </template>
    </NDrawerContent>
  </NDrawer>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.editor {
  display: flex;
  flex-direction: column;
  gap: $space-3;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lbl {
  font-size: 11px;
  color: var(--hat-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.small {
  font-size: 12px;
}

.footer {
  display: flex;
  align-items: center;
  gap: $space-2;
  width: 100%;
  .space {
    flex: 1 1 auto;
  }
}
</style>
