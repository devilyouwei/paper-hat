<script setup lang="ts">
import { onMounted, computed } from "vue";
import { NButton, NEmpty, NScrollbar, useDialog } from "naive-ui";
import { Plus, Trash } from "@vicons/tabler";
import { NIcon } from "naive-ui";
import { useSessionsStore } from "@/stores/sessions";
import { useTracesStore } from "@/stores/traces";

const sessions = useSessionsStore();
const traces = useTracesStore();
const dialog = useDialog();

onMounted(async () => {
  await sessions.refreshList();
  if (!sessions.list.length) {
    await sessions.create();
  } else {
    const id = sessions.list[0].id;
    await sessions.open(id);
    await traces.loadForSession(id);
  }
});

async function pickSession(id: string) {
  if (sessions.currentId === id) return;
  await sessions.open(id);
  await traces.loadForSession(id);
}

async function newChat() {
  await sessions.create();
  traces.clear();
}

function removeCurrent() {
  if (!sessions.currentId) return;
  const id = sessions.currentId;
  const title = sessions.current?.title || "this chat";
  dialog.warning({
    title: "Delete chat?",
    content: `“${title}” will be removed permanently.`,
    positiveText: "Delete",
    negativeText: "Cancel",
    onPositiveClick: async () => {
      await sessions.remove(id);
      if (sessions.list.length) {
        await sessions.open(sessions.list[0].id);
        await traces.loadForSession(sessions.list[0].id);
      } else {
        await sessions.create();
        traces.clear();
      }
    },
  });
}

const items = computed(() => sessions.list);
</script>

<template>
  <aside class="sidebar">
    <NButton block type="primary" @click="newChat">
      <template #icon><NIcon><Plus /></NIcon></template>
      New chat
    </NButton>

    <NScrollbar class="list-wrap">
      <NEmpty v-if="!items.length" description="No chats yet" size="small" class="hat-muted" />
      <ul v-else class="list">
        <li
          v-for="s in items"
          :key="s.id"
          class="item"
          :class="{ active: s.id === sessions.currentId }"
          @click="pickSession(s.id)"
        >
          <span class="title" :title="s.title">{{ s.title || "New chat" }}</span>
          <span class="count">{{ s.message_count }}</span>
        </li>
      </ul>
    </NScrollbar>

    <NButton
      quaternary
      type="error"
      size="small"
      :disabled="!sessions.currentId"
      @click="removeCurrent"
    >
      <template #icon><NIcon><Trash /></NIcon></template>
      Delete current
    </NButton>
  </aside>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.sidebar {
  display: flex;
  flex-direction: column;
  gap: $space-3;
  padding: $space-3;
  background: var(--hat-panel);
  border: 1px solid var(--hat-border);
  border-radius: $radius;
  min-height: 0;
  height: 100%;
}

.list-wrap {
  flex: 1 1 auto;
  min-height: 0;
}

.list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: $space-2;
  padding: $space-2 $space-3;
  border-radius: $radius-sm;
  cursor: pointer;
  font-size: 13px;
  color: var(--hat-text);
  transition: background 0.12s ease;

  .title {
    flex: 1 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .count {
    color: var(--hat-muted);
    font-size: 11px;
    background: var(--hat-panel-2);
    padding: 0 6px;
    border-radius: 999px;
    flex: 0 0 auto;
  }

  &:hover {
    background: var(--hat-panel-2);
  }
  &.active {
    background: color-mix(in srgb, var(--hat-accent) 22%, transparent);
    color: var(--hat-text);
    .count {
      background: color-mix(in srgb, var(--hat-accent) 35%, transparent);
      color: white;
    }
  }
}
</style>
