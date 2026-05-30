<script setup lang="ts">
import { ref, onMounted, watch, nextTick } from "vue";
import { NEmpty } from "naive-ui";
import ChatBubble from "./ChatBubble.vue";
import { useSessionsStore } from "@/stores/sessions";

const sessions = useSessionsStore();
const scroller = ref<HTMLElement | null>(null);
const stickBottom = ref(true);
const NEAR_BOTTOM = 60;

function nearBottom(): boolean {
  const el = scroller.value;
  if (!el) return true;
  return el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR_BOTTOM;
}

function scrollToBottom(force = false) {
  const el = scroller.value;
  if (!el) return;
  if (force || stickBottom.value) el.scrollTop = el.scrollHeight;
}

function onScroll() {
  stickBottom.value = nearBottom();
}

watch(
  () => sessions.messages.length,
  async () => {
    const wasAtBottom = stickBottom.value;
    await nextTick();
    if (wasAtBottom) scrollToBottom(true);
  },
);

// Re-scroll on each token append.
watch(
  () => sessions.messages.map((m) => m.response.length).join(","),
  async () => {
    await nextTick();
    scrollToBottom();
  },
);

onMounted(() => {
  scrollToBottom(true);
});

defineExpose({ scrollToBottom });
</script>

<template>
  <div ref="scroller" class="chatbox" @scroll.passive="onScroll">
    <NEmpty v-if="!sessions.messages.length" description="Say hi 👋" class="empty" />
    <template v-for="m in sessions.messages" :key="m.id">
      <ChatBubble v-if="m.query" role="user" :content="m.query" />
      <ChatBubble
        v-if="m.response || m.id.startsWith('local-')"
        role="assistant"
        :content="m.response"
        :hat="m.hat"
      />
    </template>
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.chatbox {
  height: 50vh;
  max-height: 50vh;
  min-height: 240px;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding: $space-4 $space-5;
  display: flex;
  flex-direction: column;
  gap: $space-4;
  background: var(--hat-bg-2);
  border: 1px solid var(--hat-border);
  border-radius: $radius;
}

.empty {
  margin: auto;
  color: var(--hat-muted);
}
</style>
