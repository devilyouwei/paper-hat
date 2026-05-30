<script setup lang="ts">
import { ref } from "vue";
import SessionSidebar from "@/components/chat/SessionSidebar.vue";
import ModelBar from "@/components/chat/ModelBar.vue";
import ChatBoard from "@/components/chat/ChatBoard.vue";
import ChatComposer from "@/components/chat/ChatComposer.vue";
import GenSettingsBar from "@/components/chat/GenSettingsBar.vue";
import TraceTimeline from "@/components/chat/TraceTimeline.vue";
import { useChatStream } from "@/composables/useChatStream";

const { sending, send, abort } = useChatStream();
const gen = ref<InstanceType<typeof GenSettingsBar> | null>(null);
const board = ref<InstanceType<typeof ChatBoard> | null>(null);

function onSend(text: string) {
  const settings = gen.value?.values || {
    temperature: 0.7,
    max_tokens: 512,
    enable_thinking: false,
  };
  send(text, settings);
  // User-initiated send always snaps to bottom, regardless of prior scroll.
  board.value?.scrollToBottom(true);
}
</script>

<template>
  <div class="chat-grid">
    <SessionSidebar class="col-side" />

    <div class="col-main">
      <ModelBar />
      <ChatBoard ref="board" class="board" />
      <ChatComposer
        :sending="sending"
        @send="onSend"
        @abort="abort('user stop')"
      />
      <GenSettingsBar ref="gen" />
    </div>

    <TraceTimeline class="col-trace" />
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.chat-grid {
  display: grid;
  grid-template-columns: 240px 1fr 320px;
  gap: $space-4;
  height: 100%;
  min-height: 0;
}

.col-side,
.col-trace {
  min-height: 0;
}

.col-main {
  display: flex;
  flex-direction: column;
  gap: $space-3;
  min-height: 0;
}

.board {
  flex: 1 1 auto;
  min-height: 0;
}

@media (max-width: 1080px) {
  .chat-grid {
    grid-template-columns: 200px 1fr;
  }
  .col-trace {
    display: none;
  }
}
</style>
