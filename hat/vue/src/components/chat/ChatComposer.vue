<script setup lang="ts">
import { ref } from "vue";
import { NInput, NButton } from "naive-ui";
import { Send, PlayerStop } from "@vicons/tabler";
import { NIcon } from "naive-ui";

const props = defineProps<{ sending: boolean }>();
const emit = defineEmits<{ (e: "send", text: string): void; (e: "abort"): void }>();

const text = ref("");

function submit() {
  const t = text.value.trim();
  if (!t || props.sending) return;
  text.value = "";
  emit("send", t);
}

function onKeydown(e: KeyboardEvent) {
  // Enter to send, Shift+Enter newline. Avoid IME composition.
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    submit();
  }
}
</script>

<template>
  <form class="composer" @submit.prevent="submit">
    <NInput
      v-model:value="text"
      type="textarea"
      placeholder="Ask anything…   (Shift+Enter for newline)"
      :autosize="{ minRows: 2, maxRows: 8 }"
      class="ta"
      @keydown="onKeydown"
    />
    <div class="actions">
      <NButton
        v-if="sending"
        type="warning"
        @click="emit('abort')"
      >
        <template #icon><NIcon><PlayerStop /></NIcon></template>
        Stop
      </NButton>
      <NButton
        v-else
        type="primary"
        :disabled="!text.trim()"
        attr-type="submit"
      >
        <template #icon><NIcon><Send /></NIcon></template>
        Send
      </NButton>
    </div>
  </form>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.composer {
  display: flex;
  gap: $space-3;
  align-items: flex-end;
}

.ta {
  flex: 1 1 auto;
}

.actions {
  flex: 0 0 auto;
}
</style>
