<script setup lang="ts">
import { NSlider, NInputNumber, NCheckbox } from "naive-ui";
import { useStorage } from "@vueuse/core";

const temperature = useStorage<number>("hat-temp", 0.7);
const maxTokens = useStorage<number>("hat-max-tokens", 512);
const enableThinking = useStorage<boolean>("hat-enable-thinking", false);

defineExpose({
  get values() {
    return {
      temperature: temperature.value,
      max_tokens: maxTokens.value,
      enable_thinking: enableThinking.value,
    };
  },
});
</script>

<template>
  <div class="gen-bar">
    <div class="cell temp">
      <label>Temp</label>
      <NSlider
        v-model:value="temperature"
        :min="0"
        :max="1.5"
        :step="0.05"
        :format-tooltip="(v: number) => v.toFixed(2)"
      />
      <span class="val">{{ temperature.toFixed(2) }}</span>
    </div>
    <div class="cell">
      <label>Max tokens</label>
      <NInputNumber
        v-model:value="maxTokens"
        size="small"
        :min="32"
        :max="4096"
        :step="32"
      />
    </div>
    <NCheckbox v-model:checked="enableThinking">
      Enable thinking
    </NCheckbox>
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.gen-bar {
  display: flex;
  align-items: center;
  gap: $space-4;
  padding: $space-2 $space-3;
  background: var(--hat-panel);
  border: 1px solid var(--hat-border);
  border-radius: $radius;
  flex-wrap: nowrap;
  white-space: nowrap;
}

.cell {
  display: flex;
  align-items: center;
  gap: $space-2;
  label {
    font-size: 11px;
    color: var(--hat-muted);
  }
}

.cell.temp {
  min-width: 180px;
  flex: 0 1 180px;
  :deep(.n-slider) {
    flex: 1;
    min-width: 90px;
  }
  .val {
    font-family: $font-mono;
    font-size: 12px;
    color: var(--hat-text);
    width: 32px;
    text-align: right;
  }
}
</style>
