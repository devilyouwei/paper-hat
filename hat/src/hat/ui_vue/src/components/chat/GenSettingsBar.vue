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
    <span class="title">Generation</span>
    <div class="cell wide">
      <label>Temperature</label>
      <div class="slider-row">
        <NSlider
          v-model:value="temperature"
          :min="0"
          :max="1.5"
          :step="0.05"
          :format-tooltip="(v: number) => v.toFixed(2)"
        />
        <span class="val">{{ temperature.toFixed(2) }}</span>
      </div>
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
      <span class="muted small">(Qwen3 &lt;think&gt;…)</span>
    </NCheckbox>
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.gen-bar {
  display: flex;
  align-items: center;
  gap: $space-4;
  padding: $space-3 $space-4;
  background: var(--hat-panel);
  border: 1px solid var(--hat-border);
  border-radius: $radius;
  flex-wrap: wrap;
}

.title {
  font-size: 11px;
  font-weight: 600;
  color: var(--hat-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  label {
    font-size: 11px;
    color: var(--hat-muted);
  }
}
.cell.wide {
  min-width: 260px;
  flex: 1 1 260px;
  max-width: 380px;
}

.slider-row {
  display: flex;
  align-items: center;
  gap: $space-2;
  .val {
    font-family: $font-mono;
    font-size: 12px;
    color: var(--hat-text);
    width: 36px;
    text-align: right;
  }
}

.muted {
  color: var(--hat-muted);
}
.small {
  font-size: 11px;
}
</style>
