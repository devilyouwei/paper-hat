<script setup lang="ts">
import { computed } from "vue";
import { NTag, NCollapse, NCollapseItem } from "naive-ui";
import { splitThink, renderMarkdown } from "@/utils/markdown";

const props = defineProps<{
  role: "user" | "assistant";
  content: string;
  hat?: { uncertainty?: number; decision?: string; reason?: string } | null;
}>();

const parts = computed(() => splitThink(props.content || ""));

const uncertaintyLabel = computed(() => {
  const u = props.hat?.uncertainty;
  if (typeof u !== "number") return null;
  let suffix = "";
  const dec = props.hat?.decision;
  if (dec === "skipped") suffix = " · skipped";
  else if (dec === "dropped") suffix = " · dropped";
  else if (dec === "created") suffix = " · create";
  else if (dec === "revised") suffix = " · revise";
  return `U=${u.toFixed(2)}${suffix}`;
});

const uncertaintyType = computed<"default" | "warning" | "error" | "success">(() => {
  const u = props.hat?.uncertainty;
  if (typeof u !== "number") return "default";
  if (u >= 0.7) return "error";
  if (u >= 0.3) return "warning";
  return "success";
});
</script>

<template>
  <div class="bubble" :class="role">
    <NTag
      v-if="uncertaintyLabel"
      :type="uncertaintyType"
      size="small"
      round
      class="u-badge"
      :title="hat?.reason || ''"
    >
      {{ uncertaintyLabel }}
    </NTag>

    <template v-for="(p, i) in parts" :key="i">
      <NCollapse v-if="p.kind === 'think'" arrow-placement="right" class="think">
        <NCollapseItem name="t" title="thinking…">
          <pre class="think-body">{{ p.value.trim() }}</pre>
        </NCollapseItem>
      </NCollapse>
      <div v-else class="md" v-html="renderMarkdown(p.value)" />
    </template>
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.bubble {
  position: relative;
  max-width: 760px;
  padding: $space-3 $space-4;
  border-radius: $radius-lg;
  word-break: break-word;
  white-space: normal;
  line-height: 1.6;
}

.bubble.user {
  align-self: flex-end;
  background: var(--hat-user-bubble);
  color: #fff;
  border-bottom-right-radius: 4px;
  :deep(a) {
    color: #fff;
    text-decoration: underline;
  }
}

.bubble.assistant {
  align-self: flex-start;
  background: var(--hat-assistant-bubble);
  border: 1px solid var(--hat-border);
  border-bottom-left-radius: 4px;
}

.u-badge {
  position: absolute;
  top: -10px;
  right: 10px;
  font-variant-numeric: tabular-nums;
}

.md {
  :deep(p) {
    margin: 0 0 0.5em;
    &:last-child {
      margin-bottom: 0;
    }
  }
  :deep(pre) {
    background: rgba(0, 0, 0, 0.35);
    color: #e8ecf3;
    padding: 10px 12px;
    border-radius: $radius-sm;
    overflow-x: auto;
    font-size: 12.5px;
  }
  :deep(code) {
    font-family: $font-mono;
    font-size: 0.92em;
    padding: 1px 5px;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 4px;
  }
  :deep(pre code) {
    background: transparent;
    padding: 0;
  }
  :deep(ul),
  :deep(ol) {
    padding-left: 1.4em;
    margin: 0.4em 0;
  }
  :deep(blockquote) {
    border-left: 3px solid var(--hat-border);
    padding-left: $space-3;
    color: var(--hat-muted);
    margin: 0.4em 0;
  }
}

.think {
  margin: $space-2 0;
  :deep(.n-collapse-item__header) {
    font-size: 12px;
    color: var(--hat-muted);
    font-style: italic;
  }
}

.think-body {
  white-space: pre-wrap;
  font-family: $font-mono;
  font-size: 12px;
  color: var(--hat-muted);
  margin: 0;
  padding: $space-2 $space-3;
  background: rgba(0, 0, 0, 0.15);
  border-radius: $radius-sm;
}
</style>
