<script setup lang="ts">
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";
import { NTabs, NTab, NTag, NTooltip } from "naive-ui";
import { useAppStore } from "@/stores/app";
import { useThemeStore } from "@/stores/theme";
import { Sun, Moon } from "@vicons/tabler";
import { NIcon } from "naive-ui";

const route = useRoute();
const router = useRouter();
const app = useAppStore();
const themeStore = useThemeStore();

const currentTab = computed({
  get: () => (route.name as string) || "chat",
  set: (v) => router.push({ name: v }),
});

const healthOk = computed(() => app.health?.status === "ok");
const healthTitle = computed(() => {
  if (app.healthError) return app.healthError;
  if (!app.health) return "loading…";
  const h = app.health;
  return `status=${h.status} · backend=${h.cortex_backend ?? "?"} · root=${h.model_root ?? "?"}`;
});

const activeLabel = computed(() => {
  if (!app.active) return "—";
  return `${app.active.backend}/${app.active.id}`;
});

const logoUrl = `${import.meta.env.BASE_URL}logo.png`;
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <a class="brand" href="#/chat" title="Hippocampus-Augmented Transformer">
        <img :src="logoUrl" alt="HAT" class="logo" />
        <span class="name">HAT</span>
      </a>

      <NTabs
        v-model:value="currentTab"
        type="segment"
        size="small"
        class="tab-nav"
        animated
      >
        <NTab name="chat">Chat</NTab>
        <NTab name="models">Models</NTab>
        <NTab name="memory">Memory</NTab>
      </NTabs>

      <div class="meta">
        <NTooltip>
          <template #trigger>
            <NTag :type="app.active ? 'success' : 'default'" size="small" round>
              {{ activeLabel }}
            </NTag>
          </template>
          Active model
        </NTooltip>

        <NTooltip>
          <template #trigger>
            <span class="dot" :class="{ ok: healthOk, err: !healthOk && app.booted }" />
          </template>
          {{ healthTitle }}
        </NTooltip>

        <button class="icon-btn" :title="themeStore.isDark ? 'Switch to light' : 'Switch to dark'" @click="themeStore.toggle">
          <NIcon :size="18">
            <Sun v-if="themeStore.isDark" />
            <Moon v-else />
          </NIcon>
        </button>
      </div>
    </header>

    <main class="content">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<style lang="scss" scoped>
@use "@/styles/tokens" as *;

.app-shell {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.topbar {
  height: $topbar-h;
  flex: 0 0 auto;
  display: grid;
  grid-template-columns: 180px 1fr 280px;
  align-items: center;
  padding: 0 $space-4;
  background: var(--hat-bg-2);
  border-bottom: 1px solid var(--hat-border);
  gap: $space-3;
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: $space-2;
  color: var(--hat-text);
  font-weight: 700;
  font-size: 16px;
  letter-spacing: 0.04em;
  text-decoration: none;
  .logo {
    height: 32px;
    width: auto;
    object-fit: contain;
    display: block;
  }
}

.tab-nav {
  justify-self: center;
  min-width: 320px;
  max-width: 420px;
  width: 100%;
}

.meta {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: $space-3;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--hat-muted);
  display: inline-block;
  &.ok {
    background: var(--hat-ok);
    box-shadow: 0 0 8px rgba(74, 222, 128, 0.5);
  }
  &.err {
    background: var(--hat-danger);
    box-shadow: 0 0 8px rgba(255, 107, 107, 0.5);
  }
}

.icon-btn {
  background: transparent;
  border: 1px solid var(--hat-border);
  color: var(--hat-text);
  border-radius: 8px;
  width: 32px;
  height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background 0.15s ease;
  &:hover {
    background: var(--hat-panel-2);
  }
}

.content {
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  padding: $space-4;
  background: var(--hat-bg);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
