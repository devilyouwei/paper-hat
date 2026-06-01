<script setup lang="ts">
import { computed, onMounted } from "vue";
import { darkTheme, lightTheme, NConfigProvider, NMessageProvider, NNotificationProvider, NDialogProvider, NLoadingBarProvider, NGlobalStyle } from "naive-ui";
import { useThemeStore } from "@/stores/theme";
import { useAppStore } from "@/stores/app";
import AppShell from "@/layouts/AppShell.vue";

const themeStore = useThemeStore();
const appStore = useAppStore();

const naiveTheme = computed(() => (themeStore.isDark ? darkTheme : lightTheme));
const themeOverrides = computed(() => ({
  common: {
    primaryColor: themeStore.isDark ? "#6c8eff" : "#3361d8",
    primaryColorHover: themeStore.isDark ? "#869fff" : "#4a73e0",
    primaryColorPressed: themeStore.isDark ? "#5478e8" : "#2a55c0",
    borderRadius: "8px",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  },
}));

onMounted(() => {
  themeStore.init();
  appStore.boot();
});
</script>

<template>
  <NConfigProvider :theme="naiveTheme" :theme-overrides="themeOverrides">
    <NLoadingBarProvider>
      <NMessageProvider>
        <NNotificationProvider>
          <NDialogProvider>
            <NGlobalStyle />
            <AppShell />
          </NDialogProvider>
        </NNotificationProvider>
      </NMessageProvider>
    </NLoadingBarProvider>
  </NConfigProvider>
</template>
