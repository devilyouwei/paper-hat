import { defineStore } from "pinia";
import { ref, computed } from "vue";

export const useThemeStore = defineStore("theme", () => {
  const stored = ref<"dark" | "light" | "system">(
    (localStorage.getItem("hat-theme") as "dark" | "light") || "system",
  );

  const systemDark = ref(
    window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches,
  );

  const isDark = computed(() =>
    stored.value === "system" ? systemDark.value : stored.value === "dark",
  );

  function init() {
    document.documentElement.dataset.theme = isDark.value ? "dark" : "light";
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", (e) => {
      systemDark.value = e.matches;
      if (stored.value === "system") sync();
    });
  }

  function sync() {
    document.documentElement.dataset.theme = isDark.value ? "dark" : "light";
  }

  function toggle() {
    stored.value = isDark.value ? "light" : "dark";
    localStorage.setItem("hat-theme", stored.value);
    sync();
  }

  return { stored, isDark, init, toggle };
});
