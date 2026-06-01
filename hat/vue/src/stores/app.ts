import { defineStore } from "pinia";
import { ref } from "vue";
import { fetchHealth, fetchActive, fetchPolicy } from "@/api/health";
import type { HealthStatus, ActiveModel, PolicyConfig } from "@/api/types";

export const useAppStore = defineStore("app", () => {
  const health = ref<HealthStatus | null>(null);
  const healthError = ref<string | null>(null);
  const active = ref<ActiveModel | null>(null);
  const policy = ref<PolicyConfig | null>(null);
  const booted = ref(false);

  async function boot() {
    const [h, a, p] = await Promise.allSettled([fetchHealth(), fetchActive(), fetchPolicy()]);
    if (h.status === "fulfilled") health.value = h.value;
    else healthError.value = String(h.reason);
    if (a.status === "fulfilled") active.value = a.value;
    if (p.status === "fulfilled") policy.value = p.value;
    booted.value = true;
  }

  async function refreshActive() {
    active.value = await fetchActive().catch(() => null);
  }

  function setActive(a: ActiveModel | null) {
    active.value = a;
  }

  return { health, healthError, active, policy, booted, boot, refreshActive, setActive };
});
