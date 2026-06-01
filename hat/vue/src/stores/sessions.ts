import { defineStore } from "pinia";
import { ref, computed } from "vue";
import * as api from "@/api/sessions";
import type { SessionSummary, SessionInteraction } from "@/api/types";

export const useSessionsStore = defineStore("sessions", () => {
  const list = ref<SessionSummary[]>([]);
  const currentId = ref<string | null>(null);
  const messages = ref<SessionInteraction[]>([]);
  const loading = ref(false);
  const status = ref<string>("");

  const current = computed(
    () => list.value.find((s) => s.id === currentId.value) ?? null,
  );

  async function refreshList(selectId: string | null = null) {
    list.value = await api.listSessions();
    if (selectId) currentId.value = selectId;
  }

  async function open(id: string) {
    loading.value = true;
    try {
      const detail = await api.getSession(id);
      currentId.value = id;
      messages.value = detail.messages;
      status.value = "";
    } finally {
      loading.value = false;
    }
  }

  async function create() {
    const s = await api.createSession();
    currentId.value = s.id;
    messages.value = [];
    await refreshList(s.id);
    return s;
  }

  async function remove(id: string) {
    await api.deleteSession(id);
    if (currentId.value === id) {
      currentId.value = null;
      messages.value = [];
    }
    await refreshList();
  }

  function appendUser(content: string): SessionInteraction {
    const it: SessionInteraction = {
      id: `local-${Date.now()}`,
      query: content,
      response: "",
      timestamp: new Date().toISOString(),
    };
    messages.value.push(it);
    return it;
  }

  function setLastResponse(content: string) {
    const last = messages.value[messages.value.length - 1];
    if (last) last.response = content;
  }

  function setLastHat(hat: NonNullable<SessionInteraction["hat"]>) {
    const last = messages.value[messages.value.length - 1];
    if (last) last.hat = { ...(last.hat ?? {}), ...hat };
  }

  function setSessionIdIfMissing(id: string) {
    if (!currentId.value) currentId.value = id;
  }

  return {
    list,
    currentId,
    messages,
    loading,
    status,
    current,
    refreshList,
    open,
    create,
    remove,
    appendUser,
    setLastResponse,
    setLastHat,
    setSessionIdIfMissing,
  };
});
