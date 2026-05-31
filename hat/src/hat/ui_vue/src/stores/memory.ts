import { defineStore } from "pinia";
import { ref, computed } from "vue";
import * as api from "@/api/neocortex";
import type { NeocortexEntry } from "@/api/types";

export const useMemoryStore = defineStore("memory", () => {
  const entries = ref<NeocortexEntry[]>([]);
  const query = ref("");
  const loading = ref(false);
  const editingId = ref<string | null>(null);
  const embedFilter = ref<string | null>(null);

  const filtered = computed(() => {
    const q = query.value.trim().toLowerCase();
    if (!q) return entries.value;
    return entries.value.filter((e) =>
      [e.trace_id, e.query, e.response].some((v) => (v || "").toLowerCase().includes(q)),
    );
  });

  async function refresh() {
    loading.value = true;
    try {
      entries.value = await api.listNeocortex(
        embedFilter.value ? { embedModel: embedFilter.value } : {},
      );
    } finally {
      loading.value = false;
    }
  }

  async function setEmbedFilter(v: string | null) {
    embedFilter.value = v;
    await refresh();
  }

  async function save(id: string, body: { query: string; response: string }) {
    const updated = await api.updateNeocortex(id, body);
    const idx = entries.value.findIndex((e) => e.trace_id === id);
    if (idx >= 0) entries.value.splice(idx, 1, updated);
  }

  async function remove(id: string) {
    await api.deleteNeocortex(id);
    entries.value = entries.value.filter((e) => e.trace_id !== id);
  }

  return {
    entries,
    query,
    loading,
    editingId,
    embedFilter,
    filtered,
    refresh,
    save,
    remove,
    setEmbedFilter,
  };
});
