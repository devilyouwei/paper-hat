import { defineStore } from "pinia";
import { ref, computed } from "vue";
import * as api from "@/api/models";
import type { CatalogItem, ActiveModel } from "@/api/types";

export interface DownloadState {
  backend: string;
  id: string;
  bytes_done: number;
  bytes_total: number;
  files_done: number;
  files_total: number;
  state: "pending" | "downloading" | "done" | "cancelled" | "error";
  message?: string;
  source?: EventSource;
}

export const useModelsStore = defineStore("models", () => {
  const backend = ref<string>("mlx");
  const items = ref<CatalogItem[]>([]);
  const active = ref<ActiveModel | null>(null);
  const downloads = ref<Map<string, DownloadState>>(new Map());
  const loading = ref(false);

  function dlKey(b: string, id: string) {
    return `${b}/${id}`;
  }

  const installedForBackend = computed(() =>
    items.value.filter((it) => it.installed).map((it) => it.id),
  );

  async function loadCatalog(b?: string) {
    if (b) backend.value = b;
    loading.value = true;
    try {
      items.value = await api.listCatalog(backend.value);
    } finally {
      loading.value = false;
    }
  }

  async function refreshActive() {
    active.value = await api.getActive().catch(() => null);
  }

  async function activate(b: string, id: string) {
    const a = await api.setActive(b, id);
    active.value = a;
    return a;
  }

  async function unload() {
    await api.unloadActive();
    active.value = null;
  }

  async function remove(b: string, id: string) {
    await api.deleteWeights(b, id);
    await loadCatalog();
  }

  function startDownload(b: string, id: string) {
    const key = dlKey(b, id);
    const existing = downloads.value.get(key);
    if (existing && (existing.state === "pending" || existing.state === "downloading")) {
      return existing;
    }

    const url = api.downloadStreamUrl(b, id);
    const src = new EventSource(url);
    const state: DownloadState = {
      backend: b,
      id,
      bytes_done: 0,
      bytes_total: 0,
      files_done: 0,
      files_total: 0,
      state: "pending",
      source: src,
    };
    downloads.value.set(key, state);
    // Reactivity for Map: replace
    downloads.value = new Map(downloads.value);

    const update = (patch: Partial<DownloadState>) => {
      const cur = downloads.value.get(key);
      if (!cur) return;
      Object.assign(cur, patch);
      downloads.value = new Map(downloads.value);
    };

    src.addEventListener("start", (ev) => {
      try {
        const d = JSON.parse((ev as MessageEvent).data || "{}");
        update({
          state: "downloading",
          bytes_total: d.bytes_total || 0,
          files_total: d.files_total || 0,
        });
      } catch {
        update({ state: "downloading" });
      }
    });
    src.addEventListener("progress", (ev) => {
      try {
        const d = JSON.parse((ev as MessageEvent).data || "{}");
        update({
          bytes_done: d.bytes_done || 0,
          bytes_total: d.bytes_total || d.bytesTotal || 0,
          files_done: d.files_done || 0,
          files_total: d.files_total || d.filesTotal || 0,
        });
      } catch {
        /* ignore */
      }
    });
    src.addEventListener("done", () => {
      update({ state: "done" });
      src.close();
      loadCatalog();
    });
    src.addEventListener("cancelled", () => {
      update({ state: "cancelled" });
      src.close();
    });
    src.addEventListener("error", (ev) => {
      let msg = "error";
      try {
        const d = JSON.parse(((ev as MessageEvent).data as string) || "{}");
        msg = d.message || msg;
      } catch {
        /* ignore */
      }
      update({ state: "error", message: msg });
      src.close();
    });

    return state;
  }

  async function cancelDownload(b: string, id: string) {
    const key = dlKey(b, id);
    const st = downloads.value.get(key);
    try {
      await api.cancelDownload(b, id);
    } catch {
      /* ignore */
    }
    if (st?.source) st.source.close();
    if (st) {
      st.state = "cancelled";
      downloads.value = new Map(downloads.value);
    }
  }

  function downloadFor(b: string, id: string): DownloadState | undefined {
    return downloads.value.get(dlKey(b, id));
  }

  return {
    backend,
    items,
    active,
    downloads,
    loading,
    installedForBackend,
    loadCatalog,
    refreshActive,
    activate,
    unload,
    remove,
    startDownload,
    cancelDownload,
    downloadFor,
  };
});
