import { jget, jpost, jdelete } from "./client";
import type { CatalogItem, ActiveModel } from "./types";

export const listCatalog = (backend: string) =>
  jget<{ backend: string; items: CatalogItem[] }>(`/api/models?backend=${backend}`).then(
    (r) => r.items,
  );

export const getActive = () => jget<ActiveModel | null>("/api/models/active");

export const setActive = (backend: string, id: string) =>
  jpost<ActiveModel>("/api/models/active", { backend, id });

export const unloadActive = () => jdelete<{ status: string }>("/api/models/active");

export const deleteWeights = (backend: string, id: string) =>
  jdelete<{ deleted: boolean }>(`/api/models/${backend}/${encodeURIComponent(id)}`);

export const cancelDownload = (backend: string, id: string) =>
  jpost<Record<string, never>>("/api/models/download/cancel", { backend, id });

export const downloadStreamUrl = (backend: string, id: string) =>
  `/api/models/download/stream?backend=${encodeURIComponent(backend)}&id=${encodeURIComponent(id)}`;
