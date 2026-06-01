import { jget, jpost, jdelete } from "./client";
import type { EmbeddingCatalogItem, ActiveEmbedder } from "./types";

export const listCatalog = (backend: string) =>
  jget<{ backend: string; items: EmbeddingCatalogItem[] }>(
    `/api/embedding-models?backend=${backend}`,
  ).then((r) => r.items);

export const getActive = () =>
  jget<ActiveEmbedder | null>("/api/embedding-models/active");

export const setActive = (backend: string, id: string) =>
  jpost<ActiveEmbedder>("/api/embedding-models/active", { backend, id });

export const unloadActive = () =>
  jdelete<{ unloaded: number }>("/api/embedding-models/active");

export const deleteWeights = (backend: string, id: string) =>
  jdelete<{ removed: boolean }>(
    `/api/embedding-models/${backend}/${encodeURIComponent(id)}`,
  );

export const cancelDownload = (backend: string, id: string) =>
  jpost<Record<string, never>>("/api/embedding-models/download/cancel", {
    backend,
    id,
  });

export const downloadStreamUrl = (backend: string, id: string) =>
  `/api/embedding-models/download/stream?backend=${encodeURIComponent(
    backend,
  )}&id=${encodeURIComponent(id)}`;
