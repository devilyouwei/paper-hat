import { jget, jpatch, jdelete } from "./client";
import type { NeocortexEntry } from "./types";

export const listNeocortex = (sessionId?: string) => {
  const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
  return jget<{ data: NeocortexEntry[] }>(`/api/neocortex${q}`).then((r) => r.data);
};

export const updateNeocortex = (id: string, body: { query: string; response: string }) =>
  jpatch<NeocortexEntry>(`/api/neocortex/${encodeURIComponent(id)}`, body);

export const deleteNeocortex = (id: string) =>
  jdelete<{ deleted: boolean }>(`/api/neocortex/${encodeURIComponent(id)}`);
