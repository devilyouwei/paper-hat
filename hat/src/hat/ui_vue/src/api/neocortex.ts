import { jget, jpatch, jdelete } from "./client";
import type { NeocortexEntry } from "./types";

export const listNeocortex = (opts: { sessionId?: string; embedModel?: string } = {}) => {
  const params = new URLSearchParams();
  if (opts.sessionId) params.set("session_id", opts.sessionId);
  if (opts.embedModel) params.set("embed_model", opts.embedModel);
  const q = params.toString();
  const suffix = q ? `?${q}` : "";
  return jget<{ data: NeocortexEntry[] }>(`/api/neocortex${suffix}`).then(
    (r) => r.data,
  );
};

export const updateNeocortex = (id: string, body: { query: string; response: string }) =>
  jpatch<NeocortexEntry>(`/api/neocortex/${encodeURIComponent(id)}`, body);

export const deleteNeocortex = (id: string) =>
  jdelete<{ deleted: boolean }>(`/api/neocortex/${encodeURIComponent(id)}`);
