import { jget, jpost, jdelete, jpatch } from "./client";
import type { SessionSummary, SessionDetail } from "./types";

export const listSessions = () =>
  jget<{ data: SessionSummary[] }>("/api/sessions").then((r) => r.data);

export const createSession = () => jpost<SessionSummary>("/api/sessions", {});

export const getSession = (id: string) => jget<SessionDetail>(`/api/sessions/${id}`);

export const renameSession = (id: string, title: string) =>
  jpatch<SessionSummary>(`/api/sessions/${id}`, { title });

export const deleteSession = (id: string) =>
  jdelete<{ deleted: boolean }>(`/api/sessions/${id}`);
