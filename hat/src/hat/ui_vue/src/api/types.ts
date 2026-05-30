export interface CatalogItem {
  id: string;
  display: string;
  repo_id: string;
  backend: string;
  installed: boolean;
  size_gb: number | null;
  notes?: string;
}

export interface ActiveModel {
  id: string;
  backend: string;
}

export interface SessionSummary {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
}

export interface SessionInteractionHat {
  decision?: string;
  uncertainty?: number;
  reason?: string;
}

export interface SessionInteraction {
  id: string;
  query: string;
  response: string;
  timestamp: string;
  hat?: SessionInteractionHat;
}

export interface SessionDetail {
  session: SessionSummary;
  messages: SessionInteraction[];
}

export interface NeocortexSignals {
  uncertainty?: number;
  novelty?: number;
  feedback?: number;
}

export interface NeocortexEntry {
  trace_id: string;
  query: string;
  response: string;
  score: number;
  signals: NeocortexSignals;
  metadata: {
    source?: string;
    timestamp?: string;
    signals?: NeocortexSignals;
    extras?: Record<string, unknown>;
    [key: string]: unknown;
  };
  session_id?: string;
  interaction_id?: string;
  interaction_ids?: string[];
}

export interface PolicyConfig {
  write_policy?: { kind?: string; threshold?: number };
  oracle?: { enabled: boolean; model?: string; threshold?: number; rps?: number; daily_calls?: number };
}

export interface HealthStatus {
  status: string;
  cortex_backend?: string;
  model_root?: string;
}

export type TraceStage =
  | "uncertainty"
  | "skipped"
  | "triage_start"
  | "triage_done"
  | "route_start"
  | "route_done"
  | "routed"
  | "scored"
  | "created"
  | "revised"
  | "rejected"
  | "dropped"
  | "abstracting"
  | string;

export interface TraceEvent {
  stage: TraceStage;
  trace_id?: string;
  interaction_id?: string;
  uncertainty?: number;
  threshold?: number;
  keep?: boolean;
  verdict?: string;
  reason?: string;
  n_priors?: number;
  decision?: string;
  rationale?: string;
  parsed?: unknown;
  score?: number;
  accepted?: boolean;
  target_response?: string;
  [k: string]: unknown;
}

export interface ChatCompletionDelta {
  choices?: Array<{ delta?: { content?: string } }>;
  hat_session_id?: string;
  hat_trace_event?: TraceEvent;
}

export interface ChatRequest {
  model: string;
  messages: Array<{ role: "user" | "assistant" | "system"; content: string }>;
  stream: boolean;
  temperature: number;
  max_tokens: number;
  chat_template_kwargs?: { enable_thinking?: boolean };
  session_id?: string;
}

export interface DownloadEvent {
  bytes_done?: number;
  bytes_total?: number;
  files_done?: number;
  files_total?: number;
  message?: string;
}
