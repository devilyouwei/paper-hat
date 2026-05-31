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

export interface EmbeddingCatalogItem {
  id: string;
  display: string;
  repo_id: string;
  backend: "mlx_embed" | "hf_embed" | string;
  installed: boolean;
  size_gb: number | null;
  notes?: string;
  local_dir?: string;
}

export interface ActiveEmbedder {
  id: string;
  backend: string;
  name?: string | null;
  index_path?: string | null;
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
  dedup?: {
    enabled: boolean;
    threshold?: number;
    active_embedder?: { backend: string; id: string } | null;
    index_path?: string | null;
  };
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
  | "extract_start"
  | "extract_done"
  | "extracted"
  | "dedup"
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
  matched_trace_id?: string;
  interaction_id?: string;
  uncertainty?: number;
  threshold?: number;
  similarity?: number;
  kp_index?: number;
  n_kps?: number;
  kps?: Array<{ trace_id?: string; query?: string; target?: string }>;
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
