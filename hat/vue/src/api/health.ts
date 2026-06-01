import { jget } from "./client";
import type { HealthStatus, ActiveModel, PolicyConfig } from "./types";

export const fetchHealth = () => jget<HealthStatus>("/healthz");
export const fetchPolicy = () => jget<PolicyConfig>("/api/policy");
export const fetchActive = () => jget<ActiveModel | null>("/api/models/active");
