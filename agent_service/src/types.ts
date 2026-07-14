export interface AgentConfig {
  enabled: boolean;
  model: string;
  maxTokens: number;
  temperature: number;
  timeoutMs: number;
}

export interface RuntimeConfig {
  host: string;
  port: number;
  projectRoot: string;
  agent: AgentConfig;
}

export interface SessionContext {
  mount_type?: "eq" | "altaz" | "unknown";
  target_angular_size?: "compact" | "extended" | "full_frame";
  camera_type?: "consumer_osc" | "astronomy_camera" | "unknown";
  calibration_darks?: boolean;
  calibration_flats?: boolean;
  calibration_bias?: boolean;
  system_ram_mb?: number;
  cpu_cores?: number;
  notes?: string;
  accepted_pi_memories?: unknown[];
  negative_pi_memories?: unknown[];
}

export interface ScanAnalysisRequest {
  schema_version?: string;
  ai_request?: unknown;
  scan_result?: unknown;
  base_config?: unknown;
  config_schema?: Record<string, unknown>;
  scan_metrics?: unknown;
  session_context?: SessionContext;
  allowed_config_paths?: string[];
  model?: string;
  send_paths?: boolean;
  force?: boolean;
}

export interface ScanAnalysisResponse {
  schema_version: "pi.scan-analysis.v1";
  summary: string;
  confidence: number;
  detected_scenarios: unknown[];
  recommendations: unknown[];
  warnings: string[];
  review_required: boolean;
  // Progress metadata
  _meta?: {
    streaming_duration_ms?: number;
    response_chars?: number;
    model?: string;
    provider?: string;
    temperature?: number;
    max_tokens?: number;
    prompt?: string;
    prompt_sha256?: string;
    request_sha256?: string;
    base_config_sha256?: string;
    config_schema_sha256?: string;
    scan_result_sha256?: string;
    scan_metrics_sha256?: string;
  };
}

// Progress events for SSE streaming
export type AnalysisProgressPhase =
  | "initializing"
  | "building_prompt"
  | "ai_thinking"
  | "receiving_tokens"
  | "parsing_response"
  | "validating"
  | "complete"
  | "error";

export interface AnalysisProgressEvent {
  phase: AnalysisProgressPhase;
  message: string;
  progress?: number; // 0-100 for phases that support it
  delta?: string; // Token delta during streaming
  charsReceived?: number;
  estimatedTotal?: number;
}

export type ProgressCallback = (event: AnalysisProgressEvent) => void;
