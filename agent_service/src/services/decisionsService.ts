import crypto from "node:crypto";

/**
 * Jev decisions adapter (docs/PI/pi_jev_implementierungsplan_de.md, M3).
 *
 * Translates the application contract pi.decisions.request.v1 into the verified provider wire format
 * (POST https://openrouter.ai/api/alpha/decisions, model typesafe/jev-1.13, see
 * docs/PI/pi_jev_m0_provider_protocol_de.md) and normalizes the answer into
 * pi.decisions.response.v1. Provider JSON is untrusted input: anything that does not satisfy the
 * documented shape is rejected wholesale, never repaired.
 *
 * Deliberately independent of @earendil-works/pi-ai (chat-completions shaped) and of PI's single
 * global provider slot. The HTTP transport, sleep and API-key resolver are injectable so the whole
 * error matrix is testable without a network.
 */

export type DecisionsMode = "off" | "shadow" | "suggest";

export interface DecisionsConfig {
  mode: DecisionsMode;
  endpoint: string;
  /** Pinned provider model id, e.g. "typesafe/jev-1.13"; aliases (latest/preview) are refused. */
  model: string;
  /** Total budget for one decision including retries. */
  deadlineMs: number;
  maxResponseBytes: number;
  maxRetries: number;
  retryBackoffMs: number;
  maxConcurrent: number;
  maxQueued: number;
  /** Echoed for the backend; the sidecar itself never filters by it. */
  allowExperimentalSuggestions: boolean;
}

export const DEFAULT_DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions";
export const DEFAULT_DECISIONS_MODEL = "typesafe/jev-1.13";

export function defaultDecisionsConfig(): DecisionsConfig {
  return {
    mode: "off",
    endpoint: DEFAULT_DECISIONS_ENDPOINT,
    model: DEFAULT_DECISIONS_MODEL,
    deadlineMs: 30_000,
    maxResponseBytes: 256 * 1024,
    maxRetries: 1,
    retryBackoffMs: 500,
    maxConcurrent: 2,
    maxQueued: 4,
    allowExperimentalSuggestions: false,
  };
}

export interface DecisionsRequest {
  request_id: string;
  state_hash: string;
  question_set_version: string;
  state_projection: Record<string, unknown>;
  allowed_candidates: string[];
}

export type DecisionsStatus = "ok" | "unavailable" | "invalid_response";

export interface DecisionsResponse {
  request_id: string;
  state_hash: string;
  status: DecisionsStatus;
  model_requested: string;
  model_reported?: string | null;
  selection?: {
    candidate_id: string;
    probabilities: Record<string, number>;
    /** The provider's own derived statistic, forwarded under its own name; never a gate here. */
    provider_confidence?: number;
  } | null;
  usage?: { input_tokens: number; output_tokens: number; cost: number };
  generation_id?: string;
  error_code?: string;
}

/** A caller/programming error in the request itself (mapped to HTTP 400), not a provider outcome. */
export class DecisionsRequestError extends Error {
  constructor(public readonly reason: string) {
    super(`invalid decisions request: ${reason}`);
    this.name = "DecisionsRequestError";
  }
}

export interface TransportRequest {
  url: string;
  headers: Record<string, string>;
  body: string;
  signal: AbortSignal;
  maxResponseBytes: number;
}

export interface TransportResponse {
  status: number;
  /** Lower-cased header names. */
  headers: Record<string, string>;
  bodyText: string;
  /** True when the body exceeded maxResponseBytes (bodyText is then incomplete and must not be parsed). */
  truncated: boolean;
}

export type DecisionsTransport = (req: TransportRequest) => Promise<TransportResponse>;
export type SleepFn = (ms: number, signal: AbortSignal) => Promise<void>;
export type ApiKeyResolver = () => string | undefined | Promise<string | undefined>;

export interface DecisionsDeps {
  transport?: DecisionsTransport;
  sleep?: SleepFn;
  resolveApiKey?: ApiKeyResolver;
  /** Never receives state, request bodies or keys. */
  log?: (line: string) => void;
  now?: () => number;
}

// --------------------------------------------------------------------------------------------
// Question sets: the sidecar owns the model-facing wording; the backend only names a version.
// --------------------------------------------------------------------------------------------

interface QuestionSet {
  questionId: string;
  instructions: string;
  describe: Record<string, string>;
}

const BASELINE_CANDIDATES = ["keep_current", "insufficient_evidence"] as const;

export const QUESTION_SETS: Record<string, QuestionSet> = {
  "decision-questions.v1": {
    questionId: "candidate_selection",
    instructions:
      "You are given pre-run acquisition statistics for an astrophotography stacking run in `state`. " +
      "Choose exactly one option. Choose keep_current unless a change is clearly warranted by the measurements; " +
      "choose insufficient_evidence when the measurements do not support a confident choice. " +
      "Use only what is present in `state`; a missing or not_applicable measurement is unknown, never zero.",
    describe: {
      keep_current: "Leave the configuration unchanged.",
      insufficient_evidence: "Abstain: the available evidence does not support a confident choice.",
      enable_adaptive_weights: "Enable adaptive per-frame quality weighting (global_metrics.adaptive_weights = true).",
    },
  },
};

// --------------------------------------------------------------------------------------------
// Validation helpers
// --------------------------------------------------------------------------------------------

const CANDIDATE_ID_RE = /^[a-z0-9_]{1,64}$/;
const LEAK_RE = /^\/|^[A-Za-z]:[\\/]|^\\\\|sk-or-|:\/\/|api[_-]?key/i;
const MAX_STATE_BYTES = 64 * 1024;

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function findLeak(value: unknown, where: string): string | undefined {
  if (typeof value === "string") return LEAK_RE.test(value) ? where : undefined;
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      const hit = findLeak(value[i], `${where}[${i}]`);
      if (hit) return hit;
    }
  } else if (isPlainObject(value)) {
    for (const [k, v] of Object.entries(value)) {
      if (LEAK_RE.test(k)) return `${where}.${k}`;
      const hit = findLeak(v, `${where}.${k}`);
      if (hit) return hit;
    }
  }
  return undefined;
}

export function isPinnedModel(model: string): boolean {
  return /^[a-z0-9._-]+\/[a-z0-9._-]+$/i.test(model) && /\d/.test(model) && !/(latest|preview)$/i.test(model);
}

export function validateRequest(req: unknown): DecisionsRequest {
  if (!isPlainObject(req)) throw new DecisionsRequestError("body_not_object");
  const { request_id, state_hash, question_set_version, state_projection, allowed_candidates } = req;
  if (typeof request_id !== "string" || !request_id || request_id.length > 128) throw new DecisionsRequestError("request_id");
  if (typeof state_hash !== "string" || !state_hash || state_hash.length > 128) throw new DecisionsRequestError("state_hash");
  if (typeof question_set_version !== "string" || !Object.hasOwn(QUESTION_SETS, question_set_version))
    throw new DecisionsRequestError("question_set_version_unknown");
  if (!isPlainObject(state_projection)) throw new DecisionsRequestError("state_projection");
  if (Buffer.byteLength(JSON.stringify(state_projection)) > MAX_STATE_BYTES) throw new DecisionsRequestError("state_projection_too_large");
  const leak = findLeak(state_projection, "state_projection");
  if (leak) throw new DecisionsRequestError(`state_projection_leak:${leak}`);
  if (!Array.isArray(allowed_candidates) || allowed_candidates.length < 2 || allowed_candidates.length > 255)
    throw new DecisionsRequestError("allowed_candidates_size");
  const ids = new Set<string>();
  for (const id of allowed_candidates) {
    if (typeof id !== "string" || !CANDIDATE_ID_RE.test(id)) throw new DecisionsRequestError("allowed_candidates_id");
    if (ids.has(id)) throw new DecisionsRequestError("allowed_candidates_duplicate");
    ids.add(id);
  }
  for (const base of BASELINE_CANDIDATES) if (!ids.has(base)) throw new DecisionsRequestError(`allowed_candidates_missing_${base}`);
  return { request_id, state_hash, question_set_version, state_projection, allowed_candidates: [...allowed_candidates] };
}

export function buildProviderBody(req: DecisionsRequest, model: string): string {
  const qs = QUESTION_SETS[req.question_set_version];
  const criteria: Record<string, string | null> = {};
  for (const id of [...req.allowed_candidates].sort()) criteria[id] = qs.describe[id] ?? null;
  return JSON.stringify({
    model,
    state: req.state_projection,
    questions: { [qs.questionId]: { type: "choice", instructions: qs.instructions, criteria } },
  });
}

type Validated = { ok: true; value: DecisionsResponse } | { ok: false; code: string };

const TOP_KEYS = new Set(["model", "answers", "usage", "id", "provider"]);
const ANSWER_KEYS = new Set(["type", "choice", "probabilities", "confidence"]);

function finiteIn01(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
}

/** Strict check against docs/PI openrouter.decisions.response.v1: an extra field is a rejection. */
export function validateProviderResponse(raw: unknown, req: DecisionsRequest, model: string): Validated {
  const bad = (code: string): Validated => ({ ok: false, code });
  if (!isPlainObject(raw)) return bad("response_not_object");
  for (const k of Object.keys(raw)) if (!TOP_KEYS.has(k)) return bad("unexpected_field");
  if (typeof raw.model !== "string" || !raw.model) return bad("model_missing");
  // Version drift: the build the provider reports must belong to the pinned model we asked for.
  if (raw.model !== model && !raw.model.startsWith(`${model}-`)) return bad("model_version_mismatch");
  if (!isPlainObject(raw.answers)) return bad("answers_missing");
  const qs = QUESTION_SETS[req.question_set_version];
  const answer = raw.answers[qs.questionId];
  if (!isPlainObject(answer)) return bad("answer_missing");
  if (Object.keys(raw.answers).length !== 1) return bad("unexpected_answer");
  for (const k of Object.keys(answer)) if (!ANSWER_KEYS.has(k)) return bad("unexpected_field");
  if (answer.type !== "choice") return bad("answer_type");
  if (typeof answer.choice !== "string" || !req.allowed_candidates.includes(answer.choice)) return bad("unknown_candidate");
  if (!isPlainObject(answer.probabilities)) return bad("probabilities_missing");
  const probabilities: Record<string, number> = {};
  for (const [id, p] of Object.entries(answer.probabilities)) {
    if (!req.allowed_candidates.includes(id)) return bad("unknown_candidate");
    if (!finiteIn01(p)) return bad("non_finite_probability");
    probabilities[id] = p;
  }
  if (!finiteIn01(answer.confidence)) return bad("non_finite_confidence");
  const usage = raw.usage;
  if (!isPlainObject(usage)) return bad("usage_missing");
  const { input_tokens, output_tokens, cost } = usage;
  if (!Number.isInteger(input_tokens) || (input_tokens as number) < 0 || !Number.isInteger(output_tokens) || (output_tokens as number) < 0 ||
      typeof cost !== "number" || !Number.isFinite(cost) || cost < 0)
    return bad("usage_invalid");
  if (typeof raw.id !== "string" || !/^gen-dec-/.test(raw.id)) return bad("generation_id_invalid");
  if (typeof raw.provider !== "string") return bad("provider_invalid");
  return {
    ok: true,
    value: {
      request_id: req.request_id,
      state_hash: req.state_hash,
      status: "ok",
      model_requested: model,
      model_reported: raw.model,
      selection: { candidate_id: answer.choice, probabilities, provider_confidence: answer.confidence },
      usage: { input_tokens: input_tokens as number, output_tokens: output_tokens as number, cost },
      generation_id: raw.id,
    },
  };
}

// --------------------------------------------------------------------------------------------
// Default transport / sleep
// --------------------------------------------------------------------------------------------

export class TransportError extends Error {
  constructor(public readonly code: "timeout" | "aborted" | "network", message?: string) {
    super(message ?? code);
    this.name = "TransportError";
  }
}

export const fetchTransport: DecisionsTransport = async (req) => {
  let res: Response;
  try {
    res = await fetch(req.url, { method: "POST", headers: req.headers, body: req.body, signal: req.signal });
  } catch (e) {
    if (req.signal.aborted) throw new TransportError(req.signal.reason === "deadline" ? "timeout" : "aborted");
    throw new TransportError("network", e instanceof Error ? e.message : String(e));
  }
  const headers: Record<string, string> = {};
  res.headers.forEach((v, k) => { headers[k.toLowerCase()] = v; });
  const chunks: Uint8Array[] = [];
  let total = 0;
  let truncated = false;
  try {
    const reader = res.body?.getReader();
    if (reader) {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        total += value.byteLength;
        if (total > req.maxResponseBytes) { truncated = true; await reader.cancel(); break; }
        chunks.push(value);
      }
    }
  } catch (e) {
    if (req.signal.aborted) throw new TransportError(req.signal.reason === "deadline" ? "timeout" : "aborted");
    throw new TransportError("network", e instanceof Error ? e.message : String(e));
  }
  return { status: res.status, headers, bodyText: Buffer.concat(chunks).toString("utf8"), truncated };
};

export const abortableSleep: SleepFn = (ms, signal) =>
  new Promise<void>((resolve, reject) => {
    if (signal.aborted) return reject(new TransportError("aborted"));
    const t = setTimeout(() => { signal.removeEventListener("abort", onAbort); resolve(); }, ms);
    const onAbort = () => { clearTimeout(t); reject(new TransportError("aborted")); };
    signal.addEventListener("abort", onAbort, { once: true });
  });

// --------------------------------------------------------------------------------------------
// Service
// --------------------------------------------------------------------------------------------

class Limiter {
  private active = 0;
  private queue: Array<() => void> = [];
  constructor(private readonly maxActive: number, private readonly maxQueued: number) {}
  /** Returns false when the queue is full (caller must answer `busy` without waiting). */
  async run<T>(fn: () => Promise<T>): Promise<{ ran: true; value: T } | { ran: false }> {
    if (this.active >= this.maxActive) {
      if (this.queue.length >= this.maxQueued) return { ran: false };
      await new Promise<void>((resolve) => this.queue.push(resolve));
    } else {
      this.active++;
    }
    try {
      return { ran: true, value: await fn() };
    } finally {
      const next = this.queue.shift();
      if (next) next(); // hand the slot over without decrementing
      else this.active--;
    }
  }
}

export class DecisionsService {
  private readonly cfg: DecisionsConfig;
  private readonly transport: DecisionsTransport;
  private readonly sleep: SleepFn;
  private readonly resolveApiKey: ApiKeyResolver;
  private readonly log: (line: string) => void;
  private readonly now: () => number;
  private readonly limiter: Limiter;
  private readonly inflight = new Map<string, Promise<DecisionsResponse>>();

  constructor(config: Partial<DecisionsConfig> = {}, deps: DecisionsDeps = {}) {
    this.cfg = { ...defaultDecisionsConfig(), ...config };
    if (!["off", "shadow", "suggest"].includes(this.cfg.mode)) throw new Error(`PI_DECISIONS_MODE invalid: ${String(this.cfg.mode)}`);
    if (!isPinnedModel(this.cfg.model)) throw new Error(`decisions model must be a pinned id (no alias): ${this.cfg.model}`);
    const url = new URL(this.cfg.endpoint);
    const localTest = url.protocol === "http:" && ["127.0.0.1", "localhost", "::1", "[::1]"].includes(url.hostname);
    if (url.protocol !== "https:" && !localTest) throw new Error("decisions endpoint must be https");
    this.transport = deps.transport ?? fetchTransport;
    this.sleep = deps.sleep ?? abortableSleep;
    this.resolveApiKey = deps.resolveApiKey ?? (() => process.env.JEV_OPENROUTER_API_KEY);
    this.log = deps.log ?? (() => {});
    this.now = deps.now ?? Date.now;
    this.limiter = new Limiter(this.cfg.maxConcurrent, this.cfg.maxQueued);
  }

  async status() {
    const key = await this.resolveApiKey();
    return {
      mode: this.cfg.mode,
      model: this.cfg.model,
      endpoint_host: new URL(this.cfg.endpoint).host,
      has_api_key: Boolean(key),
      allow_experimental_suggestions: this.cfg.allowExperimentalSuggestions,
    };
  }

  /** Throws DecisionsRequestError for malformed requests; every provider-side outcome is a response. */
  async decide(rawRequest: unknown, callerSignal?: AbortSignal): Promise<DecisionsResponse> {
    const req = validateRequest(rawRequest);
    const base = (status: DecisionsStatus, error_code?: string): DecisionsResponse => ({
      request_id: req.request_id, state_hash: req.state_hash, status, model_requested: this.cfg.model,
      model_reported: null, selection: null, ...(error_code ? { error_code } : {}),
    });
    if (this.cfg.mode === "off") return base("unavailable", "mode_off");
    const apiKey = await this.resolveApiKey();
    if (!apiKey) return base("unavailable", "no_api_key");

    // Identical concurrent decisions share one provider call.
    const key = crypto.createHash("sha256")
      .update(JSON.stringify([req.state_hash, req.question_set_version, this.cfg.model, [...req.allowed_candidates].sort()]))
      .digest("hex");
    let shared = this.inflight.get(key);
    if (!shared) {
      shared = this.callWithLimits(req, apiKey, callerSignal).finally(() => this.inflight.delete(key));
      this.inflight.set(key, shared);
    }
    const result = await shared;
    return { ...result, request_id: req.request_id };
  }

  private async callWithLimits(req: DecisionsRequest, apiKey: string, callerSignal?: AbortSignal): Promise<DecisionsResponse> {
    const started = this.now();
    const out = await this.limiter.run(() => this.call(req, apiKey, callerSignal));
    const result: DecisionsResponse = out.ran ? out.value : {
      request_id: req.request_id, state_hash: req.state_hash, status: "unavailable",
      model_requested: this.cfg.model, model_reported: null, selection: null, error_code: "busy",
    };
    // Summary only: never state, request body or key.
    this.log(`decisions request_id=${req.request_id} state_hash=${req.state_hash} candidates=${req.allowed_candidates.length} ` +
             `status=${result.status} code=${result.error_code ?? "-"} ms=${this.now() - started} cost=${result.usage?.cost ?? "-"}`);
    return result;
  }

  private async call(req: DecisionsRequest, apiKey: string, callerSignal?: AbortSignal): Promise<DecisionsResponse> {
    const fail = (status: DecisionsStatus, code: string): DecisionsResponse => ({
      request_id: req.request_id, state_hash: req.state_hash, status, model_requested: this.cfg.model,
      model_reported: null, selection: null, error_code: code,
    });
    const deadline = new AbortController();
    const timer = setTimeout(() => deadline.abort("deadline"), this.cfg.deadlineMs);
    const signal = callerSignal ? AbortSignal.any([deadline.signal, callerSignal]) : deadline.signal;
    const startedAt = this.now();
    const body = buildProviderBody(req, this.cfg.model);
    const headers = { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` };
    const classifyAbort = () => (callerSignal?.aborted ? "aborted" : "timeout");
    try {
      for (let attempt = 0; ; attempt++) {
        let res: TransportResponse;
        try {
          res = await this.transport({ url: this.cfg.endpoint, headers, body, signal, maxResponseBytes: this.cfg.maxResponseBytes });
        } catch (e) {
          if (signal.aborted) return fail("unavailable", classifyAbort());
          if (e instanceof TransportError && e.code !== "network") return fail("unavailable", e.code);
          if (attempt < this.cfg.maxRetries && await this.waitBeforeRetry(this.cfg.retryBackoffMs, startedAt, signal)) continue;
          return fail("unavailable", "network_error");
        }
        if (res.truncated) return fail("invalid_response", "response_too_large");
        if (res.status === 200) {
          let parsed: unknown;
          try { parsed = JSON.parse(res.bodyText); } catch { return fail("invalid_response", "invalid_json"); }
          const v = validateProviderResponse(parsed, req, this.cfg.model);
          return v.ok ? v.value : fail("invalid_response", v.code);
        }
        if (res.status === 401 || res.status === 403) return fail("unavailable", "auth_failed");
        if (res.status === 400 || res.status === 422) return fail("invalid_response", "provider_rejected_request");
        const retryable = res.status === 429 || res.status >= 500;
        if (!retryable) return fail("unavailable", `http_${res.status}`);
        const code = res.status === 429 ? "rate_limited" : "provider_error";
        if (attempt >= this.cfg.maxRetries) return fail("unavailable", code);
        // Retry-After is honoured only while it fits into the remaining budget.
        const retryAfter = res.status === 429 ? Number(res.headers["retry-after"]) : NaN;
        const waitMs = Number.isFinite(retryAfter) && retryAfter >= 0 ? retryAfter * 1000 : this.cfg.retryBackoffMs;
        if (!(await this.waitBeforeRetry(waitMs, startedAt, signal))) return fail("unavailable", signal.aborted ? classifyAbort() : code);
      }
    } finally {
      clearTimeout(timer);
    }
  }

  /** Sleeps and returns true iff a further attempt still fits into the deadline; never waits past it. */
  private async waitBeforeRetry(waitMs: number, startedAt: number, signal: AbortSignal): Promise<boolean> {
    const remaining = this.cfg.deadlineMs - (this.now() - startedAt);
    if (waitMs >= remaining) return false;
    try {
      await this.sleep(waitMs, signal);
    } catch {
      return false;
    }
    return !signal.aborted;
  }
}

export function decisionsConfigFromEnv(env: NodeJS.ProcessEnv = process.env): Partial<DecisionsConfig> {
  const num = (name: string, fallback: number) => {
    const raw = env[name];
    if (!raw) return fallback;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };
  const d = defaultDecisionsConfig();
  return {
    mode: (env.PI_DECISIONS_MODE || "off").toLowerCase() as DecisionsMode,
    endpoint: env.PI_DECISIONS_ENDPOINT || d.endpoint,
    model: env.PI_DECISIONS_MODEL || d.model,
    deadlineMs: num("PI_DECISIONS_DEADLINE_MS", d.deadlineMs),
    maxResponseBytes: num("PI_DECISIONS_MAX_RESPONSE_BYTES", d.maxResponseBytes),
    maxConcurrent: num("PI_DECISIONS_MAX_CONCURRENT", d.maxConcurrent),
    allowExperimentalSuggestions: ["1", "true", "yes", "on"].includes((env.PI_DECISIONS_ALLOW_EXPERIMENTAL || "").toLowerCase()),
  };
}
