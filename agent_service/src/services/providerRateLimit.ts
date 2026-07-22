import type { AgentConfig } from "../types.js";
import { appendTrafficLog } from "./trafficLog.js";

type RateLimitState = {
  tokenLimit?: number;
  remainingTokens?: number;
  resetAt?: number;
  cooldownUntil?: number;
};

const states = new Map<string, RateLimitState>();
const DEFAULT_OPENAI_TPM_LIMIT = 200_000;
const SAFETY_FACTOR = 0.85;
const MIN_429_COOLDOWN_MS = 1_500;

function keyFor(model: any): string {
  return `${String(model?.provider || "unknown")}/${String(model?.id || "unknown")}`;
}

function isOpenAiLike(model: any): boolean {
  const provider = String(model?.provider || "").toLowerCase();
  const id = String(model?.id || "").toLowerCase();
  return provider.includes("openai") || id.includes("gpt-");
}

function estimateTokens(text: string, config: AgentConfig): number {
  const outputTokens = Number.isFinite(config.maxTokens) && config.maxTokens > 0 ? config.maxTokens : 0;
  return Math.ceil(String(text || "").length / 4) + outputTokens;
}

function parseResetMs(value: string | undefined): number | null {
  const text = String(value || "").trim().toLowerCase();
  if (!text) return null;
  const numeric = Number(text);
  if (Number.isFinite(numeric)) return Math.max(0, numeric * 1000);
  const match = text.match(/([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)\s*(ms|s|sec|secs|second|seconds|m|min|minute|minutes)?/i);
  if (!match) return null;
  const amount = Number(match[1]);
  if (!Number.isFinite(amount)) return null;
  const unit = match[2] || "s";
  if (unit === "ms") return amount;
  if (unit.startsWith("m") && unit !== "ms") return amount * 60_000;
  return amount * 1000;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function waitForProviderSlot(model: any, prompt: string, config: AgentConfig, label: string): Promise<void> {
  const key = keyFor(model);
  const state = states.get(key) || {};
  const now = Date.now();
  const estimated = estimateTokens(prompt, config);
  const learnedLimit = state.tokenLimit || (isOpenAiLike(model) ? DEFAULT_OPENAI_TPM_LIMIT : undefined);
  if (learnedLimit && estimated > learnedLimit * SAFETY_FACTOR) {
    throw new Error(
      `PI rate-limit guard: request estimated at ${estimated} tokens exceeds safe budget ` +
      `${Math.floor(learnedLimit * SAFETY_FACTOR)} for ${key}. Prompt must be reduced before sending.`
    );
  }
  const waitMs = Math.max(0, (state.cooldownUntil || state.resetAt || 0) - now);
  if (waitMs > 0) {
    appendTrafficLog(`${label} rate_limit_wait model=${key} wait_ms=${Math.ceil(waitMs)} estimated_tokens=${estimated}`);
    await sleep(waitMs);
  }
}

export function recordProviderResponseHeaders(model: any, headers: Record<string, string>, label: string): void {
  const key = keyFor(model);
  const state = states.get(key) || {};
  const limit = Number(headers["x-ratelimit-limit-tokens"]);
  const remaining = Number(headers["x-ratelimit-remaining-tokens"]);
  if (Number.isFinite(limit) && limit > 0) state.tokenLimit = limit;
  if (Number.isFinite(remaining) && remaining >= 0) state.remainingTokens = remaining;
  const resetMs = parseResetMs(headers["x-ratelimit-reset-tokens"] || headers["retry-after"]);
  if (resetMs !== null && resetMs > 0) state.resetAt = Date.now() + resetMs;
  if (Number.isFinite(remaining) && remaining <= 0 && state.resetAt) {
    state.cooldownUntil = state.resetAt;
  }
  states.set(key, state);
  appendTrafficLog(`${label} rate_limit_headers model=${key} status_limit=${state.tokenLimit ?? ""} remaining=${state.remainingTokens ?? ""} reset_at=${state.resetAt ?? ""}`);
}

export function rememberProviderRateLimitError(model: any, message: string, label: string): { retryable: boolean; waitMs: number } | null {
  const text = String(message || "");
  if (!/rate limit|tokens per min|too many requests|try again/i.test(text)) return null;
  const key = keyFor(model);
  const state = states.get(key) || {};
  const limitMatch = text.match(/Limit\s+([0-9]+(?:\.[0-9]+)?)/i);
  const requestedMatch = text.match(/Requested\s+([0-9]+(?:\.[0-9]+)?)/i);
  if (limitMatch) {
    const limit = Number(limitMatch[1]);
    if (Number.isFinite(limit) && limit > 0) state.tokenLimit = limit;
  }
  const waitMatch = text.match(/try again after\s+([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)\s*seconds/i)
    || text.match(/retry after\s+([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)/i);
  const providerWaitMs = waitMatch ? parseResetMs(`${waitMatch[1]} seconds`) : null;
  const waitMs = Math.max(MIN_429_COOLDOWN_MS, Math.ceil(providerWaitMs || 0));
  state.cooldownUntil = Date.now() + waitMs;
  states.set(key, state);
  const requested = requestedMatch ? Number(requestedMatch[1]) : 0;
  const retryable = !(state.tokenLimit && requested && requested > state.tokenLimit);
  appendTrafficLog(`${label} rate_limit_error model=${key} retryable=${retryable} wait_ms=${waitMs} limit=${state.tokenLimit ?? ""} requested=${requested || ""}`);
  return { retryable, waitMs };
}
