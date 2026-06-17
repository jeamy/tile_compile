import fs from "node:fs";
import path from "node:path";
import dotenv from "dotenv";
import type { RuntimeConfig } from "./types.js";

function loadEnvFile(filePath: string) {
  if (fs.existsSync(filePath)) {
    dotenv.config({ path: filePath, override: false });
  }
}

export function loadEnv() {
  const cwd = process.cwd();
  const agentDir = path.resolve(cwd);
  const projectRoot = process.env.TILE_COMPILE_PROJECT_ROOT
    ? path.resolve(process.env.TILE_COMPILE_PROJECT_ROOT)
    : path.resolve(agentDir, "..");

  loadEnvFile(path.join(projectRoot, ".env"));
  loadEnvFile(path.join(agentDir, ".env"));
}

function envBool(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  return ["1", "true", "yes", "on"].includes(raw.toLowerCase());
}

function envNumber(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function runtimeConfig(): RuntimeConfig {
  loadEnv();
  const projectRoot = process.env.TILE_COMPILE_PROJECT_ROOT
    ? path.resolve(process.env.TILE_COMPILE_PROJECT_ROOT)
    : path.resolve(process.cwd(), "..");
  return {
    host: process.env.AI_AGENT_HOST || "127.0.0.1",
    port: envNumber("AI_AGENT_PORT", 3001),
    projectRoot,
    agent: {
      enabled: envBool("AI_SCAN_ENABLED", false),
      model: process.env.AI_SCAN_MODEL || process.env.AI_RESEARCH_MODEL || "",
      maxTokens: envNumber("AI_SCAN_MAX_TOKENS", 8000),
      temperature: envNumber("AI_SCAN_TEMPERATURE", 0.2),
      timeoutMs: envNumber("AI_SCAN_TIMEOUT_MS", 120000),
    },
  };
}

export function redactedEnvSources() {
  const providers: Record<string, string[]> = {
    anthropic: ["ANTHROPIC_API_KEY"],
    "ant-ling": ["ANT_LING_API_KEY"],
    "azure-openai-responses": ["AZURE_OPENAI_API_KEY"],
    openai: ["OPENAI_API_KEY"],
    deepseek: ["DEEPSEEK_API_KEY"],
    nvidia: ["NVIDIA_API_KEY"],
    google: ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    mistral: ["MISTRAL_API_KEY"],
    groq: ["GROQ_API_KEY"],
    cerebras: ["CEREBRAS_API_KEY"],
    "cloudflare-ai-gateway": ["CLOUDFLARE_API_KEY"],
    "cloudflare-workers-ai": ["CLOUDFLARE_API_KEY"],
    xai: ["XAI_API_KEY"],
    openrouter: ["OPENROUTER_API_KEY"],
    "vercel-ai-gateway": ["AI_GATEWAY_API_KEY"],
    zai: ["ZAI_API_KEY"],
    "zai-coding-cn": ["ZAI_CODING_CN_API_KEY"],
    opencode: ["OPENCODE_API_KEY"],
    "opencode-go": ["OPENCODE_API_KEY"],
    huggingface: ["HF_TOKEN"],
    fireworks: ["FIREWORKS_API_KEY"],
    together: ["TOGETHER_API_KEY"],
    "kimi-coding": ["KIMI_API_KEY"],
    minimax: ["MINIMAX_API_KEY"],
    "minimax-cn": ["MINIMAX_CN_API_KEY"],
    xiaomi: ["XIAOMI_API_KEY"],
    "xiaomi-token-plan-cn": ["XIAOMI_TOKEN_PLAN_CN_API_KEY"],
    "xiaomi-token-plan-ams": ["XIAOMI_TOKEN_PLAN_AMS_API_KEY"],
    "xiaomi-token-plan-sgp": ["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
    "amazon-bedrock": [
      "AWS_PROFILE",
      "AWS_ACCESS_KEY_ID",
      "AWS_BEARER_TOKEN_BEDROCK",
      "AWS_WEB_IDENTITY_TOKEN_FILE",
      "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
      "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    ],
  };
  return Object.fromEntries(
    Object.entries(providers).map(([provider, names]) => [
      provider,
      names.some((name) => Boolean(process.env[name])) ? "env" : "",
    ]),
  );
}
