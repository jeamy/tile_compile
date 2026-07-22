import fs from "node:fs";
import path from "node:path";

const projectRoot = path.resolve(
  process.env.TILE_COMPILE_PROJECT_ROOT || path.resolve(process.cwd(), ".."),
);
const trafficLogPath = path.resolve(
  process.env.AI_TRAFFIC_LOG_PATH ||
  process.env.TILE_COMPILE_PI_TRAFFIC_LOG_PATH ||
  (process.env.TILE_COMPILE_PI_STORAGE_DIR
    ? path.join(process.env.TILE_COMPILE_PI_STORAGE_DIR, "pi_agent_traffic.log")
    : path.join(projectRoot, "runs", ".pi_memory", "pi_agent_traffic.log")),
);

type PendingTrafficRepeat = {
  redactedMessage: string;
  count: number;
  lastTs: string;
};

let pendingRepeat: PendingTrafficRepeat | null = null;
let exitFlushRegistered = false;

function envBool(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  return ["1", "true", "yes", "on"].includes(raw.toLowerCase());
}

export function redactTrafficLogText(message: string): string {
  let redacted = message;

  redacted = redacted.replace(
    /("(?:api[_-]?key|token|authorization|x-api-key|secret)"\s*:\s*")([^"]*)(")/gi,
    "$1<redacted>$3",
  );
  redacted = redacted.replace(
    /\b((?:ANTHROPIC|ANT_LING|OPENAI|DEEPSEEK|GOOGLE|GEMINI|MISTRAL|GROQ|CEREBRAS|OPENROUTER|XAI|HF|FIREWORKS|TOGETHER|KIMI|KIRO|MINIMAX|NVIDIA|CLOUDFLARE|AZURE_OPENAI|AI_GATEWAY|ZAI|OPENCODE|XIAOMI|AWS)[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|CREDENTIALS)[A-Z0-9_]*)\s*=\s*([^\s,;]+)/gi,
    "$1=<redacted>",
  );
  redacted = redacted.replace(/\bBearer\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer <redacted>");

  if (projectRoot && projectRoot !== path.parse(projectRoot).root) {
    redacted = redacted.split(projectRoot).join("<PROJECT_ROOT>");
  }

  redacted = redacted.replace(
    /("(?:input_path|config_path|path|file|filename|run_dir)"\s*:\s*")((?:\/|[A-Za-z]:\\)[^"]*)(")/g,
    "$1<redacted-path>$3",
  );

  return redacted;
}

function appendTrafficLogLine(ts: string, message: string): void {
  try {
    fs.mkdirSync(path.dirname(trafficLogPath), { recursive: true });
    fs.appendFileSync(trafficLogPath, `[${ts}] ${message}\n`);
  } catch {
    // Ignore logging errors.
  }
}

function flushPendingTrafficRepeat(): void {
  if (!pendingRepeat) return;
  const pending = pendingRepeat;
  pendingRepeat = null;

  if (pending.count <= 1) return;
  if (pending.count > 2) {
    appendTrafficLogLine(
      pending.lastTs,
      `traffic_log_compacted repeated previous entry ${pending.count - 2} additional times`,
    );
  }
  appendTrafficLogLine(pending.lastTs, pending.redactedMessage);
}

function ensureExitFlushRegistered(): void {
  if (exitFlushRegistered) return;
  exitFlushRegistered = true;
  process.once("beforeExit", flushPendingTrafficRepeat);
  process.once("exit", flushPendingTrafficRepeat);
}

export function appendTrafficLog(message: string): void {
  if (!envBool("AI_TRAFFIC_LOG", true)) return;
  ensureExitFlushRegistered();
  const ts = new Date().toISOString();
  const redacted = redactTrafficLogText(message);

  if (pendingRepeat && pendingRepeat.redactedMessage === redacted) {
    pendingRepeat.count += 1;
    pendingRepeat.lastTs = ts;
    return;
  }

  flushPendingTrafficRepeat();
  appendTrafficLogLine(ts, redacted);
  pendingRepeat = { redactedMessage: redacted, count: 1, lastTs: ts };
}

export function readTrafficLog(limit = 500): { path: string; items: string[]; count: number; enabled: boolean } {
  const enabled = envBool("AI_TRAFFIC_LOG", true);
  flushPendingTrafficRepeat();
  if (!fs.existsSync(trafficLogPath)) {
    return { path: trafficLogPath, items: [], count: 0, enabled };
  }
  const text = fs.readFileSync(trafficLogPath, "utf8");
  const lines = text.split(/\r?\n/).filter(Boolean);
  const safeLimit = Math.max(1, Math.min(5000, Math.floor(limit || 500)));
  return {
    path: trafficLogPath,
    items: lines.slice(-safeLimit),
    count: lines.length,
    enabled,
  };
}
