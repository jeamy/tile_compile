import fs from "node:fs";
import path from "node:path";

const projectRoot = path.resolve(
  process.env.TILE_COMPILE_PROJECT_ROOT || path.resolve(process.cwd(), ".."),
);
const trafficLogPath = path.resolve(
  process.env.AI_TRAFFIC_LOG_PATH ||
  process.env.TILE_COMPILE_PI_TRAFFIC_LOG_PATH ||
  path.join(projectRoot, "runs", "pi_agent_traffic.log"),
);

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
    /\b((?:ANTHROPIC|ANT_LING|OPENAI|DEEPSEEK|GOOGLE|GEMINI|MISTRAL|GROQ|CEREBRAS|OPENROUTER|XAI|HF|FIREWORKS|TOGETHER|KIMI|MINIMAX|NVIDIA|CLOUDFLARE|AZURE_OPENAI|AI_GATEWAY|ZAI|OPENCODE|XIAOMI|AWS)[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|CREDENTIALS)[A-Z0-9_]*)\s*=\s*([^\s,;]+)/gi,
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

export function appendTrafficLog(message: string): void {
  if (!envBool("AI_TRAFFIC_LOG", true)) return;
  try {
    fs.mkdirSync(path.dirname(trafficLogPath), { recursive: true });
    fs.appendFileSync(trafficLogPath, `[${new Date().toISOString()}] ${redactTrafficLogText(message)}\n`);
  } catch {
    // Ignore logging errors.
  }
}

export function readTrafficLog(limit = 500): { path: string; items: string[]; count: number; enabled: boolean } {
  const enabled = envBool("AI_TRAFFIC_LOG", true);
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
