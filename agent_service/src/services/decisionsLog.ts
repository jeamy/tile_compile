import fs from "node:fs";
import path from "node:path";
import { redactTrafficLogText } from "./trafficLog.js";

// Dedicated log of every Jev call: the wire request actually sent, the raw provider response and the normalized
// result, one JSON object per line. Written by whoever runs a DecisionsService (the sidecar and the evaluation
// script), so both end up in the same file and can be followed with `tail -f`.
//
// The wire body is an allowlist projection without paths, secrets or camera names; the Authorization header is
// never part of an entry, and every line additionally passes the traffic-log redaction (keys, bearer tokens,
// project paths).

const projectRoot = path.resolve(process.env.TILE_COMPILE_PROJECT_ROOT || path.resolve(process.cwd(), ".."));

export function jevLogPath(env: NodeJS.ProcessEnv = process.env): string {
  if (env.JEV_DECISIONS_LOG_PATH) return path.resolve(env.JEV_DECISIONS_LOG_PATH);
  if (env.TILE_COMPILE_PI_STORAGE_DIR) return path.join(env.TILE_COMPILE_PI_STORAGE_DIR, "jev_decisions.log");
  return path.join(projectRoot, "runs", ".pi_memory", "jev_decisions.log");
}

export type JevLogEntry = Record<string, unknown>;

/** Returns a trace function for DecisionsService deps. `source` says who made the call ("sidecar", "eval", ...). */
export function createJevLogger(source: string, env: NodeJS.ProcessEnv = process.env): (entry: JevLogEntry) => void {
  const off = ["0", "false", "no", "off"].includes((env.JEV_LOG || "").toLowerCase());
  const file = jevLogPath(env);
  return (entry) => {
    if (off) return;
    try {
      const line = redactTrafficLogText(JSON.stringify({ ts: new Date().toISOString(), source, pid: process.pid, ...entry }));
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.appendFileSync(file, `${line}\n`);
    } catch {
      // Logging must never break a decision.
    }
  };
}
