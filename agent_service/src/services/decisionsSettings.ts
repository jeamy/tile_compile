import fs from "node:fs";
import path from "node:path";

/**
 * Persistent Jev settings written by the Jev card (Tools -> AI & API): mode, experimental flag and an
 * optional API key. Kept in its own file, separate from PI's provider auth storage, so PI and Jev stay
 * independent. The file is created with mode 0600 and the key is write-only: it is never returned by
 * any endpoint, log line or status.
 */
export interface StoredDecisionsSettings {
  mode?: "off" | "shadow" | "suggest";
  allow_experimental_suggestions?: boolean;
  api_key?: string;
}

export function decisionsSettingsPath(env: NodeJS.ProcessEnv = process.env): string {
  const projectRoot = path.resolve(env.TILE_COMPILE_PROJECT_ROOT || path.resolve(process.cwd(), ".."));
  const dir = env.TILE_COMPILE_PI_STORAGE_DIR || path.join(projectRoot, "runs", ".pi_memory");
  return path.join(dir, "jev_decisions_settings.json");
}

export function loadDecisionsSettings(file: string): StoredDecisionsSettings {
  try {
    const raw = JSON.parse(fs.readFileSync(file, "utf8"));
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return {};
    const out: StoredDecisionsSettings = {};
    if (["off", "shadow", "suggest"].includes(raw.mode)) out.mode = raw.mode;
    if (typeof raw.allow_experimental_suggestions === "boolean") out.allow_experimental_suggestions = raw.allow_experimental_suggestions;
    if (typeof raw.api_key === "string" && raw.api_key && !/\s/.test(raw.api_key)) out.api_key = raw.api_key;
    return out;
  } catch {
    return {};
  }
}

export function saveDecisionsSettings(file: string, settings: StoredDecisionsSettings): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(settings, null, 2), { mode: 0o600 });
  fs.renameSync(tmp, file);
  fs.chmodSync(file, 0o600);
}
