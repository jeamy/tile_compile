// js/utils/log.js – Log-Formatting

const LOG_LEVEL_ALIASES = { WARNING: "WARN", ERR: "ERROR", FATAL: "ERROR", CRITICAL: "ERROR", DBG: "DEBUG" };
export const LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG", "TRACE"];

/** One of LOG_LEVELS: aliases such as WARNING map to WARN, anything unknown to INFO. */
export function normalizeLogLevel(level) {
  const upper = String(level || "INFO").trim().toUpperCase();
  const mapped = LOG_LEVEL_ALIASES[upper] || upper;
  return LOG_LEVELS.includes(mapped) ? mapped : "INFO";
}
