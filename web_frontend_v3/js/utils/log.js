// js/utils/log.js – Log-Formatting

export function formatLogLevel(level) {
  return String(level || "INFO").toUpperCase().slice(0, 5);
}

