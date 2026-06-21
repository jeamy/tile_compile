// js/utils/log.js – Log-Formatting

export function formatLogLevel(level) {
  return String(level || "INFO").toUpperCase().slice(0, 5);
}

export function formatLogTime(timestamp) {
  if (!timestamp) return "";
  const d = new Date(timestamp);
  return d.toLocaleTimeString("de-DE", { hour12: false });
}

export function logLineToHtml(line) {
  const time = formatLogTime(line.timestamp || line.t);
  const level = formatLogLevel(line.level || line.l);
  const msg = line.message || line.msg || "";
  return `<span class="tc-log-time">${time}</span><span class="tc-log-level tc-log-level-${level}">${level}</span><span class="tc-log-msg">${escapeHtml(msg)}</span>`;
}

export function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = String(text || "");
  return div.innerHTML;
}
