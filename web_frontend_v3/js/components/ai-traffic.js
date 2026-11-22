// js/components/ai-traffic.js – the persistent PI/AI traffic log of the
// sidecar, shared by the AI page and the run monitor's completion analysis.

import { t } from "../i18n/i18n.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";

/**
 * Loads the last `limit` traffic lines and writes "active · N lines" (plus
 * the log path when `showPath`) into `statusEl`. Returns the lines; on an
 * error the status shows its message and the error is rethrown.
 */
export async function loadAiTrafficLines(statusEl, { limit = 500, showPath = false } = {}) {
  if (statusEl) statusEl.textContent = t("ui.state.loading", "Lädt...");
  try {
    const payload = await api.get(API_ENDPOINTS.ai.traffic(limit));
    const items = Array.isArray(payload?.items) ? payload.items : [];
    if (statusEl) {
      const enabled = payload?.enabled === false
        ? t("ui.state.disabled", "deaktiviert")
        : t("ui.state.enabled", "aktiv");
      const parts = [enabled, t("ui.pi.traffic_count", "{count} Zeilen", { count: payload?.count || items.length })];
      if (showPath && payload?.path) parts.push(payload.path);
      statusEl.textContent = parts.join(" · ");
    }
    return items;
  } catch (error) {
    if (statusEl) statusEl.textContent = error.message;
    throw error;
  }
}
