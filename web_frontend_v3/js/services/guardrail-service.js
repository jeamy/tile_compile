// js/services/guardrail-service.js – Fetch guardrail status from backend and update badges

import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { updateGuardrailBadges } from "../components/guardrail-badges.js";

export async function refreshGuardrails() {
  try {
    const result = await api.get(API_ENDPOINTS.guardrails.root);
    if (!result?.checks) return;
    const statuses = {};
    for (const check of result.checks) {
      statuses[check.id] = check.status;
    }
    updateGuardrailBadges(statuses);
  } catch {
    // silently fail – badges stay at last known state
  }
}
