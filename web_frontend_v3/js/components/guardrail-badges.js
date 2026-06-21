// js/components/guardrail-badges.js – Guardrail-Status-Badges in Sub-Tab-Leiste

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";

const GUARDRAILS = [
  { id: "scan", label: "Scan", icon: "\u25cf" },
  { id: "config", label: "Config", icon: "\u25cf" },
  { id: "calibration", label: "Cal", icon: "\u25cf" },
  { id: "bge_pcc", label: "BGE/PCC", icon: "\u25cf" },
];

export function createGuardrailBadges(statuses = {}) {
  const wrapper = el("div", { class: "tc-guardrail-badges" });
  for (const g of GUARDRAILS) {
    const status = statuses[g.id] || "pending";
    const cls = status === "ok" ? "tc-badge-success" :
                status === "error" ? "tc-badge-error" :
                status === "check" ? "tc-badge-warning" : "tc-badge-info";
    wrapper.appendChild(el("span", {
      class: `tc-badge ${cls}`,
      "data-guardrail": g.id,
    }, `${g.icon} ${g.label}`));
  }
  return wrapper;
}

export function updateGuardrailBadges(statuses) {
  for (const g of GUARDRAILS) {
    const badge = document.querySelector(`[data-guardrail="${g.id}"]`);
    if (!badge) continue;
    const status = statuses[g.id] || "pending";
    badge.className = `tc-badge ${
      status === "ok" ? "tc-badge-success" :
      status === "error" ? "tc-badge-error" :
      status === "check" ? "tc-badge-warning" : "tc-badge-info"
    }`;
  }
}
