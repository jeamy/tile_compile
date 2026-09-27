// js/components/feature-switch.js – The on/off switch shown in a card title bar (Modell & API-Key,
// Jev (Decisions API)). Reuses the existing checkbox styling; no page-specific button style.

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";

/**
 * @param {object} opts
 * @param {string} opts.id            element id of the checkbox
 * @param {boolean} opts.checked      current state
 * @param {string} opts.title         tooltip
 * @param {(checked: boolean) => void} opts.onChange
 */
export function createFeatureSwitch({ id, checked, title, onChange }) {
  const input = el("input", {
    type: "checkbox",
    id,
    checked,
    onchange: (e) => onChange(e.target.checked),
  });
  const label = el("span", { class: "tc-text-sm" }, t("ui.feature_switch.active", "Aktiv"));
  return el("label", { class: "tc-checkbox", for: id, title }, input, label);
}
