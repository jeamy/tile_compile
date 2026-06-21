// js/components/situation-assistant.js – Situation-Assistant für Parameter-Empfehlungen

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";

const SCENARIOS = [
  { id: "alt_az", label: "Alt/Az", hint: "Starke Feldrotation" },
  { id: "rotation", label: "Rotation", hint: "Frame-Rotation vorhanden" },
  { id: "bright_stars", label: "Bright Stars", hint: "Helle Sterne im Feld" },
  { id: "few_frames", label: "Few Frames", hint: "< 50 Frames" },
  { id: "gradient", label: "Gradient", hint: "Starker Gradient" },
  { id: "dithered", label: "Dithered", hint: "Frames wurden gedithert" },
  { id: "low_snr", label: "Low SNR", hint: "Niedriges Signal-Rausch-Verh\u00e4ltnis" },
  { id: "wide_field", label: "Wide Field", hint: "Weitwinkel-Feld" },
];

export function createSituationAssistant({ selected = [], onApply } = {}) {
  const state = new Set(selected);

  const wrapper = el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
    el("div", { class: "tc-card-title" }, t("ui.title.situation", "Situation")),
    el("div", { class: "tc-flex-col tc-gap-1", id: "scenario-list" },
      ...SCENARIOS.map(s => {
        const cb = el("input", { type: "checkbox", checked: state.has(s.id) });
        cb.onchange = () => {
          if (cb.checked) state.add(s.id);
          else state.delete(s.id);
        };
        return el("label", { class: "tc-checkbox", title: s.hint }, cb, s.label);
      }),
    ),
    el("button", {
      class: "tc-btn tc-btn-sm tc-mt-2",
      onclick: () => onApply?.([...state]),
    }, t("ui.button.apply", "Apply")),
  );

  return wrapper;
}

export function getSelectedScenarios() {
  const result = [];
  for (const cb of document.querySelectorAll("#scenario-list input[type=checkbox]")) {
    if (cb.checked) {
      const label = cb.parentElement?.textContent?.trim() || "";
      const scenario = SCENARIOS.find(s => s.label === label);
      if (scenario) result.push(scenario.id);
    }
  }
  return result;
}
