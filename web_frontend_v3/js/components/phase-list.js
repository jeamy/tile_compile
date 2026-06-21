// js/components/phase-list.js – Phasen-Fortschrittsanzeige

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";

const DEFAULT_PHASES = [
  "SCAN", "CALIBRATION", "REGISTRATION", "ASTROMETRY", "BGE",
  "PCC", "HYPERMETRIC_STRETCH", "DONE",
];

export function createPhaseList(phases = DEFAULT_PHASES, states = {}) {
  const wrapper = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.phases", "Phasen")),
    el("div", { class: "tc-phase-list", id: "phase-list" },
      ...phases.map(name => createPhaseItem(name, states[name] || "pending")),
    ),
  );
  return wrapper;
}

export function updatePhaseStates(states) {
  const list = document.getElementById("phase-list");
  if (!list) return;
  for (const item of list.querySelectorAll(".tc-phase-item")) {
    const name = item.dataset.phase;
    const state = states[name] || "pending";
    item.className = `tc-phase-item ${state}`;
  }
}

function createPhaseItem(name, state) {
  return el("div", {
    class: `tc-phase-item ${state}`,
    "data-phase": name,
  },
    el("span", { class: "tc-phase-dot" }),
    el("span", {}, name),
    el("span", { class: "tc-phase-progress" }, ""),
  );
}
