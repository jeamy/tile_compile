// js/components/phase-list.js – Phasen-Fortschrittsanzeige

import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";

const DEFAULT_PHASES = [
  "SCAN", "CALIBRATION", "REGISTRATION", "ASTROMETRY", "BGE",
  "PCC", "HYPERMETRIC_STRETCH", "DONE",
];

const RESUMABLE_PHASES = new Set([
  "ASTROMETRY", "BGE", "PCC", "HYPERMETRIC_STRETCH",
  "STACKING", "DEBAYER", "TILE_RECONSTRUCTION",
  "LOCAL_METRICS", "COMMON_OVERLAP", "PREWARP",
]);

let selectedPhase = null;
let phaseClickHandler = null;

export function setPhaseClickHandler(handler) {
  phaseClickHandler = handler;
}

export function getSelectedPhase() {
  return selectedPhase;
}

export function clearSelectedPhase() {
  selectedPhase = null;
  const list = document.getElementById("phase-list");
  if (list) {
    list.querySelectorAll(".tc-phase-item").forEach(item => {
      item.classList.remove("tc-phase-selected");
    });
  }
}

function isPhaseResumable(state) {
  return state === "ok" || state === "done" || state === "skipped" || state === "error" || state === "aborted";
}

export function createPhaseList(phases = DEFAULT_PHASES, states = {}) {
  const wrapper = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.phases", "Phasen")),
    el("div", { class: "tc-phase-list", id: "phase-list" },
      ...phases.map(name => createPhaseItem(name, states[name] || "pending")),
    ),
  );
  return wrapper;
}

export function setPhaseList(phases, states = {}) {
  const list = document.getElementById("phase-list");
  if (!list) return;
  clear(list);
  for (const phase of phases) {
    const name = typeof phase === "string" ? phase : (phase.phase || phase.name || "");
    const state = typeof phase === "object" ? (phase.status || states[name] || "pending") : (states[name] || "pending");
    let pct = typeof phase === "object" ? (phase.pct || 0) : 0;
    if (pct <= 1.0) pct *= 100;
    if (state === "ok" || state === "done" || state === "skipped") pct = 100;
    list.appendChild(createPhaseItem(name, state, pct));
  }
  if (selectedPhase) {
    const item = list.querySelector(`.tc-phase-item[data-phase="${selectedPhase}"]`);
    if (item) item.classList.add("tc-phase-selected");
  }
}

export function updatePhaseState(phaseName, status, pct) {
  const list = document.getElementById("phase-list");
  if (!list) return;
  for (const item of list.querySelectorAll(".tc-phase-item")) {
    if (item.dataset.phase === phaseName) {
      const wasSelected = item.classList.contains("tc-phase-selected");
      item.className = `tc-phase-item ${status}${wasSelected ? " tc-phase-selected" : ""}`;
      let displayPct = pct || 0;
      if (status === "ok" || status === "done") displayPct = 100;
      if (status === "skipped") displayPct = 100;
      const pctEl = item.querySelector(".tc-phase-progress");
      if (pctEl) pctEl.textContent = displayPct > 0 ? `${Math.round(displayPct)}%` : "";
      const barFill = item.querySelector(".tc-phase-bar-fill");
      if (barFill) barFill.style.width = `${Math.min(100, Math.round(displayPct))}%`;
      return;
    }
  }
}

export function updatePhaseStates(states) {
  const list = document.getElementById("phase-list");
  if (!list) return;
  for (const item of list.querySelectorAll(".tc-phase-item")) {
    const name = item.dataset.phase;
    const state = states[name] || "pending";
    const wasSelected = item.classList.contains("tc-phase-selected");
    item.className = `tc-phase-item ${state}${wasSelected ? " tc-phase-selected" : ""}`;
  }
}

function createPhaseItem(name, state, pct = 0) {
  const pctLabel = pct > 0 ? `${Math.round(pct)}%` : "";
  const barWidth = Math.min(100, Math.round(pct || 0));
  const resumable = RESUMABLE_PHASES.has(name) && isPhaseResumable(state);
  const item = el("div", {
    class: `tc-phase-item ${state}${resumable ? " tc-phase-clickable" : ""}`,
    "data-phase": name,
    ...(resumable ? { onclick: () => onPhaseClick(name) } : {}),
  },
    el("span", { class: "tc-phase-dot" }),
    el("span", {}, name),
    el("div", { class: "tc-phase-bar" },
      el("div", { class: "tc-phase-bar-fill", style: `width:${barWidth}%` }),
    ),
    el("span", { class: "tc-phase-progress" }, pctLabel),
  );
  if (resumable) {
    item.title = t("ui.message.resume_from_phase", "Klicken um Resume ab {phase} zu starten", { phase: name });
  }
  return item;
}

function onPhaseClick(name) {
  const list = document.getElementById("phase-list");
  if (!list) return;
  if (selectedPhase === name) {
    selectedPhase = null;
    list.querySelectorAll(".tc-phase-item").forEach(item => {
      item.classList.remove("tc-phase-selected");
    });
  } else {
    selectedPhase = name;
    list.querySelectorAll(".tc-phase-item").forEach(item => {
      item.classList.toggle("tc-phase-selected", item.dataset.phase === name);
    });
  }
  if (phaseClickHandler) phaseClickHandler(selectedPhase);
}
