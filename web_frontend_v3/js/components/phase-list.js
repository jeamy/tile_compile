// js/components/phase-list.js – Phasen-Fortschrittsanzeige

import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";

// Phase order must match tile_compile_cpp/include/tile_compile/core/types.hpp
// and the v0.2.0 AQMH methodology.
const CLASSIC_PHASES = [
  "SCAN_INPUT",
  "REGISTRATION",
  "PREWARP",
  "CHANNEL_SPLIT",
  "NORMALIZATION",
  "GLOBAL_METRICS",
  "TILE_GRID",
  "COMMON_OVERLAP",
  "LOCAL_METRICS",
  "TILE_RECONSTRUCTION",
  "STATE_CLUSTERING",
  "SYNTHETIC_FRAMES",
  "STACKING",
  "DEBAYER",
  "ASTROMETRY",
  "BGE",
  "PCC",
  "HYPERMETRIC_STRETCH",
];

// AQMH main path: GLOBAL_METRICS/TILE_GRID/LOCAL_METRICS/TILE_RECONSTRUCTION/
// STATE_CLUSTERING/SYNTHETIC_FRAMES are Classic-only and omitted.
// STACKING is still executed for AQMH (output scaling/writing).
const AQMH_PHASES = [
  "SCAN_INPUT",
  "REGISTRATION",
  "PREWARP",
  "CHANNEL_SPLIT",
  "NORMALIZATION",
  "COMMON_OVERLAP",
  "AQMH_MAPS",
  "AQMH_GLOBAL_QUALITY",
  "AQMH_RECONSTRUCTION",
  "AQMH_DIAGNOSTICS",
  "STACKING",
  "DEBAYER",
  "ASTROMETRY",
  "BGE",
  "PCC",
  "HYPERMETRIC_STRETCH",
];

const DEFAULT_PHASES = AQMH_PHASES;

const RESUMABLE_PHASES = new Set([
  "ASTROMETRY", "BGE", "PCC", "HYPERMETRIC_STRETCH",
  "STACKING", "DEBAYER", "TILE_RECONSTRUCTION",
  "LOCAL_METRICS", "COMMON_OVERLAP", "PREWARP",
  "AQMH_MAPS", "AQMH_GLOBAL_QUALITY", "AQMH_RECONSTRUCTION", "AQMH_DIAGNOSTICS",
]);

let selectedPhase = null;
let phaseClickHandler = null;

export function setPhaseClickHandler(handler) {
  phaseClickHandler = handler;
}

export function getBgeLabel(configDraft) {
  const bgeMethod = configDraft?.bge?.method || "none";
  if (bgeMethod === "none") return "BGE (Skipped)";
  if (bgeMethod === "classic") return "BGE (Classic)";
  if (bgeMethod === "autobge") return "BGE (AutoBGE)";
  return "BGE";
}

export function getPhasesForConfig(configDraft) {
  if (!configDraft || typeof configDraft !== "object") return DEFAULT_PHASES.map(p => ({ phase: p, label: p }));
  const method = configDraft.method;
  const aqmhEnabled = configDraft.aqmh && configDraft.aqmh.enabled;
  const basePhases = (method === "classic_tile_compile" || aqmhEnabled === false) ? CLASSIC_PHASES : AQMH_PHASES;
  return basePhases.map(p => p === "BGE"
    ? { phase: "BGE", label: getBgeLabel(configDraft), bgeMethod: configDraft?.bge?.method || "none" }
    : { phase: p, label: p });
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
      ...phases.map(p => {
        const phaseId = typeof p === "string" ? p : (p.phase || "");
        const label = typeof p === "string" ? p : (p.label || p.phase || "");
        return createPhaseItem(phaseId, states[phaseId] || "pending", 0, label);
      }),
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
    const label = typeof phase === "string" ? phase : (phase.label || phase.phase || "");
    const state = typeof phase === "object" ? (phase.status || states[name] || "pending") : (states[name] || "pending");
    let pct = typeof phase === "object" ? (phase.pct || 0) : 0;
    if (pct <= 1.0) pct *= 100;
    if (state === "ok" || state === "done" || state === "skipped") pct = 100;
    list.appendChild(createPhaseItem(name, state, pct, label));
  }
  if (selectedPhase) {
    const item = list.querySelector(`.tc-phase-item[data-phase="${selectedPhase}"]`);
    if (item) item.classList.add("tc-phase-selected");
  }
}

export function updatePhaseState(phaseName, status, pct, label) {
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
      if (label) {
        const labelEl = item.querySelector(".tc-phase-label");
        if (labelEl) labelEl.textContent = label;
      }
      return;
    }
  }
}

export function resetPhasesForResume(fromPhase) {
  const list = document.getElementById("phase-list");
  if (!list) return [];
  const newPhases = [];
  let found = false;
  for (const item of list.querySelectorAll(".tc-phase-item")) {
    const name = item.dataset.phase;
    const wasSelected = item.classList.contains("tc-phase-selected");
    let state, pct;
    if (name === fromPhase) {
      found = true;
      state = "running";
      pct = 0;
    } else if (!found) {
      state = "ok";
      pct = 100;
    } else {
      state = "pending";
      pct = 0;
    }
    item.className = `tc-phase-item ${state}${wasSelected ? " tc-phase-selected" : ""}`;
    const pctEl = item.querySelector(".tc-phase-progress");
    if (pctEl) pctEl.textContent = pct > 0 ? `${pct}%` : "";
    const barFill = item.querySelector(".tc-phase-bar-fill");
    if (barFill) barFill.style.width = `${pct}%`;
    const labelEl = item.querySelector(".tc-phase-label");
    const label = labelEl ? labelEl.textContent : name;
    newPhases.push({ phase: name, status: state, pct, label });
  }
  return newPhases;
}

function createPhaseItem(name, state, pct = 0, label = name) {
  const pctLabel = pct > 0 ? `${Math.round(pct)}%` : "";
  const barWidth = Math.min(100, Math.round(pct || 0));
  const resumable = RESUMABLE_PHASES.has(name) && isPhaseResumable(state);
  const item = el("div", {
    class: `tc-phase-item ${state}${resumable ? " tc-phase-clickable" : ""}`,
    "data-phase": name,
    ...(resumable ? { onclick: () => onPhaseClick(name) } : {}),
  },
    el("span", { class: "tc-phase-dot" }),
    el("span", { class: "tc-phase-label" }, label),
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
