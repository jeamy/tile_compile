import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toastError } from "./toast.js";
import { getRunState } from "../state/run-state.js";
import { parseYaml, stringifyYaml } from "../utils/yaml-parse.js";
import { setConfigValue } from "../pages/parameter.js";

// Post-run advice on request (never automatic): reads the finished run's own artifacts, changes nothing and starts nothing
// by itself. A suggestion is selectable; applying merges the selected suggestions' changes into a copy of the RUN's own
// config and hands that, plus how far the run must be redone, to `onApply` (wired by run-monitor.js to the same "prepare a
// new run" / "prepare + check a resume" actions the KI-Ergebnisanalyse card already uses). Feasibility is still the
// existing resume dry run, and starting anything stays a separate, explicit button click.

const OUTCOMES = ["no_change", "diagnose", "suggest_downstream", "suggest_reconstruction"];

function fmt(value) {
  if (value === null || value === undefined) return "—";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

/** Index of `phase` in `order` (resume_phase_order, narrowest-first); -1 if absent. */
function phaseRank(order, phase) {
  return (order || []).indexOf(phase);
}

/**
 * Combines the resume requirement of several selected suggestions into one: any full-run suggestion makes the whole
 * selection a full run; otherwise the phase that must be used is the one covering every selected suggestion's section,
 * which is the one ranked LATEST in `resume_phase_order` (that array is narrowest/latest-first; a higher rank means an
 * earlier, broader phase). Returns { mode: "full_run" } or { mode: "resume", phase }.
 */
function combineResumePlan(selected, resumePhaseOrder) {
  if (selected.some((s) => s.resume_mode === "full_run" || !s.min_resume_phase)) return { mode: "full_run" };
  let phase = selected[0].min_resume_phase;
  for (const s of selected.slice(1)) {
    if (phaseRank(resumePhaseOrder, s.min_resume_phase) > phaseRank(resumePhaseOrder, phase)) phase = s.min_resume_phase;
  }
  return { mode: "resume", phase };
}

export function createPostRunAdvicePanel(id = "post-run-advice", { onApply } = {}) {
  const status = el("div", { class: "tc-text-sm tc-text-muted", id: `${id}-status` }, t("ui.post_run.idle", "Nur auf Anforderung; ändert nichts und startet keinen Lauf."));
  const content = el("div", { class: "tc-flex-col tc-gap-2 tc-mt-2", id: `${id}-content` });
  const button = el("button", { class: "tc-btn tc-btn-sm", id: `${id}-button`, onclick: request }, t("ui.post_run.request", "Nachbetrachtung anfordern"));

  let lastRunId = "";
  let lastResumeOrder = [];
  const selected = new Set();

  function isSelectable(s) {
    return Array.isArray(s.updates) && s.updates.length > 0;
  }

  async function applySelection(ids) {
    if (!ids.length || !onApply) return;
    const suggestions = (content._suggestions || []).filter((s) => ids.includes(s.candidate_id));
    if (!suggestions.length) return;
    const applyBtn = content.querySelector(`#${id}-apply-selected`);
    const allBtn = content.querySelector(`#${id}-apply-all`);
    if (applyBtn) applyBtn.disabled = true;
    if (allBtn) allBtn.disabled = true;
    try {
      const resp = await api.get(API_ENDPOINTS.runs.config(lastRunId));
      const yaml = resp?.config_yaml || resp?.config || "";
      if (!yaml) throw new Error(t("ui.error.no_config", "Config YAML ist leer"));
      const parsed = parseYaml(yaml);
      for (const s of suggestions) for (const u of s.updates || []) setConfigValue(parsed, u.path, u.value);
      const patchedYaml = stringifyYaml(parsed);
      const plan = combineResumePlan(suggestions, lastResumeOrder);
      await onApply({ patchedYaml, resumeMode: plan.mode, minResumePhase: plan.phase || null, candidateIds: ids });
    } catch (error) {
      toastError(t("ui.post_run.apply_failed", "Übernehmen fehlgeschlagen"), error?.message || String(error));
    } finally {
      if (applyBtn) applyBtn.disabled = false;
      if (allBtn) allBtn.disabled = false;
    }
  }

  function renderAdvice(payload) {
    clear(content);
    const advice = payload?.advice;
    content._suggestions = advice?.suggestions || [];
    lastResumeOrder = advice?.resume_phase_order || [];
    selected.clear();
    for (const s of content._suggestions) if (isSelectable(s)) selected.add(s.candidate_id);
    if (!advice) {
      content.append(el("div", { class: "tc-text-sm tc-text-muted" }, t("ui.post_run.none", "Keine Auswertung.")));
      return;
    }
    const outcome = OUTCOMES.includes(advice.outcome) ? advice.outcome : "diagnose";
    content.append(el("div", { class: "tc-text-sm", "data-outcome": outcome }, t(`ui.post_run.outcome.${outcome}`, outcome)));
    for (const f of advice.findings || []) {
      const cls = f.severity === "warning" ? "tc-text-sm" : "tc-text-sm tc-text-muted";
      content.append(el("div", { class: cls }, `${t(`ui.post_run.finding.${f.code}`, f.code)}${f.detail ? `: ${f.detail}` : ""}`));
    }
    for (const s of content._suggestions) {
      const changes = (s.updates || []).map((u) => `${u.path}: ${fmt(u.old_value)} → ${fmt(u.value)}`).join("; ");
      const how = s.resume_mode === "full_run"
        ? t("ui.post_run.full_run", "neuer Lauf nötig")
        : `${t("ui.post_run.resume_from", "Resume ab spätestens")} ${s.min_resume_phase}`;
      const checkbox = isSelectable(s) ? el("input", {
        type: "checkbox", checked: true, "aria-label": s.candidate_id,
        onchange: (e) => { if (e.target.checked) selected.add(s.candidate_id); else selected.delete(s.candidate_id); },
      }) : null;
      content.append(el("div", { class: "tc-card tc-mt-2" },
        el("div", { class: "tc-flex tc-items-center tc-gap-2" },
          checkbox,
          el("div", { class: "tc-text-sm" }, `${s.candidate_id}${s.experimental ? ` (${t("ui.jev.experimental", "experimentell")})` : ""}`)),
        el("div", { class: "tc-text-sm tc-text-muted" }, changes),
        el("div", { class: "tc-text-sm tc-text-muted" }, `${how}. ${t("ui.post_run.feasibility", "Machbarkeit noch nicht geprüft (Resume-Dry-Run).")}`)));
    }
    const selectable = content._suggestions.filter(isSelectable);
    if (selectable.length && onApply) {
      content.append(el("div", { class: "tc-mt-2 tc-flex tc-gap-2" },
        el("button", { class: "tc-btn tc-btn-primary tc-btn-sm", id: `${id}-apply-selected`, onclick: () => applySelection([...selected]) },
          t("ui.post_run.apply_selected", "Ausgewählte übernehmen")),
        el("button", { class: "tc-btn tc-btn-sm", id: `${id}-apply-all`, onclick: () => applySelection(selectable.map((s) => s.candidate_id)) },
          t("ui.post_run.apply_all", "Alle übernehmen"))));
    }
    if ((advice.not_measured || []).length)
      content.append(el("div", { class: "tc-text-sm tc-text-muted tc-mt-2" },
        `${t("ui.post_run.not_measured", "Nicht gemessen")}: ${advice.not_measured.map((k) => t(`ui.post_run.metric.${k}`, k)).join(", ")}`));
  }

  async function request() {
    const { currentRunId } = getRunState();
    if (!currentRunId) return;
    lastRunId = currentRunId;
    button.disabled = true;
    status.textContent = t("ui.post_run.loading", "Auswertung läuft…");
    try {
      const payload = await api.post(API_ENDPOINTS.decisions.postRunAdvice, { run_id: currentRunId, allow_experimental: true });
      status.textContent = currentRunId;
      renderAdvice(payload);
    } catch (error) {
      status.textContent = `${t("ui.post_run.failed", "Anfrage fehlgeschlagen")}: ${error?.message || error}`;
      clear(content);
    } finally {
      button.disabled = false;
    }
  }
  return el("div", { class: "tc-card", id },
    el("div", { class: "tc-card-title tc-flex tc-items-center tc-justify-between tc-gap-2" },
      el("span", {}, t("ui.post_run.title", "Jev-Nachbetrachtung")), button),
    status, content);
}
