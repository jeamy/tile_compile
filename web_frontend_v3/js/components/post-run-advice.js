import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { getRunState } from "../state/run-state.js";

// Post-run advice on request (never automatic): reads the finished run's own artifacts, changes nothing and starts nothing.
// A suggestion only names the parameter change and how far the run has to be redone; feasibility still comes from the
// resume dry run and starting stays a separate action.

const OUTCOMES = ["no_change", "diagnose", "suggest_downstream", "suggest_reconstruction"];

function fmt(value) {
  if (value === null || value === undefined) return "—";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function renderAdvice(target, payload) {
  clear(target);
  const advice = payload?.advice;
  if (!advice) {
    target.append(el("div", { class: "tc-text-sm tc-text-muted" }, t("ui.post_run.none", "Keine Auswertung.")));
    return;
  }
  const outcome = OUTCOMES.includes(advice.outcome) ? advice.outcome : "diagnose";
  target.append(el("div", { class: "tc-text-sm", "data-outcome": outcome }, t(`ui.post_run.outcome.${outcome}`, outcome)));
  for (const f of advice.findings || []) {
    const cls = f.severity === "warning" ? "tc-text-sm" : "tc-text-sm tc-text-muted";
    target.append(el("div", { class: cls }, `${t(`ui.post_run.finding.${f.code}`, f.code)}${f.detail ? `: ${f.detail}` : ""}`));
  }
  for (const s of advice.suggestions || []) {
    const changes = (s.updates || []).map((u) => `${u.path}: ${fmt(u.old_value)} → ${fmt(u.value)}`).join("; ");
    const how = s.resume_mode === "full_run"
      ? t("ui.post_run.full_run", "neuer Lauf nötig")
      : `${t("ui.post_run.resume_from", "Resume ab spätestens")} ${s.min_resume_phase}`;
    target.append(el("div", { class: "tc-card tc-mt-2" },
      el("div", { class: "tc-text-sm" }, `${s.candidate_id}${s.experimental ? ` (${t("ui.jev.experimental", "experimentell")})` : ""}`),
      el("div", { class: "tc-text-sm tc-text-muted" }, changes),
      el("div", { class: "tc-text-sm tc-text-muted" }, `${how}. ${t("ui.post_run.feasibility", "Machbarkeit noch nicht geprüft (Resume-Dry-Run).")}`)));
  }
  if ((advice.not_measured || []).length)
    target.append(el("div", { class: "tc-text-sm tc-text-muted tc-mt-2" },
      `${t("ui.post_run.not_measured", "Nicht gemessen")}: ${advice.not_measured.map((k) => t(`ui.post_run.metric.${k}`, k)).join(", ")}`));
}

export function createPostRunAdvicePanel(id = "post-run-advice") {
  const status = el("div", { class: "tc-text-sm tc-text-muted", id: `${id}-status` }, t("ui.post_run.idle", "Nur auf Anforderung; ändert nichts und startet keinen Lauf."));
  const content = el("div", { class: "tc-flex-col tc-gap-2 tc-mt-2", id: `${id}-content` });
  const button = el("button", { class: "tc-btn tc-btn-sm", id: `${id}-button`, onclick: request }, t("ui.post_run.request", "Nachbetrachtung anfordern"));
  async function request() {
    const { currentRunId } = getRunState();
    if (!currentRunId) return;
    button.disabled = true;
    status.textContent = t("ui.post_run.loading", "Auswertung läuft…");
    try {
      const payload = await api.post(API_ENDPOINTS.decisions.postRunAdvice, { run_id: currentRunId, allow_experimental: true });
      status.textContent = currentRunId;
      renderAdvice(content, payload);
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
