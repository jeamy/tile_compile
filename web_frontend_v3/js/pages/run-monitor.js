// js/pages/run-monitor.js – Sub-Tab: Run Monitor

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { createPhaseList, updatePhaseStates } from "../components/phase-list.js";
import { createLogViewer } from "../components/log-viewer.js";
import { connectWebSocket, disconnectWebSocket, onWebSocketMessage } from "../components/ws-manager.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { getRunState, setRunState } from "../state/run-state.js";

export function createRunMonitorPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  // Run control
  const control = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.run_control", "Run Control")),
    el("div", { class: "tc-flex tc-gap-3" },
      el("button", { class: "tc-btn tc-btn-primary", onclick: () => startRun() }, t("ui.button.run_start", "Start")),
      el("button", { class: "tc-btn", onclick: () => stopRun() }, t("ui.button.stop", "Stop")),
      el("button", { class: "tc-btn", onclick: () => resumeRun() }, t("ui.button.resume", "Resume")),
    ),
  );

  // Phase progress (component-based)
  const phases = createPhaseList();

  // Stats panel
  const stats = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.stats", "Stats")),
    el("div", { class: "tc-grid-2", id: "run-stats" },
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.run_id", "Run ID")), el("div", { class: "tc-text-sm tc-mono", id: "stat-run-id" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.status", "Status")), el("div", { class: "tc-text-sm", id: "stat-status" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.elapsed", "Elapsed")), el("div", { class: "tc-text-sm tc-mono", id: "stat-elapsed" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.frames", "Frames")), el("div", { class: "tc-text-sm", id: "stat-frames" }, "\u2014")),
    ),
  );

  // Log viewer (component-based)
  const logViewer = createLogViewer();

  page.append(control, phases, stats, logViewer.wrapper);

  // WebSocket listener
  onWebSocketMessage((event) => {
    if (event.type === "ws:message") {
      handleWsMessage(event.data, logViewer, phases);
    } else if (event.type === "ws:open") {
      toast(t("ui.toast.ws_connected", "WebSocket verbunden"), "", "info");
    } else if (event.type === "ws:close") {
      toast(t("ui.toast.ws_disconnected", "WebSocket getrennt"), "", "warning");
    }
  });

  return page;
}

async function startRun() {
  try {
    toast(t("ui.toast.run_starting", "Run wird gestartet..."), "", "info");
    const result = await api.post(API_ENDPOINTS.runs.start, {});
    const runId = result?.run_id || result?.id;
    if (runId) {
      setRunState({ currentRunId: runId, status: "running" });
      updateStat("stat-run-id", runId);
      updateStat("stat-status", "running");
      connectWebSocket(runId);
      toastSuccess(t("ui.toast.run_started", "Run gestartet"), runId);
    }
  } catch (e) {
    toastError(t("ui.toast.run_start_failed", "Start fehlgeschlagen"), e.message);
  }
}

async function stopRun() {
  const { currentRunId } = getRunState();
  if (!currentRunId) return;
  try {
    await api.post(API_ENDPOINTS.runs.stop(currentRunId), {});
    setRunState({ status: "stopped" });
    updateStat("stat-status", "stopped");
    disconnectWebSocket();
    toastSuccess(t("ui.toast.run_stopped", "Run gestoppt"));
  } catch (e) {
    toastError(t("ui.toast.stop_failed", "Stop fehlgeschlagen"), e.message);
  }
}

async function resumeRun() {
  const { currentRunId } = getRunState();
  if (!currentRunId) return;
  try {
    await api.post(API_ENDPOINTS.runs.resume(currentRunId), {});
    setRunState({ status: "running" });
    updateStat("stat-status", "running");
    connectWebSocket(currentRunId);
    toastSuccess(t("ui.toast.run_resumed", "Run fortgesetzt"));
  } catch (e) {
    toastError(t("ui.toast.resume_failed", "Resume fehlgeschlagen"), e.message);
  }
}

function handleWsMessage(data, logViewer, phases) {
  if (!data) return;

  if (data.type === "log" || data.log) {
    const entry = data.log || data;
    logViewer.addLine(entry.time || formatTime(), entry.level || "INFO", entry.message || entry.text || "");
  }

  if (data.type === "phase" || data.phase) {
    const phase = data.phase || data;
    updatePhaseStates({ [phase.name]: phase.status || "active" });
    if (phase.elapsed) updateStat("stat-elapsed", phase.elapsed);
  }

  if (data.type === "stats" || data.stats) {
    const s = data.stats || data;
    if (s.run_id) updateStat("stat-run-id", s.run_id);
    if (s.status) updateStat("stat-status", s.status);
    if (s.elapsed) updateStat("stat-elapsed", s.elapsed);
    if (s.frames) updateStat("stat-frames", s.frames);
  }

  if (data.type === "done" || data.status === "done") {
    updateStat("stat-status", "done");
    disconnectWebSocket();
    toastSuccess(t("ui.toast.run_done", "Run abgeschlossen"));
  }
}

function updateStat(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(value);
}

function formatTime() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}
