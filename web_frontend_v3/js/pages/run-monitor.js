// js/pages/run-monitor.js – Sub-Tab: Run Monitor

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { createPhaseList, setPhaseList, updatePhaseState, updatePhaseStates } from "../components/phase-list.js";
import { createLogViewer } from "../components/log-viewer.js";
import { connectWebSocket, disconnectWebSocket, onWebSocketMessage } from "../components/ws-manager.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { getRunState, setRunState } from "../state/run-state.js";
import { getStore } from "../state/store.js";

export function createRunMonitorPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  // Run control
  const startBtn = el("button", { class: "tc-btn tc-btn-primary", id: "run-start-btn", onclick: () => startRun() }, t("ui.button.run_start", "Start"));
  const stopBtn = el("button", { class: "tc-btn", id: "run-stop-btn", onclick: () => stopRun() }, t("ui.button.stop", "Stop"));
  const resumeBtn = el("button", { class: "tc-btn", id: "run-resume-btn", onclick: () => resumeRun() }, t("ui.button.resume", "Resume"));
  const control = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.run_control", "Run Control")),
    el("div", { class: "tc-flex tc-gap-3" }, startBtn, stopBtn, resumeBtn),
  );

  // Run info box
  const runInfo = el("div", { class: "tc-card", id: "run-info-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.run_info", "Run Info")),
    el("div", { class: "tc-grid-2", id: "run-info-grid" },
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.run_id", "Run ID")), el("div", { class: "tc-text-sm tc-mono", id: "info-run-id" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.run_dir", "Verzeichnis")), el("div", { class: "tc-text-sm tc-mono", id: "info-run-dir" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.run_name", "Run Name")), el("div", { class: "tc-text-sm", id: "info-run-name" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.status", "Status")), el("div", { class: "tc-text-sm", id: "info-status" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.frames", "Frames")), el("div", { class: "tc-text-sm", id: "info-frames" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.color_mode", "OSC/Mono")), el("div", { class: "tc-text-sm", id: "info-color-mode" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.pipeline", "Pipeline")), el("div", { class: "tc-text-sm", id: "info-pipeline" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.output_dir", "Ausgabeordner")), el("div", { class: "tc-text-sm tc-mono", id: "info-output-dir" }, "\u2014")),
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

  page.append(control, runInfo, phases, stats, logViewer.wrapper);

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

  restoreCurrentRun();
  return page;
}

let pollTimer = null;

function setRunButtonsActive(isRunning) {
  const startBtn = document.getElementById("run-start-btn");
  const stopBtn = document.getElementById("run-stop-btn");
  const resumeBtn = document.getElementById("run-resume-btn");
  if (startBtn) startBtn.disabled = isRunning;
  if (stopBtn) stopBtn.disabled = !isRunning;
  if (resumeBtn) resumeBtn.disabled = isRunning;
}

function startPolling(runId) {
  stopPolling();
  pollTimer = setInterval(async () => {
    await refreshRunStatus(runId);
    const { status } = getRunState();
    if (status === "completed" || status === "failed" || status === "stopped" || status === "error") {
      stopPolling();
      setRunButtonsActive(false);
      disconnectWebSocket();
    }
  }, 3000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function restoreCurrentRun() {
  try {
    const appState = await api.get(API_ENDPOINTS.app.state);
    const current = appState?.run?.current;
    const scan = appState?.scan?.last_scan;

    // Always show scan info immediately, even without active run
    if (scan) {
      const frames = scan.frames_detected || scan.frames_total || 0;
      if (frames > 0) updateInfo("info-frames", frames);
      if (scan.input_path) updateInfo("info-output-dir", scan.input_path);
      if (scan.color_mode) updateInfo("info-color-mode", scan.color_mode);
    }

    // Also check input-scan store for immediate display
    const inputStore = getStore("input-scan", { scanData: {}, queueItems: [] });
    const sd = inputStore.getState().scanData || {};
    if (sd.input_dir && (!scan || !scan.input_path)) updateInfo("info-output-dir", sd.input_dir);
    if (sd.color_mode && (!scan || !scan.color_mode)) updateInfo("info-color-mode", sd.color_mode);
    if (sd.run_name) updateInfo("info-run-name", sd.run_name);

    if (current?.run_id) {
      const isRunning = current.status === "running";
      setRunState({ currentRunId: current.run_id, status: current.status || "running" });
      setRunButtonsActive(isRunning);
      updateStat("stat-run-id", current.run_id);
      updateStat("stat-status", current.status || "running");
      updateInfo("info-run-id", current.run_id);
      updateInfo("info-status", current.status || "running");
      if (current.run_dir) updateInfo("info-run-dir", current.run_dir);
      const runName = current.run_id.replace(/_\d{4}-\d{2}-\d{2}.*$/, "");
      updateInfo("info-run-name", runName);
      await refreshRunStatus(current.run_id);
      // Load existing logs from REST endpoint
      await loadInitialLogs(current.run_id);
      if (isRunning) {
        connectWebSocket(current.run_id);
        startPolling(current.run_id);
      }
    }
  } catch {}
}

async function loadInitialLogs(runId) {
  try {
    const logs = await api.get(API_ENDPOINTS.runs.logs(runId));
    if (!logs || !Array.isArray(logs.lines)) return;
    const body = document.getElementById("log-viewer-body");
    if (!body) return;
    for (const line of logs.lines) {
      let ts = formatTime();
      let level = "INFO";
      let text = "";
      if (typeof line === "string") {
        try {
          const ev = JSON.parse(line);
          ts = ev.ts ? ev.ts.split("T")[1]?.replace("Z", "") || ev.ts : formatTime();
          level = (ev.type === "error" || ev.type === "warning") ? ev.type.toUpperCase() : "INFO";
          text = formatEventMessage(ev);
        } catch {
          text = line;
        }
      } else {
        ts = line.ts || line.time || formatTime();
        level = line.level || "INFO";
        text = line.message || line.text || "";
      }
      const div = document.createElement("div");
      div.className = "tc-log-line";
      div.innerHTML = `<span class="tc-log-time">${ts}</span><span class="tc-log-level tc-log-level-${level}">${level}</span><span class="tc-log-msg">${text}</span>`;
      body.appendChild(div);
    }
    body.scrollTop = body.scrollHeight;
  } catch {}
}

function formatEventMessage(ev) {
  const type = ev.type || "";
  const phase = ev.phase_name || ev.phase || "";
  if (type === "phase_start") return `${phase} | start`;
  if (type === "phase_progress") {
    const pct = ev.pct != null ? ` (${Math.round(ev.pct)}%)` : "";
    return `${phase} | progress${pct}`;
  }
  if (type === "phase_end") {
    const status = ev.status || "ok";
    const reason = ev.reason ? ` (${ev.reason})` : "";
    return `${phase} | ${status}${reason}`;
  }
  if (type === "run_start") return "Run gestartet";
  if (type === "run_end") return `Run beendet | ${ev.status || "ok"}`;
  if (type === "queue_progress") return ev.message || "Queue progress";
  if (ev.message) return ev.message;
  return type || JSON.stringify(ev).slice(0, 200);
}

async function refreshRunStatus(runId) {
  try {
    const status = await api.get(API_ENDPOINTS.runs.status(runId));
    if (!status) return;

    if (status.phases && Array.isArray(status.phases)) {
      setPhaseList(status.phases);
    }

    updateInfo("info-run-id", status.run_id || runId);
    updateInfo("info-run-dir", status.run_dir || "\u2014");
    updateInfo("info-status", status.status || "\u2014");
    updateInfo("info-color-mode", status.color_mode || "\u2014");
    updateInfo("info-pipeline", status.method || (status.aqmh_enabled ? "AQMH" : "Classic") || "\u2014");

    const runName = (status.run_id || runId).replace(/_\d{4}-\d{2}-\d{2}.*$/, "");
    updateInfo("info-run-name", runName);

    if (status.run_dir) {
      updateInfo("info-output-dir", status.run_dir + "/outputs");
    }

    const queue = status.queue;
    if (Array.isArray(queue) && queue.length > 0) {
      let totalInputs = 0;
      for (const item of queue) {
        if (item.input_dir) totalInputs++;
      }
      if (totalInputs > 0) updateInfo("info-frames", `${queue.length} Queue Items`);
    }

    updateStat("stat-run-id", status.run_id || runId);
    updateStat("stat-status", status.status || "\u2014");

    if (status.status === "running" || status.status === "completed") {
      const appState = await api.get(API_ENDPOINTS.app.state).catch(() => null);
      const scan = appState?.scan?.last_scan;
      if (scan) {
        const frames = scan.frames_detected || scan.frames_total || 0;
        if (frames > 0) updateInfo("info-frames", frames);
      }
    }
  } catch {}
}

async function startRun() {
  try {
    const inputStore = getStore("input-scan", { scanData: {}, queueItems: [] });
    const sd = inputStore.getState().scanData || {};
    const queue = inputStore.getState().queueItems || [];

    const payload = {
      input_dir: sd.input_dir || "",
      runs_dir: sd.runs_dir || "",
      run_name: sd.run_name || "",
      color_mode: sd.color_mode || "",
      queue: queue.length > 0 ? queue : undefined,
    };

    if (!payload.input_dir && (!payload.queue || payload.queue.length === 0)) {
      toastError(t("ui.toast.run_start_failed", "Start fehlgeschlagen"), t("ui.error.no_input_dir", "Kein Eingabeordner festgelegt. Bitte zuerst unter Input & Scan konfigurieren."));
      return;
    }

    toast(t("ui.toast.run_starting", "Run wird gestartet..."), "", "info");
    const result = await api.post(API_ENDPOINTS.runs.start, payload);
    const runId = result?.run_id || result?.id;
    if (runId) {
      setRunState({ currentRunId: runId, status: "running" });
      setRunButtonsActive(true);
      updateStat("stat-run-id", runId);
      updateStat("stat-status", "running");
      connectWebSocket(runId);
      startPolling(runId);
      toastSuccess(t("ui.toast.run_started", "Run gestartet"), runId);
      refreshRunStatus(runId);
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
    setRunButtonsActive(false);
    updateStat("stat-status", "stopped");
    updateInfo("info-status", "stopped");
    stopPolling();
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
    setRunButtonsActive(true);
    updateStat("stat-status", "running");
    updateInfo("info-status", "running");
    connectWebSocket(currentRunId);
    startPolling(currentRunId);
    toastSuccess(t("ui.toast.run_resumed", "Run fortgesetzt"));
  } catch (e) {
    toastError(t("ui.toast.resume_failed", "Resume fehlgeschlagen"), e.message);
  }
}

function handleWsMessage(data, logViewer, phases) {
  if (!data) return;

  const type = data.type || "";
  const payload = data.payload || data;

  // Log lines — backend sends type "log_line" with message in payload.message
  if (type === "log_line" || type === "log") {
    const msg = payload.message || payload.text || data.message || data.text || "";
    const ts = data.ts || payload.ts || formatTime();
    const level = payload.level || data.level || (type === "error" || type === "warning" ? type.toUpperCase() : "INFO");
    logViewer.addLine(ts, level, msg);
  }

  // Warning / error events also go to log
  if (type === "warning" || type === "error") {
    const msg = payload.message || payload.text || data.message || data.text || JSON.stringify(payload);
    logViewer.addLine(data.ts || formatTime(), type.toUpperCase(), msg);
  }

  // Phase events: phase_start, phase_progress, phase_end
  // Backend sends phase at top level (data.phase), not phase_name
  if (type === "phase_start" || type === "phase_progress" || type === "phase_end") {
    const phaseName = data.phase || payload.phase_name || payload.phase || data.phase_name || "";
    if (!phaseName || phaseName === "null") return;
    const status = type === "phase_start" ? "running" : type === "phase_end" ? (payload.status || "ok") : (payload.status || "running");
    const pct = data.pct ?? payload.pct ?? payload.progress ?? data.progress ?? 0;
    updatePhaseState(phaseName, status, pct);
    if (payload.elapsed || data.elapsed) updateStat("stat-elapsed", payload.elapsed || data.elapsed);
    // Also log phase events
    const pctStr = pct > 0 ? ` (${Math.round(pct)}%)` : "";
    logViewer.addLine(data.ts || formatTime(), "INFO", `${phaseName} | ${type.replace("phase_", "")}${pctStr}`);
  }

  // Run status with full phase array
  if (type === "run_status") {
    if (payload.phases && Array.isArray(payload.phases)) {
      setPhaseList(payload.phases);
    }
    const runStatus = data.state || payload.status || data.status || "";
    if (runStatus) {
      updateStat("stat-status", runStatus);
      updateInfo("info-status", runStatus);
      setRunState({ status: runStatus });
    }
    const currentPhase = payload.current_phase || data.phase || "";
    if (currentPhase && currentPhase !== "null" && runStatus === "running") {
      updateInfo("info-status", `${runStatus} — ${currentPhase}`);
    }
  }

  // Queue progress
  if (type === "queue_progress" && payload) {
    if (payload.message) logViewer.addLine(data.ts || formatTime(), "INFO", payload.message);
  }

  // Terminal events
  const terminalStatuses = ["completed", "failed", "cancelled", "aborted", "error", "done", "finished", "ok"];
  const statusStr = String(data.status || payload.status || payload.state || "").toLowerCase();
  if (type === "run_end" || type === "run_start" || terminalStatuses.includes(statusStr)) {
    if (type === "run_start") {
      logViewer.addLine(data.ts || formatTime(), "INFO", "Run gestartet");
      return;
    }
    const finalStatus = data.status || payload.status || payload.state || "completed";
    updateStat("stat-status", finalStatus);
    updateInfo("info-status", finalStatus);
    setRunState({ status: finalStatus });
    setRunButtonsActive(false);
    stopPolling();
    if (type === "run_end") {
      disconnectWebSocket();
      toastSuccess(t("ui.toast.run_done", "Run abgeschlossen"));
      if (getRunState().currentRunId) refreshRunStatus(getRunState().currentRunId);
    }
  }
}

function updateStat(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(value);
}

function updateInfo(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(value);
}

function formatTime() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}
