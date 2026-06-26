// js/pages/run-monitor.js – Sub-Tab: Run Monitor

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { createPhaseList, setPhaseList, updatePhaseState, setPhaseClickHandler, getSelectedPhase, clearSelectedPhase, resetPhasesForResume, getPhasesForConfig } from "../components/phase-list.js";
import { createLogViewer } from "../components/log-viewer.js";
import { connectWebSocket, disconnectWebSocket, onWebSocketMessage } from "../components/ws-manager.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { getRunState, setRunState } from "../state/run-state.js";
import { getStore } from "../state/store.js";
import { getConfigState, validateConfig } from "../state/config-state.js";
import { pollJob } from "../utils/poll.js";
import { openStatsFolder, openStatsReport } from "../utils/stats-utils.js";
import { promptGrantRoot } from "../components/path-picker-modal.js";

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

  // Phase progress (component-based) — use phases matching current config
  const configPhases = getPhasesForConfig(getConfigState().draft);
  const phases = createPhaseList(configPhases);

  // Resume panel (hidden until phase selected)
  const resumePanel = el("div", { class: "tc-card", id: "resume-panel", style: "display:none" },
    el("div", { class: "tc-card-title tc-flex tc-items-center tc-justify-between" },
      el("span", {}, t("ui.title.resume", "Resume")),
      el("span", { class: "tc-badge tc-badge-info", id: "resume-phase-badge" }, ""),
    ),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.label.config_yaml", "Config YAML")),
      el("textarea", {
        class: "tc-input tc-mono",
        id: "resume-config-yaml",
        rows: 16,
        style: "width:100%;font-size:0.85em;resize:vertical",
        spellcheck: false,
      }),
    ),
    el("div", { class: "tc-mt-2 tc-flex tc-gap-2 tc-items-center tc-flex-wrap" },
      el("label", { class: "tc-label" }, t("ui.label.config_revision", "Config Revision")),
      el("select", { class: "tc-select", id: "resume-config-revision", style: "flex:1 1 auto;min-width:200px" },
        el("option", { value: "" }, t("ui.option.current", "Aktuell")),
      ),
      el("button", { class: "tc-btn tc-btn-sm", id: "resume-load-revision-btn", onclick: () => loadRevisionIntoEditor() }, t("ui.button.load_revision", "Laden")),
    ),
    el("div", { class: "tc-mt-3 tc-flex tc-gap-3 tc-items-center" },
      el("button", { class: "tc-btn tc-btn-primary", id: "resume-execute-btn", onclick: () => resumeRun() }, t("ui.button.resume_from", "Resume")),
      el("button", { class: "tc-btn tc-btn-sm", id: "resume-load-current-btn", onclick: () => loadCurrentConfig() }, t("ui.button.load_current_config", "Config laden")),
      el("span", { class: "tc-text-muted tc-text-sm", id: "resume-hint" }, ""),
    ),
  );

  // Stats panel
  const statsGenerateBtn = el("button", { class: "tc-btn tc-btn-sm", id: "stats-generate-btn", disabled: true, onclick: () => generateStats() }, t("ui.button.generate_stats", "Generate Stats"));
  const statsOpenBtn = el("button", { class: "tc-btn tc-btn-sm", id: "stats-open-btn", disabled: true, onclick: () => { const rid = getRunState().currentRunId; if (rid) openStatsFolder(rid); } }, t("ui.button.open_stats_folder", "Open Stats Folder"));
  const statsReportBtn = el("button", { class: "tc-btn tc-btn-sm", id: "stats-report-btn", disabled: true, onclick: () => { const rid = getRunState().currentRunId; if (rid) openStatsReport(rid); } }, t("ui.button.open_stats_report", "Open Report"));
  const stats = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title tc-flex tc-items-center tc-justify-between" },
      el("span", {}, t("ui.title.stats", "Stats")),
      el("div", { class: "tc-flex tc-gap-2" }, statsGenerateBtn, statsOpenBtn, statsReportBtn),
    ),
    el("div", { class: "tc-grid-2", id: "run-stats" },
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.run_id", "Run ID")), el("div", { class: "tc-text-sm tc-mono", id: "stat-run-id" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.status", "Status")), el("div", { class: "tc-text-sm", id: "stat-status" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.elapsed", "Elapsed")), el("div", { class: "tc-text-sm tc-mono", id: "stat-elapsed" }, "\u2014")),
      el("div", {}, el("div", { class: "tc-label" }, t("ui.label.frames", "Frames")), el("div", { class: "tc-text-sm", id: "stat-frames" }, "\u2014")),
    ),
  );

  // Log viewer (component-based)
  const logViewer = createLogViewer();

  // Warning banner — collects calibration/run warnings and shows them as a batch
  const warningBanner = el("div", { id: "run-warning-banner", class: "tc-card", style: "display:none" },
    el("div", { class: "tc-card-title tc-flex tc-items-center tc-gap-2" },
      el("span", { class: "tc-badge tc-badge-warning" }, "⚠"),
      el("span", {}, t("ui.title.warnings", "Warnungen")),
    ),
    el("div", { id: "run-warning-list", class: "tc-flex-col tc-gap-1" }),
  );

  page.append(control, runInfo, phases, warningBanner, resumePanel, stats, logViewer.wrapper);

  // WebSocket listener
  onWebSocketMessage((event) => {
    if (event.type === "ws:message") {
      handleWsMessage(event.data, logViewer, phases, warningBanner);
    } else if (event.type === "ws:open") {
      toast(t("ui.toast.ws_connected", "WebSocket verbunden"), "", "info");
    } else if (event.type === "ws:close") {
      toast(t("ui.toast.ws_disconnected", "WebSocket getrennt"), "", "warning");
    }
  });

  setPhaseClickHandler((phase) => onPhaseSelected(phase, logViewer));

  // Restore saved phase states from store after page is in the DOM
  // (setPhaseList uses getElementById which requires the page to be mounted)
  requestAnimationFrame(() => {
    const savedPhases = getRunState().phases;
    if (savedPhases && Array.isArray(savedPhases) && savedPhases.length > 0) {
      setPhaseList(savedPhases);
    }
    // If no saved phases, the config-based phases from createPhaseList are already shown
  });

  restoreCurrentRun();
  return page;
}

let pollTimer = null;
let resumePendingTimer = null;

// Returns the most specific run key for API calls: full path if known, else run_id.
// This allows the backend to locate runs on network drives or non-default runs_dir.
function getRunApiKey() { const { currentRunDir, currentRunId } = getRunState(); return currentRunDir || currentRunId || ""; }
function getResumePending() { return getRunState().resumePending || false; }
function setResumePending(v) { setRunState({ resumePending: v }); }
function getResumeActive() { return getRunState().resumeActive || false; }
function setResumeActive(v) { setRunState({ resumeActive: v }); }

function setRunButtonsActive(isRunning) {
  const startBtn = document.getElementById("run-start-btn");
  const stopBtn = document.getElementById("run-stop-btn");
  const resumeBtn = document.getElementById("run-resume-btn");
  const resumeExecBtn = document.getElementById("resume-execute-btn");
  if (startBtn) startBtn.disabled = isRunning;
  if (stopBtn) stopBtn.disabled = !isRunning;
  if (resumeBtn) resumeBtn.disabled = isRunning;
  if (resumeExecBtn) resumeExecBtn.disabled = isRunning;
}

function startPolling(runId) {
  stopPolling();
  pollTimer = setInterval(async () => {
    const status = await api.get(API_ENDPOINTS.runs.status(runId)).catch(() => null);
    // Safety: if backend says not running but resume flags are still set, clear them
    if (status && status.status !== "running" && (getResumeActive() || getResumePending())) {
      setResumeActive(false);
      setResumePending(false);
      if (resumePendingTimer) { clearTimeout(resumePendingTimer); resumePendingTimer = null; }
    }
    await refreshRunStatus(runId);
    if (getResumePending() || getResumeActive()) return;
    const { status: runStatus } = getRunState();
    if (runStatus === "completed" || runStatus === "failed" || runStatus === "stopped" || runStatus === "error") {
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

function savePhaseToStore(phaseName, status, pct) {
  const { phases } = getRunState();
  if (!Array.isArray(phases)) return;
  const updated = phases.map(p => {
    const name = typeof p === "string" ? p : (p.phase || p.name || "");
    if (name === phaseName) {
      if (typeof p === "string") return { phase: p, status, pct };
      return { ...p, status, pct };
    }
    return p;
  });
  setRunState({ phases: updated });
}

async function restoreCurrentRun() {
  try {
    const appState = await api.get(API_ENDPOINTS.app.state);
    const current = appState?.run?.current;
    const scan = appState?.scan?.last_scan;

    // Always show scan info immediately, even without active run
    if (scan) {
      const frames = scan.frames_detected || scan.frames_total || 0;
      if (frames > 0) updateStat("info-frames", frames);
      if (scan.input_path) updateStat("info-output-dir", scan.input_path);
      if (scan.color_mode) updateStat("info-color-mode", scan.color_mode);
    }

    // Also check input-scan store for immediate display
    const inputStore = getStore("input-scan", { scanData: {}, queueItems: [] });
    const sd = inputStore.getState().scanData || {};
    if (sd.input_dir && (!scan || !scan.input_path)) updateStat("info-output-dir", sd.input_dir);
    if (sd.color_mode && (!scan || !scan.color_mode)) updateStat("info-color-mode", sd.color_mode);
    if (sd.run_name) updateStat("info-run-name", sd.run_name);

    if (current?.run_id) {
      const backendRunning = current.status === "running";
      // If backend says not running but resume flags are still set, resume_end was missed
      if (!backendRunning && (getResumeActive() || getResumePending())) {
        setResumeActive(false);
        setResumePending(false);
        if (resumePendingTimer) { clearTimeout(resumePendingTimer); resumePendingTimer = null; }
      }
      const isRunning = backendRunning || getResumeActive() || getResumePending();
      setRunState({ currentRunId: current.run_id, currentRunDir: current.run_dir || null, status: current.status || "running" });
      setRunButtonsActive(isRunning);
      updateStat("stat-run-id", current.run_id);
      updateStat("stat-status", isRunning ? "running" : (current.status || "running"));
      updateStat("info-run-id", current.run_id);
      updateStat("info-status", isRunning ? "running" : (current.status || "running"));
      if (current.run_dir) updateStat("info-run-dir", current.run_dir);
      const runName = current.run_id.replace(/_\d{4}-\d{2}-\d{2}.*$/, "");
      updateStat("info-run-name", runName);
      await refreshRunStatus(current.run_id);
      // Load existing logs from REST endpoint
      await loadInitialLogs(current.run_id, logViewer, warningBanner);
      if (isRunning) {
        connectWebSocket(current.run_id, getResumeActive() || getResumePending());
        startPolling(current.run_id);
      } else {
        enableStatsButtons(current.run_id);
      }
    }
  } catch {}
}

async function loadInitialLogs(runId, logViewer, warningBanner) {
  try {
    const logs = await api.get(API_ENDPOINTS.runs.logs(runId));
    if (!logs || !Array.isArray(logs.lines)) return;
    for (const line of logs.lines) {
      let ts = formatTime();
      let level = "INFO";
      let text = "";
      let evType = "";
      if (typeof line === "string") {
        try {
          const ev = JSON.parse(line);
          ts = ev.ts ? ev.ts.split("T")[1]?.replace("Z", "") || ev.ts : formatTime();
          level = (ev.type === "error" || ev.type === "warning") ? ev.type.toUpperCase() : "INFO";
          evType = ev.type || "";
          text = formatEventMessage(ev);
        } catch {
          text = line;
        }
      } else {
        ts = line.ts || line.time || formatTime();
        level = line.level || "INFO";
        evType = line.type || "";
        text = line.message || line.text || "";
      }
      if (logViewer) {
        logViewer.addLine(ts, level, text);
      } else {
        const body = document.getElementById("log-viewer-body");
        if (!body) continue;
        const div = document.createElement("div");
        div.className = "tc-log-line";
        div.innerHTML = `<span class="tc-log-time">${ts}</span><span class="tc-log-level tc-log-level-${level}">${level}</span><span class="tc-log-msg">${text}</span>`;
        body.appendChild(div);
        body.scrollTop = body.scrollHeight;
      }
      // Feed warnings/errors into the warning banner
      if (warningBanner && (evType === "warning" || evType === "error" || level === "WARNING" || level === "ERROR")) {
        addRunWarning(warningBanner, text, evType === "error" || level === "ERROR" ? "error" : "warning");
      }
    }
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
  if (type === "resume_start") {
    const fromPhase = ev.from_phase || "";
    return `Resume | start | ${fromPhase}`;
  }
  if (type === "resume_end") {
    const ok = ev.success ?? false;
    const fromPhase = ev.from_phase || "";
    return `Resume | ${ok ? "OK" : "ERROR"} | ${fromPhase}`;
  }
  if (type === "queue_progress") return ev.message || "Queue progress";
  if (ev.message) return ev.message;
  return type || JSON.stringify(ev).slice(0, 200);
}

async function refreshRunStatus(runId) {
  try {
    const status = await api.get(API_ENDPOINTS.runs.status(runId));
    if (!status) return;

    // During resumePending or resumeActive, skip status/phase updates —
    // the REST endpoint still returns the old completed/failed status
    // until the resume job's events overwrite the run status
    if (getResumePending() || getResumeActive()) {
      if (status.run_dir) setRunState({ currentRunDir: status.run_dir });
      return;
    }

    if (status.phases && Array.isArray(status.phases)) {
      setPhaseList(status.phases);
      setRunState({ phases: status.phases });
    }

    updateStat("info-run-id", status.run_id || runId);
    updateStat("info-run-dir", status.run_dir || "\u2014");
    updateStat("info-status", status.status || "\u2014");
    updateStat("info-color-mode", status.color_mode || "\u2014");
    updateStat("info-pipeline", status.method || (status.aqmh_enabled ? "AQMH" : "Classic") || "\u2014");

    if (status.run_dir) {
      setRunState({ currentRunDir: status.run_dir });
      updateStat("info-output-dir", status.run_dir + "/outputs");
    }

    const runName = (status.run_id || runId).replace(/_\d{4}-\d{2}-\d{2}.*$/, "");
    updateStat("info-run-name", runName);

    const queue = status.queue;
    if (Array.isArray(queue) && queue.length > 0) {
      let totalInputs = 0;
      for (const item of queue) {
        if (item.input_dir) totalInputs++;
      }
      if (totalInputs > 0) updateStat("info-frames", `${queue.length} Queue Items`);
    }

    updateStat("stat-run-id", status.run_id || runId);
    updateStat("stat-status", status.status || "\u2014");

    if (status.status === "running" || status.status === "completed") {
      const appState = await api.get(API_ENDPOINTS.app.state).catch(() => null);
      const scan = appState?.scan?.last_scan;
      if (scan) {
        const frames = scan.frames_detected || scan.frames_total || 0;
        if (frames > 0) updateStat("info-frames", frames);
      }
    }
  } catch {}
}

async function startRun() {
  try {
    const inputStore = getStore("input-scan", { scanData: {}, queueItems: [] });
    const sd = inputStore.getState().scanData || {};
    const queue = inputStore.getState().queueItems || [];

    if (!sd.input_dir && (!queue || queue.length === 0)) {
      toastError(t("ui.toast.run_start_failed", "Start fehlgeschlagen"), t("ui.error.no_input_dir", "Kein Eingabeordner festgelegt. Bitte zuerst unter Input & Scan konfigurieren."));
      return;
    }

    // Validate config before starting
    toast(t("ui.toast.validating", "Validiere Config..."), "", "info");
    const validation = await validateConfig();
    if (validation?.errors?.length > 0) {
      const firstError = validation.errors[0];
      const errMsg = typeof firstError === "string" ? firstError : `${firstError.path || firstError.field || ""}: ${firstError.message || firstError.msg || ""}`.trim();
      toastError(t("ui.toast.run_start_failed", "Start fehlgeschlagen"), t("ui.error.config_invalid", "Config ungültig") + ` (${validation.errors.length} Fehler): ${errMsg}`);
      return;
    }

    const pccStore = getStore("pcc", { pccData: {} });
    const pccCatalogDir = pccStore.getState().pccData?.catalog_dir || "";

    const astroStore = getStore("astrometry", {});
    const astapDataDir = astroStore.getState().astapDataDir || "";

    let configYaml = getConfigState().draftYaml || getConfigState().configYaml || "";
    if (configYaml && pccCatalogDir) {
      configYaml = injectSirilCatalogDir(configYaml, pccCatalogDir);
    }
    if (configYaml && astapDataDir) {
      configYaml = injectAstapDataDir(configYaml, astapDataDir);
    }

    // Inject calibration settings from Input & Scan tab into config YAML
    const calValues = inputStore.getState().calValues || {};
    if (configYaml && calValues && Object.keys(calValues).length > 0) {
      configYaml = injectCalibrationIntoYaml(configYaml, calValues);
    }

    const payload = {
      input_dir: sd.input_dir || "",
      runs_dir: sd.runs_dir || "",
      run_name: sd.run_name || "",
      color_mode: sd.color_mode || "",
      queue: queue.length > 0 ? queue : undefined,
      config_yaml: configYaml || undefined,
    };

    toast(t("ui.toast.run_starting", "Run wird gestartet..."), "", "info");
    clearRunWarnings();
    let result;
    try {
      result = await api.post(API_ENDPOINTS.runs.start, payload);
    } catch (startErr) {
      if (startErr.status === 403 &&
          startErr.payload?.code === "PATH_NOT_ALLOWED") {
        const deniedPath = startErr.payload?.details?.path ||
                           startErr.payload?.details?.denied_path ||
                           sd.input_dir || "";
        const granted = await promptGrantRoot(deniedPath,
                                              startErr.payload?.details?.allowed_roots);
        if (granted) {
          result = await api.post(API_ENDPOINTS.runs.start, payload);
        } else {
          throw startErr;
        }
      } else {
        throw startErr;
      }
    }
    const runId = result?.run_id || result?.id;
    if (runId) {
      const newPhases = getPhasesForConfig(getConfigState().draft).map(p => ({ phase: p.phase, status: "pending", pct: 0, label: p.label }));
      setRunState({ currentRunId: runId, status: "running", phases: newPhases, resumeActive: false, resumePending: false });
      setRunButtonsActive(true);
      updateStat("stat-run-id", runId);
      updateStat("stat-status", "running");
      setPhaseList(newPhases);
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
    updateStat("info-status", "stopped");
    stopPolling();
    disconnectWebSocket();
    toastSuccess(t("ui.toast.run_stopped", "Run gestoppt"));
  } catch (e) {
    toastError(t("ui.toast.stop_failed", "Stop fehlgeschlagen"), e.message);
  }
}

async function resumeRun() {
  const { currentRunId } = getRunState();
  if (!currentRunId) {
    toastError(t("ui.toast.resume_failed", "Resume fehlgeschlagen"), t("ui.error.no_run", "Kein Run ausgewählt"));
    return;
  }
  const phase = getSelectedPhase();
  if (!phase) {
    toastError(t("ui.toast.resume_failed", "Resume fehlgeschlagen"), t("ui.error.no_phase", "Bitte eine Phase anklicken um Resume zu starten"));
    return;
  }
  const configYaml = document.getElementById("resume-config-yaml")?.value || "";
  if (!configYaml.trim()) {
    toastError(t("ui.toast.resume_failed", "Resume fehlgeschlagen"), t("ui.error.no_config", "Config YAML ist leer"));
    return;
  }
  try {
    const payload = {
      from_phase: phase,
      config_yaml: configYaml,
    };
    const { currentRunDir } = getRunState();
    if (currentRunDir) payload.run_dir = currentRunDir;
    const revSelect = document.getElementById("resume-config-revision");
    if (revSelect?.value) payload.config_revision_id = revSelect.value;
    toast(t("ui.toast.resuming", `Resume ab ${phase}...`), "", "info");
    const result = await api.post(API_ENDPOINTS.runs.resume(currentRunId), payload);
    const jobId = result?.job_id;
    setResumePending(true);
    if (resumePendingTimer) clearTimeout(resumePendingTimer);
    resumePendingTimer = setTimeout(() => { setResumePending(false); }, 15000);
    setRunState({ status: "running" });
    setRunButtonsActive(true);
    updateStat("stat-status", "running");
    updateStat("info-status", `running — ${phase}`);
    const newPhases = resetPhasesForResume(phase);
    if (newPhases.length > 0) setRunState({ phases: newPhases });
    connectWebSocket(currentRunId, true);
    startPolling(currentRunId);
    toastSuccess(t("ui.toast.run_resumed", "Run fortgesetzt"), `${phase}`);
  } catch (e) {
    toastError(t("ui.toast.resume_failed", "Resume fehlgeschlagen"), e.message);
  }
}

function onPhaseSelected(phase, logViewer) {
  const panel = document.getElementById("resume-panel");
  const badge = document.getElementById("resume-phase-badge");
  const hint = document.getElementById("resume-hint");
  if (!phase) {
    if (panel) panel.style.display = "none";
    return;
  }
  if (panel) panel.style.display = "";
  if (badge) badge.textContent = phase;
  if (hint) hint.textContent = t("ui.message.resume_hint", "Config wird geladen...");
  loadRunConfig(phase);
  loadConfigRevisions();
}

async function loadRunConfig(phase) {
  const { currentRunId } = getRunState();
  if (!currentRunId) return;
  try {
    const resp = await api.get(API_ENDPOINTS.runs.config(getRunApiKey()));
    const yaml = resp?.config_yaml || resp?.config || "";
    const editor = document.getElementById("resume-config-yaml");
    if (editor) editor.value = yaml;
    const hint = document.getElementById("resume-hint");
    if (hint) hint.textContent = yaml ? t("ui.message.resume_ready", "Bereit zum Resume") : t("ui.message.resume_no_config", "Keine Config gefunden");
  } catch (e) {
    const hint = document.getElementById("resume-hint");
    if (hint) hint.textContent = `Error: ${e.message}`;
  }
}

async function loadCurrentConfig() {
  const { currentRunId } = getRunState();
  if (!currentRunId) return;
  try {
    const resp = await api.get(API_ENDPOINTS.runs.config(getRunApiKey()));
    const yaml = resp?.config_yaml || resp?.config || "";
    const editor = document.getElementById("resume-config-yaml");
    if (editor) editor.value = yaml;
    toastSuccess(t("ui.toast.config_loaded", "Config geladen"));
  } catch (e) {
    toastError(t("ui.toast.config_load_failed", "Config laden fehlgeschlagen"), e.message);
  }
}

async function loadConfigRevisions() {
  const { currentRunId } = getRunState();
  if (!currentRunId) return;
  try {
    const resp = await api.get(API_ENDPOINTS.runs.configRevisions(getRunApiKey()));
    const select = document.getElementById("resume-config-revision");
    if (!select) return;
    select.innerHTML = "";
    select.appendChild(el("option", { value: "" }, t("ui.option.current", "Aktuell")));
    const items = resp?.items || [];
    for (const item of items) {
      const id = item.revision_id || item.id || "";
      const label = item.label || item.source || id;
      const created = item.created_at ? ` (${item.created_at.split("T")[0]})` : "";
      select.appendChild(el("option", { value: id }, `${label}${created}`));
    }
  } catch {}
}

async function loadRevisionIntoEditor() {
  const { currentRunId } = getRunState();
  if (!currentRunId) return;
  const select = document.getElementById("resume-config-revision");
  const revId = select?.value;
  if (!revId) {
    toast(t("ui.toast.select_revision", "Bitte Revision auswählen"), "", "info");
    return;
  }
  try {
    const resp = await api.get(API_ENDPOINTS.runs.configRevision(getRunApiKey(), revId));
    const yaml = resp?.config || "";
    const editor = document.getElementById("resume-config-yaml");
    if (editor) editor.value = yaml;
    toastSuccess(t("ui.toast.revision_loaded", "Revision geladen"));
  } catch (e) {
    toastError(t("ui.toast.revision_load_failed", "Revision laden fehlgeschlagen"), e.message);
  }
}

function handleWsMessage(data, logViewer, phases, warningBanner) {
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

  // Warning / error events also go to log and warning banner
  if (type === "warning" || type === "error") {
    const msg = payload.message || payload.text || data.message || data.text || JSON.stringify(payload);
    logViewer.addLine(data.ts || formatTime(), type.toUpperCase(), msg);
    if (warningBanner) addRunWarning(warningBanner, msg, type);
  }

  // Phase events: phase_start, phase_progress, phase_end
  // Backend sends phase at top level (data.phase), not phase_name
  if (type === "phase_start" || type === "phase_progress" || type === "phase_end") {
    const phaseName = data.phase || payload.phase_name || payload.phase || data.phase_name || "";
    if (!phaseName || phaseName === "null") return;
    const status = type === "phase_start" ? "running" : type === "phase_end" ? (payload.status || "ok") : (payload.status || "running");
    const pct = data.pct ?? payload.pct ?? payload.progress ?? data.progress ?? 0;
    const label = payload.label || null;
    updatePhaseState(phaseName, status, pct, label);
    savePhaseToStore(phaseName, status, pct);
    if (payload.elapsed || data.elapsed) updateStat("stat-elapsed", payload.elapsed || data.elapsed);
    // Also log phase events
    const pctStr = pct > 0 ? ` (${Math.round(pct)}%)` : "";
    const logLabel = label || phaseName;
    logViewer.addLine(data.ts || formatTime(), "INFO", `${logLabel} | ${type.replace("phase_", "")}${pctStr}`);
  }

  // Run status with full phase array
  // Skip stale run_status during resumePending or resumeActive —
  // backend sends old completed/failed phase list until resume events arrive,
  // and even after resume_start the run_status may contain stale phase data.
  // Individual phase_start/phase_progress/phase_end events drive the UI instead.
  if (type === "run_status" && !getResumePending() && !getResumeActive()) {
    if (payload.phases && Array.isArray(payload.phases)) {
      setPhaseList(payload.phases);
      setRunState({ phases: payload.phases });
    }
    const runStatus = data.state || payload.status || data.status || "";
    if (runStatus) {
      updateStat("stat-status", runStatus);
      updateStat("info-status", runStatus);
      setRunState({ status: runStatus });
    }
    const currentPhase = payload.current_phase || data.phase || "";
    if (currentPhase && currentPhase !== "null" && runStatus === "running") {
      updateStat("info-status", `${runStatus} — ${currentPhase}`);
    }
  }

  // Resume events
  if (type === "resume_start") {
    setResumePending(false);
    setResumeActive(true);
    if (resumePendingTimer) { clearTimeout(resumePendingTimer); resumePendingTimer = null; }
    const fromPhase = payload.from_phase || data.from_phase || "";
    logViewer.addLine(data.ts || formatTime(), "INFO", `Resume | start | ${fromPhase}`);
    if (fromPhase) {
      updatePhaseState(fromPhase, "running", 0);
      savePhaseToStore(fromPhase, "running", 0);
      updateStat("info-status", `running — ${fromPhase}`);
    }
    setRunState({ status: "running" });
    setRunButtonsActive(true);
  }
  if (type === "resume_end") {
    setResumePending(false);
    setResumeActive(false);
    if (resumePendingTimer) { clearTimeout(resumePendingTimer); resumePendingTimer = null; }
    const success = payload.success ?? data.success ?? false;
    const fromPhase = payload.from_phase || data.from_phase || "";
    logViewer.addLine(data.ts || formatTime(), success ? "INFO" : "ERROR", `Resume | ${success ? "OK" : "ERROR"} | ${fromPhase}`);
    if (success) {
      if (fromPhase) {
        updatePhaseState(fromPhase, "ok", 100);
        savePhaseToStore(fromPhase, "ok", 100);
      }
      updateStat("stat-status", "completed");
      updateStat("info-status", "completed");
      setRunState({ status: "completed" });
      setRunButtonsActive(false);
      stopPolling();
      disconnectWebSocket();
      toastSuccess(t("ui.toast.run_done", "Run abgeschlossen"));
      enableStatsButtons(getRunState().currentRunId);
      clearSelectedPhase();
      const panel = document.getElementById("resume-panel");
      if (panel) panel.style.display = "none";
      // Refresh full status from backend to get all final phase states
      refreshRunStatus(getRunState().currentRunId);
    } else {
      updateStat("stat-status", "failed");
      updateStat("info-status", "failed");
      setRunState({ status: "failed" });
      setRunButtonsActive(false);
      stopPolling();
      const err = payload.error || data.error || "";
      toastError(t("ui.toast.resume_failed", "Resume fehlgeschlagen"), err || fromPhase);
    }
  }

  // Queue progress
  if (type === "queue_progress" && payload) {
    if (payload.message) logViewer.addLine(data.ts || formatTime(), "INFO", payload.message);
  }

  // Terminal events — skip if resume is pending or active (stale status from previous run)
  if ((getResumePending() || getResumeActive()) && type !== "resume_end") {
    return;
  }
  const terminalStatuses = ["completed", "failed", "cancelled", "aborted", "error", "done", "finished", "ok"];
  const statusStr = String(data.status || payload.status || payload.state || "").toLowerCase();
  if (type === "run_end" || type === "run_start" || terminalStatuses.includes(statusStr)) {
    if (type === "run_start") {
      logViewer.addLine(data.ts || formatTime(), "INFO", "Run gestartet");
      return;
    }
    const finalStatus = data.status || payload.status || payload.state || "completed";
    updateStat("stat-status", finalStatus);
    updateStat("info-status", finalStatus);
    setRunState({ status: finalStatus });
    setRunButtonsActive(false);
    stopPolling();
    if (type === "run_end") {
      disconnectWebSocket();
      toastSuccess(t("ui.toast.run_done", "Run abgeschlossen"));
      enableStatsButtons(getRunState().currentRunId);
      if (getRunState().currentRunId) refreshRunStatus(getRunState().currentRunId);
    }
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

const runWarnings = new Set();

function addRunWarning(banner, msg, type) {
  if (!msg || !banner) return;
  const key = `${type}:${msg}`;
  if (runWarnings.has(key)) return;
  runWarnings.add(key);

  const list = banner.querySelector("#run-warning-list");
  if (!list) return;

  const isError = type === "error";
  const item = el("div", {
    class: `tc-text-sm tc-flex tc-items-start tc-gap-2 ${isError ? "tc-text-error" : "tc-text-warning"}`,
    style: "padding:4px 0",
  },
    el("span", { style: "flex-shrink:0" }, isError ? "✖" : "⚠"),
    el("span", {}, msg),
  );
  list.appendChild(item);
  banner.style.display = "";
}

function clearRunWarnings() {
  runWarnings.clear();
  const banner = document.getElementById("run-warning-banner");
  if (banner) {
    banner.style.display = "none";
    const list = banner.querySelector("#run-warning-list");
    if (list) list.innerHTML = "";
  }
}

function injectCalibrationIntoYaml(yaml, cal) {
  if (!yaml || !cal) return yaml;

  const normPath = (p) => (p || "").replace(/\\/g, '/');

  // Determine effective values: prefer master over dir when both set
  const biasMaster = normPath(cal.bias_master && cal.bias_master.trim() ? cal.bias_master : "");
  const biasDir = biasMaster ? "" : normPath(cal.bias_dir || "");
  const darkMaster = normPath(cal.dark_master && cal.dark_master.trim() ? cal.dark_master : "");
  const darkDir = darkMaster ? "" : normPath(cal.dark_dir || "");
  const flatMaster = normPath(cal.flat_master && cal.flat_master.trim() ? cal.flat_master : "");
  const flatDir = flatMaster ? "" : normPath(cal.flat_dir || "");

  const entries = [
    ["use_bias", cal.bias_enabled ? "true" : "false"],
    ...(cal.bias_enabled ? [
      ["bias_use_master", biasMaster ? "true" : "false"],
      ["bias_dir", biasDir],
      ["bias_master", biasMaster],
    ] : [
      ["bias_dir", ""],
      ["bias_master", ""],
    ]),
    ["use_dark", cal.dark_enabled ? "true" : "false"],
    ...(cal.dark_enabled ? [
      ["dark_use_master", darkMaster ? "true" : "false"],
      ["darks_dir", darkDir],
      ["dark_master", darkMaster],
    ] : [
      ["darks_dir", ""],
      ["dark_master", ""],
    ]),
    ["use_flat", cal.flat_enabled ? "true" : "false"],
    ...(cal.flat_enabled ? [
      ["flat_use_master", flatMaster ? "true" : "false"],
      ["flats_dir", flatDir],
      ["flat_master", flatMaster],
    ] : [
      ["flats_dir", ""],
      ["flat_master", ""],
    ]),
  ];

  for (const [yamlKey, rawVal] of entries) {
    if (rawVal === undefined || rawVal === null) continue;
    const strVal = typeof rawVal === "string" ? `"${rawVal}"` : String(rawVal);
    const replaceRegex = new RegExp(`^(\\s*)${yamlKey}:\\s*.*$`, "m");
    if (replaceRegex.test(yaml)) {
      yaml = yaml.replace(replaceRegex, `$1${yamlKey}: ${strVal}`);
    } else {
      // Key doesn't exist yet — insert under calibration: section
      yaml = insertKeyUnderSection(yaml, "calibration", yamlKey, strVal);
    }
  }
  return yaml;
}

function insertKeyUnderSection(yaml, section, key, value) {
  const sectionRegex = new RegExp(`^(${section}:\\s*\\n)`, "m");
  if (sectionRegex.test(yaml)) {
    return yaml.replace(sectionRegex, `$1  ${key}: ${value}\n`);
  }
  // Section doesn't exist — append it
  return yaml + `\n${section}:\n  ${key}: ${value}\n`;
}

function injectAstapDataDir(yaml, dataDir) {
  if (!yaml || !dataDir) return yaml;
  const safeDir = dataDir.replace(/\\/g, '/');
  // Replace any existing astap_data_dir value (empty, null, or existing path)
  if (/astap_data_dir:/m.test(yaml)) {
    return yaml.replace(/astap_data_dir:.*$/m, `astap_data_dir: "${safeDir}"`);
  }
  if (/^astrometry:/m.test(yaml)) {
    return yaml.replace(/^astrometry:/m, `astrometry:\n  astap_data_dir: "${safeDir}"`);
  }
  return yaml + `\nastrometry:\n  astap_data_dir: "${safeDir}"\n`;
}

function injectSirilCatalogDir(yaml, catalogDir) {
  if (!yaml || !catalogDir) return yaml;
  const safeCatalogDir = catalogDir.replace(/\\/g, '/');
  if (/siril_catalog_dir:\s*(?:~|null|""|'')\s*$/m.test(yaml)) {
    return yaml.replace(/siril_catalog_dir:\s*(?:~|null|""|'')\s*$/m, `siril_catalog_dir: "${safeCatalogDir}"`);
  }
  if (!/siril_catalog_dir:/m.test(yaml)) {
    if (/^pcc:/m.test(yaml)) {
      return yaml.replace(/^pcc:/m, `pcc:\n  siril_catalog_dir: "${safeCatalogDir}"`);
    }
    return yaml + `\npcc:\n  siril_catalog_dir: "${safeCatalogDir}"\n`;
  }
  return yaml;
}

function enableStatsButtons(runId) {
  const genBtn = document.getElementById("stats-generate-btn");
  const openBtn = document.getElementById("stats-open-btn");
  const reportBtn = document.getElementById("stats-report-btn");
  if (genBtn) genBtn.disabled = !runId;
  if (openBtn) openBtn.disabled = true;
  if (reportBtn) reportBtn.disabled = true;
  if (runId) checkStatsAvailable(runId);
}

async function checkStatsAvailable(runId) {
  try {
    const status = await api.get(API_ENDPOINTS.runs.statsStatus(runId));
    const openBtn = document.getElementById("stats-open-btn");
    const reportBtn = document.getElementById("stats-report-btn");
    const hasReport = !!status?.report_path;
    if (openBtn) openBtn.disabled = !hasReport;
    if (reportBtn) reportBtn.disabled = !hasReport;
  } catch {}
}

async function generateStats() {
  const runId = getRunState().currentRunId;
  if (!runId) return;
  const genBtn = document.getElementById("stats-generate-btn");
  if (genBtn) genBtn.disabled = true;
  try {
    toast(t("ui.toast.stats_generating", "Stats werden generiert..."), "", "info");
    const result = await api.post(API_ENDPOINTS.runs.stats(runId), {});
    const jobId = result?.job_id;
    if (jobId) {
      await pollJob(jobId, { intervalMs: 1000, timeoutMs: 120000 });
      toastSuccess(t("ui.toast.stats_done", "Stats generiert"));
      await checkStatsAvailable(runId);
    } else {
      await checkStatsAvailable(runId);
    }
  } catch (e) {
    toastError(t("ui.toast.stats_failed", "Stats-Generierung fehlgeschlagen"), e.message);
  } finally {
    if (genBtn) genBtn.disabled = false;
  }
}

