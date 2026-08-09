// js/pages/run-monitor.js – Sub-Tab: Run Monitor

import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { createPhaseList, setPhaseList, updatePhaseState, setPhaseClickHandler, getSelectedPhase, clearSelectedPhase, selectPhase, resetPhasesForResume, getPhasesForConfig } from "../components/phase-list.js";
import { createLogViewer } from "../components/log-viewer.js";
import { connectWebSocket, disconnectWebSocket, onWebSocketMessage } from "../components/ws-manager.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { getRunState, setRunState } from "../state/run-state.js";
import { setAiState } from "../state/ai-state.js";
import { getStore } from "../state/store.js";
import { getConfigState, validateConfig } from "../state/config-state.js";
import { pollJob } from "../utils/poll.js";
import { openStatsFolder, openStatsReport } from "../utils/stats-utils.js";
import { promptGrantRoot } from "../components/path-picker-modal.js";
import { openHmsPreview } from "../components/hms-preview.js";
import { openBgePreview } from "../components/bge-preview.js";
import { createYamlDiff } from "../components/yaml-diff.js";
import { createRunImagePreviewPanel, loadRunImagePreview } from "../components/run-image-preview.js";

export function createRunMonitorPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  // Run control
  const startBtn = el("button", { class: "tc-btn tc-btn-primary", id: "run-start-btn", onclick: () => startRun() }, t("ui.button.run_start", "Start"));
  const stopBtn = el("button", { class: "tc-btn", id: "run-stop-btn", disabled: true, onclick: () => stopRun() }, t("ui.button.stop", "Stop"));
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
      el("div", { class: "tc-flex tc-gap-2 tc-items-center" },
        el("button", { class: "tc-btn tc-btn-sm", id: "resume-hms-config-btn", style: "display:none", onclick: () => openSelectedHmsPreview() }, t("ui.button.hms_configure", "HMS konfigurieren")),
        el("button", { class: "tc-btn tc-btn-sm", id: "resume-bge-config-btn", style: "display:none", onclick: () => openSelectedBgePreview() }, t("ui.button.bge_configure", "BGE konfigurieren")),
        el("span", { class: "tc-badge tc-badge-info", id: "resume-phase-badge" }, ""),
      ),
    ),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.label.config_yaml", "Config YAML")),
      el("div", { class: "tc-config-section-nav", id: "resume-config-sections" }),
      el("textarea", {
        class: "tc-input tc-mono",
        id: "resume-config-yaml",
        rows: 16,
        style: "width:100%;font-size:0.85em;resize:vertical",
        spellcheck: false,
        oninput: (e) => updateResumeConfigSectionHighlights(e.target.value),
      }),
    ),
    el("div", { class: "tc-mt-2 tc-flex tc-gap-2 tc-items-center tc-flex-wrap" },
      el("label", { class: "tc-label" }, t("ui.label.config_revision", "Config Revision")),
      el("select", { class: "tc-select", id: "resume-config-revision", style: "flex:1 1 auto;min-width:200px" },
        el("option", { value: "" }, t("ui.option.current", "Aktuell")),
      ),
    ),
    el("div", { class: "tc-mt-4 tc-flex tc-gap-3 tc-items-center" },
      el("button", { class: "tc-btn tc-btn-primary", id: "resume-execute-btn", onclick: () => resumeRun() }, t("ui.button.resume_from", "Resume")),
      el("button", { class: "tc-btn", id: "resume-load-revision-btn", onclick: () => loadRevisionIntoEditor() }, t("ui.button.load_revision", "Revision laden")),
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

  const runPreview = createRunImagePreviewPanel("run-monitor-image-preview");
  const runChat = createRunChatPanel();

  // Log viewer (component-based)
  const logViewer = createLogViewer();
  activeLogViewer = logViewer;
  const runMonitorTabs = createRunMonitorTabs(resumePanel, runChat, logViewer.wrapper);

  // Warning banner — collects calibration/run warnings and shows them as a batch
  const warningBanner = el("div", { id: "run-warning-banner", class: "tc-card", style: "display:none" },
    el("div", { class: "tc-card-title tc-flex tc-items-center tc-gap-2" },
      el("span", { class: "tc-badge tc-badge-warning" }, "⚠"),
      el("span", {}, t("ui.title.warnings", "Warnungen")),
    ),
    el("div", { id: "run-warning-list", class: "tc-flex-col tc-gap-1" }),
  );
  activeWarningBanner = warningBanner;

  page.append(control, runInfo, phases, warningBanner, stats, runPreview, runMonitorTabs);

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

  // Defer until the page element is mounted in the DOM. createRunMonitorPage
  // returns the page node which the caller appends — any getElementById calls
  // before that (e.g. setRunButtonsActive) silently find nothing. Yielding via
  // setTimeout(0) guarantees the caller has appended the page before we try
  // to manipulate button elements.
  setTimeout(() => restoreCurrentRun(), 0);
  return page;
}

let pollTimer = null;
let logTailTimer = null;
let logTailSnapshot = null;
let resumePendingTimer = null;
let lastImagePreviewKey = "";
let monitorRunChatMessages = [];
let activeLogViewer = null;
let activeWarningBanner = null;
let runChatTrafficTimer = null;
let replayedRunLogEvents = new Set();
let resumeFeasibilitySeq = 0;
const runChatStore = getStore("run-chat", { chats: {} });
const RESUME_PENDING_TIMEOUT_MS = 120000;
const RESUME_CONFIG_SECTIONS = [
  { key: "aqmh", label: "AQMH" },
  { key: "common_overlap", label: "COMMON" },
  { key: "stacking", label: "STACKING" },
  { key: "output", label: "OUTPUT" },
  { key: "normalization", label: "NORM" },
  { key: "bge", label: "BGE" },
  { key: "pcc", label: "PCC" },
  { key: "hypermetric_stretch", label: "HMS" },
  { key: "registration", label: "REG" },
  { key: "astrometry", label: "ASTRO" },
];

// Returns the most specific run key for API calls: full path if known, else run_id.
// This allows the backend to locate runs on network drives or non-default runs_dir.
function getRunApiKey() { const { currentRunDir, currentRunId } = getRunState(); return currentRunDir || currentRunId || ""; }
function getResumePending() { return getRunState().resumePending || false; }
function setResumePending(v) { setRunState({ resumePending: v }); }
function getResumeActive() { return getRunState().resumeActive || false; }
function setResumeActive(v) { setRunState({ resumeActive: v }); }
function getResumeFromPhase() { return getRunState().resumeFromPhase || ""; }
function setResumeFromPhase(v) { setRunState({ resumeFromPhase: v || "" }); }

function isRunTerminalStatus(status) {
  return ["completed", "failed", "stopped", "error", "aborted", "unknown"].includes(String(status || "").toLowerCase());
}

function sameRunIdentity(a, b, aDir = "", bDir = "") {
  if (!a || !b || a !== b) return false;
  if (!aDir || !bDir) return true;
  return aDir === bDir;
}

function resumeErrorPayload(error) {
  const payload = error?.payload || {};
  return {
    code: payload.code || payload.error?.code || "",
    message: payload.message || payload.error?.message || error?.message || "",
    details: payload.details || payload.error?.details || {},
  };
}

function currentResumeCachePath(value) {
  const path = String(value || "");
  const legacySuffix = "/.prewarped_cache";
  if (!path.endsWith(legacySuffix)) return path;
  return `${path.slice(0, -legacySuffix.length)}/cache/prewarped_frames`;
}

function currentResumeMessage(value) {
  return String(value || "").replace(/\.prewarped_cache\b/g, "cache/prewarped_frames");
}

function formatResumeError(error, phase) {
  const parsed = resumeErrorPayload(error);
  const details = parsed.details || {};
  let title = currentResumeMessage(parsed.message || error?.message || t("ui.error.unknown", "Unbekannter Fehler"));
  let body = "";
  if (parsed.code === "RESUME_PHASE_NOT_FEASIBLE") {
    title = t("ui.error.resume_not_feasible", "Resume ab {phase} nicht möglich", { phase });
    body = currentResumeMessage(parsed.message || "");
    if (details.reason) {
      body += `\nGrund: ${details.reason}`;
    }
    if (details.effective_runner_phase && details.effective_runner_phase !== phase) {
      body += `\nRunner-Phase: ${details.effective_runner_phase}`;
    }
    if (details.cache_dir) {
      body += `\nCache: ${currentResumeCachePath(details.cache_dir)}`;
    }
    if (Array.isArray(details.missing_files) && details.missing_files.length) {
      body += "\n" + t("ui.error.missing_files", "Fehlende Dateien: {files}", {
        files: details.missing_files.join(", ")
      });
    }
    if (Array.isArray(details.feasible_phases) && details.feasible_phases.length) {
      body += "\n" + t("ui.error.feasible_phases", "Mögliche Resume-Phasen: {phases}", {
        phases: details.feasible_phases.join(", ")
      });
    }
  }
  return { title, body: body.trim() };
}

function activateRunMonitorTab(tabId) {
  const root = document.getElementById("run-monitor-work-tabs");
  if (!root) return;
  for (const btn of root.querySelectorAll(".tc-tab")) {
    btn.classList.toggle("active", btn.dataset.workTab === tabId);
    btn.setAttribute("aria-selected", btn.dataset.workTab === tabId ? "true" : "false");
  }
  for (const panel of root.querySelectorAll("[data-work-tab-panel]")) {
    panel.style.display = panel.dataset.workTabPanel === tabId ? "" : "none";
  }
}

function createRunMonitorTabs(resumePanel, runChatPanel, logPanel) {
  const tabs = [
    { id: "resume", label: t("ui.title.resume", "Resume"), node: resumePanel },
    { id: "chat", label: t("ui.title.run_chat", "Run-Chat"), node: runChatPanel },
    { id: "log", label: t("ui.title.live_log", "Live Log"), node: logPanel },
  ];
  const tabButtons = tabs.map((tab, index) => el("button", {
    class: `tc-tab${index === 0 ? " active" : ""}`,
    "data-work-tab": tab.id,
    role: "tab",
    "aria-selected": index === 0 ? "true" : "false",
    onclick: () => activateRunMonitorTab(tab.id),
  }, tab.label));
  const panels = tabs.map((tab, index) => el("div", {
    "data-work-tab-panel": tab.id,
    style: index === 0 ? "" : "display:none",
  }, tab.node));
  return el("div", { class: "tc-flex-col tc-gap-3", id: "run-monitor-work-tabs" },
    el("div", { class: "tc-subtab-bar" },
      el("div", { class: "tc-subtab-list tc-tabs", role: "tablist" }, ...tabButtons),
    ),
    ...panels,
  );
}

function findYamlTopLevelSectionLine(yaml, key) {
  const lines = String(yaml || "").split(/\r?\n/);
  const re = new RegExp(`^${key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}:\\s*(?:#.*)?$`);
  for (let i = 0; i < lines.length; i++) {
    if (re.test(lines[i])) return i;
  }
  return -1;
}

function focusResumeConfigSection(key) {
  const editor = document.getElementById("resume-config-yaml");
  if (!editor) return;
  const lines = String(editor.value || "").split(/\r?\n/);
  const line = findYamlTopLevelSectionLine(editor.value, key);
  if (line < 0) return;
  let start = 0;
  for (let i = 0; i < line; i++) start += lines[i].length + 1;
  const end = start + lines[line].length;
  editor.focus();
  editor.setSelectionRange(start, end);
  const lineHeight = 18;
  editor.scrollTop = Math.max(0, line * lineHeight - editor.clientHeight * 0.25);
}

function updateResumeConfigSectionHighlights(yaml = "") {
  const container = document.getElementById("resume-config-sections");
  if (!container) return;
  clear(container);
  const found = RESUME_CONFIG_SECTIONS
    .map(section => ({ ...section, line: findYamlTopLevelSectionLine(yaml, section.key) }))
    .filter(section => section.line >= 0);
  if (!found.length) {
    container.appendChild(el("span", { class: "tc-text-muted tc-text-sm" },
      t("ui.state.no_config_sections", "Keine Config-Abschnitte erkannt")));
    return;
  }
  container.appendChild(el("span", { class: "tc-text-muted tc-text-sm" },
    t("ui.label.config_sections", "Config-Abschnitte")));
  for (const section of found) {
    container.appendChild(el("button", {
      class: "tc-config-section-chip",
      title: t("ui.tooltip.config_section_jump", "Springt zum YAML-Abschnitt {section}.", { section: section.label }),
      onclick: () => focusResumeConfigSection(section.key),
    }, section.label));
  }
}

function phaseIndexInCurrentList(phaseName) {
  const phases = getRunState().phases;
  if (!Array.isArray(phases) || !phaseName) return -1;
  return phases.findIndex(p => {
    const name = typeof p === "string" ? p : (p.phase || p.name || "");
    return name === phaseName;
  });
}

function isBeforeResumePhase(phaseName) {
  const resumePhase = getResumeFromPhase();
  if (!resumePhase || !phaseName) return false;
  const phaseIdx = phaseIndexInCurrentList(phaseName);
  const resumeIdx = phaseIndexInCurrentList(resumePhase);
  return phaseIdx >= 0 && resumeIdx >= 0 && phaseIdx < resumeIdx;
}

function refreshCurrentImagePreview(force = false) {
  const { currentRunId, currentRunDir } = getRunState();
  if (!currentRunId) return;
  const key = `${currentRunId}|${currentRunDir || ""}`;
  const body = document.getElementById("run-monitor-image-preview-body");
  if (!force && key === lastImagePreviewKey && body?._previewBuiltKey === key) return;
  lastImagePreviewKey = key;
  loadRunImagePreview(currentRunId, currentRunDir || "", null, "run-monitor-image-preview");
}

function getRunChatKey() {
  const { currentRunId, currentRunDir } = getRunState();
  return currentRunId || currentRunDir || "";
}

function getRunChatRecord(runKey = getRunChatKey()) {
  const chats = runChatStore.getState().chats || {};
  return chats[runKey] || { messages: [], turns: [] };
}

function setRunChatRecord(runKey, record) {
  if (!runKey) return;
  const chats = { ...(runChatStore.getState().chats || {}) };
  const messages = Array.isArray(record.messages) ? record.messages.slice(-24) : [];
  const turns = Array.isArray(record.turns) ? record.turns.slice(-24) : [];
  chats[runKey] = { messages, turns, updated_at: new Date().toISOString() };
  runChatStore.setState({ chats });
}

function persistRunChatRecordToServer(runKey = getRunChatKey()) {
  const runId = getRunApiKey();
  if (!runId || !runKey) return;
  const record = getRunChatRecord(runKey);
  api.post(API_ENDPOINTS.pi.runChatHistory(runId), {
    run_id: runId,
    history: {
      messages: record.messages || [],
      turns: record.turns || [],
      updated_at: record.updated_at || new Date().toISOString(),
    },
  }).catch(() => {});
}

function appendRunChatTurn(runKey, turn) {
  const record = getRunChatRecord(runKey);
  setRunChatRecord(runKey, {
    messages: monitorRunChatMessages,
    turns: [...(record.turns || []), turn],
  });
  persistRunChatRecordToServer(runKey);
}

function monitorTextItem(item) {
  if (typeof item === "string") return { text: item, evidence: "" };
  if (item && typeof item === "object") {
    return {
      text: item.text || item.message || JSON.stringify(item),
      evidence: item.evidence_ref || item.evidence || "",
    };
  }
  return { text: String(item ?? ""), evidence: "" };
}

function monitorListSection(title, items) {
  const values = Array.isArray(items) ? items.map(monitorTextItem).filter(item => item.text) : [];
  if (!values.length) return null;
  const list = el("div", { class: "tc-run-chat-list" });
  for (const item of values) {
    list.appendChild(el("div", { class: "tc-run-chat-list-item" },
      el("div", { class: "tc-text-sm" }, item.text),
      item.evidence ? el("span", { class: "tc-badge tc-run-chat-evidence" }, item.evidence) : null,
    ));
  }
  return el("div", { class: "tc-run-chat-section" },
    el("div", { class: "tc-label" }, title),
    list,
  );
}

function appendMonitorSection(container, section) {
  if (section) container.appendChild(section);
}

function monitorStringListSection(title, items) {
  const values = Array.isArray(items) ? items.filter(Boolean).map(String) : [];
  if (!values.length) return null;
  const list = el("div", { class: "tc-run-chat-list" });
  for (const item of Array.isArray(items) ? items : []) {
    if (!item) continue;
    list.appendChild(el("div", { class: "tc-run-chat-list-item" }, el("div", { class: "tc-text-sm" }, String(item))));
  }
  return el("div", { class: "tc-run-chat-section" },
    el("div", { class: "tc-label" }, title),
    list,
  );
}

async function applyResumeRecommendation(phase, statusEl = null) {
  if (!phase) return;
  selectPhase(phase, { notify: false });
  await onPhaseSelected(phase, activeLogViewer);
  if (statusEl) statusEl.textContent = t("ui.message.phase_selected", "Phase ausgewählt");
}

function renderResumeRecommendation(result) {
  const rec = result?.resume_recommendation || result?.resume || null;
  const phase = rec?.from_phase || rec?.phase || "";
  if (!phase) return null;
  const status = el("span", { class: "tc-text-success tc-text-sm tc-mt-2" }, "");
  return el("div", { class: "tc-card tc-mt-3 tc-run-chat-resume-rec" },
    el("div", { class: "tc-label" }, t("ui.title.resume_recommendation", "Resume-Empfehlung")),
    el("div", { class: "tc-text-sm tc-mono" }, phase),
    rec.reason ? el("div", { class: "tc-text-sm tc-text-muted tc-mt-1" }, rec.reason) : null,
    rec.execution_note ? el("div", { class: "tc-text-sm tc-text-warning tc-mt-1" }, rec.execution_note) : null,
    el("div", { class: "tc-flex tc-gap-2 tc-items-center tc-mt-2 tc-flex-wrap" },
      el("button", {
        class: "tc-btn tc-btn-sm",
        title: t("ui.tooltip.run_chat_use_resume_phase", "Wählt diese Phase im Resume-Panel aus."),
        onclick: () => applyResumeRecommendation(phase, status),
      }, t("ui.button.use_resume_phase", "Resume ab Phase wählen")),
      status,
    ),
  );
}

function renderMonitorRunChatResult(container, result, opts = {}) {
  if (!opts.append) clear(container);
  if (result?.summary) {
    container.appendChild(el("div", { class: "tc-run-chat-summary tc-text-sm" }, result.summary));
  }

  const hints = result?.context?.problem_hints || [];
  if (Array.isArray(hints) && hints.length) {
    container.appendChild(el("div", { class: "tc-flex tc-gap-2 tc-mt-2 tc-flex-wrap" },
      ...hints.map(h => el("span", { class: "tc-badge", title: h.confidence || "" }, h.label || h.id || "-")),
    ));
  }

  appendMonitorSection(container, monitorStringListSection(t("ui.pi.run_chat.image_observations", "Bildbeobachtungen"), result?.image_observations));
  appendMonitorSection(container, monitorListSection(t("ui.pi.run_chat.likely_causes", "Wahrscheinliche Ursachen"), result?.likely_causes));
  appendMonitorSection(container, monitorListSection(t("ui.pi.run_chat.checks", "Prüfen"), result?.checks));
  appendMonitorSection(container, monitorListSection(t("ui.pi.run_chat.recommendations", "Empfehlungen"), result?.recommendations));
  appendMonitorSection(container, monitorStringListSection(t("ui.pi.run_chat.warnings", "Hinweise"), result?.warnings));

  const resumeRec = renderResumeRecommendation(result);
  if (resumeRec) container.appendChild(resumeRec);

  const actionPlan = monitorRunChatActionPlanWithTextRecommendations(result);
  const actionCount = monitorActionPlanUpdates(actionPlan).length;
  container.appendChild(createMonitorRunChatActionControls(actionPlan, actionCount));

  const evidence = Array.isArray(result?.evidence) ? result.evidence : [];
  if (evidence.length) {
    container.appendChild(el("details", { class: "tc-mt-2" },
      el("summary", { class: "tc-text-sm" }, t("ui.pi.run_chat.evidence", "Evidenz")),
      el("pre", { class: "tc-log-viewer tc-text-sm", style: { maxHeight: "220px" } }, JSON.stringify(evidence, null, 2)),
    ));
  }
}

function monitorParseActionValue(raw) {
  const value = String(raw || "").trim().replace(/^`|`$/g, "");
  if (value === "true") return true;
  if (value === "false") return false;
  if (value === "null") return null;
  if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
    return value.slice(1, -1);
  }
  if (/^-?\d+(?:\.\d+)?$/.test(value)) return Number(value);
  return value;
}

function monitorExtractTextConfigActions(result) {
  const sourceItems = [
    ...(Array.isArray(result?.recommendations) ? result.recommendations : []),
    ...(Array.isArray(result?.checks) ? result.checks : []),
  ];
  const actions = [];
  const seen = new Set();
  const addAction = (path, value, text, evidence, source = "text") => {
    if (!path) return;
    if (source === "directive" && typeof value === "boolean" && !/(^|\.)(enabled|enable|disabled|disable)$|clipping|correction|cleanup|use_|apply_/i.test(path)) {
      return;
    }
    const key = `${path}:${JSON.stringify(value)}`;
    if (seen.has(key)) return;
    seen.add(key);
    actions.push({
      id: `run_chat_${source}_${actions.length + 1}`,
      type: "config.set",
      path,
      value,
      rationale: evidence ? `${text} (${evidence})` : text,
    });
  };

  const exactPattern = /`?([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)`?\s*=\s*`?("([^"\\]|\\.)*"|'([^'\\]|\\.)*'|true|false|null|-?\d+(?:\.\d+)?)`?/g;
  const exampleNumberPattern = /`?([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)`?[^.\n]{0,180}?(?:z\.B\.|zum Beispiel|beispielsweise)[^\d-]{0,30}(-?\d+(?:\.\d+)?)/gi;
  const boolDirectivePattern = /`?([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)`?[^.\n]{0,160}?\b(deaktivieren|abschalten|disable|aktivieren|einschalten|enable)\b/gi;
  for (const item of sourceItems) {
    const { text, evidence } = monitorTextItem(item);
    let match;
    while ((match = exactPattern.exec(text)) !== null) {
      addAction(match[1], monitorParseActionValue(match[2]), text, evidence, "text");
    }
    while ((match = exampleNumberPattern.exec(text)) !== null) {
      addAction(match[1], Number(match[2]), text, evidence, "example");
    }
    while ((match = boolDirectivePattern.exec(text)) !== null) {
      const directive = String(match[2] || "").toLowerCase();
      const value = ["aktivieren", "einschalten", "enable"].includes(directive);
      addAction(match[1], value, text, evidence, "directive");
    }
  }
  return actions;
}

function monitorRunChatActionPlanWithTextRecommendations(result) {
  const plan = result?.action_plan && typeof result.action_plan === "object"
    ? JSON.parse(JSON.stringify(result.action_plan))
    : {
        schema_version: "pi.action-plan.v1",
        source: "pi.run-chat.text-recommendations",
        run_id: result?.run_id || getRunApiKey(),
        mutation_free: true,
        actions: [],
      };
  if (!Array.isArray(plan.actions)) plan.actions = [];
  const existing = new Set(monitorActionPlanUpdates(plan).map(change => `${change.path}:${JSON.stringify(change.value)}`));
  for (const action of monitorExtractTextConfigActions(result)) {
    const key = `${action.path}:${JSON.stringify(action.value)}`;
    if (existing.has(key)) continue;
    plan.actions.push(action);
    existing.add(key);
  }
  return plan;
}

function monitorActionPlanUpdates(plan) {
  const changes = [];
  const actions = Array.isArray(plan?.actions) ? plan.actions : [];
  actions.forEach((action, actionIndex) => {
    if (!action || typeof action !== "object") return;
    if (action.type === "config.set" && typeof action.path === "string") {
      changes.push({
        id: `a${actionIndex}`,
        actionIndex,
        updateIndex: null,
        path: action.path,
        value: action.value,
        current: action.current_value ?? action.current,
        rationale: action.rationale || action.reason || "",
        selected: true,
      });
    } else if (action.type === "config.patch" && Array.isArray(action.updates)) {
      action.updates.forEach((update, updateIndex) => {
        if (!update || typeof update.path !== "string") return;
        changes.push({
          id: `a${actionIndex}_u${updateIndex}`,
          actionIndex,
          updateIndex,
          path: update.path,
          value: update.value,
          current: update.current_value ?? update.current,
          rationale: update.rationale || update.reason || action.rationale || "",
          selected: true,
        });
      });
    }
  });
  return changes;
}

function selectedMonitorActionPlan(plan, changes) {
  const selectedIds = new Set(changes.filter(c => c.selected).map(c => c.id));
  const copy = JSON.parse(JSON.stringify(plan || {}));
  copy.actions = (Array.isArray(copy.actions) ? copy.actions : []).flatMap((action, actionIndex) => {
    if (action?.type === "config.set") {
      return selectedIds.has(`a${actionIndex}`) ? [action] : [];
    }
    if (action?.type === "config.patch" && Array.isArray(action.updates)) {
      const updates = action.updates.filter((_, updateIndex) => selectedIds.has(`a${actionIndex}_u${updateIndex}`));
      return updates.length ? [{ ...action, updates }] : [];
    }
    return [];
  });
  return copy.actions.length ? copy : null;
}

function formatMonitorActionValue(value) {
  if (value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function createMonitorRunChatActionControls(plan, actionCount) {
  const changes = monitorActionPlanUpdates(plan);
  if (!changes.length) {
    return el("div", { class: "tc-run-chat-section tc-text-sm tc-text-muted" },
      el("div", { class: "tc-label" }, t("ui.title.parameter_changes", "Parameteränderungen")),
      el("div", {}, t("ui.pi.run_chat.no_parameter_changes", "Keine konkreten Parameterwerte zum direkten Übernehmen erkannt.")),
    );
  }

  const previewTarget = el("div", { class: "tc-flex-col tc-gap-2 tc-mt-2" });
  const applyButton = el("button", {
    class: "tc-btn tc-btn-sm",
    disabled: true,
    title: t("ui.tooltip.run_chat_apply_resume", "Übernimmt die validierte PI-Preview in den Resume-Config-Editor."),
  }, t("ui.button.apply_to_resume", "In Resume übernehmen"));
  let lastPreview = null;

  const clearPreviewState = () => {
    lastPreview = null;
    applyButton.disabled = true;
    clear(previewTarget);
  };

  const previewSelection = async () => {
    const selectedPlan = selectedMonitorActionPlan(plan, changes);
    if (!selectedPlan) {
      toastError(t("ui.toast.preview_failed", "Preview fehlgeschlagen"), t("ui.state.no_selection", "Keine Empfehlung ausgewählt."));
      return null;
    }
    try {
      const yaml = await getMonitorResumeYaml();
      const preview = await previewMonitorRunChatActionPlan(selectedPlan, previewTarget, yaml);
      lastPreview = preview ? { plan: selectedPlan, preview } : null;
      applyButton.disabled = !lastPreview?.preview?.config_valid;
      return lastPreview;
    } catch (e) {
      clearPreviewState();
      clear(previewTarget);
      previewTarget.appendChild(el("div", { class: "tc-text-error tc-text-sm" }, e.message));
      toastError(t("ui.toast.preview_failed", "Preview fehlgeschlagen"), e.message);
      return null;
    }
  };

  applyButton.onclick = async () => {
    const state = lastPreview || await previewSelection();
    const patchedYaml = state?.preview?.patched_yaml || "";
    if (!state?.preview?.config_valid || !patchedYaml) {
      toastError(t("ui.toast.apply_failed", "Anwenden fehlgeschlagen"), t("ui.state.preview_required", "Erst PI Preview ausführen."));
      return;
    }
    const editor = document.getElementById("resume-config-yaml");
    if (!editor) {
      toastError(t("ui.toast.apply_failed", "Anwenden fehlgeschlagen"), t("ui.error.no_config", "Config YAML ist leer"));
      return;
    }
    editor.value = patchedYaml;
    updateResumeConfigSectionHighlights(patchedYaml);
    const panel = document.getElementById("resume-panel");
    if (panel) panel.style.display = "";
    const hint = document.getElementById("resume-hint");
    if (hint) hint.textContent = t("ui.message.resume_ready", "Bereit zum Resume");
    toastSuccess(t("ui.toast.applied_to_resume", "In Resume übernommen"));
  };

  const items = changes.map((change) => el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
    el("label", { class: "tc-checkbox" },
      el("input", {
        type: "checkbox",
        checked: change.selected,
        title: t("ui.tooltip.run_chat_change_select", "Legt fest, ob diese Parameteränderung in die PI-Preview eingeht."),
        onchange: (e) => {
          change.selected = e.target.checked;
          clearPreviewState();
        },
      }),
      el("span", { class: "tc-mono tc-text-sm" }, change.path),
    ),
    change.current !== undefined ? el("div", { class: "tc-mt-2 tc-text-sm" },
      el("span", { class: "tc-text-muted" }, t("ui.label.current", "Aktuell") + ": "),
      el("span", {}, formatMonitorActionValue(change.current)),
    ) : null,
    el("div", { class: "tc-text-sm" },
      el("span", { class: "tc-text-muted" }, t("ui.label.recommended", "Empfohlen") + ": "),
      el("span", {}, formatMonitorActionValue(change.value)),
    ),
    change.rationale ? el("div", { class: "tc-mt-1 tc-text-sm tc-text-muted" }, change.rationale) : null,
  ));

  return el("div", { class: "tc-mt-3" },
    el("div", { class: "tc-label" }, t("ui.title.parameter_changes", "Parameteränderungen")),
    el("div", { class: "tc-text-sm tc-text-muted tc-mb-2" },
      t("ui.pi.run_chat.action_plan", "{count} optionale PI-Action-Plan-Schritte erzeugt.", { count: changes.length || actionCount }),
    ),
    el("div", { class: "tc-flex-col tc-gap-2" }, ...items),
    el("div", { class: "tc-flex tc-gap-2 tc-mt-2" },
      el("button", {
        class: "tc-btn tc-btn-sm",
        title: t("ui.tooltip.run_chat_preview", "Validiert den Action-Plan und zeigt den YAML-Diff ohne Speichern."),
        onclick: () => previewSelection(),
      }, t("ui.button.pi_preview", "PI Preview")),
      applyButton,
    ),
    previewTarget,
  );
}

async function getMonitorResumeYaml() {
  const editor = document.getElementById("resume-config-yaml");
  const yaml = String(editor?.value || "").trim();
  if (yaml) return yaml;
  const runId = getRunApiKey();
  if (!runId) throw new Error(t("ui.error.no_run", "Kein Run ausgewählt"));
  const resp = await api.get(API_ENDPOINTS.runs.config(runId));
  const loaded = resp?.config_yaml || resp?.config || "";
  if (!loaded) throw new Error(t("ui.error.no_config", "Config YAML ist leer"));
  if (editor) {
    editor.value = loaded;
    updateResumeConfigSectionHighlights(loaded);
  }
  return loaded;
}

async function previewMonitorRunChatActionPlan(plan, container, yaml = "") {
  if (!plan || !container) return;
  clear(container);
  container.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.loading", "Lädt...")));
  try {
    const result = await api.post(API_ENDPOINTS.pi.actionPlanPreview, yaml ? { plan, yaml } : { plan });
    const preview = result?.preview || result;
    clear(container);
    const valid = preview?.config_valid === true
      ? t("ui.state.config_valid", "Config gültig")
      : t("ui.state.config_invalid", "Config ungültig");
    container.appendChild(el("div", { class: preview?.config_valid === true ? "tc-text-success tc-text-sm" : "tc-text-error tc-text-sm" }, valid));
    container.appendChild(createYamlDiff(preview?.base_yaml || "", preview?.patched_yaml || ""));
    return preview;
  } catch (e) {
    clear(container);
    container.appendChild(el("div", { class: "tc-text-error tc-text-sm" }, e.message));
    throw e;
  }
}

function createRunChatPanel() {
  const inputId = "run-monitor-chat-input";
  const outputId = "run-monitor-chat-output";
  const trafficId = "run-monitor-chat-traffic";
  const placeholder = t(
    "ui.placeholder.run_chat",
    "z.B. Sterne oben haben schwarzen Kern, der Nebel oben wird beschnitten und ist kaum sichtbar. Was kann man tun?",
  );
  return el("div", { class: "tc-card", id: "run-monitor-chat-card" },
    el("div", { class: "tc-card-title tc-flex tc-items-center tc-justify-between" },
      el("span", {}, t("ui.title.run_chat", "Run-Chat")),
      el("button", {
        class: "tc-btn tc-btn-sm",
        title: t("ui.tooltip.run_chat_clear", "Leert den lokalen Chat-Verlauf."),
        onclick: () => clearMonitorRunChat(outputId),
      }, t("ui.button.clear", "Zurücksetzen")),
    ),
    el("textarea", {
      class: "tc-input",
      id: inputId,
      rows: 3,
      placeholder,
      title: t("ui.tooltip.run_chat_input", "Beschreibe sichtbare Bildprobleme in normaler Sprache."),
    }),
    el("div", { class: "tc-flex tc-gap-2 tc-mt-2" },
      el("button", {
        class: "tc-btn tc-btn-sm",
        title: t("ui.tooltip.run_chat_send", "Analysiert deine Beschreibung mit Run-Report, Artefakten und PI Memories."),
        onclick: () => submitMonitorRunChat(inputId, outputId),
      }, t("ui.button.ask_pi", "PI fragen")),
    ),
    el("div", { class: "tc-flex-col tc-gap-3 tc-mt-2", id: outputId },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.run_chat_empty", "Noch keine Frage gestellt.")),
    ),
    el("div", { class: "tc-mt-3" },
      el("div", { class: "tc-flex tc-items-center tc-justify-between tc-gap-2" },
        el("div", { class: "tc-label" }, t("ui.title.ai_traffic", "KI-Datenverkehr")),
        el("div", { class: "tc-flex tc-items-center tc-gap-2" },
          el("span", { class: "tc-text-muted tc-text-sm", id: `${trafficId}-status` }, t("ui.state.not_loaded", "nicht geladen")),
          el("button", {
            class: "tc-btn tc-btn-sm",
            title: t("ui.tooltip.ai.refresh_traffic", "Lädt den persistenten PI/KI-Traffic-Log aus dem Sidecar."),
            onclick: () => loadRunChatTrafficLog(),
          }, t("ui.button.refresh", "Aktualisieren")),
        ),
      ),
      el("div", { class: "tc-log-viewer tc-mt-2", id: trafficId, style: { maxHeight: "220px" } },
        el("div", { class: "tc-text-muted" }, t("ui.state.no_traffic", "Keine Daten")),
      ),
    ),
  );
}

function createMonitorRunChatAnswer(result, { collapsed = false, loading = false } = {}) {
  if (collapsed) {
    const details = el("details", { class: "tc-run-chat-answer tc-run-chat-answer-collapsible" },
      el("summary", { class: "tc-run-chat-answer-summary" },
        el("span", { class: "tc-run-chat-role" }, t("ui.label.answer", "Antwort")),
        result?.summary ? el("span", { class: "tc-run-chat-answer-summary-text" }, result.summary) : null,
      ),
    );
    if (result) renderMonitorRunChatResult(details, result, { append: true });
    return details;
  }

  const answer = el("div", { class: "tc-run-chat-answer" },
    el("div", { class: "tc-run-chat-role" }, t("ui.label.answer", "Antwort")),
  );
  if (loading) {
    answer.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.loading", "Lädt...")));
  } else if (result) {
    renderMonitorRunChatResult(answer, result, { append: true });
  }
  return answer;
}

function collapseExistingMonitorRunChatAnswers(output) {
  if (!output) return;
  for (const answer of output.querySelectorAll(".tc-run-chat-answer-collapsible")) {
    answer.open = false;
  }
  for (const answer of output.querySelectorAll(".tc-run-chat-answer:not(.tc-run-chat-answer-collapsible)")) {
    const turn = answer.closest(".tc-run-chat-turn");
    const summaryText = answer.querySelector(".tc-run-chat-summary")?.textContent || "";
    const details = el("details", { class: "tc-run-chat-answer tc-run-chat-answer-collapsible" },
      el("summary", { class: "tc-run-chat-answer-summary" },
        el("span", { class: "tc-run-chat-role" }, t("ui.label.answer", "Antwort")),
        summaryText ? el("span", { class: "tc-run-chat-answer-summary-text" }, summaryText) : null,
      ),
    );
    while (answer.firstChild) {
      const child = answer.firstChild;
      if (child.classList?.contains("tc-run-chat-role")) {
        answer.removeChild(child);
      } else {
        details.appendChild(child);
      }
    }
    if (turn) turn.replaceChild(details, answer);
  }
}

function renderMonitorRunChatTurn(output, turn, { collapsed = false } = {}) {
  const card = el("div", { class: "tc-run-chat-turn" },
    el("div", { class: "tc-run-chat-question" },
      el("div", { class: "tc-run-chat-role" }, t("ui.label.question", "Frage")),
      el("div", { class: "tc-text-sm tc-run-chat-question-text" }, turn?.message || ""),
    ),
  );
  if (turn?.result) {
    card.appendChild(createMonitorRunChatAnswer(turn.result, { collapsed }));
  } else if (turn?.error) {
    card.appendChild(el("div", { class: "tc-text-error tc-text-sm tc-mt-2" }, turn.error));
  }
  output.appendChild(card);
}

function restoreMonitorRunChat(outputId = "run-monitor-chat-output") {
  const output = document.getElementById(outputId);
  if (!output) return;
  const runKey = getRunChatKey();
  const record = getRunChatRecord(runKey);
  monitorRunChatMessages = Array.isArray(record.messages) ? [...record.messages] : [];
  clear(output);
  const turns = Array.isArray(record.turns) ? record.turns : [];
  if (!runKey || !turns.length) {
    output.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.run_chat_empty", "Noch keine Frage gestellt.")));
    return;
  }
  [...turns].reverse().forEach((turn, index) => {
    renderMonitorRunChatTurn(output, turn, { collapsed: index > 0 });
  });
}

async function restoreMonitorRunChatFromServer(outputId = "run-monitor-chat-output") {
  const runKey = getRunChatKey();
  const runId = getRunApiKey();
  if (!runKey || !runId) return;
  try {
    const history = await api.get(API_ENDPOINTS.pi.runChatHistory(runId));
    const serverTurns = Array.isArray(history?.turns) ? history.turns : [];
    const localTurns = Array.isArray(getRunChatRecord(runKey).turns) ? getRunChatRecord(runKey).turns : [];
    if (serverTurns.length >= localTurns.length) {
      setRunChatRecord(runKey, {
        messages: Array.isArray(history?.messages) ? history.messages : [],
        turns: serverTurns,
      });
      restoreMonitorRunChat(outputId);
    }
  } catch {}
}

function restoreMonitorRunChatAll(outputId = "run-monitor-chat-output") {
  restoreMonitorRunChat(outputId);
  restoreMonitorRunChatFromServer(outputId);
}

function clearMonitorRunChat(outputId = "run-monitor-chat-output") {
  monitorRunChatMessages = [];
  const runKey = getRunChatKey();
  if (runKey) {
    const chats = { ...(runChatStore.getState().chats || {}) };
    delete chats[runKey];
    runChatStore.setState({ chats });
    persistRunChatRecordToServer(runKey);
  }
  const output = document.getElementById(outputId);
  if (output) {
    clear(output);
    output.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.run_chat_empty", "Noch keine Frage gestellt.")));
  }
}

function renderRunChatTrafficLog(items) {
  const container = document.getElementById("run-monitor-chat-traffic");
  if (!container) return;
  clear(container);
  if (!Array.isArray(items) || items.length === 0) {
    container.appendChild(el("div", { class: "tc-text-muted" }, t("ui.state.no_traffic", "Keine Daten")));
    return;
  }
  for (const line of items.slice(-80)) {
    container.appendChild(el("div", { class: "tc-text-sm tc-mono" }, String(line)));
  }
  container.scrollTop = container.scrollHeight;
}

async function loadRunChatTrafficLog() {
  const status = document.getElementById("run-monitor-chat-traffic-status");
  if (status) status.textContent = t("ui.state.loading", "Lädt...");
  try {
    const payload = await api.get(API_ENDPOINTS.ai.traffic(500));
    const items = Array.isArray(payload?.items) ? payload.items : [];
    renderRunChatTrafficLog(items);
    if (status) {
      const enabled = payload?.enabled === false ? t("ui.state.disabled", "deaktiviert") : t("ui.state.enabled", "aktiv");
      status.textContent = `${enabled} · ${t("ui.pi.traffic_count", "{count} Zeilen", { count: payload?.count || items.length })}`;
    }
  } catch (e) {
    if (status) status.textContent = e.message;
  }
}

function startRunChatTrafficPolling() {
  if (runChatTrafficTimer) clearInterval(runChatTrafficTimer);
  loadRunChatTrafficLog();
  runChatTrafficTimer = setInterval(() => loadRunChatTrafficLog(), 1500);
}

function stopRunChatTrafficPolling() {
  if (!runChatTrafficTimer) return;
  clearInterval(runChatTrafficTimer);
  runChatTrafficTimer = null;
  loadRunChatTrafficLog();
}

async function submitMonitorRunChat(inputId, outputId) {
  const runKey = getRunChatKey();
  const runId = getRunApiKey();
  if (!runId) {
    toastError(t("ui.toast.run_chat_failed", "Run-Chat fehlgeschlagen"), t("ui.error.no_run", "Kein Run ausgewählt"));
    return;
  }
  const input = document.getElementById(inputId);
  const output = document.getElementById(outputId);
  const message = String(input?.value || "").trim();
  if (!message) {
    toast(t("ui.toast.run_chat_empty", "Bitte erst ein Problem beschreiben."), "", "info");
    return;
  }
  if (!monitorRunChatMessages.length && output) {
    clear(output);
  }
  collapseExistingMonitorRunChatAnswers(output);
  monitorRunChatMessages.push({ role: "user", content: message });
  const turn = output ? el("div", { class: "tc-run-chat-turn" },
    el("div", { class: "tc-run-chat-question" },
      el("div", { class: "tc-run-chat-role" }, t("ui.label.question", "Frage")),
      el("div", { class: "tc-text-sm tc-run-chat-question-text" }, message),
    ),
    createMonitorRunChatAnswer(null, { loading: true }),
  ) : null;
  if (output && turn) output.prepend(turn);
  startRunChatTrafficPolling();
  try {
    const result = await api.post(API_ENDPOINTS.pi.runChat, {
      run_id: runId,
      message,
      messages: monitorRunChatMessages.slice(-12),
      object_name: getStore("input-scan", { scanData: {} }).getState().scanData?.object_name || "",
    });
    monitorRunChatMessages.push({ role: "assistant", content: result?.summary || "", result });
    appendRunChatTurn(runKey, { message, result, created_at: new Date().toISOString() });
    if (input) input.value = "";
    if (turn) {
      clear(turn);
      turn.appendChild(el("div", { class: "tc-run-chat-question" },
        el("div", { class: "tc-run-chat-role" }, t("ui.label.question", "Frage")),
        el("div", { class: "tc-text-sm tc-run-chat-question-text" }, message),
      ));
      turn.appendChild(createMonitorRunChatAnswer(result));
    } else if (output) {
      renderMonitorRunChatResult(output, result);
    }
  } catch (e) {
    appendRunChatTurn(runKey, { message, error: e.message, created_at: new Date().toISOString() });
    if (turn) turn.appendChild(el("div", { class: "tc-text-error tc-text-sm" }, e.message));
    else if (output) output.appendChild(el("div", { class: "tc-text-error tc-text-sm" }, e.message));
    toastError(t("ui.toast.run_chat_failed", "Run-Chat fehlgeschlagen"), e.message);
  } finally {
    stopRunChatTrafficPolling();
  }
}

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
  startLogTailPolling(runId, getRunState().currentRunDir || "");
  pollTimer = setInterval(async () => {
    const status = await api.get(API_ENDPOINTS.runs.status(runId)).catch(() => null);
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
  stopLogTailPolling();
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

function resetRunMonitorNoRun() {
  stopPolling();
  disconnectWebSocket();
  logTailSnapshot = null;
  clearSelectedPhase();
  const neutralPhases = getPhasesForConfig(getConfigState().draft)
    .map(p => ({ ...p, status: "pending", pct: 0 }));
  setPhaseList(neutralPhases);
  setRunState({
    currentRunId: null,
    currentRunDir: null,
    status: null,
    phases: [],
    resumeActive: false,
    resumePending: false,
    resumeFromPhase: "",
  });
  setRunButtonsActive(false);
  for (const [id, value] of [
    ["stat-run-id", "\u2014"],
    ["stat-status", "\u2014"],
    ["stat-elapsed", "\u2014"],
    ["stat-frames", "\u2014"],
    ["info-run-id", "\u2014"],
    ["info-run-dir", "\u2014"],
    ["info-run-name", "\u2014"],
    ["info-status", "\u2014"],
    ["info-pipeline", "\u2014"],
  ]) {
    updateStat(id, value);
  }
  restoreMonitorRunChat();
  if (activeLogViewer) activeLogViewer.clearLines();
  const preview = document.getElementById("run-monitor-image-preview");
  if (preview) {
    const status = preview.querySelector("[data-preview-status]");
    if (status) status.textContent = t("ui.state.no_run_selected", "Kein Run ausgewählt");
  }
}

async function restoreCurrentRun() {
  try {
    const appState = await api.get(API_ENDPOINTS.app.state);
    const storedRun = getRunState();
    const backendCurrent = appState?.run?.current;
    const current = backendCurrent?.run_id
      ? backendCurrent
      : (storedRun.currentRunId ? {
          run_id: storedRun.currentRunId,
          run_dir: storedRun.currentRunDir || "",
          status: storedRun.status || storedRun.runStatus || "unknown",
        } : null);
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
      const runDir = current.run_dir || storedRun.currentRunDir || "";
      const storedResumeForCurrent = sameRunIdentity(
        storedRun.currentRunId,
        current.run_id,
        storedRun.currentRunDir || "",
        runDir || "",
      );
      const allowStoredResumeState =
        storedResumeForCurrent && !isRunTerminalStatus(current.status);
      const isRunning =
        backendRunning ||
        (allowStoredResumeState && (getResumeActive() || getResumePending()));
      if (!allowStoredResumeState && (getResumeActive() || getResumePending())) {
        setRunState({ resumeActive: false, resumePending: false, resumeFromPhase: "" });
      }
      setRunState({ currentRunId: current.run_id, currentRunDir: runDir || null, status: current.status || "running" });
      restoreMonitorRunChatAll();
      setRunButtonsActive(isRunning);
      updateStat("stat-run-id", current.run_id);
      updateStat("stat-status", isRunning ? "running" : (current.status || "running"));
      updateStat("info-run-id", current.run_id);
      updateStat("info-status", isRunning ? "running" : (current.status || "running"));
      if (current.run_dir) updateStat("info-run-dir", current.run_dir);
      const runName = current.run_id.replace(/_\d{4}-\d{2}-\d{2}.*$/, "");
      updateStat("info-run-name", runName);
      await refreshRunStatus(current.run_id);
      refreshCurrentImagePreview();
      setRunButtonsActive(isRunning);
      // Load existing logs from REST endpoint
      await loadInitialLogs(current.run_id, activeLogViewer, activeWarningBanner, runDir || getRunState().currentRunDir || "");
      if (isRunning) {
        connectWebSocket(current.run_id, getResumeActive() || getResumePending(), runDir || getRunState().currentRunDir || "");
        startPolling(current.run_id);
      } else {
        enableStatsButtons(current.run_id);
      }
    } else {
      resetRunMonitorNoRun();
    }
  } catch (e) {
    console.error("Run monitor restore failed:", e);
  }
}

function normalizeLogLine(line) {
  return typeof line === "string" ? line : JSON.stringify(line);
}

function formatEventTimestamp(value) {
  const raw = String(value || "");
  const isoMatch = raw.match(/T(\d{2}:\d{2}:\d{2})/);
  if (isoMatch) return isoMatch[1];
  const timeMatch = raw.match(/^(\d{2}:\d{2}:\d{2})/);
  return timeMatch ? timeMatch[1] : (raw || formatTime());
}

function normalizedEventPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return numeric >= 0 && numeric <= 1 ? numeric * 100 : numeric;
}

function eventDedupeKey(event) {
  const payload = event?.payload || event || {};
  const values = [
    event?.run_id || payload.run_id || "",
    event?.ts || payload.ts || "",
    event?.type || payload.type || "",
    event?.phase_name || event?.phase || payload.phase_name || payload.phase || "",
    event?.current ?? payload.current ?? "",
    event?.total ?? payload.total ?? "",
    event?.pass || payload.pass || "",
    event?.substep || payload.substep || "",
    event?.status || payload.status || "",
  ];
  return `event:${values.map(value => String(value)).join("|")}`;
}

function addStructuredLogLine(logViewer, event, level, text) {
  if (!logViewer) return;
  const payload = event?.payload || event || {};
  const timestamp = formatEventTimestamp(event?.ts || payload.ts);
  logViewer.addLine(timestamp, level, text, eventDedupeKey(event));
}

function findLogTailStart(previousLines, currentLines) {
  if (!previousLines || previousLines.length === 0) return 0;
  const samePrefix = previousLines.length <= currentLines.length &&
    previousLines.every((line, index) => line === currentLines[index]);
  if (samePrefix) return previousLines.length;

  const maxOverlap = Math.min(previousLines.length, currentLines.length);
  for (let overlap = maxOverlap; overlap > 0; overlap--) {
    let matches = true;
    for (let index = 0; index < overlap; index++) {
      if (previousLines[previousLines.length - overlap + index] !== currentLines[index]) {
        matches = false;
        break;
      }
    }
    if (matches) return overlap;
  }
  return 0;
}

async function loadInitialLogs(runId, logViewer, warningBanner, runDir = "") {
  try {
    const logs = await api.get(API_ENDPOINTS.runs.logs(runId, 250, runDir));
    if (!logs || !Array.isArray(logs.lines)) return;
    const currentLines = logs.lines.map(normalizeLogLine);
    const sameRun = logTailSnapshot &&
      logTailSnapshot.runId === runId &&
      logTailSnapshot.runDir === runDir;
    const firstNewLine = sameRun
      ? findLogTailStart(logTailSnapshot.lines, currentLines)
      : 0;
    logTailSnapshot = { runId, runDir, lines: currentLines };

    for (const line of logs.lines.slice(firstNewLine)) {
      let ts = formatTime();
      let level = "INFO";
      let text = "";
      let evType = "";
      let parsedEvent = null;
      if (typeof line === "string") {
        try {
          const ev = JSON.parse(line);
          parsedEvent = ev;
          ts = formatEventTimestamp(ev.ts);
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
        if (parsedEvent) addStructuredLogLine(logViewer, parsedEvent, level, text);
        else logViewer.addLine(formatEventTimestamp(ts), level, text);
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
      if ((getResumePending() || getResumeActive()) && parsedEvent && shouldReplayResumeLogEvent(parsedEvent)) {
        const key = typeof line === "string" ? line : JSON.stringify(parsedEvent);
        if (!replayedRunLogEvents.has(key)) {
          replayedRunLogEvents.add(key);
          handleWsMessage(parsedEvent, activeLogViewer || logViewer, null, activeWarningBanner || warningBanner);
        }
      }
    }
  } catch {}
}

function shouldReplayResumeLogEvent(ev) {
  const type = ev?.type || "";
  return type === "resume_start" ||
    type === "resume_end" ||
    type === "phase_start" ||
    type === "phase_progress" ||
    type === "phase_end" ||
    type === "warning" ||
    type === "error";
}

function startLogTailPolling(runId, runDir = "") {
  stopLogTailPolling();
  if (!runId || !activeLogViewer) return;
  const refresh = () => loadInitialLogs(runId, activeLogViewer, null, runDir || getRunState().currentRunDir || "");
  refresh();
  logTailTimer = setInterval(refresh, 5000);
}

function stopLogTailPolling() {
  if (!logTailTimer) return;
  clearInterval(logTailTimer);
  logTailTimer = null;
}

const PHASE_I18N_KEYS = {
  AQMH_MAPS: "phase.aqmh_maps",
  AQMH_GLOBAL_QUALITY: "phase.aqmh_global_quality",
  AQMH_RECONSTRUCTION: "phase.aqmh_reconstruction",
  AQMH_DIAGNOSTICS: "phase.aqmh_diagnostics",
};

function localizedPhaseName(value) {
  const raw = String(value || "");
  const key = PHASE_I18N_KEYS[raw];
  return key ? t(key, raw) : raw;
}

function localizedAqmhSubstep(pass, rawSubstep) {
  const substep = String(rawSubstep || "");
  const key = `monitor.log.aqmh.${pass || ""}`;
  const params = {};
  const rowMatch = substep.match(/(\d+)\/(\d+)$/);
  if (rowMatch) {
    params.current = rowMatch[1];
    params.total = rowMatch[2];
  }
  const alphaMatch = substep.match(/alpha=([0-9.eE+-]+)/);
  if (alphaMatch) params.alpha = alphaMatch[1];
  const iterationMatch = substep.match(/(?:Schritt|step)\s+(\d+)\/4/i);
  if (iterationMatch) params.iteration = iterationMatch[1];
  const translated = t(key, "", params);
  return translated || substep;
}

function formatEventMessage(ev) {
  const type = ev.type || "";
  const payload = ev.payload || ev;
  const phase = localizedPhaseName(ev.phase_name || ev.phase || payload.phase_name || payload.phase);
  if (type === "phase_start") return `${phase} | ${t("monitor.log.start", "start")}`;
  if (type === "phase_progress") {
    const pctValue = normalizedEventPercent(ev.pct ?? payload.pct ?? ev.progress ?? payload.progress);
    const pct = pctValue != null ? ` (${Math.round(pctValue)}%)` : "";
    const substep = payload.substep || ev.substep || "";
    const pass = payload.pass || ev.pass || "";
    const detail = pass.startsWith("core_") || pass.startsWith("rgb_") || pass
      ? localizedAqmhSubstep(pass, substep)
      : substep;
    return `${phase} | ${t("monitor.log.progress", "progress")}${pct}${detail ? ` | ${detail}` : ""}`;
  }
  if (type === "phase_end") {
    const status = payload.status || ev.status || "ok";
    const reason = payload.reason || ev.reason ? ` (${payload.reason || ev.reason})` : "";
    return `${phase} | ${t(`monitor.log.status.${status}`, status)}${reason}`;
  }
  if (type === "run_start") return t("monitor.log.run_started", "Run started");
  if (type === "run_end") return `${t("monitor.log.run_finished", "Run finished")} | ${t(`monitor.log.status.${payload.status || ev.status || "ok"}`, payload.status || ev.status || "ok")}`;
  if (type === "resume_start") {
    const fromPhase = localizedPhaseName(payload.from_phase || ev.from_phase);
    return `${t("monitor.log.resume", "Resume")} | ${t("monitor.log.start", "start")} | ${fromPhase}`;
  }
  if (type === "resume_end") {
    const ok = payload.success ?? ev.success ?? false;
    const fromPhase = localizedPhaseName(payload.from_phase || ev.from_phase);
    return `${t("monitor.log.resume", "Resume")} | ${ok ? t("monitor.log.ok", "OK") : t("monitor.log.error", "ERROR")} | ${fromPhase}`;
  }
  if (type === "queue_progress") return payload.message || ev.message || t("monitor.log.queue_progress", "Queue progress");
  if (ev.message) return ev.message;
  return type || JSON.stringify(ev).slice(0, 200);
}

function getEventPhaseName(data, payload) {
  const candidates = [
    data?.phase_name,
    payload?.phase_name,
    data?.phase,
    payload?.phase,
  ];
  return candidates.find((value) =>
    typeof value === "string" && value.length > 0 && value !== "null"
  ) || "";
}

async function refreshRunStatus(runId) {
  try {
    const status = await api.get(API_ENDPOINTS.runs.status(runId));
    if (!status) return;

    // While a resume is only queued, the REST endpoint may still expose the
    // old terminal status. Once it reports running, its phase state is the
    // authoritative source and must restore the UI after a page refresh.
    if ((getResumePending() || getResumeActive()) && status.status !== "running") {
      if (status.run_dir) setRunState({ currentRunDir: status.run_dir });
      return;
    }

    if (status.phases && Array.isArray(status.phases)) {
      // Merge backend statuses into the correct phase order for the run method.
      // This prevents backend-specific or out-of-order phases (e.g. GLOBAL_METRICS
      // for AQMH) from appearing at the bottom of the list.
      const method = status.method || (status.aqmh_enabled ? "aqmh" : "classic_tile_compile");
      const basePhases = getPhasesForConfig({ method, aqmh: { enabled: method === "aqmh" } });
      const statusMap = new Map();
      for (const p of status.phases) {
        const name = p.phase || p.phase_name || "";
        if (name) statusMap.set(name, p);
      }
      const mergedPhases = basePhases.map(p => {
        const s = statusMap.get(p.phase);
        return {
          phase: p.phase,
          label: p.label,
          status: s?.status || "pending",
          pct: s?.pct ?? s?.progress ?? 0,
        };
      });
      setPhaseList(mergedPhases);
      setRunState({ phases: mergedPhases });
    }

    updateStat("info-run-id", status.run_id || runId);
    updateStat("info-run-dir", status.run_dir || "\u2014");
    updateStat("info-status", status.status || "\u2014");
    updateStat("info-color-mode", status.color_mode || "\u2014");
    updateStat("info-pipeline", status.method || (status.aqmh_enabled ? "AQMH" : "Classic") || "\u2014");

    if (status.run_dir) {
      setRunState({ currentRunDir: status.run_dir });
      updateStat("info-output-dir", status.run_dir + "/outputs");
      refreshCurrentImagePreview();
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
      object_name: sd.object_name || "",
      target: sd.object_name || "",
      color_mode: sd.color_mode || "",
      queue: queue.length > 0 ? queue : undefined,
      config_yaml: configYaml || undefined,
    };

    toast(t("ui.toast.run_starting", "Run wird gestartet..."), "", "info");
    setAiState({ currentAnalysis: null });
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
      replayedRunLogEvents = new Set();
      const newPhases = getPhasesForConfig(getConfigState().draft).map(p => ({ phase: p.phase, status: "pending", pct: 0, label: p.label }));
      setRunState({ currentRunId: runId, currentRunDir: result?.run_dir || getRunState().currentRunDir || null, status: "running", phases: newPhases, resumeActive: false, resumePending: false, resumeFromPhase: "" });
      restoreMonitorRunChatAll();
      setRunButtonsActive(true);
      updateStat("stat-run-id", runId);
      updateStat("stat-status", "running");
      setPhaseList(newPhases);
      connectWebSocket(runId, false, result?.run_dir || getRunState().currentRunDir || "");
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
    setRunState({ status: "stopped", resumeFromPhase: "" });
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
    return false;
  }
  const phase = getSelectedPhase();
  if (!phase) {
    toastError(t("ui.toast.resume_failed", "Resume fehlgeschlagen"), t("ui.error.no_phase", "Bitte eine Phase anklicken um Resume zu starten"));
    return false;
  }
  const configYaml = document.getElementById("resume-config-yaml")?.value || "";
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
    if (activeLogViewer) activeLogViewer.addLine(formatTime(), "INFO", `Resume | queued | ${phase}`);
    const result = await api.post(API_ENDPOINTS.runs.resume(currentRunId), payload);
    const jobId = result?.job_id;
    replayedRunLogEvents = new Set();
    setResumePending(true);
    setResumeFromPhase(phase);
    if (resumePendingTimer) clearTimeout(resumePendingTimer);
    resumePendingTimer = setTimeout(() => {
      if (getResumePending() && activeLogViewer) {
        activeLogViewer.addLine(formatTime(), "WARN", `Resume | waiting for live events | ${phase}`);
      }
    }, RESUME_PENDING_TIMEOUT_MS);
    setRunState({ status: "running" });
    setRunButtonsActive(true);
    updateStat("stat-status", "running");
    updateStat("info-status", `running — ${phase}`);
    const newPhases = resetPhasesForResume(phase);
    if (newPhases.length > 0) setRunState({ phases: newPhases });
    connectWebSocket(currentRunId, true, currentRunDir || "");
    startPolling(currentRunId);
    activateRunMonitorTab("log");
    toastSuccess(t("ui.toast.run_resumed", "Run fortgesetzt"), `${phase}`);
    return true;
  } catch (e) {
    const formatted = formatResumeError(e, phase);
    toastError(t("ui.toast.resume_failed", "Resume fehlgeschlagen"), formatted.body || formatted.title);
    return false;
  }
}

async function checkResumeFeasibility(phase) {
  const seq = ++resumeFeasibilitySeq;
  const { currentRunId, currentRunDir, status } = getRunState();
  const hint = document.getElementById("resume-hint");
  const resumeExecBtn = document.getElementById("resume-execute-btn");
  if (!phase || !currentRunId) return false;

  if (resumeExecBtn) resumeExecBtn.disabled = true;
  if (hint) hint.textContent = t("ui.message.resume_checking", "Prüfe Resume-Möglichkeit...");

  try {
    const payload = { from_phase: phase, dry_run: true };
    if (currentRunDir) payload.run_dir = currentRunDir;
    const editor = document.getElementById("resume-config-yaml");
    if (editor?.value?.trim()) payload.config_yaml = editor.value;
    const revSelect = document.getElementById("resume-config-revision");
    if (revSelect?.value) payload.config_revision_id = revSelect.value;

    await api.post(API_ENDPOINTS.runs.resume(currentRunId), payload);
    if (seq !== resumeFeasibilitySeq || getSelectedPhase() !== phase) return false;
    if (hint) hint.textContent = t("ui.message.resume_feasible", "Resume ab {phase} ist möglich.", { phase });
    if (resumeExecBtn) resumeExecBtn.disabled = status === "running";
    return true;
  } catch (e) {
    if (seq !== resumeFeasibilitySeq || getSelectedPhase() !== phase) return false;
    const formatted = formatResumeError(e, phase);
    if (hint) hint.textContent = formatted.body || formatted.title;
    if (resumeExecBtn) resumeExecBtn.disabled = true;
    toastError(formatted.title, formatted.body || e.message);
    return false;
  }
}

async function onPhaseSelected(phase, logViewer) {
  activateRunMonitorTab("resume");
  const panel = document.getElementById("resume-panel");
  const badge = document.getElementById("resume-phase-badge");
  const hint = document.getElementById("resume-hint");
  const hmsButton = document.getElementById("resume-hms-config-btn");
  const bgeButton = document.getElementById("resume-bge-config-btn");
  const resumeExecBtn = document.getElementById("resume-execute-btn");
  if (!phase) {
    ++resumeFeasibilitySeq;
    if (panel) panel.style.display = "none";
    if (hmsButton) hmsButton.style.display = "none";
    if (bgeButton) bgeButton.style.display = "none";
    if (resumeExecBtn) resumeExecBtn.disabled = getRunState().status === "running";
    return;
  }
  if (panel) panel.style.display = "";
  if (badge) badge.textContent = phase;
  if (hmsButton) hmsButton.style.display = phase === "HYPERMETRIC_STRETCH" ? "" : "none";
  if (bgeButton) bgeButton.style.display = phase === "BGE" ? "" : "none";
  if (hint) hint.textContent = t("ui.message.resume_hint", "Config wird geladen...");
  if (resumeExecBtn) resumeExecBtn.disabled = true;
  await loadRunConfig(phase);
  await checkResumeFeasibility(phase);
  loadConfigRevisions();
}

function openSelectedHmsPreview() {
  const { currentRunId, currentRunDir } = getRunState();
  const editor = document.getElementById("resume-config-yaml");
  if (!currentRunId || !editor?.value.trim()) {
    toastError(t("ui.toast.hms_preview_failed", "HMS-Vorschau fehlgeschlagen"), t("ui.error.no_config", "Config YAML ist leer"));
    return;
  }
  openHmsPreview({
    runId: currentRunId, runDir: currentRunDir, yaml: editor.value,
    onApply: async (updatedYaml) => { editor.value = updatedYaml; updateResumeConfigSectionHighlights(updatedYaml); if (!await resumeRun()) throw new Error("Resume failed"); },
  });
}

function openSelectedBgePreview() {
  const { currentRunId, currentRunDir } = getRunState();
  const editor = document.getElementById("resume-config-yaml");
  if (!currentRunId || !editor?.value.trim()) {
    toastError(t("ui.toast.bge_preview_failed", "BGE-Vorschau fehlgeschlagen"), t("ui.error.no_config", "Config YAML ist leer"));
    return;
  }
  openBgePreview({
    runId: currentRunId, runDir: currentRunDir, yaml: editor.value,
    onApply: async (updatedYaml) => { editor.value = updatedYaml; updateResumeConfigSectionHighlights(updatedYaml); if (!await resumeRun()) throw new Error(t("ui.toast.resume_failed", "Resume failed")); },
  });
}

async function loadRunConfig(phase) {
  const { currentRunId } = getRunState();
  if (!currentRunId) return;
  try {
    const resp = await api.get(API_ENDPOINTS.runs.config(getRunApiKey()));
    const yaml = resp?.config_yaml || resp?.config || "";
    const editor = document.getElementById("resume-config-yaml");
    if (editor) {
      editor.value = yaml;
      updateResumeConfigSectionHighlights(yaml);
    }
    const hint = document.getElementById("resume-hint");
    if (hint) hint.textContent = yaml ? t("ui.message.resume_ready", "Bereit zum Resume") : t("ui.message.resume_no_config", "Keine Config gefunden");
    return true;
  } catch (e) {
    const hint = document.getElementById("resume-hint");
    if (hint) hint.textContent = `Error: ${e.message}`;
    return false;
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
    if (await loadRunConfig(getSelectedPhase()))
      toastSuccess(t("ui.toast.config_loaded", "Config geladen"));
    return;
  }
  try {
    const resp = await api.get(API_ENDPOINTS.runs.configRevision(getRunApiKey(), revId));
    const yaml = resp?.config || "";
    const editor = document.getElementById("resume-config-yaml");
    if (editor) {
      editor.value = yaml;
      updateResumeConfigSectionHighlights(yaml);
    }
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
    const level = payload.level || data.level || (type === "error" || type === "warning" ? type.toUpperCase() : "INFO");
    addStructuredLogLine(logViewer, data, level, msg);
  }

  // Warning / error events also go to log and warning banner
  if (type === "warning" || type === "error") {
    const msg = payload.message || payload.text || data.message || data.text || JSON.stringify(payload);
    addStructuredLogLine(logViewer, data, type.toUpperCase(), msg);
    if (warningBanner) addRunWarning(warningBanner, msg, type);
  }

  // Phase events: phase_start, phase_progress, phase_end
  // Backend sends both a numeric phase id and a readable phase_name. Prefer
  // the readable name so it can address the phase-list entry directly.
  if (type === "phase_start" || type === "phase_progress" || type === "phase_end") {
    const phaseName = getEventPhaseName(data, payload);
    if (!phaseName || phaseName === "null") return;
    if (getResumePending() && !getResumeActive()) return;
    const pct = normalizedEventPercent(data.pct ?? payload.pct ?? payload.progress ?? data.progress) ?? 0;
    const label = payload.label || null;
    if (getResumeActive() && isBeforeResumePhase(phaseName)) {
      addStructuredLogLine(logViewer, data, "INFO", formatEventMessage(data));
      return;
    }
    if ((getResumePending() || getResumeActive()) && type === "phase_start") {
      const existing = getRunState().phases;
      if (Array.isArray(existing) && existing.length > 0) {
        setPhaseList(existing);
      }
    }
    const status = type === "phase_start" ? "running" : type === "phase_end" ? (payload.status || "ok") : (payload.status || "running");
    updatePhaseState(phaseName, status, pct, label);
    savePhaseToStore(phaseName, status, pct);
    if (payload.elapsed || data.elapsed) updateStat("stat-elapsed", payload.elapsed || data.elapsed);
    // Also log phase events through the shared localized formatter.
    addStructuredLogLine(logViewer, data, "INFO", formatEventMessage(data));
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
      setRunButtonsActive(runStatus === "running");
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
    setResumeFromPhase(fromPhase);
    addStructuredLogLine(logViewer, data, "INFO", formatEventMessage(data));
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
    setResumeFromPhase("");
    if (resumePendingTimer) { clearTimeout(resumePendingTimer); resumePendingTimer = null; }
    const success = payload.success ?? data.success ?? false;
    const fromPhase = payload.from_phase || data.from_phase || "";
    addStructuredLogLine(logViewer, data, success ? "INFO" : "ERROR", formatEventMessage(data));
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
      refreshCurrentImagePreview();
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
      refreshCurrentImagePreview();
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

  const sourceFor = (type) => {
    const explicitSource = cal[`${type}_source`];
    if (explicitSource === "master" || explicitSource === "dir") return explicitSource;
    const explicitUseMaster = cal[`${type}_use_master`];
    if (typeof explicitUseMaster === "boolean") return explicitUseMaster ? "master" : "dir";
    const hasMaster = Boolean((cal[`${type}_master`] || "").trim());
    const hasDir = Boolean((cal[`${type}_dir`] || "").trim());
    return hasMaster && !hasDir ? "master" : "dir";
  };

  const effective = (type) => {
    const source = sourceFor(type);
    const master = normPath((cal[`${type}_master`] || "").trim());
    const dir = normPath(cal[`${type}_dir`] || "");
    return {
      useMaster: source === "master",
      dir: source === "master" ? "" : dir,
      master: source === "master" ? master : "",
    };
  };

  const bias = effective("bias");
  const dark = effective("dark");
  const flat = effective("flat");

  const entries = [
    ["use_bias", cal.bias_enabled ? "true" : "false"],
    ...(cal.bias_enabled ? [
      ["bias_use_master", bias.useMaster ? "true" : "false"],
      ["bias_dir", bias.dir],
      ["bias_master", bias.master],
    ] : [
      ["bias_use_master", "false"],
      ["bias_dir", ""],
      ["bias_master", ""],
    ]),
    ["use_dark", cal.dark_enabled ? "true" : "false"],
    ...(cal.dark_enabled ? [
      ["dark_use_master", dark.useMaster ? "true" : "false"],
      ["darks_dir", dark.dir],
      ["dark_master", dark.master],
    ] : [
      ["dark_use_master", "false"],
      ["darks_dir", ""],
      ["dark_master", ""],
    ]),
    ["use_flat", cal.flat_enabled ? "true" : "false"],
    ...(cal.flat_enabled ? [
      ["flat_use_master", flat.useMaster ? "true" : "false"],
      ["flats_dir", flat.dir],
      ["flat_master", flat.master],
    ] : [
      ["flat_use_master", "false"],
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
