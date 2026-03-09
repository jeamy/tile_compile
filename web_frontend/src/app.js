import { ApiClient } from "./api.js";
import { API_ENDPOINTS } from "./constants.js";
import { applyLocaleMessages, t } from "./i18n.js";

const api = new ApiClient(localStorage.getItem("gui2.backendBase") || "");
const CONFIG_DRAFT_KEY = "gui2.configYamlDraft";
const CONFIG_VALIDATION_STATE_KEY = "gui2.configValidationState";
const LOCALE_KEY = "gui2.locale";
const LAST_INPUT_DIRS_KEY = "gui2.lastInputDirs";

const uiState = {
  currentRunId: localStorage.getItem("gui2.currentRunId") || "",
  currentRunDir: "",
  defaultConfigPath: "",
  selectedHistoryRunId: "",
  compareHistoryRunId: "",
  configYaml: "",
  configObject: null,
  parameterDirty: {},
  runSocket: null,
  runLogLines: [],
  runLogPending: [],
  runLogFlushTimer: null,
  liveSocket: null,
  liveLines: [],
  livePendingLines: [],
  liveLogFlushTimer: null,
  liveFilter: "all",
  lastAstrometryWcs: "",
  lastPccOutput: "",
  lastPccChannels: [],
  lastPccResult: null,
  locale: localStorage.getItem(LOCALE_KEY) || "de",
  projectRunsDir: "",
  monitorStatsStatus: null,
};

const PARAM_CONTROL_PATHS = {
  "parameter.registration.engine": "registration.engine",
  "parameter.registration.allow_rotation": "registration.allow_rotation",
  "parameter.registration.star_topk": "registration.star_topk",
  "parameter.registration.star_inlier_tol_px": "registration.star_inlier_tol_px",
  "parameter.registration.reject_cc_min_abs": "registration.reject_cc_min_abs",
  "parameter.bge.enabled": "bge.enabled",
  "parameter.bge.fit_method": "bge.fit.method",
  "parameter.bge.rbf_lambda": "bge.fit.rbf_lambda",
  "parameter.pcc.source": "pcc.source",
  "parameter.pcc.sigma_clip": "pcc.sigma_clip",
  "parameter.pcc.k_max": "pcc.k_max",
  "input_scan.pattern": "input.pattern",
  "input_scan.max_frames": "input.max_frames",
  "input_scan.color_mode_confirm": "data.color_mode",
  "input_scan.bayer_pattern": "data.bayer_pattern",
  "input_scan.calibration.use_bias": "calibration.use_bias",
  "input_scan.calibration.bias_dir": "calibration.bias_dir",
  "input_scan.calibration.use_dark": "calibration.use_dark",
  "input_scan.calibration.darks_dir": "calibration.darks_dir",
  "input_scan.calibration.use_flat": "calibration.use_flat",
  "input_scan.calibration.flats_dir": "calibration.flats_dir",
};

const PARAM_ID_PATHS = {
  "parameter-bge-sample-quantile": "bge.sample_quantile",
  "parameter-bge-min-tiles": "bge.min_tiles_per_cell",
  "parameter-pcc-min-stars": "pcc.min_stars",
  "parameter-input-pattern": "input.pattern",
  "parameter-input-max-frames": "input.max_frames",
  "parameter-data-bayer": "data.bayer_pattern",
  "parameter-runtime-workers": "runtime_limits.parallel_workers",
  "parameter-runtime-memory": "runtime_limits.memory_budget",
  "parameter-runtime-hard-abort": "runtime_limits.hard_abort_hours",
  "parameter-cal-use-dark": "calibration.use_dark",
  "parameter-cal-darks-dir": "calibration.darks_dir",
  "parameter-cal-use-flat": "calibration.use_flat",
  "parameter-cal-flats-dir": "calibration.flats_dir",
  "parameter-ass-pipeline-profile": "assumptions.pipeline_profile",
  "parameter-ass-frames-min": "assumptions.frames_min",
  "parameter-ass-frames-optimal": "assumptions.frames_optimal",
};

const ASSUMPTION_ID_PATHS = {
  "asmpt-profile": "assumptions.pipeline_profile",
  "asmpt-min": "assumptions.frames_min",
  "asmpt-opt": "assumptions.frames_optimal",
  "asmpt-reduced": "assumptions.frames_reduced_threshold",
  "asmpt-skip-cluster": "assumptions.reduced_mode_skip_clustering",
  "asmpt-cluster-range": "assumptions.reduced_mode_cluster_range",
  "asmpt-exp-tol": "assumptions.exposure_time_tolerance_percent",
};

const SCENARIO_DELTAS = {
  altaz: [
    ["registration.allow_rotation", true],
    ["registration.star_topk", 180],
    ["registration.reject_shift_px_min", 120],
    ["registration.reject_shift_median_multiplier", 5.0],
  ],
  rotation: [
    ["registration.engine", "robust_phase_ecc"],
    ["registration.allow_rotation", true],
    ["registration.star_inlier_tol_px", 4.0],
    ["registration.reject_cc_min_abs", 0.3],
  ],
  bright_stars: [
    ["pcc.mag_bright_limit", 6.0],
    ["pcc.k_max", 2.4],
    ["pcc.sigma_clip", 2.7],
    ["bge.mask.star_dilate_px", 6],
  ],
  few_frames: [
    ["assumptions.frames_reduced_threshold", 200],
    ["assumptions.reduced_mode_skip_clustering", true],
    ["synthetic.frames_min", 4],
    ["synthetic.clustering.cluster_count_range", [3, 10]],
  ],
  gradient: [
    ["bge.enabled", true],
    ["bge.fit.method", "rbf"],
    ["bge.fit.rbf_lambda", "1e-2"],
    ["bge.sample_quantile", 0.15],
    ["bge.structure_thresh_percentile", 0.8],
  ],
};

const $ = (id) => document.getElementById(id);

function pageName() {
  const raw = window.location.pathname.split("/").pop() || "index.html";
  return raw.toLowerCase();
}

function errorText(err) {
  return err?.payload?.detail?.error?.message || err?.payload?.error?.message || err?.message || String(err);
}

function apiErrorCode(err) {
  return String(err?.payload?.detail?.error?.code || err?.payload?.error?.code || "").trim();
}

function apiErrorDetails(err) {
  return err?.payload?.detail?.error?.details || err?.payload?.error?.details || {};
}

async function withPathGrantRetry(fn, { fallbackPath = "" } = {}) {
  try {
    return await fn();
  } catch (err) {
    if (apiErrorCode(err) !== "PATH_NOT_ALLOWED") throw err;
    const details = apiErrorDetails(err);
    const candidatePath = String(details?.path || fallbackPath || "").trim();
    if (!candidatePath || !isAbsolutePath(candidatePath)) throw err;
    const allow = window.confirm(
      `Pfad ist aktuell nicht freigegeben:\n${candidatePath}\n\nZugriff fuer diese Sitzung erlauben?`,
    );
    if (!allow) throw err;
    await api.post(API_ENDPOINTS.fs.grantRoot, { path: candidatePath });
    return fn();
  }
}

function setCurrentRunId(runId) {
  if (!runId) return;
  uiState.currentRunId = String(runId);
  localStorage.setItem("gui2.currentRunId", uiState.currentRunId);
}

function clearCurrentRunId() {
  uiState.currentRunId = "";
  localStorage.removeItem("gui2.currentRunId");
}

function footerEl() {
  return $("scan-note") || document.querySelector(".footer-note");
}

function setFooter(text, isError = false) {
  const el = footerEl();
  if (!el) return;
  el.textContent = String(text);
  el.style.color = isError ? "#b91c1c" : "";
}

function setRunReady(status, runStatus = "") {
  const chip = $("status-run-ready");
  if (!chip) return;
  const runNormalized = String(runStatus || "").toLowerCase();
  if (["running", "queued", "starting"].includes(runNormalized)) {
    chip.textContent = t("ui.status.run_ready_running", "Run Running");
    chip.style.background = "#dbeafe";
    chip.style.borderColor = "#bfdbfe";
    chip.style.color = "#1d4ed8";
    return;
  }
  const normalized = String(status || "check").toLowerCase();
  chip.textContent = normalized === "ok"
    ? t("ui.status.run_ready_ok", "Run Ready")
    : normalized === "error"
      ? t("ui.status.run_ready_blocked", "Run Blocked")
      : t("ui.status.run_ready_check", "Run Check");
  if (normalized === "ok") {
    chip.style.background = "#dff7e8";
    chip.style.borderColor = "#b7e8cc";
    chip.style.color = "#166534";
    return;
  }
  if (normalized === "error") {
    chip.style.background = "#fee2e2";
    chip.style.borderColor = "#fecaca";
    chip.style.color = "#991b1b";
    return;
  }
  chip.style.background = "#fef3c7";
  chip.style.borderColor = "#fde68a";
  chip.style.color = "#92400e";
}

async function waitForJob(jobId, { timeoutMs = 240000, onTick, allowMissing = false } = {}) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    let job;
    try {
      job = await api.get(API_ENDPOINTS.jobs.byId(jobId));
    } catch (err) {
      if (allowMissing && Number(err?.status) === 404) {
        return { job_id: jobId, state: "missing", data: {} };
      }
      throw err;
    }
    onTick?.(job);
    if (["ok", "error", "cancelled"].includes(String(job.state))) {
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
  throw new Error(`job timeout: ${jobId}`);
}

function findLogBoxBySectionTitle(titlePrefix) {
  const sections = Array.from(document.querySelectorAll(".ps-section"));
  const sec = sections.find((s) => {
    const t = s.querySelector(".ps-section-title");
    return t && String(t.textContent || "").trim().toLowerCase().startsWith(titlePrefix.toLowerCase());
  });
  if (!sec) return null;
  return sec.querySelector("div[style*='font-family:monospace']");
}

function appendLine(el, line) {
  if (!el) return;
  const old = el.textContent ? `${el.textContent}\n` : "";
  const merged = `${old}${line}`.split("\n");
  el.textContent = merged.slice(-300).join("\n");
}

function scrollLogToEnd(el) {
  if (!el) return;
  if (typeof el.matches === "function" && el.matches(":hover")) return;
  el.scrollTop = el.scrollHeight;
}

function flushRunMonitorLog() {
  if (uiState.runLogFlushTimer) {
    clearTimeout(uiState.runLogFlushTimer);
    uiState.runLogFlushTimer = null;
  }
  if (uiState.runLogPending.length === 0) return;
  uiState.runLogLines.push(...uiState.runLogPending);
  uiState.runLogLines = uiState.runLogLines.slice(-300);
  uiState.runLogPending = [];
  const logBox = runMonitorLogBox();
  if (!logBox) return;
  logBox.textContent = uiState.runLogLines.join("\n");
  scrollLogToEnd(logBox);
}

function scheduleRunMonitorLogFlush() {
  if (uiState.runLogFlushTimer) return;
  uiState.runLogFlushTimer = window.setTimeout(() => {
    flushRunMonitorLog();
  }, 5000);
}

function enqueueRunMonitorLogLine(line) {
  uiState.runLogPending.push(String(line));
  scheduleRunMonitorLogFlush();
}

function setText(el, value) {
  if (!el) return;
  el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function getConfigDraft() {
  return localStorage.getItem(CONFIG_DRAFT_KEY) || "";
}

function setConfigDraft(yamlText) {
  if (!yamlText) return;
  uiState.configYaml = String(yamlText);
  localStorage.setItem(CONFIG_DRAFT_KEY, uiState.configYaml);
}

function getConfigValidationState() {
  try {
    const raw = localStorage.getItem(CONFIG_VALIDATION_STATE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function setConfigValidationState({ yaml = "", ok = false } = {}) {
  localStorage.setItem(
    CONFIG_VALIDATION_STATE_KEY,
    JSON.stringify({
      yaml: String(yaml || ""),
      ok: Boolean(ok),
      updated_at: new Date().toISOString(),
    }),
  );
}

function clearConfigValidationState() {
  localStorage.removeItem(CONFIG_VALIDATION_STATE_KEY);
}

function setDisabledLike(el, disabled) {
  if (!el) return;
  const isOff = Boolean(disabled);
  if ("disabled" in el) el.disabled = isOff;
  el.style.opacity = isOff ? "0.55" : "";
  el.style.pointerEvents = isOff ? "none" : "";
}

function timestampSuffix(now = new Date()) {
  const pad = (n) => String(n).padStart(2, "0");
  const yyyy = String(now.getFullYear());
  const mm = pad(now.getMonth() + 1);
  const dd = pad(now.getDate());
  const hh = pad(now.getHours());
  const mi = pad(now.getMinutes());
  const ss = pad(now.getSeconds());
  return `${yyyy}${mm}${dd}_${hh}${mi}${ss}`;
}

function readFieldValue(el) {
  if (!el) return "";
  if (el.type === "checkbox") return Boolean(el.checked);
  if (el.tagName === "SELECT") {
    const raw = String(el.value || "").trim();
    if (raw.toLowerCase() === "true") return true;
    if (raw.toLowerCase() === "false") return false;
    return raw;
  }
  const raw = String(el.value || "").trim();
  if (el.type === "number") {
    if (raw === "") return "";
    const n = Number(raw);
    return Number.isFinite(n) ? n : raw;
  }
  return raw;
}

function writeFieldValue(el, value) {
  if (!el || value === undefined || value === null) return;
  if (el.type === "checkbox") {
    el.checked = Boolean(value);
    return;
  }
  if (el.tagName === "SELECT") {
    const txt = String(value);
    const opt = Array.from(el.options || []).find((o) => String(o.value) === txt || String(o.textContent) === txt);
    if (opt) {
      el.value = opt.value;
      return;
    }
  }
  el.value = Array.isArray(value) || typeof value === "object" ? JSON.stringify(value) : String(value);
}

function getByPath(root, dotted) {
  let cur = root;
  for (const key of String(dotted || "").split(".").filter(Boolean)) {
    if (!cur || typeof cur !== "object" || !(key in cur)) return undefined;
    cur = cur[key];
  }
  return cur;
}

function updatesFromMap(pathBySelector) {
  const updates = [];
  for (const [selector, path] of pathBySelector) {
    const el = selector.startsWith("#")
      ? document.getElementById(selector.slice(1))
      : document.querySelector(selector);
    if (!el) continue;
    updates.push({ path, value: readFieldValue(el) });
  }
  return updates;
}

function parseInputDirs(value) {
  const raw = String(value || "");
  const dirs = raw
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
  return dirs;
}

function isAbsolutePath(value) {
  const s = String(value || "").trim();
  return s.startsWith("/") || /^[A-Za-z]:[\\/]/.test(s) || s.startsWith("\\\\");
}

function allAbsolutePaths(paths) {
  return Array.isArray(paths) && paths.length > 0 && paths.every((p) => isAbsolutePath(p));
}

function persistLastInputDirs(rawValue) {
  const value = String(rawValue || "").trim();
  if (!value) return;
  const dirs = parseInputDirs(value);
  if (!allAbsolutePaths(dirs)) return;
  localStorage.setItem(LAST_INPUT_DIRS_KEY, value);
}

function restoreLastInputDirs(...ids) {
  const value = String(localStorage.getItem(LAST_INPUT_DIRS_KEY) || "").trim();
  if (!value) return;
  const dirs = parseInputDirs(value);
  if (!allAbsolutePaths(dirs)) {
    localStorage.removeItem(LAST_INPUT_DIRS_KEY);
    return;
  }
  ids.forEach((id) => {
    const el = $(id);
    if (el) el.value = value;
  });
}

function firstNonEmptyText(...values) {
  for (const value of values) {
    const text = String(value || "").trim();
    if (text) return text;
  }
  return "";
}

function sanitizeRunName(raw) {
  return String(raw || "")
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_+/g, "_");
}

function suggestRunNameFromInputs(dirs) {
  const firstDir = Array.isArray(dirs) && dirs.length > 0 ? String(dirs[0] || "").trim() : "";
  const leaf = firstDir
    ? firstDir.replace(/[\\/]+$/, "").split(/[\\/]/).filter(Boolean).pop() || "run"
    : "run";
  return sanitizeRunName(leaf) || "run";
}

async function resolveConfigYamlForRun() {
  return await ensureConfigYaml();
}

async function startRunFromCurrentForm({ source = "" } = {}) {
  const normalizedSource = String(source || "").trim().toLowerCase();
  const useDashboardFields = normalizedSource === "dashboard";
  const useWizardFields = normalizedSource === "wizard";
  const inputDirsText = useDashboardFields
    ? String($("dashboard-input-dirs")?.value || "")
    : useWizardFields
      ? String($("inp-dirs")?.value || "")
      : firstNonEmptyText(localStorage.getItem(LAST_INPUT_DIRS_KEY), $("dashboard-input-dirs")?.value, $("inp-dirs")?.value);
  const inputDirs = parseInputDirs(inputDirsText);
  if (inputDirs.length === 0) {
    throw new Error("Bitte mindestens einen Eingabeordner setzen.");
  }
  persistLastInputDirs(inputDirsText);

  const runNameEl = useDashboardFields ? $("dashboard-run-name") : useWizardFields ? $("wizard-run-name") : null;
  const runsDirEl = useDashboardFields ? $("dashboard-run-runs-dir") : useWizardFields ? $("wizard-runs-dir") : null;
  const runName = sanitizeRunName(runNameEl?.value || "") || suggestRunNameFromInputs(inputDirs);
  if (runNameEl) runNameEl.value = runName;
  const runsDir = firstNonEmptyText(runsDirEl?.value, uiState.projectRunsDir);
  if (runsDirEl && !String(runsDirEl.value || "").trim() && runsDir) {
    runsDirEl.value = runsDir;
  }
  const colorMode = firstNonEmptyText(
    useDashboardFields ? $("dashboard-color-mode")?.value : "",
    useWizardFields ? $("inp-colormode")?.value : "",
    $("dashboard-color-mode")?.value,
    $("inp-colormode")?.value,
    "OSC",
  ).toUpperCase();

  const payload = {
    color_mode: colorMode,
    run_name: runName || undefined,
    runs_dir: runsDir || undefined,
    config_yaml: await resolveConfigYamlForRun(),
  };
  const queue = useDashboardFields || useWizardFields ? collectQueueRows() : [];
  if (queue.length > 0) {
    payload.queue = queue;
  } else if (inputDirs.length > 1 && colorMode === "MONO") {
    payload.input_dirs = inputDirs.map((dir) => ({ input_dir: dir }));
  } else {
    payload.input_dir = inputDirs[0] || "";
  }
  if (!payload.input_dir && !payload.queue && !payload.input_dirs) {
    throw new Error("Bitte mindestens einen Eingabeordner setzen.");
  }
  return withPathGrantRetry(() => api.post(API_ENDPOINTS.runs.start, payload), {
    fallbackPath: String(payload.runs_dir || inputDirs[0] || ""),
  });
}

function summarizeScanResult(raw, fallbackInputPath = "") {
  const src = raw && typeof raw === "object" ? raw : {};
  const errors = Array.isArray(src.errors) ? src.errors : [];
  const warnings = Array.isArray(src.warnings) ? src.warnings : [];
  const candidates = Array.isArray(src.color_mode_candidates)
    ? src.color_mode_candidates.map((x) => String(x))
    : [];
  const width = Number(src.image_width || 0);
  const height = Number(src.image_height || 0);
  const framesDetected = Number(src.frames_detected || 0);
  const hasScan = typeof src.has_scan === "boolean" ? src.has_scan : Object.keys(src).length > 0;
  const ok = typeof src.ok === "boolean" ? src.ok : errors.length === 0;
  const inputDirs = Array.isArray(src.input_dirs) ? src.input_dirs.map((x) => String(x || "").trim()).filter(Boolean) : [];
  const colorMode = String(src.color_mode || "UNKNOWN");
  const normalizedColorMode = normalizeDetectedColorMode(colorMode);
  if (normalizedColorMode) {
    localStorage.setItem("gui2.lastScanColorMode", normalizedColorMode);
  }
  return {
    has_scan: hasScan,
    ok,
    input_path: String(src.input_path || fallbackInputPath || ""),
    input_dirs: inputDirs,
    frames_detected: Number.isFinite(framesDetected) ? framesDetected : 0,
    color_mode: colorMode,
    color_mode_candidates: candidates,
    image_width: Number.isFinite(width) ? width : 0,
    image_height: Number.isFinite(height) ? height : 0,
    bayer_pattern: src.bayer_pattern ?? null,
    requires_user_confirmation: Boolean(src.requires_user_confirmation),
    errors,
    warnings,
  };
}

function renderScanSummary(prefix, summary) {
  const data = summarizeScanResult(summary);
  const status = !data.has_scan
    ? t("ui.status.scan_none", "Kein Scan")
    : data.ok
      ? t("ui.status.scan_ok", "OK")
      : data.errors.length > 0
        ? t("ui.status.scan_error", "ERROR")
        : t("ui.status.scan_check", "CHECK");
  const sizeText =
    data.image_width > 0 && data.image_height > 0 ? `${data.image_width} x ${data.image_height}` : t("ui.value.unknown_size", "unbekannt");
  const candidates = data.color_mode_candidates.length > 0 ? data.color_mode_candidates.join(", ") : "-";
  setText($(`${prefix}-status`), status);
  setText($(`${prefix}-input-path`), data.input_path || "-");
  setText($(`${prefix}-frames`), String(data.frames_detected));
  setText($(`${prefix}-color-mode`), data.color_mode || t("ui.value.unknown_color_mode", "UNKNOWN"));
  setText($(`${prefix}-candidates`), candidates);
  setText($(`${prefix}-size`), sizeText);
  setText($(`${prefix}-bayer`), data.bayer_pattern || "-");
  setText($(`${prefix}-confirm`), data.requires_user_confirmation ? t("ui.value.yes", "ja") : t("ui.value.no", "nein"));
  setText($(`${prefix}-errors`), String(data.errors.length));
  setText($(`${prefix}-warnings`), String(data.warnings.length));
  return data;
}

function normalizeDetectedColorMode(value) {
  const normalized = String(value || "").trim().toUpperCase();
  return normalized === "MONO" || normalized === "OSC" ? normalized : "";
}

function applyDetectedColorModeToSelect(selectEl, scanSummary) {
  if (!selectEl) return false;
  const detected = normalizeDetectedColorMode(scanSummary?.color_mode);
  if (!detected) return false;
  const option = Array.from(selectEl.options || []).find((opt) => String(opt.value || "").trim().toUpperCase() === detected);
  if (!option) return false;
  const current = String(selectEl.value || "").trim().toUpperCase();
  if (current === detected) return true;
  selectEl.value = option.value;
  selectEl.dispatchEvent(new Event("change", { bubbles: true }));
  return true;
}

function renderDashboardScanKpis(summary, qualityScore) {
  const data = summarizeScanResult(summary);
  const framesKpi = document.querySelector("#dashboard-kpi-scan-quality div:nth-child(2)");
  if (framesKpi) framesKpi.textContent = String(data.frames_detected);
  const colorChip = $("dashboard-kpi-color-mode");
  if (colorChip) colorChip.textContent = `Color: ${data.color_mode || "UNKNOWN"}`;

  const qualityKpi = document.querySelector("#dashboard-kpi-open-warnings div:nth-child(2)");
  if (qualityKpi) qualityKpi.textContent = Number.isFinite(Number(qualityScore)) ? Number(qualityScore).toFixed(3) : "0.000";
  const sizeChip = $("dashboard-kpi-scan-size");
  if (sizeChip) sizeChip.textContent = `${data.image_width || 0} x ${data.image_height || 0} px`;

  const warningCount = data.errors.length + data.warnings.length;
  const warnKpi = document.querySelector("#dashboard-kpi-guardrail-warnings div:nth-child(2)");
  if (warnKpi) warnKpi.textContent = String(warningCount);
  const pathState = $("dashboard-kpi-path-state");
  if (pathState) pathState.textContent = data.input_path || "kein Scan";
}

function renderDashboardLastRunKpi(appState) {
  const card = $("dashboard-kpi-last-run");
  const statusEl = $("dashboard-kpi-last-run-status");
  const metaEl = $("dashboard-kpi-last-run-meta");
  if (!card || !statusEl || !metaEl) return;
  const currentRun = appState?.run?.current || {};
  const runId = String(currentRun?.run_id || "").trim();
  if (!runId) {
    statusEl.textContent = "-";
    metaEl.textContent = "kein aktueller Projekt-Run";
    card.onclick = () => {
      window.location.href = "history-tools.html";
    };
    return;
  }
  const statusText = String(currentRun?.status || "unknown").toUpperCase();
  const progressValue = Number(currentRun?.progress);
  const progressText = Number.isFinite(progressValue)
    ? `${(progressValue <= 1 ? progressValue * 100 : progressValue).toFixed(1)}%`
    : "-";
  const phaseText = String(currentRun?.current_phase || "").trim();
  statusEl.textContent = statusText;
  metaEl.textContent = [runId, phaseText || null, progressText !== "-" ? progressText : null].filter(Boolean).join(" • ");
  card.onclick = () => {
    window.location.href = "history-tools.html";
  };
}

function formatUiDateTime(isoRaw) {
  const iso = String(isoRaw || "").trim();
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(uiState.locale === "en" ? "en-GB" : "de-DE", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function deriveOutputPath(inputPath, suffix) {
  const s = String(inputPath || "").trim();
  if (!s) return "";
  const idx = s.lastIndexOf(".");
  if (idx <= 0) return `${s}${suffix}`;
  return `${s.slice(0, idx)}${suffix}${s.slice(idx)}`;
}

function parentDirOfPath(pathValue) {
  const s = String(pathValue || "").trim();
  if (!s) return "";
  const slash = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  if (slash <= 0) return "";
  return s.slice(0, slash);
}

function pathBaseName(pathValue) {
  const s = String(pathValue || "")
    .trim()
    .replace(/\\/g, "/")
    .replace(/\/+$/, "");
  if (!s) return "";
  const idx = s.lastIndexOf("/");
  return idx >= 0 ? s.slice(idx + 1) : s;
}

function joinPath(basePath, childName) {
  const base = String(basePath || "").trim().replace(/\\/g, "/").replace(/\/+$/, "");
  const child = String(childName || "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
  if (!base) return child ? `/${child}` : "/";
  if (!child) return base;
  return `${base}/${child}`;
}

function ensureYamlFileName(fileName) {
  const trimmed = String(fileName || "").trim();
  if (!trimmed) return "";
  if (/\.(yaml|yml)$/i.test(trimmed)) return trimmed;
  return `${trimmed}.yaml`;
}

function deriveParameterSaveDefaultDir() {
  const presetDir = parentDirOfPath(String($("parameter-preset-select")?.value || "").trim());
  if (presetDir) return presetDir;
  return firstNonEmptyText(uiState.projectRunsDir, parentDirOfPath(uiState.defaultConfigPath));
}

function deriveParameterSaveDefaultName() {
  return firstNonEmptyText(pathBaseName(uiState.defaultConfigPath), "tile_compile.example.yaml");
}

async function pickDirectoryPath(initialPath) {
  if (typeof window.gui2PickPathValue === "function") {
    return window.gui2PickPathValue(initialPath, "dir");
  }
  const typed = window.prompt("Verzeichnis eingeben", initialPath || "");
  return String(typed || "").trim() || null;
}

async function chooseConfigSaveAsPath() {
  const defaultDir = deriveParameterSaveDefaultDir();
  const defaultName = deriveParameterSaveDefaultName();
  const defaultPath = joinPath(defaultDir, defaultName);
  if (typeof window.gui2PickPathValue === "function") {
    const pickedPath = await window.gui2PickPathValue(defaultPath, { mode: "save-file", defaultFileName: defaultName });
    const normalizedPickedPath = ensureYamlFileName(String(pickedPath || "").trim());
    return normalizedPickedPath || null;
  }
  const typedPath = window.prompt("Dateipfad fuer Speichern unter", defaultPath);
  const normalizedPath = ensureYamlFileName(typedPath);
  return normalizedPath || null;
}

function ensureRunIdFromHeader() {
  if (uiState.currentRunId) return uiState.currentRunId;
  const sub = document.querySelector(".app-content .ps-sub");
  if (!sub) return "";
  const codeNodes = Array.from(sub.querySelectorAll("code"));
  for (const node of codeNodes) {
    const token = String(node.textContent || "").trim();
    if (!token) continue;
    if (token === "-" || token.toLowerCase() === "running" || token.toLowerCase() === "unknown") continue;
    if (token === "2/5 (R)") continue;
    if (!/[A-Za-z0-9]/.test(token)) continue;
    if (!/^[A-Za-z0-9._/-]+$/.test(token)) continue;
    if (!token.includes("/") && !token.includes("_") && !/\d/.test(token)) continue;
    if (!token.includes("running")) {
      setCurrentRunId(token);
      return token;
    }
  }
  return "";
}

async function initGlobalState() {
  try {
    const [guardrails, appState] = await Promise.all([
      api.get(API_ENDPOINTS.guardrails.root),
      api.get(API_ENDPOINTS.app.state),
    ]);
    setRunReady(guardrails?.status || "check", appState?.run?.current?.status || "");
    const rid = String(appState?.project?.current_run_id || "").trim();
    if (rid) setCurrentRunId(rid);
    else clearCurrentRunId();
    const runsDir = String(appState?.project?.runs_dir || "").trim();
    if (runsDir) uiState.projectRunsDir = runsDir;
    const defaultConfigPath = String(appState?.project?.default_config_path || "").trim();
    if (defaultConfigPath) uiState.defaultConfigPath = defaultConfigPath;
    const scanPath = String(appState?.scan?.last_input_path || "").trim();
    if (scanPath) persistLastInputDirs(scanPath);
  } catch (err) {
    setFooter(`Backend nicht erreichbar: ${errorText(err)}`, true);
  }
}

async function applyLocale(localeRaw) {
  const locale = String(localeRaw || "de").toLowerCase() === "en" ? "en" : "de";
  uiState.locale = locale;
  localStorage.setItem(LOCALE_KEY, locale);
  document.documentElement.setAttribute("lang", locale);
  $("locale-de")?.classList.toggle("active", locale === "de");
  $("locale-en")?.classList.toggle("active", locale === "en");
  await applyLocaleMessages(locale);
}

function bindLocaleControls() {
  void applyLocale(uiState.locale);
  $("locale-de")?.addEventListener("click", () => {
    void applyLocale("de");
  });
  $("locale-en")?.addEventListener("click", () => {
    void applyLocale("en");
  });
}

function buildScanPayloadFromDirs(dirs, framesMin, withChecksums) {
  const payload = {
    frames_min: Number.isFinite(framesMin) ? Math.max(1, framesMin) : 1,
    with_checksums: withChecksums,
  };
  if (dirs.length <= 1) {
    payload.input_path = dirs[0] || "";
    return payload;
  }
  payload.input_dirs = dirs;
  payload.input_path = dirs[0];
  return payload;
}

async function executeScanFlow({
  inputDirsId = "inp-dirs",
  resultPanelId = "scan-result",
  resultBodyId = "scan-result-body",
  summaryPrefix = "scan-summary",
  framesMinId = "inp-frames-min",
  checksumsId = "inp-checksums",
} = {}) {
  const dirText = String($(inputDirsId)?.value || "");
  const dirs = parseInputDirs(dirText);
  if (dirs.length === 0) {
    setFooter("Bitte mindestens einen Eingabeordner setzen.", true);
    return;
  }
  persistLastInputDirs(dirText);

  const resultPanel = $(resultPanelId);
  const resultBody = $(resultBodyId);
  const framesMin = Number($(framesMinId)?.value || 1);
  const withChecksums = Boolean($(checksumsId)?.checked);
  const payload = buildScanPayloadFromDirs(dirs, framesMin, withChecksums);

  try {
    const accepted = await withPathGrantRetry(() => api.post(API_ENDPOINTS.scan.root, payload), {
      fallbackPath: dirs[0] || "",
    });
    if (resultPanel) resultPanel.style.display = "block";
    renderScanSummary(summaryPrefix, { has_scan: true, input_path: payload.input_path, color_mode: "UNKNOWN" });
    setText(resultBody, { state: accepted.state, message: "Scan gestartet..." });
    const job = await waitForJob(accepted.job_id, { allowMissing: true });
    if (String(job?.state) === "missing") {
      const latest = await api.get(API_ENDPOINTS.scan.latest);
      const summary = summarizeScanResult(latest, payload.input_path);
      renderScanSummary(summaryPrefix, summary);
      applyDetectedColorModeToSelect($("inp-colormode"), summary);
      applyDetectedColorModeToSelect($("dashboard-color-mode"), summary);
      setText(resultBody, latest);
      setFooter(
        "Scan-Status war kurzzeitig nicht abrufbar (Backend-Reload). Letztes Scan-Ergebnis wurde geladen.",
        true,
      );
      const mergedInputText = summary.input_dirs?.length > 0 ? summary.input_dirs.join(", ") : summary.input_path;
      if (mergedInputText) {
        if ($(inputDirsId)) $(inputDirsId).value = mergedInputText;
        persistLastInputDirs(summary.input_path);
      }
      await initGlobalState();
      return;
    }
    const result = job?.data?.result || {};
    setText(resultBody, result);
    let summary = summarizeScanResult(result, payload.input_path);
    try {
      const latest = await api.get(API_ENDPOINTS.scan.latest);
      summary = summarizeScanResult(latest, payload.input_path);
    } catch {
      // keep local summary from job payload
    }
    renderScanSummary(summaryPrefix, summary);
    applyDetectedColorModeToSelect($("inp-colormode"), summary);
    applyDetectedColorModeToSelect($("dashboard-color-mode"), summary);
    const mergedInputText = summary.input_dirs?.length > 0 ? summary.input_dirs.join(", ") : summary.input_path;
    if (mergedInputText) {
      if ($(inputDirsId)) $(inputDirsId).value = mergedInputText;
      persistLastInputDirs(summary.input_path);
    }
    setFooter(job.state === "ok" ? "Scan abgeschlossen." : `Scan beendet mit Status: ${job.state}`, job.state !== "ok");
    await initGlobalState();
  } catch (err) {
    const code = apiErrorCode(err);
    const details = apiErrorDetails(err);
    if (code === "PATH_NOT_FOUND" && Array.isArray(details?.tried) && details.tried.length > 0) {
      setFooter(`Scan-Pfad nicht gefunden. Geprueft: ${details.tried.join(" | ")}`, true);
    } else {
      setFooter(`Scan fehlgeschlagen: ${errorText(err)}`, true);
    }
    setText(resultBody, err?.payload || { error: errorText(err) });
    if (resultPanel) resultPanel.style.display = "block";
  }
}

function bindInputDirMemory(...ids) {
  restoreLastInputDirs(...ids);
  ids.forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("input", () => persistLastInputDirs(el.value));
    el.addEventListener("change", () => persistLastInputDirs(el.value));
  });
}

function bindScanPages() {
  bindInputDirMemory("inp-dirs");
  if (!$("btn-scan")) return;
  window.runScan = () => {
    void executeScanFlow();
  };
  const syncScanConfigField = async (el) => {
    const path = parameterPathFromElement(el);
    if (!path) return;
    try {
      await patchConfig({ updates: [{ path, value: readFieldValue(el) }], persist: false });
    } catch (err) {
      setFooter(`Input-Config-Update fehlgeschlagen: ${errorText(err)}`, true);
    }
  };
  document.querySelectorAll(".app-content [data-control]").forEach((el) => {
    const path = parameterPathFromElement(el);
    if (!path) return;
    el.addEventListener("input", () => {
      void syncScanConfigField(el);
    });
    el.addEventListener("change", () => {
      void syncScanConfigField(el);
    });
  });
  const colorModeEl = $("inp-colormode");
  if (colorModeEl) {
    const updateQueue = () => setMonoQueueVisible(colorModeEl.value || "");
    colorModeEl.addEventListener("change", updateQueue);
    updateQueue();
  }
  void (async () => {
    try {
      const parsed = await patchConfig({ updates: [], persist: false });
      if (parsed?.config) {
        syncParameterFieldsFromConfig(parsed.config);
      }
      const latest = await api.get(API_ENDPOINTS.scan.latest);
      const summary = summarizeScanResult(latest);
      if (summary.has_scan) {
        $("scan-result").style.display = "block";
        renderScanSummary("scan-summary", summary);
        applyDetectedColorModeToSelect($("inp-colormode"), summary);
        applyDetectedColorModeToSelect($("dashboard-color-mode"), summary);
        setText($("scan-result-body"), latest);
        const mergedInputText = summary.input_dirs?.length > 0 ? summary.input_dirs.join(", ") : summary.input_path;
        if (mergedInputText) {
          $("inp-dirs").value = mergedInputText;
          persistLastInputDirs(summary.input_path);
        }
      }
    } catch {
      // page still works without preloaded summary
    }
  })();
}

function parameterDiffBox() {
  return document.querySelector("#parameter-diff-panel div[style*='font-family:monospace']");
}

function parameterValidateStatusEl() {
  return $("parameter-validate-status");
}

function parameterValidateDetailsEl() {
  return $("parameter-validate-details");
}

function parameterSituationApplyStatusEl() {
  return $("parameter-situation-apply-status");
}

function monitorStartValidationEl() {
  return $("monitor-start-validation");
}

function setParameterPreview(value) {
  const box = parameterDiffBox();
  if (!box) return;
  setText(box, value);
}

function clearChildren(node) {
  while (node?.firstChild) node.removeChild(node.firstChild);
}

function formatValidationIssue(issue) {
  if (typeof issue === "string") return issue;
  if (!issue || typeof issue !== "object") return String(issue || "");
  const path = firstNonEmptyText(issue.path, issue.instance_path, issue.schema_path, issue.field, issue.param);
  const code = firstNonEmptyText(issue.code, issue.keyword, issue.type);
  const message = firstNonEmptyText(issue.message, issue.error, issue.detail, issue.reason);
  const parts = [];
  if (path) parts.push(path);
  if (code) parts.push(`[${code}]`);
  if (message) parts.push(message);
  if (parts.length > 0) return parts.join(": ");
  try {
    return JSON.stringify(issue);
  } catch {
    return String(issue);
  }
}

function setParameterValidateDetails(result) {
  const el = parameterValidateDetailsEl();
  if (!el) return;
  clearChildren(el);
  if (!result || typeof result !== "object") {
    el.style.display = "none";
    return;
  }
  const groups = [
    { label: "Fehler", items: Array.isArray(result.errors) ? result.errors : [], color: "#b91c1c" },
    { label: "Warnungen", items: Array.isArray(result.warnings) ? result.warnings : [], color: "#b45309" },
  ].filter((group) => group.items.length > 0);
  if (groups.length === 0) {
    el.style.display = "none";
    return;
  }
  groups.forEach((group) => {
    const title = document.createElement("div");
    title.textContent = `${group.label} (${group.items.length})`;
    title.style.marginTop = "8px";
    title.style.fontWeight = "600";
    title.style.color = group.color;
    el.appendChild(title);

    const list = document.createElement("ul");
    list.style.margin = "6px 0 0 18px";
    list.style.padding = "0";
    group.items.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = formatValidationIssue(item);
      li.style.marginBottom = "4px";
      list.appendChild(li);
    });
    el.appendChild(list);
  });
  el.style.display = "block";
}

function setSituationApplyStatus(applied, text = "") {
  const el = parameterSituationApplyStatusEl();
  if (!el) return;
  if (!applied) {
    el.style.display = "none";
    el.textContent = text || t("ui.status.situation_idle", "Noch nicht angewendet");
    return;
  }
  el.style.display = "inline-flex";
  el.textContent = text || t("ui.status.situation_applied", "Angewendet");
}

function setMonitorStartValidationMessage(text = "") {
  const el = monitorStartValidationEl();
  if (!el) return;
  const message = String(text || "").trim();
  el.textContent = message;
  el.style.display = message ? "block" : "none";
}

function monitorReportBtn() {
  return $("monitor-report");
}

function setInlineAsyncStatus(el, text = "", tone = "idle") {
  if (!el) return;
  const message = String(text || "").trim();
  el.textContent = message;
  el.style.display = message ? "inline-flex" : "none";
  if (!message) return;
  if (tone === "ok") {
    el.style.color = "#166534";
    return;
  }
  if (tone === "error") {
    el.style.color = "#b91c1c";
    return;
  }
  if (tone === "running") {
    el.style.color = "#b45309";
    return;
  }
  el.style.color = "#475569";
}

function statsStartedMessage(jobId) {
  return t("ui.message.stats_started", "Stats-Generierung gestartet (Job {job_id}).")
    .replace("{job_id}", String(jobId || "-"));
}

function statsFailedMessage(err) {
  return t("ui.message.stats_failed", "Stats-Generierung fehlgeschlagen: {error}")
    .replace("{error}", errorText(err));
}

function isRunActiveStatus(status) {
  return ["running", "queued", "starting"].includes(String(status || "").trim().toLowerCase());
}

async function isMonitorRunCurrentlyActive() {
  try {
    const appState = await api.get(API_ENDPOINTS.app.state);
    if (isRunActiveStatus(appState?.run?.current?.status)) return true;
  } catch {
    // Ignore status probe errors and fall back to local validation state.
  }
  return false;
}

async function getRunStartValidationBlockReason() {
  if (await isMonitorRunCurrentlyActive()) return "";
  const yaml = await resolveConfigYamlForRun();
  const validation = getConfigValidationState();
  if (!validation || String(validation.yaml || "") !== String(yaml || "")) {
    return t("ui.message.monitor_validation_required", "Run blockiert: Konfiguration im Parameter Studio validieren.");
  }
  if (!validation.ok) {
    return t("ui.message.monitor_validation_failed", "Run blockiert: letzte Validierung der Konfiguration ist fehlgeschlagen.");
  }
  return "";
}

async function refreshRunMonitorValidationMessage() {
  const message = await getRunStartValidationBlockReason();
  setMonitorStartValidationMessage(message);
  return message;
}

function setParameterValidateStatus(result, fallbackText = "") {
  const el = parameterValidateStatusEl();
  if (!el) return;
  if (!result || typeof result !== "object") {
    el.textContent = fallbackText || "Validierung: nicht geprüft";
    el.style.color = "";
    return;
  }
  const errors = Array.isArray(result.errors) ? result.errors.length : 0;
  const warnings = Array.isArray(result.warnings) ? result.warnings.length : 0;
  if (errors > 0) {
    const firstError = formatValidationIssue(result.errors?.[0]);
    el.textContent = `Validierung: ERROR (${errors} Fehler, ${warnings} Warnungen)${firstError ? ` - ${firstError}` : ""}`;
    el.style.color = "#b91c1c";
    return;
  }
  if (warnings > 0) {
    const firstWarning = formatValidationIssue(result.warnings?.[0]);
    el.textContent = `Validierung: WARN (${warnings} Warnungen)${firstWarning ? ` - ${firstWarning}` : ""}`;
    el.style.color = "#b45309";
    return;
  }
  el.textContent = "Validierung: OK";
  el.style.color = "#166534";
}

async function ensureConfigYaml() {
  if (uiState.configYaml) return uiState.configYaml;
  const draft = getConfigDraft();
  if (draft) {
    uiState.configYaml = draft;
    return draft;
  }
  const current = await api.get(API_ENDPOINTS.config.current);
  uiState.configYaml = String(current?.config || "");
  setConfigDraft(uiState.configYaml);
  return uiState.configYaml;
}

async function patchConfig({ updates = [], persist = false, yamlText } = {}) {
  const baseYaml = yamlText !== undefined ? String(yamlText || "") : await ensureConfigYaml();
  const result = await api.post(API_ENDPOINTS.config.patch, {
    yaml: baseYaml,
    updates,
    parse_values: true,
    persist,
  });
  if (result?.config_yaml) {
    setConfigDraft(result.config_yaml);
  }
  if (result?.config && typeof result.config === "object") {
    uiState.configObject = result.config;
  }
  return result;
}

async function saveParameterConfig(targetPath = "") {
  const patched = await patchConfig({ updates: collectParameterDirtyUpdates(), persist: false });
  const result = await api.post(API_ENDPOINTS.config.save, {
    yaml: patched?.config_yaml || "",
    path: targetPath || undefined,
  });
  uiState.configYaml = String(patched?.config_yaml || "");
  setConfigDraft(uiState.configYaml);
  uiState.parameterDirty = {};
  setParameterPreview(uiState.configYaml);
  return result;
}

function parameterPathFromElement(el) {
  if (!el) return "";
  const dynRow = el.closest(".ps-dyn-row[data-path]");
  if (dynRow) return String(dynRow.getAttribute("data-path") || "");
  const control = String(el.getAttribute("data-control") || "");
  if (control && PARAM_CONTROL_PATHS[control]) return PARAM_CONTROL_PATHS[control];
  const id = String(el.id || "");
  if (id && PARAM_ID_PATHS[id]) return PARAM_ID_PATHS[id];
  return "";
}

function bindParameterDirtyTracking() {
  const root = document.querySelector(".app-content") || document.body;
  if (!root) return;
  const onAny = (ev) => {
    const el = ev.target;
    if (!(el instanceof HTMLElement)) return;
    const path = parameterPathFromElement(el);
    if (!path) return;
    uiState.parameterDirty[path] = readFieldValue(el);
  };
  root.addEventListener("input", onAny);
  root.addEventListener("change", onAny);
}

function collectParameterDirtyUpdates() {
  const out = [];
  for (const [path, value] of Object.entries(uiState.parameterDirty)) {
    out.push({ path, value });
  }
  return out;
}

function syncParameterFieldsFromConfig(config) {
  if (!config || typeof config !== "object") return;

  for (const [control, path] of Object.entries(PARAM_CONTROL_PATHS)) {
    const el = document.querySelector(`[data-control='${control}']`);
    if (!el) continue;
    const value = getByPath(config, path);
    writeFieldValue(el, value);
  }
  for (const [id, path] of Object.entries(PARAM_ID_PATHS)) {
    const el = document.getElementById(id);
    if (!el) continue;
    const value = getByPath(config, path);
    writeFieldValue(el, value);
  }
  document.querySelectorAll(".ps-dyn-row[data-path]").forEach((row) => {
    const path = String(row.getAttribute("data-path") || "");
    if (!path) return;
    const el = row.querySelector("input,select,textarea");
    if (!el) return;
    writeFieldValue(el, getByPath(config, path));
  });
}

function activeScenarioKeys(scopeSelector = "#parameter-studio-root") {
  return Array.from(document.querySelectorAll(`${scopeSelector} [data-scenario].ps-chip-btn.active`))
    .map((el) => String(el.getAttribute("data-scenario") || "").trim())
    .filter(Boolean);
}

async function bindParameterStudio() {
  const presetSelect = $("parameter-preset-select");
  if (!presetSelect) return;

  bindParameterDirtyTracking();

  const applyPreview = async ({ persist = false } = {}) => {
    const updates = collectParameterDirtyUpdates();
    const patched = await patchConfig({ updates, persist });
    setParameterPreview(patched?.config_yaml || "");
    if (patched?.config) {
      syncParameterFieldsFromConfig(patched.config);
    }
    if (persist) {
      uiState.parameterDirty = {};
    }
    return patched;
  };

  try {
    const presets = await api.get(API_ENDPOINTS.config.presets);
    if (Array.isArray(presets?.items) && presets.items.length > 0) {
      const old = presetSelect.value;
      presetSelect.innerHTML = "";
      for (const item of presets.items) {
        const opt = document.createElement("option");
        opt.value = item.path;
        opt.textContent = item.name;
        presetSelect.appendChild(opt);
      }
      if (old) presetSelect.value = old;
    }
    const currentYaml = await ensureConfigYaml();
    const parsed = await patchConfig({ yamlText: currentYaml, updates: [] });
    if (parsed?.config) {
      syncParameterFieldsFromConfig(parsed.config);
    }
    setParameterPreview(currentYaml);
    setParameterValidateStatus(null, "Validierung: nicht geprüft");
    setParameterValidateDetails(null);
    setSituationApplyStatus(false);
    clearConfigValidationState();
  } catch (err) {
    setFooter(`Preset-Liste konnte nicht geladen werden: ${errorText(err)}`, true);
  }

  $("parameter-yaml-sync")?.addEventListener("click", async () => {
    try {
      const current = await api.get(API_ENDPOINTS.config.current);
      uiState.configYaml = String(current?.config || "");
      setConfigDraft(uiState.configYaml);
      uiState.parameterDirty = {};
      const parsed = await patchConfig({ yamlText: uiState.configYaml, updates: [] });
      if (parsed?.config) syncParameterFieldsFromConfig(parsed.config);
      setParameterPreview(uiState.configYaml);
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterValidateDetails(null);
      setSituationApplyStatus(false);
      clearConfigValidationState();
      setFooter("YAML aus Backend synchronisiert.");
    } catch (err) {
      setFooter(`YAML Sync fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-preset-apply")?.addEventListener("click", async () => {
    try {
      const path = String(presetSelect.value || "").trim();
      if (!path) {
        setFooter("Kein Preset ausgewaehlt.", true);
        return;
      }
      const applied = await api.post(API_ENDPOINTS.config.applyPreset, { path });
      uiState.configYaml = String(applied?.config || "");
      setConfigDraft(uiState.configYaml);
      uiState.parameterDirty = {};
      const parsed = await patchConfig({ yamlText: uiState.configYaml, updates: [] });
      if (parsed?.config) syncParameterFieldsFromConfig(parsed.config);
      setParameterPreview(String(parsed?.config_yaml || uiState.configYaml));
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterValidateDetails(null);
      setSituationApplyStatus(false);
      clearConfigValidationState();
      setFooter("Preset angewendet.");
    } catch (err) {
      setFooter(`Preset anwenden fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-validate")?.addEventListener("click", async () => {
    try {
      const patched = await applyPreview({ persist: false });
      const result = await api.post(API_ENDPOINTS.config.validate, { yaml: patched?.config_yaml || "" });
      setParameterValidateStatus(result);
      setParameterValidateDetails(result);
      setConfigValidationState({ yaml: patched?.config_yaml || "", ok: Boolean(result?.ok) });
      setFooter(result.ok ? "Validierung OK." : "Validierung hat Fehler.");
    } catch (err) {
      setParameterValidateStatus(null, "Validierung: fehlgeschlagen");
      setParameterValidateDetails(null);
      clearConfigValidationState();
      setFooter(`Validierung fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-save")?.addEventListener("click", async () => {
    try {
      const result = await saveParameterConfig("");
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterValidateDetails(null);
      setSituationApplyStatus(false);
      clearConfigValidationState();
      setFooter(`Config gespeichert. Revision: ${result?.revision_id || "-"}`);
    } catch (err) {
      setFooter(`Speichern fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-save-as")?.addEventListener("click", async () => {
    try {
      const targetPath = await chooseConfigSaveAsPath();
      if (!targetPath) return;
      const result = await saveParameterConfig(targetPath);
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterValidateDetails(null);
      setSituationApplyStatus(false);
      clearConfigValidationState();
      setFooter(`Config gespeichert unter ${result?.path || targetPath}. Revision: ${result?.revision_id || "-"}`);
    } catch (err) {
      setFooter(`Speichern unter fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-review-changes")?.addEventListener("click", async () => {
    try {
      const result = await applyPreview({ persist: false });
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterValidateDetails(null);
      setSituationApplyStatus(false);
      clearConfigValidationState();
      setFooter(`YAML-Vorschau aktualisiert (${result?.applied?.length || 0} Aenderungen).`);
    } catch (err) {
      setFooter(`Vorschau fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-reset-default")?.addEventListener("click", async () => {
    try {
      const current = await api.get(API_ENDPOINTS.config.current);
      uiState.parameterDirty = {};
      uiState.configYaml = String(current?.config || "");
      setConfigDraft(uiState.configYaml);
      const parsed = await patchConfig({ yamlText: uiState.configYaml, updates: [] });
      if (parsed?.config) syncParameterFieldsFromConfig(parsed.config);
      setParameterPreview(uiState.configYaml);
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterValidateDetails(null);
      setSituationApplyStatus(false);
      clearConfigValidationState();
      setFooter("Werte auf aktuelle Config zurueckgesetzt.");
    } catch (err) {
      setFooter(`Reset fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-situation-apply")?.addEventListener("click", async () => {
    try {
      const scenarioUpdates = [];
      for (const key of activeScenarioKeys(".app-content")) {
        for (const [path, value] of SCENARIO_DELTAS[key] || []) {
          scenarioUpdates.push({ path, value });
        }
      }
      if (scenarioUpdates.length === 0) {
        setFooter("Keine Situation ausgewaehlt.", true);
        return;
      }
      const patched = await patchConfig({ updates: scenarioUpdates, persist: false });
      uiState.parameterDirty = {};
      if (patched?.config) syncParameterFieldsFromConfig(patched.config);
      setParameterPreview(patched?.config_yaml || "");
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterValidateDetails(null);
      setSituationApplyStatus(true, `${t("ui.status.situation_applied", "Angewendet")} (${scenarioUpdates.length})`);
      clearConfigValidationState();
      setFooter(`Situation angewendet (${scenarioUpdates.length} Deltas).`);
    } catch (err) {
      setFooter(`Situation anwenden fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelectorAll(".ps-chip-btn[data-scenario]").forEach((btn) => {
    btn.addEventListener("click", () => {
      window.setTimeout(() => setSituationApplyStatus(false), 0);
    });
  });

  document.addEventListener("gui2:locale-changed", () => {
    const statusEl = parameterSituationApplyStatusEl();
    if (!statusEl || statusEl.style.display === "none") return;
    const countMatch = String(statusEl.textContent || "").match(/\((\d+)\)\s*$/);
    const countText = countMatch ? ` (${countMatch[1]})` : "";
    setSituationApplyStatus(true, `${t("ui.status.situation_applied", "Angewendet")}${countText}`);
  });

  document.querySelectorAll("#parameter-category-list button[data-category]").forEach((btn) => {
    btn.addEventListener("click", () => {
      window.setTimeout(() => {
        if (uiState.configObject) syncParameterFieldsFromConfig(uiState.configObject);
      }, 0);
    });
  });
}

async function populatePresetSelect(selectId, preserveCurrentValue = true) {
  const select = $(selectId);
  if (!select) return;
  const oldValue = String(select.value || "").trim();
  const presets = await api.get(API_ENDPOINTS.config.presets);
  const items = Array.isArray(presets?.items) ? presets.items : [];
  if (items.length === 0) return;
  select.innerHTML = "";
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = String(item?.path || "");
    opt.textContent = String(item?.name || item?.path || "preset");
    select.appendChild(opt);
  }
  if (preserveCurrentValue && oldValue) {
    const matching = Array.from(select.options).find((opt) => String(opt.value || "") === oldValue || String(opt.textContent || "") === oldValue);
    if (matching) {
      select.value = matching.value;
      return;
    }
  }
  if (!preserveCurrentValue && items[0]?.path) {
    select.value = String(items[0].path);
  }
}

function runMonitorSelectedPhase() {
  const selected = document.querySelector(".ps-phase-row.is-selected .phase-name");
  return selected ? String(selected.textContent || "").trim().toUpperCase() : "";
}

function runMonitorSelectedFilter() {
  const chipRow = $("monitor-filter-row");
  if (chipRow && chipRow.style.display === "none") return "";
  const selected = document.querySelector(".ps-chip-btn.active[id^='monitor-filter-']");
  if (!selected) return "";
  return String(selected.textContent || "").trim().toUpperCase();
}

function setRunMonitorFilterVisibility(colorModeRaw) {
  const chipRow = $("monitor-filter-row");
  if (!chipRow) return;
  const colorMode = String(colorModeRaw || "").trim().toUpperCase();
  const hideFilters = colorMode === "OSC";
  chipRow.style.display = hideFilters ? "none" : "";
  const chipButtons = Array.from(document.querySelectorAll(".ps-chip-btn[id^='monitor-filter-']"));
  if (hideFilters) {
    chipButtons.forEach((btn) => btn.classList.remove("active"));
    return;
  }
  if (!document.querySelector(".ps-chip-btn.active[id^='monitor-filter-']")) {
    chipButtons[0]?.classList.add("active");
  }
}

function runMonitorLogBox() {
  return findLogBoxBySectionTitle("Live Log");
}

function artifactPathFromAbsolutePath(baseDir, targetPath, fallbackPath = "") {
  const base = String(baseDir || "").trim().replace(/\\/g, "/").replace(/\/+$/, "");
  const target = String(targetPath || "").trim().replace(/\\/g, "/");
  if (!target) return String(fallbackPath || "");
  if (!base) return target;
  if (target === base) return "";
  if (target.startsWith(`${base}/`)) return target.slice(base.length + 1);
  return String(fallbackPath || target);
}

function setMonitorReportAvailable(enabled) {
  setDisabledLike(monitorReportBtn(), !enabled);
}

async function openExternalPath(path) {
  const targetPath = String(path || "").trim();
  if (!targetPath) throw new Error("Pfad fehlt.");
  return withPathGrantRetry(
    () => api.post(API_ENDPOINTS.fs.openPath, { path: targetPath }),
    { fallbackPath: targetPath },
  );
}

function reportArtifactPath(runDir, reportPath) {
  return artifactPathFromAbsolutePath(runDir, reportPath, "artifacts/report.html");
}

function openRunReportInNewTab(runId, runDir, reportPath) {
  const artifactPath = reportArtifactPath(runDir, reportPath);
  if (!runId || !artifactPath) {
    throw new Error("Report nicht verfuegbar.");
  }
  const reportUrl = api.httpUrl(API_ENDPOINTS.runs.artifactRaw(runId, artifactPath));
  const targetWindow = window.open(reportUrl, "_blank");
  if (!targetWindow) {
    throw new Error("Report konnte nicht in neuem Tab geoeffnet werden.");
  }
  return { artifactPath, reportUrl };
}

function findReportArtifactPath(artifacts) {
  const items = Array.isArray(artifacts) ? artifacts : [];
  const match = items.find((item) => {
    const relativePath = String(item?.relative_path || item?.filename || item?.path || "").replace(/\\/g, "/").toLowerCase();
    return relativePath === "artifacts/report.html" || relativePath.endsWith("/artifacts/report.html");
  });
  return match ? String(match.relative_path || match.filename || match.path || "").replace(/\\/g, "/") : "";
}

function setPhaseRow(phaseName, status, pctRaw) {
  const row = Array.from(document.querySelectorAll(".ps-phase-row")).find((el) => {
    const name = el.querySelector(".phase-name");
    return name && String(name.textContent || "").trim().toUpperCase() === String(phaseName || "").toUpperCase();
  });
  if (!row) return;

  row.classList.remove("done", "running", "pending", "error");
  const normalized = String(status || "pending").toLowerCase();
  if (normalized === "ok" || normalized === "completed" || normalized === "done" || normalized === "skipped") {
    row.classList.add("done");
    const stateEl = row.querySelector(".state");
    if (stateEl) stateEl.textContent = "OK";
  } else if (normalized === "running") {
    row.classList.add("running");
    const stateEl = row.querySelector(".state");
    if (stateEl) stateEl.textContent = "RUN";
  } else if (normalized === "error" || normalized === "failed" || normalized === "aborted") {
    row.classList.add("error");
    const stateEl = row.querySelector(".state");
    if (stateEl) stateEl.textContent = "ERR";
  } else {
    row.classList.add("pending");
    const stateEl = row.querySelector(".state");
    if (stateEl) stateEl.textContent = "P";
  }

  const pctEl = row.querySelector(".phase-progress");
  if (!pctEl) return;
  let pct = Number(pctRaw || 0);
  if (Number.isFinite(pct) && pct <= 1.0) pct *= 100.0;
  if (!Number.isFinite(pct)) pct = 0;
  pct = Math.max(0, Math.min(100, pct));
  pctEl.textContent = `${pct.toFixed(0)}%`;
}

async function loadRunStatus(runId) {
  const status = await api.get(API_ENDPOINTS.runs.status(runId));
  uiState.currentRunDir = String(status?.run_dir || "");
  const fallbackScanColorMode = String(localStorage.getItem("gui2.lastScanColorMode") || "").trim().toUpperCase();
  const effectiveColorMode = String(status?.color_mode || "").trim().toUpperCase() || fallbackScanColorMode;
  setRunMonitorFilterVisibility(effectiveColorMode);
  if (Array.isArray(status?.phases)) {
    for (const p of status.phases) {
      setPhaseRow(p.phase, p.status, p.pct);
    }
  }
  const sub = document.querySelector(".app-content .ps-sub");
  if (sub) {
    sub.innerHTML = `Run-ID <code>${runId}</code>, Status <code>${status.status || "unknown"}</code>, Phase <code>${status.current_phase || "-"}</code>.`;
  }
  return status;
}

async function loadRunRevisions() {
  const sel = $("monitor-resume-config-revision");
  if (!sel) return;
  const old = sel.value;
  const revisions = await api.get(API_ENDPOINTS.config.revisions);
  sel.innerHTML = "";
  for (const item of revisions.items || []) {
    const opt = document.createElement("option");
    opt.value = item.revision_id;
    opt.textContent = item.revision_id;
    sel.appendChild(opt);
  }
  if (old) sel.value = old;
}

function connectRunMonitorStream(runId) {
  if (!runId) return;
  if (uiState.runSocket) uiState.runSocket.close();
  const logBox = runMonitorLogBox();
  if (logBox) scrollLogToEnd(logBox);
  uiState.runSocket = api.ws(
    API_ENDPOINTS.ws.run(runId),
    (event) => {
      enqueueRunMonitorLogLine(JSON.stringify(event));
      if (event?.type === "phase_progress" || event?.type === "phase_end" || event?.type === "phase_start") {
        const payload = event.payload || {};
        const phase = payload.phase_name || payload.phase || event.phase || "";
        const status = payload.status || (event.type === "phase_start" ? "running" : event.type === "phase_end" ? "ok" : "running");
        const pct = payload.progress ?? payload.pct ?? event.pct ?? 0;
        if (phase) setPhaseRow(phase, status, pct);
      }
      if (event?.type === "run_status" && event?.payload?.phases) {
        for (const p of event.payload.phases) {
          setPhaseRow(p.phase, p.status, p.pct);
        }
      }
    },
    (err) => {
      enqueueRunMonitorLogLine(`ws_error: ${String(err)}`);
    },
  );
}

async function bindRunMonitor() {
  if (!$("monitor-stop")) return;

  const startBtn = $("monitor-start");
  const stopBtn = $("monitor-stop");
  const statsGenerateBtn = $("monitor-stats-generate");
  const statsOpenFolderBtn = $("monitor-stats-open-folder");
  const statsStatusEl = $("monitor-stats-status");
  const sub = document.querySelector(".app-content .ps-sub");
  const updateResumeEnabled = () => {
    const phase = runMonitorSelectedPhase();
    const revisionId = $("monitor-resume-config-revision")?.value || "";
    setDisabledLike($("monitor-resume"), !uiState.currentRunId || !phase || !revisionId);
    setDisabledLike($("monitor-resume-restore-revision"), !revisionId);
  };
  document.querySelectorAll(".ps-phase-row").forEach((row) => {
    row.addEventListener("click", () => {
      document.querySelectorAll(".ps-phase-row").forEach((x) => x.classList.remove("is-selected"));
      row.classList.add("is-selected");
      updateResumeEnabled();
    });
  });
  document.querySelectorAll(".ps-chip-btn[id^='monitor-filter-']").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".ps-chip-btn[id^='monitor-filter-']").forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
    });
  });
  $("monitor-resume-config-revision")?.addEventListener("change", updateResumeEnabled);

  const artifactSection = Array.from(document.querySelectorAll(".ps-section")).find((sec) => {
    const title = sec.querySelector(".ps-section-title");
    return title && String(title.textContent || "").trim() === "Artefakte";
  });
  const artifactList = $("monitor-artifact-list") || artifactSection?.querySelector("ul.ps-list") || null;
  const artifactViewer = $("monitor-artifact-viewer");
  const artifactViewerTitle = $("monitor-artifact-viewer-title");
  const artifactViewerBody = $("monitor-artifact-viewer-body");
  const artifactViewerClose = $("monitor-artifact-viewer-close");
  const formatBytes = (sizeRaw) => {
    const size = Number(sizeRaw);
    if (!Number.isFinite(size) || size < 0) return "-";
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };
  const isDisplayArtifact = (item) => {
    const filename = String(item?.filename || item?.relative_path || item?.path || "").trim();
    if (!filename) return false;
    if (/\.(fit|fits)$/i.test(filename)) return false;
    if (/^frame_\d+\.(fit|fits)$/i.test(filename)) return false;
    if (/(^|\/|\\)frame_\d+\.(fit|fits)$/i.test(String(item?.path || ""))) return false;
    return true;
  };
  const formatArtifactContent = (payload) => {
    if (payload?.is_json && payload?.json !== null && payload?.json !== undefined) {
      return JSON.stringify(payload.json, null, 2);
    }
    const filename = String(payload?.filename || "").toLowerCase();
    const text = String(payload?.text || "");
    if (filename.endsWith(".jsonl")) {
      const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
      try {
        return JSON.stringify(lines.map((line) => JSON.parse(line)), null, 2);
      } catch {
        return text;
      }
    }
    try {
      return JSON.stringify(JSON.parse(text), null, 2);
    } catch {
      return text;
    }
  };
  const closeArtifactViewer = () => {
    if (!artifactViewer) return;
    artifactViewer.hidden = true;
  };
  const openArtifactViewer = async (path, title) => {
    if (!uiState.currentRunId || !artifactViewer || !artifactViewerBody) return;
    artifactViewer.hidden = false;
    if (artifactViewerTitle) artifactViewerTitle.textContent = title || "Artefakt";
    artifactViewerBody.textContent = "Lade Artefakt ...";
    try {
      const payload = await api.get(API_ENDPOINTS.runs.artifactView(uiState.currentRunId, path));
      if (artifactViewerTitle) artifactViewerTitle.textContent = String(payload?.filename || title || "Artefakt");
      artifactViewerBody.textContent = formatArtifactContent(payload);
    } catch (err) {
      artifactViewerBody.textContent = `Artefakt konnte nicht geladen werden:\n${errorText(err)}`;
    }
  };
  const ensureCurrentRunStatus = async () => {
    if (uiState.currentRunDir && uiState.currentRunId) {
      return { run_dir: uiState.currentRunDir, run_id: uiState.currentRunId };
    }
    if (!uiState.currentRunId) return null;
    const status = await loadRunStatus(uiState.currentRunId);
    return status;
  };
  artifactViewerClose?.addEventListener("click", closeArtifactViewer);
  artifactViewer?.addEventListener("click", (ev) => {
    if (ev.target === artifactViewer) closeArtifactViewer();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeArtifactViewer();
  });
  const renderArtifacts = (items) => {
    if (!artifactList) return;
    const artifacts = (Array.isArray(items) ? items : []).filter(isDisplayArtifact);
    if (artifacts.length === 0) {
      artifactList.innerHTML = "<li><button>Keine Artefakte gefunden</button></li>";
      return;
    }
    artifactList.innerHTML = artifacts
      .slice(0, 50)
      .map((item) => {
        const filename = String(item?.filename || item?.relative_path || item?.path || "artifact");
        const relativePath = String(item?.relative_path || filename);
        const artifactPath = String(item?.relative_path || item?.filename || item?.path || "");
        const sizeText = formatBytes(item?.size_bytes);
        return `<li><button data-artifact-path="${artifactPath.replace(/"/g, "&quot;")}" title="${relativePath}">${filename} (${sizeText})</button></li>`;
      })
      .join("");
    artifactList.querySelectorAll("button[data-artifact-path]").forEach((btn) => {
      btn.addEventListener("click", () => {
        void openArtifactViewer(
          btn.getAttribute("data-artifact-path") || "",
          btn.textContent || btn.getAttribute("title") || "Artefakt",
        );
      });
    });
  };
  const refreshArtifacts = async () => {
    if (!uiState.currentRunId) {
      renderArtifacts([]);
      return;
    }
    const result = await api.get(API_ENDPOINTS.runs.artifacts(uiState.currentRunId));
    renderArtifacts(result?.items || []);
  };
  const refreshStatsActions = async () => {
    if (!uiState.currentRunId) {
      uiState.monitorStatsStatus = null;
      setMonitorReportAvailable(false);
      setInlineAsyncStatus(statsStatusEl, "");
      return null;
    }
    const status = await api.get(API_ENDPOINTS.runs.statsStatus(uiState.currentRunId)).catch(() => null);
    uiState.monitorStatsStatus = status;
    const hasReport = Boolean(String(status?.report_path || "").trim());
    setMonitorReportAvailable(hasReport);
    if (String(status?.state || "").toLowerCase() === "running") {
      setInlineAsyncStatus(statsStatusEl, t("ui.status.stats_running", "Stats laeuft"), "running");
    } else if (hasReport) {
      setInlineAsyncStatus(statsStatusEl, t("ui.status.stats_completed", "Stats beendet"), "ok");
    } else {
      setInlineAsyncStatus(statsStatusEl, "");
    }
    return status;
  };
  const setMonitorActionState = (isActive) => {
    const hasRun = Boolean(String(uiState.currentRunId || "").trim());
    setDisabledLike(startBtn, isActive);
    setDisabledLike(stopBtn, !isActive);
    setDisabledLike(statsGenerateBtn, isActive || !hasRun);
    setDisabledLike(statsOpenFolderBtn, isActive || !hasRun);
    if (isActive || !hasRun) {
      setMonitorReportAvailable(false);
      if (!hasRun) setInlineAsyncStatus(statsStatusEl, "");
    }
  };
  const resetPhaseRows = () => {
    document.querySelectorAll(".ps-phase-row").forEach((row) => {
      row.classList.remove("done", "running", "error", "is-selected");
      row.classList.add("pending");
      const stateEl = row.querySelector(".state");
      if (stateEl) stateEl.textContent = "P";
      const pctEl = row.querySelector(".phase-progress");
      if (pctEl) pctEl.textContent = "0%";
    });
  };
  const renderNoRunState = (text) => {
    if (uiState.runSocket) {
      uiState.runSocket.close();
      uiState.runSocket = null;
    }
    if (uiState.runLogFlushTimer) {
      clearTimeout(uiState.runLogFlushTimer);
      uiState.runLogFlushTimer = null;
    }
    uiState.runLogLines = [];
    uiState.runLogPending = [];
    uiState.currentRunDir = "";
    resetPhaseRows();
    renderArtifacts([]);
    setMonitorReportAvailable(false);
    const logBox = runMonitorLogBox();
    if (logBox) {
      logBox.textContent = "";
      scrollLogToEnd(logBox);
    }
    if (sub) sub.textContent = text;
  };

  updateResumeEnabled();
  void refreshRunMonitorValidationMessage();

  $("monitor-start")?.addEventListener("click", async () => {
    try {
      const validationMessage = await refreshRunMonitorValidationMessage();
      if (validationMessage) {
        setFooter(validationMessage, true);
        return;
      }
      const appState = await api.get(API_ENDPOINTS.app.state).catch(() => ({ run: { current: {} }, project: {} }));
      const currentStatus = String(appState?.run?.current?.status || "").trim().toLowerCase();
      if (currentStatus === "running") {
        setMonitorActionState(true);
        setFooter("Es läuft bereits ein aktiver Run.", true);
        return;
      }
      const latestGuardrails = await api.get(API_ENDPOINTS.guardrails.root);
      if (String(latestGuardrails?.status || "").toLowerCase() === "error") {
        setFooter("Run blockiert: Guardrail-Status ist ERROR.", true);
        return;
      }
      const accepted = await startRunFromCurrentForm({ source: "monitor" });
      setCurrentRunId(accepted?.run_id || uiState.currentRunId);
      setMonitorStartValidationMessage("");
      setFooter(`Run gestartet (Job ${accepted?.job_id || "-"}).`);
      window.location.reload();
    } catch (err) {
      setFooter(`Run-Start fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-stop")?.addEventListener("click", async () => {
    if (!uiState.currentRunId) return;
    try {
      const result = await api.post(API_ENDPOINTS.runs.stop(uiState.currentRunId), {});
      if (result.ok) {
        const stoppedJobs = Array.isArray(result.cancelled_jobs) ? result.cancelled_jobs.length : 0;
        const killedPids = Array.isArray(result.killed_pids) ? result.killed_pids.length : 0;
        setFooter(`Stop gesendet. Jobs beendet: ${stoppedJobs}, verwaiste Prozesse beendet: ${killedPids}.`);
      } else {
        setFooter("Kein laufender Job/Prozess fuer diesen Run gefunden.", true);
      }
      const status = await loadRunStatus(uiState.currentRunId);
      setMonitorActionState(String(status?.status || "").toLowerCase() === "running");
    } catch (err) {
      setFooter(`Stop fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-resume")?.addEventListener("click", async () => {
    const phase = runMonitorSelectedPhase();
    const revisionId = $("monitor-resume-config-revision")?.value || "";
    if (!phase || !revisionId) {
      setFooter("Bitte Phase und Config-Revision waehlen.", true);
      return;
    }
    try {
      const accepted = await api.post(API_ENDPOINTS.runs.resume(uiState.currentRunId), {
        from_phase: phase,
        config_revision_id: revisionId,
        run_dir: uiState.currentRunDir || undefined,
        filter_context: runMonitorSelectedFilter() || undefined,
      });
      setFooter(`Resume gestartet (Job ${accepted.job_id}).`);
      await waitForJob(accepted.job_id);
      await loadRunStatus(uiState.currentRunId);
    } catch (err) {
      setFooter(`Resume fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-resume-restore-revision")?.addEventListener("click", async () => {
    const revisionId = $("monitor-resume-config-revision")?.value || "";
    if (!revisionId) {
      setFooter("Bitte Config-Revision waehlen.", true);
      return;
    }
    try {
      await api.post(API_ENDPOINTS.runs.restoreRevision(uiState.currentRunId, revisionId), {});
      const current = await api.get(API_ENDPOINTS.config.current);
      setConfigDraft(String(current?.config || ""));
      setFooter(`Revision ${revisionId} wiederhergestellt.`);
    } catch (err) {
      setFooter(`Revision-Restore fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-stats-generate")?.addEventListener("click", async () => {
    try {
      const appState = await api.get(API_ENDPOINTS.app.state).catch(() => ({ run: { current: {} } }));
      const currentStatus = String(appState?.run?.current?.status || "").trim().toLowerCase();
      if (isRunActiveStatus(currentStatus)) {
        setMonitorActionState(true);
        setFooter("Stats erst nach beendetem Run verfuegbar.", true);
        return;
      }
      const accepted = await api.post(API_ENDPOINTS.runs.stats(uiState.currentRunId), {
        run_dir: uiState.currentRunDir || undefined,
      });
      setInlineAsyncStatus(statsStatusEl, t("ui.status.stats_running", "Stats laeuft"), "running");
      setFooter(statsStartedMessage(accepted.job_id));
      await waitForJob(accepted.job_id);
      await refreshArtifacts();
      await refreshStatsActions();
      setFooter(t("ui.message.stats_completed", "Stats-Generierung beendet."));
    } catch (err) {
      setFooter(statsFailedMessage(err), true);
    }
  });

  $("monitor-stats-open-folder")?.addEventListener("click", async () => {
    try {
      const appState = await api.get(API_ENDPOINTS.app.state).catch(() => ({ run: { current: {} } }));
      const currentStatus = String(appState?.run?.current?.status || "").trim().toLowerCase();
      if (isRunActiveStatus(currentStatus)) {
        setMonitorActionState(true);
        setFooter("Stats-Ordner erst nach beendetem Run verfuegbar.", true);
        return;
      }
      const status = await api.get(API_ENDPOINTS.runs.statsStatus(uiState.currentRunId));
      const targetDir = String(status.output_dir || "").trim();
      if (!targetDir) {
        setFooter("Stats-Ordner nicht verfuegbar.", true);
        return;
      }
      await openExternalPath(targetDir);
      setFooter(`Stats-Ordner: ${targetDir}`);
    } catch (err) {
      setFooter(`Stats-Status fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.addEventListener("gui2:locale-changed", () => {
    void refreshRunMonitorValidationMessage();
  });

  $("monitor-report")?.addEventListener("click", async () => {
    try {
      const status = uiState.monitorStatsStatus;
      if (!status?.report_path) {
        setFooter("Report erst nach Generate Stats verfuegbar.", true);
        return;
      }
      const runStatus = uiState.currentRunDir ? { run_dir: uiState.currentRunDir } : await ensureCurrentRunStatus();
      const { artifactPath } = openRunReportInNewTab(
        uiState.currentRunId,
        runStatus?.run_dir || uiState.currentRunDir,
        status.report_path,
      );
      setFooter(`Report: ${status.report_path || artifactPath}`);
    } catch (err) {
      setFooter(`Report-Status fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-open-run-folder")?.addEventListener("click", async () => {
    try {
      const runStatus = await ensureCurrentRunStatus();
      const runDir = String(runStatus?.run_dir || uiState.currentRunDir || "").trim();
      if (!runDir) {
        setFooter("Run-Ordner nicht verfuegbar.", true);
        return;
      }
      await openExternalPath(runDir);
      setFooter(`Run-Ordner: ${runDir}`);
    } catch (err) {
      setFooter(`Run-Ordner konnte nicht geoeffnet werden: ${errorText(err)}`, true);
    }
  });

  try {
    await loadRunRevisions();
    const appState = await api.get(API_ENDPOINTS.app.state).catch(() => ({ project: {}, run: { current: {} } }));
    const currentRunId = String(appState?.project?.current_run_id || "").trim();
    if (currentRunId) setCurrentRunId(currentRunId);
    const hintedRunId = currentRunId || ensureRunIdFromHeader();
    if (!currentRunId && !hintedRunId) {
      clearCurrentRunId();
      setMonitorActionState(false);
      renderNoRunState("Kein aktiver Run. Start über Run starten.");
      updateResumeEnabled();
      return;
    }
    if (!currentRunId && hintedRunId) setCurrentRunId(hintedRunId);
    const status = await loadRunStatus(uiState.currentRunId);
    await refreshArtifacts();
    await refreshStatsActions();
    const isActive = isRunActiveStatus(status?.status || appState?.run?.current?.status || "");
    setMonitorActionState(isActive);
    if (isActive) {
      setMonitorStartValidationMessage("");
      connectRunMonitorStream(uiState.currentRunId);
    } else if (uiState.runSocket) {
      uiState.runSocket.close();
      uiState.runSocket = null;
    }
    updateResumeEnabled();
  } catch (err) {
    setFooter(`Run-Monitor Initialisierung fehlgeschlagen: ${errorText(err)}`, true);
  }
}

async function bindHistoryPage() {
  const list = document.querySelector(".ps-section ul.ps-list");
  if (!list || !$("history-refresh")) return;

  const historySourcePath = $("history-source-path");
  const selectedRunIdField = $("history-selected-run-id");
  const selectedStatusField = $("history-selected-status");
  const selectedPhaseField = $("history-selected-phase");
  const selectedProgressField = $("history-selected-progress");
  const selectedArtifactsField = $("history-selected-artifacts");
  const selectedReportField = $("history-selected-report");
  const selectedRunDirField = $("history-selected-run-dir");
  const historyStatsGenerateBtn = $("history-stats-generate");
  const historyStatsOpenFolderBtn = $("history-stats-open-folder");
  const historyOpenReportBtn = $("history-open-report");
  const historyStatsStatusEl = $("history-stats-status");
  const compareRunSelect = $("history-compare-run-id");
  const compareStatusField = $("history-compare-status");
  const comparePhaseField = $("history-compare-phase");
  const compareProgressField = $("history-compare-progress");
  const compareArtifactsField = $("history-compare-artifacts");
  const compareReportField = $("history-compare-report");
  const compareRunDirField = $("history-compare-run-dir");
  const compareSummaryField = $("history-compare-summary");
  let selectedSnapshotCache = null;

  const setHistoryFieldValue = (el, value) => {
    if (!el) return;
    el.value = value === null || value === undefined || value === "" ? "-" : String(value);
  };
  const formatHistoryProgress = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "-";
    const pct = numeric <= 1 ? numeric * 100 : numeric;
    return `${pct.toFixed(1)}%`;
  };
  const clearHistoryDetails = (refs, summaryText = "-") => {
    setHistoryFieldValue(refs.runIdField, "-");
    setHistoryFieldValue(refs.statusField, "-");
    setHistoryFieldValue(refs.phaseField, "-");
    setHistoryFieldValue(refs.progressField, "-");
    setHistoryFieldValue(refs.artifactsField, "-");
    setHistoryFieldValue(refs.reportField, "-");
    setHistoryFieldValue(refs.runDirField, "-");
    if (refs.summaryField) setHistoryFieldValue(refs.summaryField, summaryText);
  };
  const applyHistorySnapshot = (snapshot, refs) => {
    if (!snapshot) {
      clearHistoryDetails(refs);
      return;
    }
    setHistoryFieldValue(refs.runIdField, snapshot.runId);
    setHistoryFieldValue(refs.statusField, snapshot.status);
    setHistoryFieldValue(refs.phaseField, snapshot.currentPhase);
    setHistoryFieldValue(refs.progressField, snapshot.progressText);
    setHistoryFieldValue(refs.artifactsField, String(snapshot.artifactCount));
    setHistoryFieldValue(refs.reportField, snapshot.reportPath);
    setHistoryFieldValue(refs.runDirField, snapshot.runDir);
  };
  const updateHistoryActionState = (snapshot) => {
    const hasRun = Boolean(String(snapshot?.runId || uiState.selectedHistoryRunId || "").trim());
    const hasReport = Boolean(String(snapshot?.reportPath || "").trim() && String(snapshot?.reportPath || "").trim() !== "-");
    setDisabledLike(historyStatsGenerateBtn, !hasRun);
    setDisabledLike(historyStatsOpenFolderBtn, !hasReport);
    setDisabledLike(historyOpenReportBtn, !hasReport);
    if (!hasRun) setInlineAsyncStatus(historyStatsStatusEl, "");
    else if (String(snapshot?.statsState || "").toLowerCase() === "running") setInlineAsyncStatus(historyStatsStatusEl, t("ui.status.stats_running", "Stats laeuft"), "running");
    else if (hasReport) setInlineAsyncStatus(historyStatsStatusEl, t("ui.status.stats_completed", "Stats beendet"), "ok");
    else setInlineAsyncStatus(historyStatsStatusEl, "");
  };
  const loadRunSnapshot = async (runId) => {
    if (!runId) return null;
    const [runStatus, statsStatus, artifactResult] = await Promise.all([
      api.get(API_ENDPOINTS.runs.status(runId)),
      api.get(API_ENDPOINTS.runs.statsStatus(runId)).catch(() => ({ report_path: "", output_dir: "", state: "unknown" })),
      api.get(API_ENDPOINTS.runs.artifacts(runId)).catch(() => ({ items: [] })),
    ]);
    const artifacts = Array.isArray(artifactResult?.items) ? artifactResult.items : [];
    const runDir = String(runStatus?.run_dir || "-");
    const reportArtifactPath = findReportArtifactPath(artifacts);
    const resolvedReportPath = String(statsStatus?.report_path || "").trim()
      || (reportArtifactPath && runDir && runDir !== "-" ? `${runDir}/${reportArtifactPath}` : "");
    const resolvedStatsOutputDir = String(statsStatus?.output_dir || "").trim()
      || (resolvedReportPath ? parentDirOfPath(resolvedReportPath) : "");
    const resolvedStatsState = String(statsStatus?.state || "").trim()
      || (resolvedReportPath ? "ok" : "unknown");
    const progressValue = Number(runStatus?.progress);
    return {
      runId,
      status: runStatus?.status || "-",
      currentPhase: runStatus?.current_phase || "-",
      progressValue,
      progressText: formatHistoryProgress(runStatus?.progress),
      artifactCount: artifacts.length,
      reportPath: resolvedReportPath || "-",
      statsOutputDir: resolvedStatsOutputDir || "",
      statsState: resolvedStatsState || "unknown",
      runDir,
    };
  };
  const selectedRefs = {
    runIdField: selectedRunIdField,
    statusField: selectedStatusField,
    phaseField: selectedPhaseField,
    progressField: selectedProgressField,
    artifactsField: selectedArtifactsField,
    reportField: selectedReportField,
    runDirField: selectedRunDirField,
  };
  const compareRefs = {
    runIdField: compareRunSelect,
    statusField: compareStatusField,
    phaseField: comparePhaseField,
    progressField: compareProgressField,
    artifactsField: compareArtifactsField,
    reportField: compareReportField,
    runDirField: compareRunDirField,
    summaryField: compareSummaryField,
  };
  const renderSelectedRunDetails = async () => {
    if (!uiState.selectedHistoryRunId) {
      clearHistoryDetails(selectedRefs);
      selectedSnapshotCache = null;
      updateHistoryActionState(null);
      return null;
    }
    const snapshot = await loadRunSnapshot(uiState.selectedHistoryRunId);
    selectedSnapshotCache = snapshot;
    applyHistorySnapshot(snapshot, selectedRefs);
    updateHistoryActionState(snapshot);
    return snapshot;
  };
  const renderCompareOptions = (items) => {
    if (!compareRunSelect) return;
    const compareCandidates = items.filter((item) => item.run_id !== uiState.selectedHistoryRunId);
    compareRunSelect.innerHTML = [
      '<option value="">-</option>',
      ...compareCandidates.map(
        (item) => `<option value="${item.run_id}">${item.status.toUpperCase()} ${item.run_id} | ${item.name}</option>`,
      ),
    ].join("");
    if (!compareCandidates.some((item) => item.run_id === uiState.compareHistoryRunId)) {
      uiState.compareHistoryRunId = "";
    }
    compareRunSelect.value = uiState.compareHistoryRunId || "";
  };
  const renderCompareDetails = async (selectedSnapshot) => {
    if (!uiState.compareHistoryRunId || uiState.compareHistoryRunId === uiState.selectedHistoryRunId) {
      clearHistoryDetails(compareRefs, "Vergleichs-Run wählen");
      if (compareRunSelect) compareRunSelect.value = "";
      return;
    }
    const snapshot = await loadRunSnapshot(uiState.compareHistoryRunId);
    applyHistorySnapshot(snapshot, compareRefs);
    const baseProgress = Number(selectedSnapshot?.progressValue);
    const compareProgress = Number(snapshot?.progressValue);
    const progressDelta = Number.isFinite(baseProgress) && Number.isFinite(compareProgress)
      ? `${compareProgress >= baseProgress ? "+" : ""}${((compareProgress - baseProgress) * 100).toFixed(1)} pp`
      : "-";
    const artifactDelta = Number(snapshot?.artifactCount || 0) - Number(selectedSnapshot?.artifactCount || 0);
    const artifactDeltaText = `${artifactDelta >= 0 ? "+" : ""}${artifactDelta}`;
    const statusText = selectedSnapshot && snapshot && String(selectedSnapshot.status) === String(snapshot.status)
      ? `Status gleich (${snapshot.status})`
      : `Status ${selectedSnapshot?.status || "-"} vs ${snapshot?.status || "-"}`;
    setHistoryFieldValue(compareSummaryField, `${statusText} | Δ Artefakte ${artifactDeltaText} | Δ Fortschritt ${progressDelta}`);
  };

  const render = async () => {
    const [runs, appState] = await Promise.all([
      api.get(API_ENDPOINTS.runs.list),
      api.get(API_ENDPOINTS.app.state).catch(() => ({ project: {} })),
    ]);
    if (historySourcePath) {
      const runsDir = String(appState?.project?.runs_dir || "").trim();
      historySourcePath.textContent = runsDir ? `Quelle: ${runsDir}` : "Quelle: -";
    }
    const items = Array.isArray(runs?.items) ? runs.items : [];
    if (items.length === 0) {
      list.innerHTML = "<li><button>Keine Runs gefunden</button></li>";
      clearHistoryDetails(selectedRefs);
      clearHistoryDetails(compareRefs, "Vergleichs-Run wählen");
      if (compareRunSelect) compareRunSelect.innerHTML = '<option value="">-</option>';
      selectedSnapshotCache = null;
      updateHistoryActionState(null);
      return;
    }
    if (!items.some((item) => item.run_id === uiState.selectedHistoryRunId)) {
      uiState.selectedHistoryRunId = uiState.currentRunId && items.some((item) => item.run_id === uiState.currentRunId)
        ? uiState.currentRunId
        : items[0].run_id;
    }
    if (uiState.compareHistoryRunId === uiState.selectedHistoryRunId) {
      uiState.compareHistoryRunId = "";
    }
    list.innerHTML = items
      .slice(0, 50)
      .map((item) => {
        const active = item.run_id === uiState.selectedHistoryRunId ? " is-active" : "";
        return `<li><button class="${active}" data-run-id="${item.run_id}">${item.status.toUpperCase()} ${item.run_id} | ${item.name}</button></li>`;
      })
      .join("");
    list.querySelectorAll("button[data-run-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        uiState.selectedHistoryRunId = btn.getAttribute("data-run-id") || "";
        if (uiState.compareHistoryRunId === uiState.selectedHistoryRunId) uiState.compareHistoryRunId = "";
        render().catch((err) => {
          setFooter(`History laden fehlgeschlagen: ${errorText(err)}`, true);
        });
      });
    });
    renderCompareOptions(items);
    const selectedSnapshot = await renderSelectedRunDetails();
    await renderCompareDetails(selectedSnapshot);
  };

  $("history-refresh").addEventListener("click", () => void render());
  $("history-set-current")?.addEventListener("click", async () => {
    if (!uiState.selectedHistoryRunId) return;
    try {
      await api.post(API_ENDPOINTS.runs.setCurrent(uiState.selectedHistoryRunId), {});
      setCurrentRunId(uiState.selectedHistoryRunId);
      setFooter(`Current Run gesetzt: ${uiState.selectedHistoryRunId}`);
    } catch (err) {
      setFooter(`Set Current fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("history-stats-generate")?.addEventListener("click", async () => {
    const runId = String(uiState.selectedHistoryRunId || "").trim();
    if (!runId) return;
    try {
      const runDir = String(selectedSnapshotCache?.runDir || "").trim();
      const accepted = await api.post(API_ENDPOINTS.runs.stats(runId), {
        run_dir: runDir && runDir !== "-" ? runDir : undefined,
      });
      setInlineAsyncStatus(historyStatsStatusEl, t("ui.status.stats_running", "Stats laeuft"), "running");
      setFooter(statsStartedMessage(accepted.job_id));
      await waitForJob(accepted.job_id);
      await render();
      setFooter(t("ui.message.stats_completed", "Stats-Generierung beendet."));
    } catch (err) {
      setFooter(statsFailedMessage(err), true);
    }
  });

  $("history-stats-open-folder")?.addEventListener("click", async () => {
    const snapshot = selectedSnapshotCache;
    const targetDir = String(snapshot?.statsOutputDir || "").trim();
    if (!targetDir) {
      setFooter("Stats-Ordner nicht verfuegbar.", true);
      return;
    }
    try {
      await openExternalPath(targetDir);
      setFooter(`Stats-Ordner: ${targetDir}`);
    } catch (err) {
      setFooter(`Stats-Ordner konnte nicht geoeffnet werden: ${errorText(err)}`, true);
    }
  });

  $("history-open-report")?.addEventListener("click", async () => {
    if (!uiState.selectedHistoryRunId) return;
    try {
      const snapshot = selectedSnapshotCache;
      const reportPath = String(snapshot?.reportPath || "").trim();
      if (!reportPath || reportPath === "-") {
        setFooter("Report erst nach Generate Stats verfuegbar.", true);
        return;
      }
      const { artifactPath } = openRunReportInNewTab(uiState.selectedHistoryRunId, snapshot?.runDir, reportPath);
      setHistoryFieldValue(selectedReportField, reportPath);
      setFooter(`Report: ${reportPath || artifactPath}`);
    } catch (err) {
      setFooter(`Report-Status fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("history-delete-run")?.addEventListener("click", async () => {
    const runId = String(uiState.selectedHistoryRunId || "").trim();
    if (!runId) return;
    const confirmed = window.confirm(`Run wirklich löschen?\n${runId}`);
    if (!confirmed) return;
    try {
      await api.post(API_ENDPOINTS.runs.delete(runId), {});
      if (uiState.currentRunId === runId) clearCurrentRunId();
      if (uiState.compareHistoryRunId === runId) uiState.compareHistoryRunId = "";
      if (uiState.selectedHistoryRunId === runId) uiState.selectedHistoryRunId = "";
      setFooter(`Run gelöscht: ${runId}`);
      await render();
    } catch (err) {
      setFooter(`Run-Löschen fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  compareRunSelect?.addEventListener("change", () => {
    uiState.compareHistoryRunId = String(compareRunSelect.value || "").trim();
    render().catch((err) => {
      setFooter(`History laden fehlgeschlagen: ${errorText(err)}`, true);
    });
  });

  $("history-compare-use-current")?.addEventListener("click", () => {
    if (!uiState.currentRunId) {
      setFooter("Kein Current Run gesetzt.", true);
      return;
    }
    if (uiState.currentRunId === uiState.selectedHistoryRunId) {
      setFooter("Current Run ist bereits der ausgewählte Run. Bitte anderen Haupt-Run wählen.", true);
      return;
    }
    uiState.compareHistoryRunId = uiState.currentRunId;
    render().catch((err) => {
      setFooter(`History laden fehlgeschlagen: ${errorText(err)}`, true);
    });
  });

  $("history-compare-clear")?.addEventListener("click", () => {
    uiState.compareHistoryRunId = "";
    render().catch((err) => {
      setFooter(`History laden fehlgeschlagen: ${errorText(err)}`, true);
    });
  });

  try {
    await render();
  } catch (err) {
    setFooter(`History laden fehlgeschlagen: ${errorText(err)}`, true);
  }
}

async function bindAstrometryPage() {
  if (!$("tools-astrometry-bin")) return;
  const logBox = findLogBoxBySectionTitle("Log");
  const statusChip = document.querySelector("[data-control='tools.astrometry.status']");

  const raField = $("tools-astrometry-ra");
  const decField = $("tools-astrometry-dec");
  const pixelScaleField = $("tools-astrometry-pixel-scale");
  const rotationField = $("tools-astrometry-rotation");
  const fovField = $("tools-astrometry-fov");

  const append = (msg) => appendLine(logBox, typeof msg === "string" ? msg : JSON.stringify(msg));
  const setFieldValue = (el, value) => {
    if (!el) return;
    el.value = value === null || value === undefined || value === "" ? "-" : String(value);
  };
  const formatDeg = (value) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric.toFixed(6)} deg` : "-";
  };
  const formatPixelScale = (value) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric.toFixed(3)} arcsec/px` : "-";
  };
  const formatFov = (widthDeg, heightDeg) => {
    const w = Number(widthDeg);
    const h = Number(heightDeg);
    return Number.isFinite(w) && Number.isFinite(h) ? `${w.toFixed(3)} x ${h.toFixed(3)} deg` : "-";
  };
  const applyAstrometryResult = (payload) => {
    if (!payload || typeof payload !== "object") return;
    setFieldValue(raField, formatDeg(payload.ra_deg));
    setFieldValue(decField, formatDeg(payload.dec_deg));
    setFieldValue(pixelScaleField, formatPixelScale(payload.pixel_scale_arcsec));
    setFieldValue(rotationField, formatDeg(payload.rotation_deg));
    setFieldValue(fovField, formatFov(payload.fov_width_deg, payload.fov_height_deg));
    if (payload.wcs_path) {
      uiState.lastAstrometryWcs = String(payload.wcs_path);
    }
  };

  async function detect() {
    const payload = {
      astap_cli: $("tools-astrometry-bin")?.value || "",
      astap_data_dir: $("tools-astrometry-data-dir")?.value || "",
    };
    const result = await withPathGrantRetry(
      () => api.post(API_ENDPOINTS.astrometry.detect, payload),
      { fallbackPath: payload.astap_cli || payload.astap_data_dir },
    );
    if (statusChip) statusChip.textContent = result.installed ? "Installed" : "Missing";
    append(result);
  }

  document.querySelector("[data-control='tools.astrometry.detect']")?.addEventListener("click", async () => {
    try {
      await detect();
      setFooter("Astrometry-Detection aktualisiert.");
    } catch (err) {
      setFooter(`Astrometry detect fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.astrometry.install_cli']")?.addEventListener("click", async () => {
    try {
      const astapDataDir = $("tools-astrometry-data-dir")?.value || "";
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.astrometry.installCli, { astap_data_dir: astapDataDir }),
        { fallbackPath: astapDataDir },
      );
      append(accepted);
      const job = await waitForJob(accepted.job_id, { onTick: (j) => append({ state: j.state, progress: j.data?.progress ?? null }) });
      append(job);
      await detect();
    } catch (err) {
      setFooter(`ASTAP Install fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.astrometry.download_catalog']")?.addEventListener("click", async () => {
    try {
      const sel = $("tools-astrometry-catalog");
      const txt = String(sel?.value || "").toLowerCase();
      const match = txt.match(/d\d+/);
      const catalogId = match ? match[0] : "d50";
      const astapDataDir = $("tools-astrometry-data-dir")?.value || "";
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.astrometry.downloadCatalog, {
          catalog_id: catalogId,
          astap_data_dir: astapDataDir,
        }),
        { fallbackPath: astapDataDir },
      );
      append(accepted);
      const job = await waitForJob(accepted.job_id, { onTick: (j) => append({ state: j.state, current_chunk: j.data?.current_chunk }) });
      append(job);
    } catch (err) {
      setFooter(`Catalog-Download fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.astrometry.cancel_download']")?.addEventListener("click", async () => {
    try {
      const result = await api.post(API_ENDPOINTS.astrometry.cancelDownload, {});
      append(result);
    } catch (err) {
      setFooter(`Catalog-Cancel fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.astrometry.solve']")?.addEventListener("click", async () => {
    try {
      const payload = {
        solve_file: $("tools-astrometry-file")?.value || "",
        astap_cli: $("tools-astrometry-bin")?.value || "",
        astap_data_dir: $("tools-astrometry-data-dir")?.value || "",
      };
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.astrometry.solve, payload),
        { fallbackPath: payload.solve_file || payload.astap_cli || payload.astap_data_dir },
      );
      append(accepted);
      const job = await waitForJob(accepted.job_id);
      const jobResult = job?.data?.result;
      if (jobResult) {
        applyAstrometryResult(jobResult);
        append(jobResult);
      }
      uiState.lastAstrometryWcs = String(jobResult?.wcs_path || job?.data?.wcs_path || "");
      append(job);
      if (String(job?.state || "") !== "ok") {
        throw new Error(jobResult?.error || job?.data?.stderr || "ASTAP solve failed");
      }
      setFooter(`Solve erfolgreich: ${uiState.lastAstrometryWcs || "WCS erstellt"}`);
    } catch (err) {
      setFooter(`Solve fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.astrometry.save_solved']")?.addEventListener("click", async () => {
    try {
      const input = $("tools-astrometry-file")?.value || "";
      const defaultOutput = deriveOutputPath(input, "_solved");
      const output = window.prompt("Output-FITS Pfad:", defaultOutput);
      if (!output) return;
      const result = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.astrometry.saveSolved, {
          input_path: input,
          output_path: output,
          wcs_path: uiState.lastAstrometryWcs || undefined,
        }),
        { fallbackPath: input || uiState.lastAstrometryWcs || parentDirOfPath(output) },
      );
      append(result);
      setFooter(`Saved: ${result.output_path || output}`);
    } catch (err) {
      setFooter(`Save Solved fehlgeschlagen: ${errorText(err)}`, true);
    }
  });
}

async function bindPccPage() {
  if (!$("tools-pcc-rgb")) return;
  const logBox = findLogBoxBySectionTitle("Result + Log");
  const statusField = document.querySelector("[data-control='tools.pcc.siril_status']");

  const missingField = $("tools-pcc-missing-chunks");
  const starsMatchedField = $("tools-pcc-stars-matched");
  const starsUsedField = $("tools-pcc-stars-used");
  const residualField = $("tools-pcc-residual-rms");
  const matrixField = $("tools-pcc-matrix");

  const append = (msg) => appendLine(logBox, typeof msg === "string" ? msg : JSON.stringify(msg));
  const setInputValue = (el, value) => {
    if (!el) return;
    el.value = value === null || value === undefined ? "" : String(value);
  };
  const readNumber = (id) => {
    const raw = String($(id)?.value || "").trim();
    return raw === "" ? undefined : Number(raw);
  };
  const readInteger = (id) => {
    const raw = String($(id)?.value || "").trim();
    return raw === "" ? undefined : parseInt(raw, 10);
  };
  const formatMatrix = (matrix) => {
    if (!Array.isArray(matrix)) return "-";
    return matrix
      .map((row) => (Array.isArray(row) ? `[${row.map((value) => Number(value).toFixed(6)).join(", ")}]` : ""))
      .filter(Boolean)
      .join("\n") || "-";
  };
  const applyPccResult = (payload) => {
    if (!payload || typeof payload !== "object") return;
    setInputValue(starsMatchedField, payload.stars_matched ?? payload.n_stars_matched ?? "");
    setInputValue(starsUsedField, payload.stars_used ?? payload.n_stars_used ?? "");
    setInputValue(residualField, payload.residual_rms ?? "");
    setInputValue(matrixField, formatMatrix(payload.matrix));
    if (payload.output_rgb) {
      uiState.lastPccOutput = String(payload.output_rgb);
    }
    if (Array.isArray(payload.output_channels)) {
      uiState.lastPccChannels = payload.output_channels.map((item) => String(item));
    }
    uiState.lastPccResult = payload;
  };

  const refreshStatus = async () => {
    const catalogDir = $("tools-pcc-catalog-dir")?.value || "";
    const status = await withPathGrantRetry(
      () => api.get(API_ENDPOINTS.pcc.sirilStatus(catalogDir)),
      { fallbackPath: catalogDir },
    );
    if (statusField) statusField.value = `${status.installed}/${status.total} installiert`;
    if (missingField) missingField.value = String(Array.isArray(status.missing) ? status.missing.length : "");
    if (status.catalog_dir && !String($("tools-pcc-catalog-dir")?.value || "").trim()) {
      setInputValue($("tools-pcc-catalog-dir"), status.catalog_dir);
    }
    append(status);
  };

  try {
    await refreshStatus();
  } catch (err) {
    setFooter(`PCC Status fehlgeschlagen: ${errorText(err)}`, true);
  }

  document.querySelector("[data-control='tools.pcc.download_missing']")?.addEventListener("click", async () => {
    try {
      const catalogDir = $("tools-pcc-catalog-dir")?.value || "";
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.pcc.downloadMissing, { catalog_dir: catalogDir }),
        { fallbackPath: catalogDir },
      );
      append(accepted);
      const job = await waitForJob(accepted.job_id, { onTick: (j) => append({ state: j.state, current_chunk: j.data?.current_chunk }) });
      append(job);
      await refreshStatus();
    } catch (err) {
      setFooter(`PCC Download fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.pcc.cancel_download']")?.addEventListener("click", async () => {
    try {
      const result = await api.post(API_ENDPOINTS.pcc.cancelDownload, {});
      append(result);
    } catch (err) {
      setFooter(`PCC Cancel fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.pcc.check_online']")?.addEventListener("click", async () => {
    try {
      const result = await api.post(API_ENDPOINTS.pcc.checkOnline, {});
      append(result);
      setFooter(result.ok ? `Online source OK (${result.latency_ms} ms)` : "Online source nicht erreichbar.", !result.ok);
    } catch (err) {
      setFooter(`Online-Check fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.pcc.run']")?.addEventListener("click", async () => {
    try {
      const input = $("tools-pcc-rgb")?.value || "";
      const defaultOutput = deriveOutputPath(input, "_pcc");
      const output = window.prompt("Output RGB FITS Pfad:", defaultOutput) || defaultOutput;
      const payload = {
        input_rgb: input,
        output_rgb: output,
        wcs_file: $("tools-pcc-wcs")?.value || "",
        source: $("tools-pcc-source")?.value || "siril",
        catalog_dir: $("tools-pcc-catalog-dir")?.value || "",
        mag_limit: readNumber("tools-pcc-mag-limit"),
        mag_bright_limit: readNumber("tools-pcc-mag-bright"),
        min_stars: readInteger("tools-pcc-min-stars"),
        sigma_clip: readNumber("tools-pcc-sigma"),
        aperture_radius_px: readNumber("tools-pcc-aperture"),
        annulus_inner_px: readNumber("tools-pcc-annulus-in"),
        annulus_outer_px: readNumber("tools-pcc-annulus-out"),
      };
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.pcc.run, payload),
        { fallbackPath: input || payload.wcs_file || payload.catalog_dir || parentDirOfPath(output) },
      );
      append(accepted);
      const job = await waitForJob(accepted.job_id);
      const jobResult = job?.data?.result;
      if (jobResult) {
        applyPccResult(jobResult);
        append(jobResult);
      }
      append(job);
      if (String(job?.state || "") !== "ok") {
        throw new Error(jobResult?.error || job?.data?.stderr || "PCC job failed");
      }
      setFooter(`PCC abgeschlossen: ${jobResult?.stars_used ?? "-"} Sterne genutzt.`);
    } catch (err) {
      setFooter(`Run PCC fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.pcc.save_corrected']")?.addEventListener("click", async () => {
    try {
      const output = uiState.lastPccOutput || $("tools-pcc-rgb")?.value || "";
      const result = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.pcc.saveCorrected, { output_rgb: output, output_channels: uiState.lastPccChannels }),
        { fallbackPath: output },
      );
      append(result);
      setFooter(`Save Corrected: ${result.output_rgb || "-"}`);
    } catch (err) {
      setFooter(`Save Corrected fehlgeschlagen: ${errorText(err)}`, true);
    }
  });
}

async function bindLiveLogPage() {
  const page = pageName();
  if (page !== "live-log.html") return;
  const box = document.querySelector(".app-content .ps-section div[style*='font-family:monospace']");
  if (!box) return;

  const allButtons = Array.from(document.querySelectorAll(".app-content .ps-section .ps-btn.ps-btn-secondary"));
  const levelButtons = allButtons.filter((btn) => {
    const t = String(btn.textContent || "").trim().toLowerCase();
    return ["all", "info", "warning", "error", "clear"].includes(t);
  });

  function detectLevel(line) {
    const lower = String(line).toLowerCase();
    if (lower.includes("error")) return "error";
    if (lower.includes("warn")) return "warning";
    return "info";
  }

  const escapeHtml = (text) =>
    String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  function render() {
    const lines = uiState.liveLines.filter((item) => uiState.liveFilter === "all" || item.level === uiState.liveFilter);
    const colorByLevel = {
      info: "#e5edf6",
      warning: "#f59e0b",
      error: "#ef4444",
    };
    box.innerHTML = lines
      .map((item) => {
        const level = String(item.level || "info");
        const color = colorByLevel[level] || colorByLevel.info;
        return `<div style="color:${color};white-space:pre-wrap;">${escapeHtml(item.line)}</div>`;
      })
      .join("");
    scrollLogToEnd(box);
  }

  function flushLiveLog() {
    if (uiState.liveLogFlushTimer) {
      clearTimeout(uiState.liveLogFlushTimer);
      uiState.liveLogFlushTimer = null;
    }
    if (uiState.livePendingLines.length === 0) return;
    uiState.liveLines.push(...uiState.livePendingLines);
    if (uiState.liveLines.length > 600) uiState.liveLines = uiState.liveLines.slice(-600);
    uiState.livePendingLines = [];
    render();
  }

  function scheduleLiveLogFlush() {
    if (uiState.liveLogFlushTimer) return;
    uiState.liveLogFlushTimer = window.setTimeout(() => {
      flushLiveLog();
    }, 5000);
  }

  const renderEmptyState = (text) => {
    uiState.liveLines = [];
    uiState.livePendingLines = [];
    box.innerHTML = `<div style="color:#9ca3af;white-space:pre-wrap;">${escapeHtml(text)}</div>`;
    scrollLogToEnd(box);
  };

  levelButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const t = String(btn.textContent || "").trim().toLowerCase();
      if (t === "clear") {
        uiState.liveLines = [];
        uiState.livePendingLines = [];
        if (uiState.liveLogFlushTimer) {
          clearTimeout(uiState.liveLogFlushTimer);
          uiState.liveLogFlushTimer = null;
        }
        render();
        return;
      }
      uiState.liveFilter = t;
      render();
    });
  });

  if (uiState.liveSocket) {
    uiState.liveSocket.close();
    uiState.liveSocket = null;
  }
  if (uiState.liveLogFlushTimer) {
    clearTimeout(uiState.liveLogFlushTimer);
    uiState.liveLogFlushTimer = null;
  }
  uiState.livePendingLines = [];

  let runId = "";
  try {
    const appState = await api.get(API_ENDPOINTS.app.state);
    runId = String(appState?.project?.current_run_id || "").trim();
    if (runId) setCurrentRunId(runId);
    else clearCurrentRunId();
  } catch (err) {
    renderEmptyState("Live Log nicht verfügbar.");
    setFooter(`Live Log konnte nicht initialisiert werden: ${errorText(err)}`, true);
    return;
  }

  if (!runId) {
    renderEmptyState("Kein aktiver Run.");
    setFooter("Kein aktueller Run gesetzt. Bitte in History einen Run als Current markieren.", true);
    return;
  }
  try {
    const logs = await api.get(API_ENDPOINTS.runs.logs(runId, 250));
    uiState.liveLines = (logs.lines || []).map((line) => ({ line, level: detectLevel(line) }));
    render();
    if (uiState.liveSocket) uiState.liveSocket.close();
    uiState.liveSocket = api.ws(
      API_ENDPOINTS.ws.run(runId),
      (event) => {
        const line = typeof event === "string" ? event : JSON.stringify(event);
        uiState.livePendingLines.push({ line, level: detectLevel(line) });
        scheduleLiveLogFlush();
      },
      () => {},
    );
  } catch (err) {
    setFooter(`Live Log konnte nicht geladen werden: ${errorText(err)}`, true);
  }
}

function findMonoQueueSection() {
  return Array.from(document.querySelectorAll(".ps-section")).find((sec) => {
    const title = sec.querySelector(".ps-section-title");
    const normalized = String(title?.textContent || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
    return normalized.includes("mono filter queue");
  });
}

function getPersistedDetectedColorMode() {
  return normalizeDetectedColorMode(localStorage.getItem("gui2.lastScanColorMode") || "");
}

function shouldShowMonoUi(selectedModeRaw) {
  const selectedMode = normalizeDetectedColorMode(selectedModeRaw);
  const detectedMode = getPersistedDetectedColorMode();
  if (detectedMode === "OSC") return false;
  return selectedMode === "MONO";
}

function setMonoQueueVisible(selectedModeRaw) {
  const isMono = shouldShowMonoUi(selectedModeRaw);
  const dashboardAdvancedBtn = $("dashboard-guided-mode-advanced");
  const dashboardHasModeToggle = Boolean($("dashboard-guided-mode-simple") || dashboardAdvancedBtn);
  const dashboardAdvancedVisible = !dashboardHasModeToggle || Boolean(dashboardAdvancedBtn?.classList.contains("active"));
  document.querySelectorAll(".guided-mono-only").forEach((el) => {
    el.style.display = isMono && dashboardAdvancedVisible ? "" : "none";
  });
  const sec = findMonoQueueSection();
  if (!sec) return;
  sec.style.display = isMono ? "" : "none";
}

function collectQueueRows() {
  const rows = Array.from(document.querySelectorAll(".ps-queue-row"));
  const out = [];
  for (const row of rows) {
    const select = row.querySelector("select");
    const inputs = Array.from(row.querySelectorAll("input[type='text']"));
    const enabled = row.querySelector("input[type='checkbox']");
    const filter = String(select?.value || "").trim();
    const inputDir = String(inputs[0]?.value || "").trim();
    const pattern = String(inputs[1]?.value || "").trim();
    const runLabel = String(inputs[2]?.value || "").trim();
    const isOn = enabled ? Boolean(enabled.checked) : true;
    if (!isOn || !inputDir) continue;
    const item = { filter, input_dir: inputDir };
    if (pattern) item.pattern = pattern;
    if (runLabel) item.run_id = runLabel;
    out.push(item);
  }
  return out;
}

function renderGuardrailRow(row, status, label) {
  if (!row) return;
  const chip = row.querySelector("span");
  const txt = row.querySelector("span:last-child");
  const s = String(status || "check").toLowerCase();
  if (chip) {
    chip.textContent = s === "ok" ? "OK" : s === "error" ? "ERR" : "CHECK";
    if (s === "ok") {
      chip.style.background = "#d1f4e0";
      chip.style.color = "#15808d";
    } else if (s === "error") {
      chip.style.background = "#fee2e2";
      chip.style.color = "#b91c1c";
    } else {
      chip.style.background = "#fde68a";
      chip.style.color = "#d97706";
    }
  }
  if (txt && label) txt.textContent = label;
}

async function renderDashboardDerivedGuardrails(appState) {
  let configValidation = null;
  try {
    const yaml = await ensureConfigYaml();
    if (yaml) {
      configValidation = await api.post(API_ENDPOINTS.config.validate, { yaml });
    }
  } catch {
    configValidation = null;
  }

  const validationErrors = Array.isArray(configValidation?.errors) ? configValidation.errors.length : 0;
  const validationWarnings = Array.isArray(configValidation?.warnings) ? configValidation.warnings.length : 0;
  const currentRunId = String(appState?.run?.current?.run_id || "").trim();

  renderGuardrailRow(
    $("dashboard-guardrail-config-valid"),
    !configValidation ? "check" : validationErrors > 0 ? "error" : validationWarnings > 0 ? "check" : "ok",
    !configValidation
      ? "Config nicht geprüft"
      : validationErrors > 0
        ? `Config mit ${validationErrors} Fehlern`
        : validationWarnings > 0
          ? `Config validiert (${validationWarnings} Warnungen)`
          : "Config validiert",
  );
  renderGuardrailRow($("dashboard-guardrail-calibration-paths"), "check", "Kalibrierpfade nicht separat geprüft");
  renderGuardrailRow(
    $("dashboard-guardrail-bge-pcc"),
    "check",
    currentRunId ? "BGE/PCC nicht automatisch bewertet" : "BGE/PCC nicht geprüft (kein Run)",
  );
}

async function bindDashboard() {
  if (!$("dashboard-kpi-scan-quality")) return;
  bindInputDirMemory("dashboard-input-dirs");
  const runsDirInput = $("dashboard-run-runs-dir");
  if (runsDirInput && !String(runsDirInput.value || "").trim() && uiState.projectRunsDir) {
    runsDirInput.value = uiState.projectRunsDir;
  }
  try {
    const [quality, guardrails, latestScan, appState] = await Promise.all([
      api.get(API_ENDPOINTS.scan.quality),
      api.get(API_ENDPOINTS.guardrails.root),
      api.get(API_ENDPOINTS.scan.latest),
      api.get(API_ENDPOINTS.app.state),
    ]);
    setRunReady(guardrails?.status || "check", appState?.run?.current?.status || "");
    const summary = summarizeScanResult(
      latestScan?.has_scan ? latestScan : quality?.scan || {},
      String($("dashboard-input-dirs")?.value || "").trim(),
    );
    renderDashboardScanKpis(summary, quality?.score ?? 0);
    renderDashboardLastRunKpi(appState);
    renderScanSummary("dashboard-scan", summary);
    applyDetectedColorModeToSelect($("dashboard-color-mode"), summary);
    applyDetectedColorModeToSelect($("inp-colormode"), summary);
    const mergedInputText = summary.input_dirs?.length > 0 ? summary.input_dirs.join(", ") : summary.input_path;
    if (mergedInputText) {
      $("dashboard-input-dirs") && ($("dashboard-input-dirs").value = mergedInputText);
      persistLastInputDirs(mergedInputText);
      restoreLastInputDirs("dashboard-input-dirs");
    }

    const runStart = $("dashboard-run-start");
    setDisabledLike(runStart, String(guardrails?.status || "").toLowerCase() === "error");
    const scanCheck = (guardrails?.checks || []).find((c) => c.id === "scan_ok");
    const warnCheck = (guardrails?.checks || []).find((c) => c.id === "scan_warnings");
    const colorModeCheck = (guardrails?.checks || []).find((c) => c.id === "color_mode");
    renderGuardrailRow($("dashboard-guardrail-scan-ok"), scanCheck?.status || "check", scanCheck?.label || "Scan ausstehend");
    renderGuardrailRow(
      $("dashboard-guardrail-color-mode"),
      colorModeCheck?.status || warnCheck?.status || "check",
      colorModeCheck?.label || "Color mode bestaetigen",
    );
    await renderDashboardDerivedGuardrails(appState);

    await populatePresetSelect("dashboard-preset", false);

    const preview = () => {
      const runsDir = String($("dashboard-run-runs-dir")?.value || "").trim();
      const runName = sanitizeRunName(String($("dashboard-run-name")?.value || ""));
      if ($("dashboard-run-name")) $("dashboard-run-name").value = runName;
      if (!$("dashboard-run-path-preview")) return;
      if (!runsDir || !runName) {
        $("dashboard-run-path-preview").value = "";
        return;
      }
      $("dashboard-run-path-preview").value = `${runsDir}/${runName}_${timestampSuffix()}`;
    };
    $("dashboard-run-runs-dir")?.addEventListener("input", preview);
    $("dashboard-run-name")?.addEventListener("input", preview);
    preview();

    $("dashboard-color-mode")?.addEventListener("change", () => {
      setMonoQueueVisible($("dashboard-color-mode")?.value || "");
    });
    setMonoQueueVisible($("dashboard-color-mode")?.value || "");
    if ($("dashboard-run-name") && !String($("dashboard-run-name").value || "").trim()) {
      $("dashboard-run-name").value = suggestRunNameFromInputs(parseInputDirs($("dashboard-input-dirs")?.value || ""));
    }
    $("dashboard-input-dirs")?.addEventListener("change", preview);
    $("dashboard-input-dirs")?.addEventListener("input", preview);
    preview();

    $("dashboard-preset")?.addEventListener("change", async () => {
      try {
        const path = String($("dashboard-preset")?.value || "").trim();
        if (!path) return;
        const applied = await api.post(API_ENDPOINTS.config.applyPreset, { path });
        setConfigDraft(String(applied?.config || ""));
        setFooter("Preset fuer Guided Run aktualisiert.");
      } catch (err) {
        setFooter(`Preset-Laden fehlgeschlagen: ${errorText(err)}`, true);
      }
    });

    $("dashboard-run-start")?.addEventListener("click", async (ev) => {
      ev.preventDefault();
      const runStartButton = $("dashboard-run-start");
      try {
        setDisabledLike(runStartButton, true);
        const latestGuardrails = await api.get(API_ENDPOINTS.guardrails.root);
        if (String(latestGuardrails?.status || "").toLowerCase() === "error") {
          setDisabledLike(runStartButton, false);
          setFooter("Run blockiert: Guardrail-Status ist ERROR.", true);
          return;
        }
        const accepted = await startRunFromCurrentForm({ source: "dashboard" });
        setCurrentRunId(accepted?.run_id || uiState.currentRunId);
        setRunReady(latestGuardrails?.status || "check", "running");
        setFooter(`Run gestartet (Job ${accepted?.job_id || "-"}).`);
        window.location.href = "run-monitor.html";
      } catch (err) {
        setDisabledLike(runStartButton, false);
        setFooter(`Run-Start fehlgeschlagen: ${errorText(err)}`, true);
      }
    });

    $("dashboard-scan-refresh")?.addEventListener("click", async (ev) => {
      ev.preventDefault();
      try {
        const dirs = parseInputDirs(String($("dashboard-input-dirs")?.value || ""));
        if (dirs.length === 0) {
          setFooter("Bitte mindestens einen Eingabeordner setzen.", true);
          return;
        }
        const accepted = await withPathGrantRetry(
          () =>
            api.post(
              API_ENDPOINTS.scan.root,
              buildScanPayloadFromDirs(
                dirs,
                1,
                false,
              ),
            ),
          { fallbackPath: dirs[0] || "" },
        );
        setFooter(`Scan gestartet (Job ${accepted.job_id}).`);
        const job = await waitForJob(accepted.job_id, { allowMissing: true });
        const [quality2, guardrails2, latest2] = await Promise.all([
          api.get(API_ENDPOINTS.scan.quality),
          api.get(API_ENDPOINTS.guardrails.root),
          api.get(API_ENDPOINTS.scan.latest),
        ]);
        const summary2 = summarizeScanResult(latest2?.has_scan ? latest2 : quality2?.scan || {}, dirs[0] || "");
        const mergedInputText2 = summary2.input_dirs?.length > 0 ? summary2.input_dirs.join(", ") : summary2.input_path;
        if (mergedInputText2 && $("dashboard-input-dirs")) {
          $("dashboard-input-dirs").value = mergedInputText2;
          persistLastInputDirs(mergedInputText2);
        }
        renderDashboardScanKpis(summary2, quality2?.score ?? 0);
        renderScanSummary("dashboard-scan", summary2);
        applyDetectedColorModeToSelect($("dashboard-color-mode"), summary2);
        applyDetectedColorModeToSelect($("inp-colormode"), summary2);
        const appState2 = await api.get(API_ENDPOINTS.app.state).catch(() => appState);
        setRunReady(guardrails2?.status || "check", appState2?.run?.current?.status || "");
        const scanCheck2 = (guardrails2?.checks || []).find((c) => c.id === "scan_ok");
        const warnCheck2 = (guardrails2?.checks || []).find((c) => c.id === "scan_warnings");
        const colorModeCheck2 = (guardrails2?.checks || []).find((c) => c.id === "color_mode");
        renderGuardrailRow(
          $("dashboard-guardrail-scan-ok"),
          scanCheck2?.status || "check",
          scanCheck2?.label || "Scan ausstehend",
        );
        renderGuardrailRow(
          $("dashboard-guardrail-color-mode"),
          colorModeCheck2?.status || warnCheck2?.status || "check",
          colorModeCheck2?.label || "Color mode bestaetigen",
        );
        renderDashboardLastRunKpi(appState2);
        await renderDashboardDerivedGuardrails(appState2);
        if (String(job?.state) === "missing") {
          setFooter(
            "Scan-Status war kurzzeitig nicht abrufbar (Backend-Reload). Letztes Scan-Ergebnis wurde geladen.",
            true,
          );
        } else {
          setFooter(
            job?.state === "ok" ? "Scan abgeschlossen." : `Scan beendet mit Status: ${job?.state || "unknown"}`,
            job?.state !== "ok",
          );
        }
      } catch (err) {
        setFooter(`Scan fehlgeschlagen: ${errorText(err)}`, true);
      }
    });
    const deepLinks = {
      "dashboard-guardrail-scan-ok": "input-scan.html",
      "dashboard-guardrail-color-mode": "input-scan.html",
      "dashboard-guardrail-config-valid": "parameter-studio.html",
      "dashboard-guardrail-calibration-paths": "input-scan.html",
      "dashboard-guardrail-bge-pcc": "parameter-studio.html",
    };
    Object.entries(deepLinks).forEach(([id, href]) => {
      $(id)?.addEventListener("click", () => {
        window.location.href = href;
      });
    });
  } catch (err) {
    setFooter(`Dashboard-Daten konnten nicht geladen werden: ${errorText(err)}`, true);
  }
}

async function bindWizard() {
  if (pageName() !== "wizard.html") return;
  const wizardRunsDir = $("wizard-runs-dir");
  if (wizardRunsDir && !String(wizardRunsDir.value || "").trim() && uiState.projectRunsDir) {
    wizardRunsDir.value = uiState.projectRunsDir;
  }
  const updateWizardPreview = () => {
    const runsDir = String($("wizard-runs-dir")?.value || "").trim();
    const dirs = parseInputDirs(String($("inp-dirs")?.value || ""));
    const suggested = sanitizeRunName(String($("wizard-run-name")?.value || "")) || suggestRunNameFromInputs(dirs);
    if ($("wizard-run-name")) $("wizard-run-name").value = suggested;
    const previewEl = $("wizard-run-path-preview");
    if (!previewEl) return;
    if (!runsDir || !suggested) {
      previewEl.value = "";
      return;
    }
    previewEl.value = `${runsDir}/${suggested}_${timestampSuffix()}`;
  };
  try {
    await populatePresetSelect("wizard-preset-select", true);
  } catch (err) {
    setFooter(`Wizard-Presetliste konnte nicht geladen werden: ${errorText(err)}`, true);
  }

  $("inp-colormode")?.addEventListener("change", () => {
    setMonoQueueVisible($("inp-colormode")?.value || "");
  });
  setMonoQueueVisible($("inp-colormode")?.value || "");
  $("wizard-runs-dir")?.addEventListener("input", updateWizardPreview);
  $("wizard-run-name")?.addEventListener("input", updateWizardPreview);
  $("inp-dirs")?.addEventListener("input", updateWizardPreview);
  $("inp-dirs")?.addEventListener("change", updateWizardPreview);
  updateWizardPreview();

  $("wizard-nav-next")?.addEventListener("click", () => {
    const step4 = Array.from(document.querySelectorAll(".ps-section")).find((sec) => {
      const title = sec.querySelector(".ps-section-title");
      return title && String(title.textContent || "").includes("Step 4");
    });
    step4?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $("wizard-nav-back")?.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  $("wizard-preset-select")?.addEventListener("change", async () => {
    try {
      const path = String($("wizard-preset-select")?.value || "").trim();
      if (!path) return;
      const applied = await api.post(API_ENDPOINTS.config.applyPreset, { path });
      setConfigDraft(String(applied?.config || ""));
      const v = await api.post(API_ENDPOINTS.config.validate, { yaml: String(applied?.config || "") });
      const box = $("wizard-validation-result");
      if (box) box.innerHTML = `<div class="ps-result-title">Validation</div><div>Schema: <b>${v.ok ? "OK" : "ERROR"}</b> | Fehler: <b>${(v.errors || []).length}</b> | Warnungen: <b>${(v.warnings || []).length}</b></div>`;
      setFooter("Wizard-Preset angewendet.");
    } catch (err) {
      setFooter(`Wizard-Preset fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("wizard-situation-apply")?.addEventListener("click", async () => {
    try {
      let keys = activeScenarioKeys(".app-content");
      if (keys.length === 0) keys = ["altaz", "rotation", "bright_stars"];
      const updates = [];
      for (const key of keys) {
        for (const [path, value] of SCENARIO_DELTAS[key] || []) updates.push({ path, value });
      }
      if (updates.length === 0) {
        setFooter("Keine Wizard-Situation aktiv.", true);
        return;
      }
      const patched = await patchConfig({ updates, persist: false });
      const v = await api.post(API_ENDPOINTS.config.validate, { yaml: patched?.config_yaml || "" });
      const box = $("wizard-validation-result");
      if (box) box.innerHTML = `<div class="ps-result-title">Validation</div><div>Schema: <b>${v.ok ? "OK" : "ERROR"}</b> | Fehler: <b>${(v.errors || []).length}</b> | Warnungen: <b>${(v.warnings || []).length}</b></div>`;
      setFooter(`Wizard-Szenario angewendet (${updates.length} Deltas).`);
    } catch (err) {
      setFooter(`Wizard-Szenario fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("wizard-start")?.addEventListener("click", async (ev) => {
    ev.preventDefault();
    try {
      const accepted = await startRunFromCurrentForm({ source: "wizard" });
      setCurrentRunId(accepted?.run_id || uiState.currentRunId);
      setFooter(`Wizard-Run gestartet (Job ${accepted?.job_id || "-"}).`);
      window.location.href = "run-monitor.html";
    } catch (err) {
      setFooter(`Wizard-Runstart fehlgeschlagen: ${errorText(err)}`, true);
    }
  });
}

async function bindAssumptions() {
  if (pageName() !== "assumptions.html") return;
  const ids = Object.keys(ASSUMPTION_ID_PATHS);
  if (ids.length === 0) return;
  try {
    const parsed = await patchConfig({ updates: [], persist: false });
    if (parsed?.config) {
      for (const [id, path] of Object.entries(ASSUMPTION_ID_PATHS)) {
        writeFieldValue($(id), getByPath(parsed.config, path));
      }
    }
  } catch {
    // ignore, page can still operate with defaults
  }
  const onChange = async () => {
    try {
      const updates = updatesFromMap(Object.entries(ASSUMPTION_ID_PATHS).map(([id, path]) => [`#${id}`, path]));
      await patchConfig({ updates, persist: false });
      setFooter("Assumptions im Config-Draft aktualisiert.");
    } catch (err) {
      setFooter(`Assumptions-Update fehlgeschlagen: ${errorText(err)}`, true);
    }
  };
  ids.forEach((id) => {
    $(id)?.addEventListener("input", () => void onChange());
    $(id)?.addEventListener("change", () => void onChange());
  });
}

async function init() {
  bindLocaleControls();
  await initGlobalState();
  bindScanPages();
  await bindParameterStudio();
  await bindRunMonitor();
  await bindHistoryPage();
  await bindAstrometryPage();
  await bindPccPage();
  await bindLiveLogPage();
  await bindDashboard();
  await bindWizard();
  await bindAssumptions();
}

document.addEventListener("DOMContentLoaded", () => {
  void init();
});
