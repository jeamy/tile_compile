import { ApiClient } from "./api.js";
import { API_ENDPOINTS } from "./constants.js";
import { applyLocaleMessages, t } from "./i18n.js";

const api = new ApiClient(localStorage.getItem("gui2.backendBase") || "");
const CONFIG_DRAFT_KEY = "gui2.configYamlDraft";
const CONFIG_VALIDATION_STATE_KEY = "gui2.configValidationState";
const PARAMETER_DIRTY_STATE_KEY = "gui2.parameterDirtyState";
const HISTORY_CURRENT_RUN_KEY = "gui2.historyCurrentRunId";
const LOCALE_KEY = "gui2.locale";
const LAST_INPUT_DIRS_KEY = "gui2.lastInputDirs";
const PRESETS_DIR_KEY = "gui2.presetsDir";
const CALIBRATION_PATH_KEY_PREFIX = "gui2.calibrationPath";
const LAST_SCAN_COLOR_MODE_KEY = "gui2.lastScanColorMode";
const ASTROMETRY_LAST_RESULT_KEY = "gui2.tools.astrometry.lastResult";
const ASTROMETRY_LAST_WCS_KEY = "gui2.tools.astrometry.lastWcs";
const PCC_LAST_OUTPUT_KEY = "gui2.tools.pcc.lastOutput";
const PCC_LAST_CHANNELS_KEY = "gui2.tools.pcc.lastChannels";
const PCC_LAST_RESULT_KEY = "gui2.tools.pcc.lastResult";
const UI_STORAGE_KEYS = {
  dashboardRunsDir: "gui2.run.runsDir",
  dashboardRunName: "gui2.run.runName",
  dashboardQueue: "gui2.dashboard.queueDraft",
  dashboardPreset: "gui2.dashboard.presetPath",
  parameterPreset: "gui2.parameter.presetPath",
  wizardRunsDir: "gui2.run.runsDir",
  wizardRunName: "gui2.run.runName",
  wizardQueue: "gui2.wizard.queueDraft",
  wizardPreset: "gui2.wizard.presetPath",
  historySelectedRunId: "gui2.history.selectedRunId",
  historyCompareRunId: "gui2.history.compareRunId",
  liveFilter: "gui2.live.filter",
  astrometryBinary: "gui2.tools.astrometry.binary",
  astrometryDataDir: "gui2.tools.astrometry.dataDir",
  astrometryFile: "gui2.tools.astrometry.file",
  astrometryCatalog: "gui2.tools.astrometry.catalog",
  pccRgb: "gui2.tools.pcc.rgb",
  pccWcs: "gui2.tools.pcc.wcs",
  pccSource: "gui2.tools.pcc.source",
  pccCatalogDir: "gui2.tools.pcc.catalogDir",
  pccMagLimit: "gui2.tools.pcc.magLimit",
  pccMagBrightLimit: "gui2.tools.pcc.magBrightLimit",
  pccMinStars: "gui2.tools.pcc.minStars",
  pccSigma: "gui2.tools.pcc.sigma",
  pccAperture: "gui2.tools.pcc.aperture",
  pccAnnulusInner: "gui2.tools.pcc.annulusInner",
  pccAnnulusOuter: "gui2.tools.pcc.annulusOuter",
  astrometryLastResult: ASTROMETRY_LAST_RESULT_KEY,
  astrometryLastWcs: ASTROMETRY_LAST_WCS_KEY,
  pccLastOutput: PCC_LAST_OUTPUT_KEY,
  pccLastChannels: PCC_LAST_CHANNELS_KEY,
  pccLastResult: PCC_LAST_RESULT_KEY,
};

const uiState = {
  currentRunId: "",
  currentRunDir: "",
  currentRunColorMode: "",
  parameterBaseYaml: "",
  missingHistoryRunIds: new Set(),
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
  locale: "de",
  projectRunsDir: "",
  projectPresetsDir: "",
  monitorStatsStatus: null,
  dashboardGuardrailStatus: "",
  runReadyStatus: "check",
  runProcessStatus: "",
  configSchemaPaths: null,
};

let serverUiState = {};
let serverUiStateLoaded = false;
let serverUiStateSaveTimer = null;
let serverUiStateSavePromise = Promise.resolve();
const SERVER_UI_STATE_MIGRATION_KEYS = [
  CONFIG_DRAFT_KEY,
  CONFIG_VALIDATION_STATE_KEY,
  PARAMETER_DIRTY_STATE_KEY,
  HISTORY_CURRENT_RUN_KEY,
  LOCALE_KEY,
  LAST_INPUT_DIRS_KEY,
  PRESETS_DIR_KEY,
  LAST_SCAN_COLOR_MODE_KEY,
  "gui2.currentRunId",
  ...Object.values(UI_STORAGE_KEYS),
];

function legacyStorageGet(key) {
  return localStorage.getItem(key);
}

function storedJsonValue(key, fallback = null) {
  try {
    const raw = readServerUiStateValue(key);
    if (!raw) return fallback;
    return JSON.parse(String(raw));
  } catch {
    writeServerUiStateValue(key, "");
    return fallback;
  }
}

function persistJsonValue(key, value) {
  if (value === undefined || value === null) {
    writeServerUiStateValue(key, "");
    return;
  }
  writeServerUiStateValue(key, JSON.stringify(value));
}

function legacyStorageRemove(key) {
  localStorage.removeItem(key);
}

function hasServerUiStateKey(key) {
  return Object.prototype.hasOwnProperty.call(serverUiState, key);
}

function readServerUiStateValue(key) {
  if (hasServerUiStateKey(key)) return serverUiState[key];
  return legacyStorageGet(key);
}

function writeServerUiStateValue(key, value) {
  if (value === undefined || value === null || value === "") delete serverUiState[key];
  else serverUiState[key] = value;
  if (!serverUiStateLoaded) return;
  if (serverUiStateSaveTimer) window.clearTimeout(serverUiStateSaveTimer);
  serverUiStateSaveTimer = window.setTimeout(() => {
    serverUiStateSaveTimer = null;
    const snapshot = { ...serverUiState };
    serverUiStateSavePromise = api.post(API_ENDPOINTS.app.uiState, { state: snapshot })
      .then((result) => {
        const nextState = result?.state;
        if (nextState && typeof nextState === "object" && !Array.isArray(nextState)) {
          serverUiState = nextState;
        }
      })
      .catch(() => {});
  }, 120);
}

async function flushServerUiState() {
  if (!serverUiStateLoaded) return;
  if (serverUiStateSaveTimer) {
    window.clearTimeout(serverUiStateSaveTimer);
    serverUiStateSaveTimer = null;
    const snapshot = { ...serverUiState };
    serverUiStateSavePromise = api.post(API_ENDPOINTS.app.uiState, { state: snapshot })
      .then((result) => {
        const nextState = result?.state;
        if (nextState && typeof nextState === "object" && !Array.isArray(nextState)) {
          serverUiState = nextState;
        }
      })
      .catch(() => {});
  }
  await serverUiStateSavePromise;
}

function hydrateServerUiState(nextState) {
  serverUiState = nextState && typeof nextState === "object" && !Array.isArray(nextState)
    ? { ...nextState }
    : {};
  let migrated = false;
  SERVER_UI_STATE_MIGRATION_KEYS.forEach((key) => {
    if (hasServerUiStateKey(key)) return;
    const legacy = legacyStorageGet(key);
    if (legacy === null || legacy === undefined || legacy === "") return;
    serverUiState[key] = legacy;
    migrated = true;
  });
  serverUiStateLoaded = true;
  uiState.currentRunId = String(readServerUiStateValue("gui2.currentRunId") || "");
  uiState.selectedHistoryRunId = String(readServerUiStateValue(UI_STORAGE_KEYS.historySelectedRunId) || "");
  uiState.compareHistoryRunId = String(readServerUiStateValue(UI_STORAGE_KEYS.historyCompareRunId) || "");
  uiState.liveFilter = String(readServerUiStateValue(UI_STORAGE_KEYS.liveFilter) || "all") || "all";
  uiState.locale = String(readServerUiStateValue(LOCALE_KEY) || "de") || "de";
  if (migrated) {
    SERVER_UI_STATE_MIGRATION_KEYS.forEach((key) => legacyStorageRemove(key));
    writeServerUiStateValue("__migration_marker__", "v1");
    delete serverUiState.__migration_marker__;
  }
}

const DASHBOARD_PIPELINE_GROUPS = [
  { key: "SCAN", phases: ["SCAN_INPUT", "CHANNEL_SPLIT", "NORMALIZATION", "GLOBAL_METRICS"] },
  { key: "REG", phases: ["REGISTRATION", "PREWARP", "COMMON_OVERLAP"] },
  { key: "TILES", phases: ["TILE_GRID", "LOCAL_METRICS", "TILE_RECONSTRUCTION", "STATE_CLUSTERING", "SYNTHETIC_FRAMES"] },
  { key: "STACK", phases: ["STACKING", "DEBAYER"] },
  { key: "ASTROM", phases: ["ASTROMETRY"] },
  { key: "BGE", phases: ["BGE"] },
  { key: "PCC", phases: ["PCC"] },
  { key: "DONE", phases: [] },
];

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
  "input_scan.calibration.bias_use_master": "calibration.bias_use_master",
  "input_scan.calibration.use_dark": "calibration.use_dark",
  "input_scan.calibration.dark_use_master": "calibration.dark_use_master",
  "input_scan.calibration.use_flat": "calibration.use_flat",
  "input_scan.calibration.flat_use_master": "calibration.flat_use_master",
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

const SCAN_CALIBRATION_BINDINGS = [
  {
    storageKey: "bias",
    sourceId: "cal-bias-source",
    inputId: "cal-bias-dir",
    useMasterPath: "calibration.bias_use_master",
    dirPath: "calibration.bias_dir",
    masterPath: "calibration.bias_master",
    dirPlaceholder: "Bias-Ordner waehlen",
    masterPlaceholder: "Master-Bias-Datei waehlen",
    dirTitle: "Bias-Ordner setzen.",
    masterTitle: "Master-Bias-Datei setzen.",
  },
  {
    storageKey: "dark",
    sourceId: "cal-dark-source",
    inputId: "cal-dark-dir",
    useMasterPath: "calibration.dark_use_master",
    dirPath: "calibration.darks_dir",
    masterPath: "calibration.dark_master",
    dirPlaceholder: "Dark-Ordner waehlen",
    masterPlaceholder: "Master-Dark-Datei waehlen",
    dirTitle: "Dark-Ordner setzen.",
    masterTitle: "Master-Dark-Datei setzen.",
  },
  {
    storageKey: "flat",
    sourceId: "cal-flat-source",
    inputId: "cal-flat-dir",
    useMasterPath: "calibration.flat_use_master",
    dirPath: "calibration.flats_dir",
    masterPath: "calibration.flat_master",
    dirPlaceholder: "Flat-Ordner waehlen",
    masterPlaceholder: "Master-Flat-Datei waehlen",
    dirTitle: "Flat-Ordner setzen.",
    masterTitle: "Master-Flat-Datei setzen.",
  },
];

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
  writeServerUiStateValue("gui2.currentRunId", uiState.currentRunId);
}

function markCurrentRunFromHistory(runId) {
  const value = String(runId || "").trim();
  if (!value) return;
  writeServerUiStateValue(HISTORY_CURRENT_RUN_KEY, value);
}

function clearCurrentRunHistoryMark() {
  writeServerUiStateValue(HISTORY_CURRENT_RUN_KEY, "");
}

function isCurrentRunFromHistory() {
  const marked = String(readServerUiStateValue(HISTORY_CURRENT_RUN_KEY) || "").trim();
  return Boolean(marked) && marked === String(uiState.currentRunId || "").trim();
}

function clearCurrentRunId() {
  uiState.currentRunId = "";
  writeServerUiStateValue("gui2.currentRunId", "");
  clearCurrentRunHistoryMark();
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

function scanErrorFromResult(result) {
  const errors = Array.isArray(result?.errors) ? result.errors : [];
  const warnings = Array.isArray(result?.warnings) ? result.warnings : [];
  const firstError = errors.find((e) => String(e?.message || "").trim()) || errors[0];
  if (firstError) return String(firstError.message || firstError.code || "Unbekannter Scan-Fehler");
  const firstWarn = warnings.find((w) => String(w?.message || "").trim()) || warnings[0];
  if (firstWarn) return String(firstWarn.message || firstWarn.code || "Scan-Warnung");
  return "";
}

function setRunReady(status, runStatus = "") {
  uiState.runReadyStatus = String(status || "check");
  uiState.runProcessStatus = String(runStatus || "");
  const chip = $("status-run-ready");
  const guardrailChip = $("status-guardrail");
  if (!chip && !guardrailChip) return;
  const runNormalized = String(runStatus || "").toLowerCase();
  const applyChip = (node, variant, text) => {
    if (!node) return;
    node.textContent = text;
    node.className = `shell-status-chip shell-status-chip-${variant}`;
  };
  const guardrailNormalized = String(status || "check").toLowerCase();
  const guardrailText = guardrailNormalized === "ok"
    ? t("ui.status.guardrail_ok", "Guardrails: OK")
    : guardrailNormalized === "error"
      ? t("ui.status.guardrail_error", "Guardrails: blocked")
      : t("ui.status.guardrail_check", "Guardrails: check");
  applyChip(
    guardrailChip,
    guardrailNormalized === "ok" ? "ok" : guardrailNormalized === "error" ? "error" : "check",
    guardrailText,
  );
  if (["running", "queued", "starting"].includes(runNormalized)) {
    applyChip(chip, "running", t("ui.status.run_ready_running", "Status: run running"));
    return;
  }
  const normalized = String(status || "check").toLowerCase();
  const statusText = normalized === "ok"
    ? t("ui.status.run_ready_ok", "Status: ready to run")
    : normalized === "error"
      ? t("ui.status.run_ready_blocked", "Status: blocked")
      : t("ui.status.run_ready_check", "Status: check");
  applyChip(chip, normalized === "ok" ? "ok" : normalized === "error" ? "error" : "check", statusText);
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
  const text = String(line ?? "").trim();
  if (!text) return;
  const lines = String(el.textContent || "")
    .split("\n")
    .filter(Boolean);
  if (lines[lines.length - 1] === text) return;
  lines.push(text);
  el.textContent = lines.slice(-300).join("\n");
}

function compactLogMessage(raw) {
  return String(raw ?? "")
    .replace(/\r/g, "")
    .split("\n")
    .map((part) => part.trim())
    .filter(Boolean)
    .join(" | ");
}

function maybeParseJsonLine(raw) {
  if (typeof raw !== "string") return raw;
  const trimmed = raw.trim();
  if (!trimmed || (!trimmed.startsWith("{") && !trimmed.startsWith("["))) return raw;
  try {
    return JSON.parse(trimmed);
  } catch {
    return raw;
  }
}

function humanizeLogToken(raw) {
  return String(raw || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function shortLogTimestamp(isoRaw) {
  const iso = String(isoRaw || "").trim();
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (value) => String(value).padStart(2, "0");
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function formatLogPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "";
  const pct = numeric <= 1 ? numeric * 100 : numeric;
  return `${pct.toFixed(pct >= 10 ? 0 : 1)}%`;
}

function formatLogBytes(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = numeric;
  let unitIndex = 0;
  while (amount >= 1024 && unitIndex < units.length - 1) {
    amount /= 1024;
    unitIndex += 1;
  }
  const digits = amount >= 100 || unitIndex === 0 ? 0 : amount >= 10 ? 1 : 2;
  return `${amount.toFixed(digits)} ${units[unitIndex]}`;
}

function formatCatalogSummary(catalogs) {
  if (!catalogs || typeof catalogs !== "object") return "";
  const items = Object.entries(catalogs)
    .map(([key, value]) => `${String(key || "").toUpperCase()}:${value ? "ok" : "missing"}`)
    .filter(Boolean);
  return items.join(", ");
}

function genericLogSummary(entry) {
  if (!entry || typeof entry !== "object") return "";
  const simpleParts = [];
  const simpleState = humanizeLogToken(entry.state || entry.status || "");
  const simplePct = formatLogPercent(entry.progress ?? entry.pct);
  if (simpleState) simpleParts.push(simpleState);
  if (entry.stage) simpleParts.push(humanizeLogToken(entry.stage));
  if (simplePct) simpleParts.push(simplePct);
  if (Number.isFinite(Number(entry.current_chunk))) simpleParts.push(`chunk ${entry.current_chunk}`);
  const simpleError = compactLogMessage(entry.error || "");
  if (simpleError) simpleParts.push(simpleError);
  if (simpleParts.length > 0) return simpleParts.join(" | ");

  const parts = [];
  for (const [key, value] of Object.entries(entry)) {
    if (value === null || value === undefined) continue;
    if (typeof value === "object") continue;
    if (["stdout", "stderr", "command", "matrix"].includes(key)) continue;
    parts.push(`${humanizeLogToken(key)}=${value}`);
    if (parts.length >= 6) break;
  }
  return parts.join(" | ");
}

function formatLogNumber(value, digits = 3) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "";
  return numeric.toFixed(digits);
}

function pushLogPart(parts, value) {
  const text = compactLogMessage(value);
  if (text) parts.push(text);
}

function normalizeLogLevel(raw) {
  const level = String(raw || "").trim().toLowerCase();
  if (["error", "err", "fatal"].includes(level)) return "error";
  if (["warning", "warn", "skipped"].includes(level)) return "warning";
  return "info";
}

function detectStructuredLogLevel(entry) {
  const parsed = maybeParseJsonLine(entry);
  if (typeof parsed === "string") {
    const lower = parsed.toLowerCase();
    if (lower.includes("error") || lower.includes("failed") || lower.includes("fatal")) return "error";
    if (lower.includes("warning") || lower.includes("warn") || lower.includes("skipped")) return "warning";
    return "info";
  }
  if (!parsed || typeof parsed !== "object") return "info";

  const type = String(parsed.type || "").trim().toLowerCase();
  const payload = parsed.payload && typeof parsed.payload === "object" ? parsed.payload : parsed;
  const status = String(payload.status || parsed.status || "").trim().toLowerCase();
  const severity = String(payload.severity || parsed.severity || "").trim().toLowerCase();
  const message = compactLogMessage(payload.message || parsed.message || payload.error || parsed.error || "").toLowerCase();

  if (type === "error" || type === "run_stream_error") return "error";
  if (type === "warning") return "warning";
  if (["error", "failed", "aborted", "cancelled"].includes(status)) return "error";
  if (["warning", "warn", "skipped"].includes(status)) return "warning";
  if (severity) return normalizeLogLevel(severity);
  if (message.includes("error") || message.includes("failed") || message.includes("fatal")) return "error";
  if (message.includes("warning") || message.includes("warn") || message.includes("skipped")) return "warning";
  return "info";
}

function liveLogTag(level) {
  if (level === "error") return "ERR";
  if (level === "warning") return "WARN";
  return "INFO";
}

function summarizePhaseEndPayload(phaseRaw, payload) {
  if (!payload || typeof payload !== "object") return [];
  const phase = String(phaseRaw || "").trim().toUpperCase();
  const parts = [];

  pushLogPart(parts, payload.reason);
  pushLogPart(parts, payload.error);

  if (Number.isFinite(Number(payload.exit_code))) {
    parts.push(`exit ${payload.exit_code}`);
  }

  if (phase === "ASTROMETRY") {
    const ra = formatLogNumber(payload.ra, 6);
    const dec = formatLogNumber(payload.dec, 6);
    const scale = formatLogNumber(payload.pixel_scale_arcsec, 3);
    const rotation = formatLogNumber(payload.rotation_deg, 2);
    if (ra && dec) parts.push(`RA ${ra} deg`, `Dec ${dec} deg`);
    if (scale) parts.push(`Scale ${scale} arcsec/px`);
    if (rotation) parts.push(`Rot ${rotation} deg`);
    pushLogPart(parts, payload.astap_bin);
    pushLogPart(parts, payload.wcs_file);
  } else if (phase === "REGISTRATION") {
    if (Number.isFinite(Number(payload.ref_frame))) parts.push(`ref ${payload.ref_frame}`);
    pushLogPart(parts, payload.ref_frame_strategy);
    if (Number.isFinite(Number(payload.frames_cc_positive))) parts.push(`cc>0 ${payload.frames_cc_positive}`);
    if (Number.isFinite(Number(payload.frames_cc_zero))) parts.push(`cc=0 ${payload.frames_cc_zero}`);
    const rejected = [
      ["orient", payload.reg_reject_orientation_outliers],
      ["reflect", payload.reg_reject_reflection_outliers],
      ["scale", payload.reg_reject_scale_outliers],
      ["cc", payload.reg_reject_cc_outliers],
      ["shift", payload.reg_reject_shift_outliers],
    ]
      .filter(([, value]) => Number.isFinite(Number(value)) && Number(value) > 0)
      .map(([label, value]) => `${label}=${value}`);
    if (rejected.length > 0) parts.push(`reject ${rejected.join(",")}`);
    const modeled = [
      ["pred", payload.reg_model_predicted],
      ["local", payload.reg_model_local_refined],
      ["interp", payload.reg_model_interpolated],
      ["blend", payload.reg_model_blended],
    ]
      .filter(([, value]) => Number.isFinite(Number(value)) && Number(value) > 0)
      .map(([label, value]) => `${label}=${value}`);
    if (modeled.length > 0) parts.push(`model ${modeled.join(",")}`);
  } else if (phase === "PREWARP") {
    if (Number.isFinite(Number(payload.num_frames_with_data)) && Number.isFinite(Number(payload.num_frames))) {
      parts.push(`frames ${payload.num_frames_with_data}/${payload.num_frames}`);
    }
    if (Number.isFinite(Number(payload.canvas_width)) && Number.isFinite(Number(payload.canvas_height))) {
      parts.push(`canvas ${payload.canvas_width}x${payload.canvas_height}`);
    }
    if (Number.isFinite(Number(payload.tile_offset_x)) && Number.isFinite(Number(payload.tile_offset_y))) {
      parts.push(`offset ${payload.tile_offset_x},${payload.tile_offset_y}`);
    }
    if (Number.isFinite(Number(payload.workers))) parts.push(`workers ${payload.workers}`);
  } else if (phase === "PCC") {
    if (Number.isFinite(Number(payload.stars_matched))) parts.push(`matched ${payload.stars_matched}`);
    if (Number.isFinite(Number(payload.stars_used))) parts.push(`used ${payload.stars_used}`);
    const rms = formatLogNumber(payload.residual_rms, 4);
    const det = formatLogNumber(payload.determinant, 4);
    const cond = formatLogNumber(payload.condition_number, 3);
    if (rms) parts.push(`RMS ${rms}`);
    if (det) parts.push(`det ${det}`);
    if (cond) parts.push(`cond ${cond}`);
    pushLogPart(parts, payload.apply_mode);
    pushLogPart(parts, payload.source);
    pushLogPart(parts, payload.input_rgb_bge);
  }

  return parts;
}

function formatRunStreamLog(entry, { suppressRunStatus = false } = {}) {
  if (!entry || typeof entry !== "object") return "";
  const type = String(entry.type || "").trim().toLowerCase();
  if (!type) return "";
  const ts = shortLogTimestamp(entry.ts);
  const prefix = ts ? `${ts} | ` : "";
  const payload = entry.payload && typeof entry.payload === "object" ? entry.payload : {};
  const phase = humanizeLogToken(entry.phase || payload.phase_name || payload.phase || "");
  const pct = formatLogPercent(entry.pct ?? payload.progress ?? payload.pct);
  const message = compactLogMessage(payload.message || entry.message || payload.substep || entry.substep || "");

  if (type === "phase_start") {
    return `${prefix}${phase || "Phase"} | start`;
  }
  if (type === "phase_progress") {
    const parts = [phase || "Phase"];
    if (pct) parts.push(pct);
    const current = Number(entry.current ?? payload.current);
    const total = Number(entry.total ?? payload.total);
    if (Number.isFinite(current) && Number.isFinite(total) && total > 0) parts.push(`${current}/${total}`);
    pushLogPart(parts, payload.pass || entry.pass);
    if (message) parts.push(message);
    return `${prefix}${parts.join(" | ")}`;
  }
  if (type === "phase_end") {
    const status = String(payload.status || entry.status || "ok").trim().toUpperCase();
    const parts = [phase || "Phase", status || "OK"];
    if (pct) parts.push(pct);
    if (message) parts.push(message);
    parts.push(...summarizePhaseEndPayload(phase, payload));
    return `${prefix}${parts.join(" | ")}`;
  }
  if (type === "queue_progress") {
    const done = Number(payload.done);
    const total = Number(payload.total);
    const parts = ["Queue"];
    if (Number.isFinite(done) && Number.isFinite(total) && total > 0) parts.push(`${done}/${total}`);
    if (pct) parts.push(pct);
    const filter = humanizeLogToken(entry.filter || "");
    if (filter) parts.push(`filter ${filter}`);
    return `${prefix}${parts.join(" | ")}`;
  }
  if (type === "run_start") {
    const parts = ["Run", "start"];
    pushLogPart(parts, payload.run_dir || entry.run_dir);
    return `${prefix}${parts.join(" | ")}`;
  }
  if (type === "run_end") {
    const parts = ["Run beendet", humanizeLogToken(payload.state || entry.status || "done")];
    const currentPhase = humanizeLogToken(payload.current_phase || "");
    if (currentPhase) parts.push(currentPhase);
    return `${prefix}${parts.filter(Boolean).join(" | ")}`;
  }
  if (type === "resume_start") {
    const fromPhase = humanizeLogToken(payload.from_phase || entry.from_phase || "");
    return `${prefix}${["Resume", "start", fromPhase].filter(Boolean).join(" | ")}`;
  }
  if (type === "resume_end") {
    const ok = payload.success ?? entry.success;
    const fromPhase = humanizeLogToken(payload.from_phase || entry.from_phase || "");
    const parts = ["Resume", ok ? "OK" : "ERROR"];
    if (fromPhase) parts.push(fromPhase);
    pushLogPart(parts, payload.error || entry.error);
    return `${prefix}${parts.join(" | ")}`;
  }
  if (type === "warning" || type === "error") {
    const label = type === "error" ? "Fehler" : "Warnung";
    const parts = [label];
    if (phase) parts.push(phase);
    if (message) parts.push(message);
    return `${prefix}${parts.join(" | ")}`;
  }
  if (type === "run_stream_error") {
    return `${prefix}Stream error | ${message || "unbekannt"}`;
  }
  if (type === "run_status") {
    const state = String(payload.status || entry.state || "").trim().toLowerCase();
    const terminal = ["completed", "failed", "cancelled", "aborted", "error", "done", "finished"].includes(state);
    if (suppressRunStatus && !terminal) return "";
    const parts = ["Run", humanizeLogToken(state || "status")];
    if (phase) parts.push(phase);
    if (pct) parts.push(pct);
    return `${prefix}${parts.join(" | ")}`;
  }
  if (type === "log_line") {
    const parts = [];
    if (phase) parts.push(phase);
    if (message) parts.push(message);
    return parts.length > 0 ? `${prefix}${parts.join(" | ")}` : "";
  }
  return "";
}

function formatAstrometryLog(entry) {
  if (!entry || typeof entry !== "object") return "";
  if (Object.prototype.hasOwnProperty.call(entry, "installed") && (Object.prototype.hasOwnProperty.call(entry, "binary") || Object.prototype.hasOwnProperty.call(entry, "catalogs"))) {
    const parts = [entry.installed ? "ASTAP gefunden" : "ASTAP fehlt"];
    if (entry.binary) parts.push(String(entry.binary));
    if (entry.data_dir) parts.push(`dir ${entry.data_dir}`);
    const catalogs = formatCatalogSummary(entry.catalogs);
    if (catalogs) parts.push(`catalogs ${catalogs}`);
    return parts.join(" | ");
  }
  if (Object.prototype.hasOwnProperty.call(entry, "ra_deg") || Object.prototype.hasOwnProperty.call(entry, "wcs_path")) {
    const parts = ["Plate solve"];
    if (Number.isFinite(Number(entry.ra_deg)) && Number.isFinite(Number(entry.dec_deg))) {
      parts.push(`RA ${Number(entry.ra_deg).toFixed(6)} deg`);
      parts.push(`Dec ${Number(entry.dec_deg).toFixed(6)} deg`);
    }
    if (Number.isFinite(Number(entry.pixel_scale_arcsec))) parts.push(`Scale ${Number(entry.pixel_scale_arcsec).toFixed(3)} arcsec/px`);
    if (entry.wcs_path) parts.push(String(entry.wcs_path));
    return parts.join(" | ");
  }
  if (entry.output_path || entry.saved) {
    return `Solved FITS gespeichert | ${entry.output_path || "-"}`;
  }
  return "";
}

function formatPccLog(entry) {
  if (!entry || typeof entry !== "object") return "";
  if (Object.prototype.hasOwnProperty.call(entry, "installed") && Object.prototype.hasOwnProperty.call(entry, "total") && Array.isArray(entry.missing)) {
    const parts = [`Siril catalog ${entry.installed}/${entry.total}`];
    if (entry.missing.length > 0) parts.push(`missing ${entry.missing.length}`);
    if (entry.catalog_dir) parts.push(String(entry.catalog_dir));
    return parts.join(" | ");
  }
  if (Object.prototype.hasOwnProperty.call(entry, "latency_ms") && Object.prototype.hasOwnProperty.call(entry, "ok")) {
    return `Online source ${entry.ok ? "OK" : "fehler"} | ${entry.latency_ms} ms${entry.error ? ` | ${entry.error}` : ""}`;
  }
  if (Object.prototype.hasOwnProperty.call(entry, "stars_used") || Object.prototype.hasOwnProperty.call(entry, "stars_matched") || Object.prototype.hasOwnProperty.call(entry, "residual_rms")) {
    const parts = ["PCC"];
    if (entry.stars_matched ?? entry.n_stars_matched) parts.push(`matched ${entry.stars_matched ?? entry.n_stars_matched}`);
    if (entry.stars_used ?? entry.n_stars_used) parts.push(`used ${entry.stars_used ?? entry.n_stars_used}`);
    if (entry.residual_rms !== undefined && entry.residual_rms !== null && entry.residual_rms !== "") parts.push(`RMS ${entry.residual_rms}`);
    if (entry.output_rgb) parts.push(String(entry.output_rgb));
    return parts.join(" | ");
  }
  if (entry.output_rgb && Array.isArray(entry.output_channels)) {
    return `PCC gespeichert | ${entry.output_rgb}`;
  }
  return "";
}

function formatJobLog(entry) {
  if (!entry || typeof entry !== "object" || !entry.job_id) return "";
  const state = humanizeLogToken(entry.state || "");
  const data = entry.data && typeof entry.data === "object" ? entry.data : {};
  const parts = [`Job ${entry.job_id}`];
  if (state) parts.push(state);
  if (data.stage) parts.push(humanizeLogToken(data.stage));
  if (data.catalog_id) parts.push(String(data.catalog_id).toUpperCase());
  if (Number.isFinite(Number(data.current_chunk))) parts.push(`chunk ${data.current_chunk}`);
  const pct = formatLogPercent(data.progress);
  if (pct) parts.push(pct);
  const received = formatLogBytes(data.bytes_received);
  const total = formatLogBytes(data.bytes_total);
  if (received && total) parts.push(`${received}/${total}`);
  else if (received) parts.push(received);
  if (data.resumed) parts.push("resume");
  if (Number.isFinite(Number(data.attempt))) parts.push(`attempt ${data.attempt}`);
  if (Number.isFinite(Number(data.status_code)) && Number(data.status_code) > 0) parts.push(`HTTP ${data.status_code}`);
  if (data.retrying) parts.push("retry");
  const error = compactLogMessage(data.error || entry.error || "");
  if (error) parts.push(error);
  return parts.join(" | ");
}

function formatStructuredLogLine(entry, options = {}) {
  const parsed = maybeParseJsonLine(entry);
  if (typeof parsed === "string") return compactLogMessage(parsed);
  if (!parsed || typeof parsed !== "object") return String(parsed ?? "");
  return formatRunStreamLog(parsed, options)
    || formatAstrometryLog(parsed)
    || formatPccLog(parsed)
    || formatJobLog(parsed)
    || genericLogSummary(parsed)
    || compactLogMessage(JSON.stringify(parsed));
}

function appendStructuredLog(el, entry, options = {}) {
  const line = formatStructuredLogLine(entry, options);
  if (!line) return;
  appendLine(el, line);
  scrollLogToEnd(el);
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
  return String(readServerUiStateValue(CONFIG_DRAFT_KEY) || "");
}

function setConfigDraft(yamlText) {
  const value = String(yamlText || "");
  if (!value) return;
  uiState.configYaml = value;
  writeServerUiStateValue(CONFIG_DRAFT_KEY, uiState.configYaml);
}

function getConfigValidationState() {
  try {
    const raw = readServerUiStateValue(CONFIG_VALIDATION_STATE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function setConfigValidationState({ yaml = "", ok = false, errors = [], warnings = [] } = {}) {
  writeServerUiStateValue(
    CONFIG_VALIDATION_STATE_KEY,
    JSON.stringify({
      yaml: String(yaml || ""),
      ok: Boolean(ok),
      errors: Array.isArray(errors) ? errors : [],
      warnings: Array.isArray(warnings) ? warnings : [],
      updated_at: new Date().toISOString(),
    }),
  );
}

function clearConfigValidationState() {
  writeServerUiStateValue(CONFIG_VALIDATION_STATE_KEY, "");
}

function getParameterDirtyState() {
  try {
    const raw = readServerUiStateValue(PARAMETER_DIRTY_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    writeServerUiStateValue(PARAMETER_DIRTY_STATE_KEY, "");
    return {};
  }
}

function setParameterDirtyState(dirtyMap) {
  const payload = dirtyMap && typeof dirtyMap === "object" && !Array.isArray(dirtyMap) ? dirtyMap : {};
  if (Object.keys(payload).length === 0) {
    writeServerUiStateValue(PARAMETER_DIRTY_STATE_KEY, "");
    return;
  }
  writeServerUiStateValue(PARAMETER_DIRTY_STATE_KEY, JSON.stringify(payload));
}

function clearParameterDirtyState() {
  writeServerUiStateValue(PARAMETER_DIRTY_STATE_KEY, "");
}

function setDisabledLike(el, disabled) {
  if (!el) return;
  const isOff = Boolean(disabled);
  if ("disabled" in el) el.disabled = isOff;
  el.setAttribute("aria-disabled", isOff ? "true" : "false");
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
  let displayValue = Array.isArray(value) || typeof value === "object" ? JSON.stringify(value) : String(value);
  el.value = displayValue;
}

function getByPath(root, dotted) {
  let cur = root;
  for (const key of String(dotted || "").split(".").filter(Boolean)) {
    if (!cur || typeof cur !== "object" || !(key in cur)) return undefined;
    cur = cur[key];
  }
  return cur;
}

function scanCalibrationBindingForElement(el) {
  if (!el) return null;
  const id = String(el.id || "");
  return SCAN_CALIBRATION_BINDINGS.find((binding) => binding.sourceId === id || binding.inputId === id) || null;
}

function scanCalibrationUseMaster(binding) {
  return readFieldValue($(binding?.sourceId)) === true;
}

function scanCalibrationActivePath(binding, useMaster = scanCalibrationUseMaster(binding)) {
  return useMaster ? binding.masterPath : binding.dirPath;
}

function calibrationStorageKey(binding, useMaster) {
  const stem = String(binding?.storageKey || binding?.inputId || "cal").trim();
  return `${CALIBRATION_PATH_KEY_PREFIX}.${stem}.${useMaster ? "master" : "dir"}`;
}

function storedCalibrationPath(binding, useMaster) {
  const value = String(readServerUiStateValue(calibrationStorageKey(binding, useMaster)) || "").trim();
  if (!value) return "";
  if (!isAbsolutePath(value)) {
    writeServerUiStateValue(calibrationStorageKey(binding, useMaster), "");
    return "";
  }
  return value;
}

function persistCalibrationPath(binding, useMaster, rawValue) {
  if (!binding) return;
  const key = calibrationStorageKey(binding, useMaster);
  const value = String(rawValue || "").trim();
  if (!value) {
    writeServerUiStateValue(key, "");
    return;
  }
  if (!isAbsolutePath(value)) return;
  writeServerUiStateValue(key, value);
}

function syncScanCalibrationInputPresentation(binding, useMaster) {
  const input = $(binding?.inputId);
  if (!input) return;
  input.placeholder = useMaster ? binding.masterPlaceholder : binding.dirPlaceholder;
  input.title = useMaster ? binding.masterTitle : binding.dirTitle;
}

function syncScanCalibrationUiFromConfig(config) {
  SCAN_CALIBRATION_BINDINGS.forEach((binding) => {
    const sourceEl = $(binding.sourceId);
    const inputEl = $(binding.inputId);
    if (!sourceEl || !inputEl) return;
    const useMaster = Boolean(getByPath(config, binding.useMasterPath));
    writeFieldValue(sourceEl, useMaster);
    syncScanCalibrationInputPresentation(binding, useMaster);
    const activeValue = getByPath(config, scanCalibrationActivePath(binding, useMaster));
    const preferredValue =
      activeValue === undefined || activeValue === null || String(activeValue).trim() === ""
        ? storedCalibrationPath(binding, useMaster)
        : String(activeValue);
    inputEl.value = preferredValue;
  });
}

async function restoreStoredCalibrationPathsIntoConfig(config) {
  if (!config || typeof config !== "object") return config;
  const updates = [];
  SCAN_CALIBRATION_BINDINGS.forEach((binding) => {
    const useMaster = Boolean(getByPath(config, binding.useMasterPath));
    const activePath = scanCalibrationActivePath(binding, useMaster);
    const currentValue = String(getByPath(config, activePath) || "").trim();
    if (currentValue) {
      persistCalibrationPath(binding, useMaster, currentValue);
      return;
    }
    const storedValue = storedCalibrationPath(binding, useMaster);
    if (!storedValue) return;
    updates.push({ path: activePath, value: storedValue });
  });
  if (updates.length === 0) return config;
  const patched = await patchConfig({ updates, persist: false });
  return patched?.config && typeof patched.config === "object" ? patched.config : config;
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
  writeServerUiStateValue(LAST_INPUT_DIRS_KEY, value);
}

function persistPresetsDir(rawValue) {
  const value = String(rawValue || "").trim();
  if (!value) {
    writeServerUiStateValue(PRESETS_DIR_KEY, "");
    return;
  }
  if (!isAbsolutePath(value)) return;
  writeServerUiStateValue(PRESETS_DIR_KEY, value);
}

function selectedPresetsDir() {
  const stored = String(readServerUiStateValue(PRESETS_DIR_KEY) || "").trim();
  if (stored) return stored;
  return String(uiState.projectPresetsDir || "").trim();
}

function syncPresetDirInputs() {
  const value = selectedPresetsDir();
  ["dashboard-preset-dir", "parameter-preset-dir", "wizard-preset-dir", "monitor-resume-preset-dir"].forEach((id) => {
    const el = $(id);
    if (el) el.value = value;
  });
}

function storedTextValue(key, { absolute = false } = {}) {
  const value = String(readServerUiStateValue(key) || "").trim();
  if (!value) return "";
  if (absolute && !isAbsolutePath(value)) {
    writeServerUiStateValue(key, "");
    return "";
  }
  return value;
}

function persistTextValue(key, rawValue, { absolute = false } = {}) {
  const value = String(rawValue || "").trim();
  if (!value) {
    writeServerUiStateValue(key, "");
    return;
  }
  if (absolute && !isAbsolutePath(value)) return;
  writeServerUiStateValue(key, value);
}

function bindStoredField(id, key, { absolute = false, normalize = null, overwrite = false } = {}) {
  const el = $(id);
  if (!el) return;
  const stored = storedTextValue(key, { absolute });
  if ((overwrite || !String(el.value || "").trim()) && stored) {
    el.value = stored;
  }
  const persist = () => {
    const raw = normalize ? normalize(el.value) : String(el.value || "").trim();
    if (normalize && raw !== el.value) el.value = raw;
    persistTextValue(key, raw, { absolute });
  };
  el.addEventListener("input", persist);
  el.addEventListener("change", persist);
}

function restoreStoredSelectValue(selectId, key, { absolute = false } = {}) {
  const select = $(selectId);
  if (!select) return "";
  const stored = storedTextValue(key, { absolute });
  if (!stored) return "";
  const option = Array.from(select.options || []).find((item) => String(item.value || "") === stored);
  if (!option) return "";
  select.value = option.value;
  return option.value;
}

function bindStoredSelect(selectId, key, { absolute = false } = {}) {
  const select = $(selectId);
  if (!select) return;
  restoreStoredSelectValue(selectId, key, { absolute });
  persistTextValue(key, String(select.value || "").trim(), { absolute });
  const persist = () => persistTextValue(key, String(select.value || "").trim(), { absolute });
  select.addEventListener("input", persist);
  select.addEventListener("change", persist);
}

function collectQueueDraftRows(scope = document) {
  return Array.from(scope.querySelectorAll(".ps-queue-row")).map((row) => {
    const select = row.querySelector("select");
    const inputs = Array.from(row.querySelectorAll("input[type='text']"));
    const enabled = row.querySelector("input[type='checkbox']");
    return {
      filter: String(select?.value || "").trim(),
      input_dir: String(inputs[0]?.value || "").trim(),
      pattern: String(inputs[1]?.value || "").trim(),
      run_id: String(inputs[2]?.value || "").trim(),
      enabled: enabled ? Boolean(enabled.checked) : true,
    };
  });
}

function restoreQueueDraftRows(key, scope = document) {
  let rows = [];
  try {
    const parsed = JSON.parse(String(readServerUiStateValue(key) || "[]"));
    rows = Array.isArray(parsed) ? parsed : [];
  } catch {
    writeServerUiStateValue(key, "");
    return;
  }
  if (rows.length === 0) return;
  Array.from(scope.querySelectorAll(".ps-queue-row")).forEach((row, index) => {
    const item = rows[index];
    if (!item || typeof item !== "object") return;
    const select = row.querySelector("select");
    const inputs = Array.from(row.querySelectorAll("input[type='text']"));
    const enabled = row.querySelector("input[type='checkbox']");
    if (select) select.value = String(item.filter || "");
    if (inputs[0]) inputs[0].value = String(item.input_dir || "");
    if (inputs[1]) inputs[1].value = String(item.pattern || "");
    if (inputs[2]) inputs[2].value = String(item.run_id || "");
    if (enabled) enabled.checked = item.enabled !== false;
  });
}

function bindQueueDraftPersistence(key, scope = document) {
  const rows = Array.from(scope.querySelectorAll(".ps-queue-row"));
  if (rows.length === 0) return;
  restoreQueueDraftRows(key, scope);
  const persist = () => {
    const items = collectQueueDraftRows(scope);
    const hasContent = items.some((item) => !item.enabled || item.filter || item.input_dir || item.pattern || item.run_id);
    if (!hasContent) {
      writeServerUiStateValue(key, "");
      return;
    }
    writeServerUiStateValue(key, JSON.stringify(items));
  };
  rows.forEach((row) => {
    row.querySelectorAll("input,select").forEach((el) => {
      el.addEventListener("input", persist);
      el.addEventListener("change", persist);
    });
  });
}

function persistHistorySelectionState() {
  persistTextValue(UI_STORAGE_KEYS.historySelectedRunId, uiState.selectedHistoryRunId);
  persistTextValue(UI_STORAGE_KEYS.historyCompareRunId, uiState.compareHistoryRunId);
}

function restoreLastInputDirs(...ids) {
  const value = String(readServerUiStateValue(LAST_INPUT_DIRS_KEY) || "").trim();
  if (!value) return;
  const dirs = parseInputDirs(value);
  if (!allAbsolutePaths(dirs)) {
    writeServerUiStateValue(LAST_INPUT_DIRS_KEY, "");
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

function explicitRunNameValue(inputId = "") {
  return sanitizeRunName(String((inputId ? $(inputId) : null)?.value || ""));
}

function preferredStoredRunName() {
  const dashboardName = sanitizeRunName(storedTextValue(UI_STORAGE_KEYS.dashboardRunName));
  if (dashboardName) return dashboardName;
  const wizardName = sanitizeRunName(storedTextValue(UI_STORAGE_KEYS.wizardRunName));
  if (wizardName) return wizardName;
  return "";
}

function preferredStoredRunsDir() {
  const sharedRunsDir = String(storedTextValue(UI_STORAGE_KEYS.dashboardRunsDir, { absolute: true }) || "").trim();
  if (sharedRunsDir) return sharedRunsDir;
  const wizardRunsDir = String(storedTextValue(UI_STORAGE_KEYS.wizardRunsDir, { absolute: true }) || "").trim();
  if (wizardRunsDir) return wizardRunsDir;
  return "";
}

function preferredStoredPresetPath() {
  return firstNonEmptyText(
    storedTextValue(UI_STORAGE_KEYS.dashboardPreset, { absolute: true }),
    storedTextValue(UI_STORAGE_KEYS.parameterPreset, { absolute: true }),
    storedTextValue(UI_STORAGE_KEYS.wizardPreset, { absolute: true }),
  );
}

function persistUnifiedPresetPath(path = "") {
  persistTextValue(UI_STORAGE_KEYS.dashboardPreset, path, { absolute: true });
  persistTextValue(UI_STORAGE_KEYS.parameterPreset, path, { absolute: true });
  persistTextValue(UI_STORAGE_KEYS.wizardPreset, path, { absolute: true });
}

function syncPresetSelectValues(path = "") {
  const normalized = String(path || "").trim();
  ["dashboard-preset", "parameter-preset-select", "wizard-preset-select", "monitor-resume-preset-select"].forEach((id) => {
    const select = $(id);
    if (!select || !normalized) return;
    const option = Array.from(select.options || []).find((item) => String(item.value || "") === normalized);
    if (option) select.value = normalized;
  });
}

function syncUnifiedPresetSelection(path = "") {
  const normalized = String(path || "").trim();
  persistUnifiedPresetPath(normalized);
  syncPresetSelectValues(normalized);
}

function restoreUnifiedPresetSelectValue(selectId) {
  const select = $(selectId);
  if (!select) return "";
  const stored = preferredStoredPresetPath();
  if (!stored) return "";
  const option = Array.from(select.options || []).find((item) => String(item.value || "") === stored);
  if (!option) return "";
  select.value = option.value;
  return option.value;
}

function bindUnifiedPresetSelect(selectId) {
  const select = $(selectId);
  if (!select) return;
  restoreUnifiedPresetSelectValue(selectId);
  syncUnifiedPresetSelection(String(select.value || "").trim());
  const persist = () => syncUnifiedPresetSelection(String(select.value || "").trim());
  select.addEventListener("input", persist);
  select.addEventListener("change", persist);
}

function preferredRunName({ inputId = "", storageKey = "", fallbackDirs = [] } = {}) {
  const inputValue = explicitRunNameValue(inputId);
  if (inputValue) return inputValue;
  const storedValue = storageKey ? storedTextValue(storageKey) : "";
  if (storedValue) return sanitizeRunName(storedValue);
  return suggestRunNameFromInputs(fallbackDirs);
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
      : firstNonEmptyText(readServerUiStateValue(LAST_INPUT_DIRS_KEY), $("dashboard-input-dirs")?.value, $("inp-dirs")?.value);
  const inputDirs = parseInputDirs(inputDirsText);
  if (inputDirs.length === 0) {
    throw new Error("Bitte mindestens einen Eingabeordner setzen.");
  }
  persistLastInputDirs(inputDirsText);
  await flushServerUiState();

  const runNameEl = useDashboardFields
    ? $("dashboard-run-name")
    : useWizardFields
      ? $("wizard-run-name")
      : $("scan-run-name");
  const runsDirEl = useDashboardFields
    ? $("dashboard-run-runs-dir")
    : useWizardFields
      ? $("wizard-runs-dir")
      : $("scan-runs-dir");
  const configYaml = await resolveConfigYamlForRun();
  const explicitRunName = useDashboardFields
    ? explicitRunNameValue("dashboard-run-name")
    : useWizardFields
      ? explicitRunNameValue("wizard-run-name")
      : sanitizeRunName(runNameEl?.value || "");
  const runName = useDashboardFields
    ? preferredRunName({ inputId: "dashboard-run-name", storageKey: UI_STORAGE_KEYS.dashboardRunName, fallbackDirs: inputDirs })
    : useWizardFields
      ? preferredRunName({ inputId: "wizard-run-name", storageKey: UI_STORAGE_KEYS.wizardRunName, fallbackDirs: inputDirs })
      : explicitRunName || preferredStoredRunName() || suggestRunNameFromInputs(inputDirs);
  if (runNameEl && explicitRunName) runNameEl.value = explicitRunName;
  const runsDir = firstNonEmptyText(runsDirEl?.value, preferredStoredRunsDir(), uiState.projectRunsDir);
  if (runsDirEl && !String(runsDirEl.value || "").trim() && runsDir) {
    runsDirEl.value = runsDir;
  }
  if (useDashboardFields) {
    persistTextValue(UI_STORAGE_KEYS.dashboardRunName, explicitRunName);
    persistTextValue(UI_STORAGE_KEYS.dashboardRunsDir, runsDir, { absolute: true });
  }
  if (useWizardFields) {
    persistTextValue(UI_STORAGE_KEYS.wizardRunName, explicitRunName);
    persistTextValue(UI_STORAGE_KEYS.wizardRunsDir, runsDir, { absolute: true });
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
    config_yaml: configYaml,
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
  const colorMode = String(src.color_mode || "");
  const normalizedColorMode = normalizeDetectedColorMode(colorMode);
  if (normalizedColorMode) {
    writeServerUiStateValue(LAST_SCAN_COLOR_MODE_KEY, normalizedColorMode);
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
  const sizeText = data.image_width > 0 && data.image_height > 0 ? `${data.image_width} x ${data.image_height}` : "-";
  const candidates = data.color_mode_candidates.length > 0 ? data.color_mode_candidates.join(", ") : "-";
  const framesText = data.has_scan ? String(data.frames_detected) : "-";
  const colorModeText = data.color_mode || "-";
  const errorCountText = data.has_scan ? String(data.errors.length) : "-";
  const warningCountText = data.has_scan ? String(data.warnings.length) : "-";
  setText($(`${prefix}-status`), status);
  setText($(`${prefix}-input-path`), data.input_path || "-");
  setText($(`${prefix}-frames`), framesText);
  setText($(`${prefix}-color-mode`), colorModeText);
  setText($(`${prefix}-candidates`), candidates);
  setText($(`${prefix}-size`), sizeText);
  setText($(`${prefix}-bayer`), data.bayer_pattern || "-");
  setText($(`${prefix}-confirm`), data.requires_user_confirmation ? t("ui.value.yes", "ja") : t("ui.value.no", "nein"));
  setText($(`${prefix}-errors`), errorCountText);
  setText($(`${prefix}-warnings`), warningCountText);
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
  if (framesKpi) framesKpi.textContent = data.has_scan ? String(data.frames_detected) : "-";
  const colorChip = $("dashboard-kpi-color-mode");
  if (colorChip) colorChip.textContent = `Color: ${data.color_mode || "-"}`;

  const qualityKpi = document.querySelector("#dashboard-kpi-open-warnings div:nth-child(2)");
  if (qualityKpi) qualityKpi.textContent = data.has_scan && Number.isFinite(Number(qualityScore)) ? Number(qualityScore).toFixed(3) : "-";
  const sizeChip = $("dashboard-kpi-scan-size");
  if (sizeChip) {
    sizeChip.textContent = data.image_width > 0 && data.image_height > 0 ? `${data.image_width} x ${data.image_height} px` : "-";
  }

  const warningCount = data.errors.length + data.warnings.length;
  const warnKpi = document.querySelector("#dashboard-kpi-guardrail-warnings div:nth-child(2)");
  if (warnKpi) warnKpi.textContent = data.has_scan ? String(warningCount) : "-";
  const pathState = $("dashboard-kpi-path-state");
  if (pathState) pathState.textContent = data.input_path || "-";
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

function shouldKeepAstapSelection(rawInput, detectedBinary) {
  const selected = String(rawInput || "").trim().replace(/\\/g, "/").replace(/\/+$/, "");
  const binary = String(detectedBinary || "").trim().replace(/\\/g, "/");
  if (!selected || !binary) return false;
  if (selected === binary) return false;
  return binary.startsWith(`${selected}/`);
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
  const explicitPresetDir = String($("parameter-preset-dir")?.value || "").trim();
  if (explicitPresetDir) return explicitPresetDir;
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

async function fetchPresetsForDir(dir = "") {
  return api.get(API_ENDPOINTS.config.presets(dir));
}

async function refreshPresetSelect(selectId, preserveCurrentValue = true, dir = "") {
  const select = $(selectId);
  if (!select) return null;
  const oldValue = String(select.value || "").trim();
  const presets = await withPathGrantRetry(
    () => fetchPresetsForDir(dir),
    { fallbackPath: dir },
  );
  const items = Array.isArray(presets?.items) ? presets.items : [];
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
    }
  } else if (items[0]?.path) {
    select.value = String(items[0].path);
  }
  return presets;
}

async function bindPresetDirectoryControl({ inputId, browseId, reloadId, selectId }) {
  const input = $(inputId);
  if (!input) return;
  input.value = selectedPresetsDir();
  const reload = async ({ preserveCurrentValue = true } = {}) => {
    const dir = String(input.value || "").trim();
    persistPresetsDir(dir);
    syncPresetDirInputs();
    const result = await refreshPresetSelect(selectId, preserveCurrentValue, dir);
    if (result?.dir && result.fallback_used) {
      persistPresetsDir(result.dir);
      syncPresetDirInputs();
    }
    return result;
  };
  input.addEventListener("change", () => {
    const dir = String(input.value || "").trim();
    persistPresetsDir(dir);
    syncPresetDirInputs();
  });
  $(browseId)?.addEventListener("click", async () => {
    try {
      const chosen = await pickDirectoryPath(String(input.value || "").trim() || selectedPresetsDir());
      if (!chosen) return;
      input.value = chosen;
      await reload({ preserveCurrentValue: false });
      setFooter("Preset-Verzeichnis aktualisiert.");
    } catch (err) {
      setFooter(`Preset-Verzeichnis konnte nicht geladen werden: ${errorText(err)}`, true);
    }
  });
  $(reloadId)?.addEventListener("click", async () => {
    try {
      await reload({ preserveCurrentValue: true });
      setFooter("Preset-Liste aktualisiert.");
    } catch (err) {
      setFooter(`Preset-Liste konnte nicht geladen werden: ${errorText(err)}`, true);
    }
  });
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

async function chooseRunMonitorTemplateSavePath() {
  const runStem = String(uiState.currentRunId || "resume").trim().replace(/[^A-Za-z0-9._-]+/g, "_");
  const defaultDir = firstNonEmptyText(
    String($("monitor-resume-preset-dir")?.value || "").trim(),
    selectedPresetsDir(),
    parentDirOfPath(uiState.defaultConfigPath),
  );
  const defaultName = ensureYamlFileName(`${runStem || "resume"}_resume_template.yaml`);
  const defaultPath = joinPath(defaultDir, defaultName);
  if (typeof window.gui2PickPathValue === "function") {
    const pickedPath = await window.gui2PickPathValue(defaultPath, { mode: "save-file", defaultFileName: defaultName });
    const normalizedPickedPath = ensureYamlFileName(String(pickedPath || "").trim());
    return normalizedPickedPath || null;
  }
  const typedPath = window.prompt("Dateipfad fuer Template speichern", defaultPath);
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
    hydrateServerUiState(appState?.ui_state || {});
    setRunReady(guardrails?.status || "check", appState?.run?.current?.status || "");
    const rid = String(appState?.project?.current_run_id || "").trim();
    if (rid) setCurrentRunId(rid);
    else clearCurrentRunId();
    const runsDir = String(appState?.project?.runs_dir || "").trim();
    if (runsDir) uiState.projectRunsDir = runsDir;
    const presetsDir = String(appState?.project?.presets_dir || "").trim();
    if (presetsDir) uiState.projectPresetsDir = presetsDir;
    const defaultConfigPath = String(appState?.project?.default_config_path || "").trim();
    if (defaultConfigPath) uiState.defaultConfigPath = defaultConfigPath;
    syncPresetDirInputs();
    const scanPath = String(appState?.scan?.last_input_path || "").trim();
    if (scanPath) persistLastInputDirs(scanPath);
  } catch (err) {
    setFooter(`Backend nicht erreichbar: ${errorText(err)}`, true);
  }
}

async function applyLocale(localeRaw) {
  const locale = String(localeRaw || "de").toLowerCase() === "en" ? "en" : "de";
  uiState.locale = locale;
  writeServerUiStateValue(LOCALE_KEY, locale);
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

document.addEventListener("gui2:locale-changed", () => {
  setRunReady(uiState.runReadyStatus, uiState.runProcessStatus);
});

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
    renderScanSummary(summaryPrefix, { has_scan: true, input_path: payload.input_path });
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
    if (job.state === "ok") {
      setFooter("Scan abgeschlossen.");
    } else {
      const detail = scanErrorFromResult(result);
      setFooter(detail ? `Scan fehlgeschlagen: ${detail}` : `Scan beendet mit Status: ${job.state}`, true);
    }
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
  bindStoredField("scan-runs-dir", UI_STORAGE_KEYS.dashboardRunsDir, {
    absolute: true,
  });
  bindStoredField("scan-run-name", UI_STORAGE_KEYS.dashboardRunName, {
    normalize: sanitizeRunName,
  });
  const scanRunsDir = $("scan-runs-dir");
  if (scanRunsDir && !String(scanRunsDir.value || "").trim() && uiState.projectRunsDir) {
    scanRunsDir.value = uiState.projectRunsDir;
  }
  if (!$("btn-scan")) return;
  window.runScan = () => {
    void executeScanFlow();
  };
  const syncScanConfigField = async (el) => {
    const calibrationBinding = scanCalibrationBindingForElement(el);
    try {
      if (calibrationBinding) {
        const updates = [];
        if (String(el.id || "") === calibrationBinding.sourceId) {
          updates.push({
            path: calibrationBinding.useMasterPath,
            value: readFieldValue(el),
          });
        } else if (String(el.id || "") === calibrationBinding.inputId) {
          persistCalibrationPath(
            calibrationBinding,
            scanCalibrationUseMaster(calibrationBinding),
            readFieldValue(el),
          );
          updates.push({
            path: scanCalibrationActivePath(calibrationBinding),
            value: readFieldValue(el),
          });
        }
        if (updates.length === 0) return;
        const patched = await patchConfig({ updates, persist: false });
        if (patched?.config) {
          const hydratedConfig = await restoreStoredCalibrationPathsIntoConfig(patched.config);
          syncScanCalibrationUiFromConfig(hydratedConfig);
        } else {
          syncScanCalibrationInputPresentation(calibrationBinding, scanCalibrationUseMaster(calibrationBinding));
          const inputEl = $(calibrationBinding.inputId);
          if (inputEl && !String(inputEl.value || "").trim()) {
            inputEl.value = storedCalibrationPath(
              calibrationBinding,
              scanCalibrationUseMaster(calibrationBinding),
            );
          }
        }
        return;
      }
      const path = parameterPathFromElement(el);
      if (!path) return;
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
        const hydratedConfig = await restoreStoredCalibrationPathsIntoConfig(parsed.config);
        syncParameterFieldsFromConfig(hydratedConfig);
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

function setParameterBaseYaml(yamlText) {
  uiState.parameterBaseYaml = String(yamlText || "");
}

function splitYamlLines(text) {
  const normalized = String(text || "").replace(/\r/g, "");
  return normalized === "" ? [] : normalized.split("\n");
}

function computeYamlDiffOperations(beforeText, afterText) {
  const before = splitYamlLines(beforeText);
  const after = splitYamlLines(afterText);
  let prefix = 0;
  while (prefix < before.length && prefix < after.length && before[prefix] === after[prefix]) prefix += 1;

  let suffix = 0;
  while (
    suffix < before.length - prefix
    && suffix < after.length - prefix
    && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  ) {
    suffix += 1;
  }

  const beforeMid = before.slice(prefix, before.length - suffix);
  const afterMid = after.slice(prefix, after.length - suffix);
  const dp = Array.from({ length: beforeMid.length + 1 }, () => Array(afterMid.length + 1).fill(0));

  for (let i = beforeMid.length - 1; i >= 0; i -= 1) {
    for (let j = afterMid.length - 1; j >= 0; j -= 1) {
      dp[i][j] = beforeMid[i] === afterMid[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const ops = [];
  let oldLine = 1;
  let newLine = 1;

  for (let i = 0; i < prefix; i += 1) {
    ops.push({ type: "context", oldLine, newLine, text: before[i] });
    oldLine += 1;
    newLine += 1;
  }

  let i = 0;
  let j = 0;
  while (i < beforeMid.length && j < afterMid.length) {
    if (beforeMid[i] === afterMid[j]) {
      ops.push({ type: "context", oldLine, newLine, text: beforeMid[i] });
      i += 1;
      j += 1;
      oldLine += 1;
      newLine += 1;
      continue;
    }
    if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: "remove", oldLine, newLine: "", text: beforeMid[i] });
      i += 1;
      oldLine += 1;
    } else {
      ops.push({ type: "add", oldLine: "", newLine, text: afterMid[j] });
      j += 1;
      newLine += 1;
    }
  }
  while (i < beforeMid.length) {
    ops.push({ type: "remove", oldLine, newLine: "", text: beforeMid[i] });
    i += 1;
    oldLine += 1;
  }
  while (j < afterMid.length) {
    ops.push({ type: "add", oldLine: "", newLine, text: afterMid[j] });
    j += 1;
    newLine += 1;
  }
  for (let k = before.length - suffix; k < before.length; k += 1) {
    ops.push({ type: "context", oldLine, newLine, text: before[k] });
    oldLine += 1;
    newLine += 1;
  }
  return ops;
}

function renderYamlDiffHtml(beforeText, afterText) {
  const escapeHtml = (text) => String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
  const ops = computeYamlDiffOperations(beforeText, afterText);
  const added = ops.filter((item) => item.type === "add").length;
  const removed = ops.filter((item) => item.type === "remove").length;
  const summary = added === 0 && removed === 0
    ? t("page.parameter_studio.diff.no_changes", "Keine lokalen YAML-Aenderungen.")
    : t("page.parameter_studio.diff.summary", "Aenderungen: +{added} / -{removed}")
      .replace("{added}", String(added))
      .replace("{removed}", String(removed));

  const rows = ops.map((item) => {
    const tone = item.type === "add"
      ? { bg: "rgba(34,197,94,0.15)", fg: "#bbf7d0", sign: "+" }
      : item.type === "remove"
        ? { bg: "rgba(248,113,113,0.16)", fg: "#fecaca", sign: "-" }
        : { bg: "transparent", fg: "#e5edf6", sign: " " };
    return `<div style="display:grid;grid-template-columns:28px 44px 44px minmax(0,1fr);gap:10px;padding:2px 8px;background:${tone.bg};color:${tone.fg};border-radius:6px;">
      <span>${tone.sign}</span>
      <span style="color:#94a3b8;">${item.oldLine || ""}</span>
      <span style="color:#94a3b8;">${item.newLine || ""}</span>
      <span style="white-space:pre-wrap;word-break:break-word;">${escapeHtml(item.text)}</span>
    </div>`;
  }).join("");

  return `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;color:#cbd5e1;font-weight:700;">
      <span>${escapeHtml(summary)}</span>
      <span style="font-size:11px;color:#94a3b8;">old | new</span>
    </div>
    <div style="display:grid;gap:2px;">${rows || `<div style="color:#94a3b8;">${escapeHtml(summary)}</div>`}</div>
  `;
}

function parameterValidateStatusEl() {
  return $("parameter-validate-status");
}

function parameterPresetStatusEl() {
  return $("parameter-preset-status");
}

function parameterValidateDetailsEl() {
  return $("parameter-validate-details");
}

function dashboardValidateStatusEl() {
  return $("dashboard-validate-status");
}

function dashboardValidateDetailsEl() {
  return $("dashboard-validate-details");
}

function wizardValidationResultEl() {
  return $("wizard-validation-result");
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
  const previewYaml = String(value || "");
  const baseYaml = uiState.parameterBaseYaml || previewYaml;
  box.innerHTML = renderYamlDiffHtml(baseYaml, previewYaml);
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

function setValidationDetailsBox(el, result) {
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

function setValidationStatusText(el, result, fallbackText = "") {
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

function setParameterValidateDetails(result) {
  setValidationDetailsBox(parameterValidateDetailsEl(), result);
}

function setParameterPresetStatus(text = "") {
  const el = parameterPresetStatusEl();
  if (!el) return;
  const message = String(text || "").trim();
  el.textContent = message;
  el.style.display = message ? "inline-flex" : "none";
  el.style.color = message ? "#166534" : "";
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

function historyDeleteStartedMessage(runId) {
  return t("ui.message.history_delete_started", "Eintrag wird gelöscht: {run_id}")
    .replace("{run_id}", String(runId || "-"));
}

function historyDeleteDoneMessage(runId) {
  return t("ui.message.history_delete_done", "Eintrag gelöscht: {run_id}")
    .replace("{run_id}", String(runId || "-"));
}

function historyDeleteFailedMessage(err) {
  return t("ui.message.history_delete_failed", "Eintrag-Löschen fehlgeschlagen: {error}")
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
  if (isCurrentRunFromHistory()) return "";
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
  setValidationStatusText(parameterValidateStatusEl(), result, fallbackText);
}

function setDashboardValidateStatus(result, fallbackText = "") {
  setValidationStatusText(dashboardValidateStatusEl(), result, fallbackText);
}

function setDashboardValidateDetails(result) {
  setValidationDetailsBox(dashboardValidateDetailsEl(), result);
}

function updateWizardStartState(validationState) {
  const wizardStart = $("wizard-start");
  if (!wizardStart) return;
  const validationOk = Boolean(validationState?.ok);
  setDisabledLike(wizardStart, !validationOk);
  if (!validationState) {
    wizardStart.title = "Run mit aktuellem Wizard-Draft starten (zuerst erfolgreiche Validierung erforderlich).";
  } else if (!validationOk) {
    wizardStart.title = "Run mit aktuellem Wizard-Draft starten (Validierung hat Fehler).";
  } else {
    wizardStart.title = "Run mit aktuellem Wizard-Draft starten.";
  }
}

function setWizardValidationResult(result, fallbackText = "") {
  const box = wizardValidationResultEl();
  if (!box) return;

  const title = `<div class="ps-result-title">Validation</div>`;
  if (!result || typeof result !== "object") {
    const text = String(fallbackText || "Validierung ausstehend.");
    box.innerHTML = `${title}<div>${text}</div>`;
    return;
  }

  const errors = Array.isArray(result.errors) ? result.errors : [];
  const warnings = Array.isArray(result.warnings) ? result.warnings : [];
  const state = errors.length > 0 ? "ERROR" : result.ok ? "OK" : "ERROR";
  const firstIssue = errors[0] || warnings[0] || null;
  const issueText = firstIssue ? formatValidationIssue(firstIssue) : "";
  box.innerHTML =
    `${title}<div>Schema: <b>${state}</b> | Fehler: <b>${errors.length}</b> | Warnungen: <b>${warnings.length}</b>${issueText ? ` | Hinweis: <b>${issueText}</b>` : ""}</div>`;
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
  await flushServerUiState();
  const patched = await patchConfig({ updates: collectParameterDirtyUpdates(), persist: false });
  const result = await api.post(API_ENDPOINTS.config.save, {
    yaml: patched?.config_yaml || "",
    path: targetPath || undefined,
  });
  uiState.configYaml = String(patched?.config_yaml || "");
  setConfigDraft(uiState.configYaml);
  setParameterBaseYaml(uiState.configYaml);
  uiState.parameterDirty = {};
  clearParameterDirtyState();
  setParameterPreview(uiState.configYaml);
  return result;
}

function flattenConfigSchemaPaths(node, prefix = [], out = new Set()) {
  if (!node || typeof node !== "object" || !node.properties || typeof node.properties !== "object") return out;
  for (const [key, value] of Object.entries(node.properties)) {
    const path = [...prefix, key];
    if (value && typeof value === "object" && value.type === "object" && value.properties) {
      flattenConfigSchemaPaths(value, path, out);
      continue;
    }
    out.add(path.join("."));
  }
  return out;
}

async function ensureConfigSchemaPaths() {
  if (uiState.configSchemaPaths instanceof Set) return uiState.configSchemaPaths;
  try {
    const schema = await api.get(API_ENDPOINTS.config.schema);
    uiState.configSchemaPaths = flattenConfigSchemaPaths(schema);
  } catch {
    uiState.configSchemaPaths = null;
  }
  return uiState.configSchemaPaths;
}

function isKnownConfigSchemaPath(path) {
  const normalized = String(path || "").trim();
  if (!normalized) return false;
  if (!(uiState.configSchemaPaths instanceof Set)) return true;
  return uiState.configSchemaPaths.has(normalized);
}

function sanitizeParameterDirtyState(dirty) {
  const source = dirty && typeof dirty === "object" ? dirty : {};
  if (!(uiState.configSchemaPaths instanceof Set)) return { ...source };
  const sanitized = {};
  for (const [path, value] of Object.entries(source)) {
    if (isKnownConfigSchemaPath(path)) sanitized[path] = value;
  }
  return sanitized;
}

function parameterPathFromElement(el) {
  if (!el) return "";
  const dynRow = el.closest(".ps-dyn-row[data-path]");
  if (dynRow) {
    const path = String(dynRow.getAttribute("data-path") || "");
    return isKnownConfigSchemaPath(path) ? path : "";
  }
  const control = String(el.getAttribute("data-control") || "");
  if (control && PARAM_CONTROL_PATHS[control]) {
    const path = PARAM_CONTROL_PATHS[control];
    return isKnownConfigSchemaPath(path) ? path : "";
  }
  const id = String(el.id || "");
  if (id && PARAM_ID_PATHS[id]) {
    const path = PARAM_ID_PATHS[id];
    return isKnownConfigSchemaPath(path) ? path : "";
  }
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
    setParameterDirtyState(uiState.parameterDirty);
  };
  root.addEventListener("input", onAny);
  root.addEventListener("change", onAny);
}

function collectParameterDirtyUpdates() {
  const out = [];
  for (const [path, value] of Object.entries(uiState.parameterDirty)) {
    if (!isKnownConfigSchemaPath(path)) continue;
    out.push({ path, value });
  }
  return out;
}

function syncParameterFieldsFromConfig(config) {
  if (!config || typeof config !== "object") return;

  for (const [control, path] of Object.entries(PARAM_CONTROL_PATHS)) {
    if (!isKnownConfigSchemaPath(path)) continue;
    const el = document.querySelector(`[data-control='${control}']`);
    if (!el) continue;
    const value = getByPath(config, path);
    writeFieldValue(el, value);
  }
  for (const [id, path] of Object.entries(PARAM_ID_PATHS)) {
    if (!isKnownConfigSchemaPath(path)) continue;
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
  syncScanCalibrationUiFromConfig(config);
}

function activeScenarioKeys(scopeSelector = "#parameter-studio-root") {
  return Array.from(document.querySelectorAll(`${scopeSelector} [data-scenario].ps-chip-btn.active`))
    .map((el) => String(el.getAttribute("data-scenario") || "").trim())
    .filter(Boolean);
}

async function bindParameterStudio() {
  const presetSelect = $("parameter-preset-select");
  if (!presetSelect) return;

  await ensureConfigSchemaPaths();
  bindParameterDirtyTracking();
  await bindPresetDirectoryControl({
    inputId: "parameter-preset-dir",
    browseId: "parameter-preset-dir-browse",
    reloadId: "parameter-preset-dir-reload",
    selectId: "parameter-preset-select",
  });

  const applyPreview = async ({ persist = false } = {}) => {
    const updates = collectParameterDirtyUpdates();
    const patched = await patchConfig({ updates, persist });
    setParameterPreview(patched?.config_yaml || "");
    if (patched?.config) {
      syncParameterFieldsFromConfig(patched.config);
    }
    if (persist) {
      uiState.parameterDirty = {};
      clearParameterDirtyState();
    }
    return patched;
  };

  try {
    await populatePresetSelect("parameter-preset-select", true);
    restoreUnifiedPresetSelectValue("parameter-preset-select");
    bindUnifiedPresetSelect("parameter-preset-select");
    uiState.parameterDirty = sanitizeParameterDirtyState(getParameterDirtyState());
    setParameterDirtyState(uiState.parameterDirty);
    const currentYaml = await ensureConfigYaml();
    const parsed = await patchConfig({ yamlText: currentYaml, updates: collectParameterDirtyUpdates() });
    if (parsed?.config) {
      syncParameterFieldsFromConfig(parsed.config);
    }
    setParameterBaseYaml(currentYaml);
    setParameterPreview(parsed?.config_yaml || currentYaml);
    setParameterValidateStatus(null, "Validierung: nicht geprüft");
    setParameterPresetStatus("");
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
      clearParameterDirtyState();
      const parsed = await patchConfig({ yamlText: uiState.configYaml, updates: [] });
      if (parsed?.config) syncParameterFieldsFromConfig(parsed.config);
      setParameterBaseYaml(uiState.configYaml);
      setParameterPreview(uiState.configYaml);
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterPresetStatus("");
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
      syncUnifiedPresetSelection(path);
      const applied = await api.post(API_ENDPOINTS.config.applyPreset, { path });
      uiState.configYaml = String(applied?.config || "");
      setConfigDraft(uiState.configYaml);
      uiState.parameterDirty = {};
      clearParameterDirtyState();
      const parsed = await patchConfig({ yamlText: uiState.configYaml, updates: [] });
      if (parsed?.config) syncParameterFieldsFromConfig(parsed.config);
      setParameterBaseYaml(String(parsed?.config_yaml || uiState.configYaml));
      setParameterPreview(String(parsed?.config_yaml || uiState.configYaml));
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterPresetStatus(t("ui.status.parameter_preset_applied", "Preset wurde angewendet."));
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
      setParameterPresetStatus("");
      setParameterValidateStatus(result);
      setParameterValidateDetails(result);
      setConfigValidationState({
        yaml: patched?.config_yaml || "",
        ok: Boolean(result?.ok),
        errors: Array.isArray(result?.errors) ? result.errors : [],
        warnings: Array.isArray(result?.warnings) ? result.warnings : [],
      });
      setFooter(result.ok ? "Validierung OK." : "Validierung hat Fehler.");
    } catch (err) {
      setParameterPresetStatus("");
      setParameterValidateStatus(null, "Validierung: fehlgeschlagen");
      setParameterValidateDetails(null);
      clearConfigValidationState();
      setFooter(`Validierung fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-save")?.addEventListener("click", async () => {
    try {
      const result = await saveParameterConfig("");
      setParameterPresetStatus("");
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
      setParameterPresetStatus("");
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
      setParameterPresetStatus("");
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
      clearParameterDirtyState();
      uiState.configYaml = String(current?.config || "");
      setConfigDraft(uiState.configYaml);
      const parsed = await patchConfig({ yamlText: uiState.configYaml, updates: [] });
      if (parsed?.config) syncParameterFieldsFromConfig(parsed.config);
      setParameterBaseYaml(uiState.configYaml);
      setParameterPreview(uiState.configYaml);
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterPresetStatus("");
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
      const droppedPaths = new Set();
      for (const key of activeScenarioKeys(".app-content")) {
        for (const [path, value] of SCENARIO_DELTAS[key] || []) {
          if (!isKnownConfigSchemaPath(path)) {
            droppedPaths.add(path);
            continue;
          }
          scenarioUpdates.push({ path, value });
        }
      }
      if (scenarioUpdates.length === 0) {
        setFooter(
          droppedPaths.size > 0
            ? "Keine anwendbaren Szenario-Deltas im aktuellen Schema gefunden."
            : "Keine Situation ausgewaehlt.",
          true,
        );
        return;
      }
      const patched = await patchConfig({ updates: scenarioUpdates, persist: false });
      uiState.parameterDirty = {};
      clearParameterDirtyState();
      if (patched?.config) syncParameterFieldsFromConfig(patched.config);
      setParameterPreview(patched?.config_yaml || "");
      setParameterValidateStatus(null, "Validierung: nicht geprüft");
      setParameterPresetStatus("");
      setParameterValidateDetails(null);
      setSituationApplyStatus(true, `${t("ui.status.situation_applied", "Angewendet")} (${scenarioUpdates.length})`);
      clearConfigValidationState();
      setFooter(
        droppedPaths.size > 0
          ? `Situation angewendet (${scenarioUpdates.length} Deltas, ${droppedPaths.size} veraltete Pfade ignoriert).`
          : `Situation angewendet (${scenarioUpdates.length} Deltas).`,
      );
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
  await refreshPresetSelect(selectId, preserveCurrentValue, selectedPresetsDir());
}

function runMonitorSelectedPhase() {
  const selected = document.querySelector(".ps-phase-row.is-selected .phase-name");
  return selected ? String(selected.textContent || "").trim().toUpperCase() : "";
}

function setMonitorResumeInfo(message = "") {
  const el = $("monitor-resume-info");
  if (!el) return;
  const text = String(message || "").trim();
  el.textContent = text;
  el.style.display = text ? "" : "none";
}

function applyRunMonitorResumePhaseAvailability(resumePhases) {
  const allowed = new Set(
    Array.isArray(resumePhases) && resumePhases.length > 0
      ? resumePhases.map((phase) => String(phase || "").trim().toUpperCase()).filter(Boolean)
      : ["ASTROMETRY", "BGE", "PCC"],
  );
  document.querySelectorAll(".ps-phase-row").forEach((row) => {
    const phaseName = String(row.querySelector(".phase-name")?.textContent || "").trim().toUpperCase();
    const resumable = allowed.has(phaseName);
    row.dataset.resumeAllowed = resumable ? "1" : "0";
    if (!resumable) {
      row.classList.remove("is-selected");
      row.style.opacity = "0.6";
      row.style.cursor = "not-allowed";
      row.title = `Resume aktuell nur ab ${Array.from(allowed).join(", ")} unterstützt.`;
      return;
    }
    row.style.opacity = "";
    row.style.cursor = "pointer";
  });
}

function runMonitorSelectedFilter() {
  const chipRow = $("monitor-filter-row");
  if (chipRow && chipRow.style.display === "none") return "";
  const selected = document.querySelector(".ps-chip-btn.active[id^='monitor-filter-']");
  if (!selected) return "";
  return String(selected.textContent || "").trim().toUpperCase();
}

function runMonitorFilterButtons() {
  return Array.from(document.querySelectorAll(".ps-chip-btn[id^='monitor-filter-']"));
}

function normalizeMonitorFilterName(raw) {
  const token = String(raw || "")
    .trim()
    .toUpperCase()
    .replace(/[\s_-]+/g, "");
  if (!token) return "";
  if (token === "HALPHA") return "HA";
  return token;
}

function collectActiveRunMonitorFilters(queueItemsRaw = null) {
  const source = Array.isArray(queueItemsRaw) ? queueItemsRaw : collectQueueRows();
  const out = [];
  const seen = new Set();
  for (const item of source) {
    const rawFilter = typeof item === "string" ? item : item?.filter || item?.filter_name || "";
    const filter = normalizeMonitorFilterName(rawFilter);
    if (!filter || seen.has(filter)) continue;
    seen.add(filter);
    out.push(filter);
  }
  return out;
}

function runMonitorEffectiveColorMode(colorModeRaw = "") {
  return normalizeDetectedColorMode(
    firstNonEmptyText(colorModeRaw, uiState.currentRunColorMode, getPersistedDetectedColorMode(), ""),
  );
}

function setRunMonitorFilterVisibility(colorModeRaw, queueItemsRaw = null) {
  const chipRow = $("monitor-filter-row");
  if (!chipRow) return;
  const chipButtons = runMonitorFilterButtons();
  const colorMode = runMonitorEffectiveColorMode(colorModeRaw);
  const activeFilters = collectActiveRunMonitorFilters(queueItemsRaw).filter((filter) => chipButtons.some((btn) => normalizeMonitorFilterName(btn.textContent) === filter));
  const hideFilters = colorMode !== "MONO" || activeFilters.length === 0;
  chipRow.style.display = hideFilters ? "none" : "";
  if (hideFilters) {
    chipButtons.forEach((btn) => {
      btn.style.display = "";
      btn.classList.remove("active");
    });
    return;
  }
  const activeSet = new Set(activeFilters);
  chipButtons.forEach((btn) => {
    const filter = normalizeMonitorFilterName(btn.textContent);
    const visible = activeSet.has(filter);
    btn.style.display = visible ? "" : "none";
    if (!visible) btn.classList.remove("active");
  });
  if (!document.querySelector(".ps-chip-btn.active[id^='monitor-filter-']")) {
    chipButtons.find((btn) => btn.style.display !== "none")?.classList.add("active");
  }
}

function bindRunMonitorFilterSync() {
  const refresh = () => {
    setRunMonitorFilterVisibility(
      firstNonEmptyText($("dashboard-color-mode")?.value, $("inp-colormode")?.value, uiState.currentRunColorMode, ""),
    );
  };
  document.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (target.id === "dashboard-color-mode" || target.id === "inp-colormode" || target.closest(".ps-queue-row")) {
      refresh();
    }
  });
  document.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (target.closest(".ps-queue-row")) refresh();
  });
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

  row.classList.remove("done", "running", "pending", "error", "skipped");
  const normalized = String(status || "pending").toLowerCase();
  if (normalized === "skipped") {
    row.classList.add("skipped");
    const stateEl = row.querySelector(".state");
    if (stateEl) stateEl.textContent = "SKIP";
  } else if (normalized === "ok" || normalized === "completed" || normalized === "done") {
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
  const fallbackScanColorMode = String(readServerUiStateValue(LAST_SCAN_COLOR_MODE_KEY) || "").trim().toUpperCase();
  const effectiveColorMode = String(status?.color_mode || "").trim().toUpperCase() || fallbackScanColorMode;
  uiState.currentRunColorMode = effectiveColorMode;
  setRunMonitorFilterVisibility(effectiveColorMode, Array.isArray(status?.queue_filters) ? status.queue_filters : null);
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
  if (!uiState.currentRunId) {
    sel.innerHTML = "";
    return;
  }
  const old = sel.value;
  const revisions = await api.get(API_ENDPOINTS.runs.configRevisions(uiState.currentRunId));
  sel.innerHTML = "";
  for (const item of revisions.items || []) {
    const opt = document.createElement("option");
    opt.value = item.revision_id;
    const source = String(item?.source || "revision").trim();
    const created = String(item?.created_at || "").trim();
    opt.textContent = created ? `${item.revision_id} | ${source} | ${created}` : `${item.revision_id} | ${source}`;
    sel.appendChild(opt);
  }
  if (old) sel.value = old;
}

function setRunMonitorConfigStatus(text = "") {
  const el = $("monitor-resume-config-status");
  if (!el) return;
  const value = String(text || "").trim();
  el.textContent = value;
  el.style.display = value ? "" : "none";
}

function runMonitorConfigEditorValue() {
  return String($("monitor-resume-config-editor")?.value || "");
}

function setRunMonitorConfigEditor(yamlText = "", { source = "", revisionId = "" } = {}) {
  const editor = $("monitor-resume-config-editor");
  if (!editor) return;
  const value = String(yamlText || "");
  editor.value = value;
  editor.dataset.source = String(source || "");
  editor.dataset.revisionId = String(revisionId || "");
  const parts = [];
  if (source) parts.push(`Quelle: ${source}`);
  if (revisionId) parts.push(`Revision: ${revisionId}`);
  parts.push(`Zeilen: ${value ? value.split(/\r?\n/).length : 0}`);
  setRunMonitorConfigStatus(parts.join(" | "));
}

async function loadRunMonitorCurrentConfig() {
  if (!uiState.currentRunId) return;
  const current = await api.get(API_ENDPOINTS.runs.config(uiState.currentRunId));
  const sourcePath = String(current?.path || "").trim();
  setRunMonitorConfigEditor(String(current?.config || ""), {
    source: sourcePath || "run/config.yaml",
    revisionId: "",
  });
}

async function loadRunMonitorSelectedRevision() {
  const revisionId = String($("monitor-resume-config-revision")?.value || "").trim();
  if (!uiState.currentRunId || !revisionId) return;
  const revision = await api.get(API_ENDPOINTS.runs.configRevision(uiState.currentRunId, revisionId));
  setRunMonitorConfigEditor(String(revision?.config || ""), {
    source: String(revision?.source || "run_revision").trim(),
    revisionId,
  });
}

function connectRunMonitorStream(runId) {
  if (!runId) return;
  if (uiState.runSocket) uiState.runSocket.close();
  const logBox = runMonitorLogBox();
  if (logBox) scrollLogToEnd(logBox);
  uiState.runSocket = api.ws(
    API_ENDPOINTS.ws.run(runId),
    (event) => {
      const eventType = String(event?.type || "").trim().toLowerCase();
      const line = formatStructuredLogLine(event, { suppressRunStatus: true });
      if (line) enqueueRunMonitorLogLine(line);
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
      if (eventType === "queue_progress") {
        setRunMonitorFilterVisibility(uiState.currentRunColorMode, Array.isArray(event?.payload?.queue) ? event.payload.queue : null);
      }
      const terminalRunStatus = String(event?.payload?.status || event?.status || "").trim().toLowerCase();
      const isTerminalRunEvent =
        eventType === "run_end"
        || eventType === "resume_end"
        || (
          eventType === "run_status"
          && ["completed", "failed", "cancelled", "aborted", "error", "done", "finished"].includes(terminalRunStatus)
        );
      if (isTerminalRunEvent) {
        window.setTimeout(() => {
          document.dispatchEvent(
            new CustomEvent("gui2:run-monitor-terminal", {
              detail: {
                eventType,
                status: terminalRunStatus,
                runId,
              },
            }),
          );
        }, 250);
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
  const resumeEditor = $("monitor-resume-config-editor");
  const resumePresetSelect = $("monitor-resume-preset-select");
  const resumeLoadCurrentBtn = $("monitor-resume-load-current");
  const resumeApplyTemplateBtn = $("monitor-resume-apply-template");
  const resumeSaveTemplateBtn = $("monitor-resume-save-template");
  const sub = document.querySelector(".app-content .ps-sub");
  const updateResumeEnabled = () => {
    const phase = runMonitorSelectedPhase();
    const selectedRow = document.querySelector(".ps-phase-row.is-selected");
    const resumable = String(selectedRow?.dataset?.resumeAllowed || "") === "1";
    const showHistoryResumeHint = isCurrentRunFromHistory();
    const hasYaml = String(runMonitorConfigEditorValue() || "").trim().length > 0;
    const isActive = isRunActiveStatus(uiState.runProcessStatus || "");
    setDisabledLike($("monitor-resume"), !uiState.currentRunId || !phase || !resumable || !hasYaml || isActive);
    setDisabledLike($("monitor-resume-restore-revision"), !$("monitor-resume-config-revision")?.value || isActive);
    setDisabledLike(resumeLoadCurrentBtn, !uiState.currentRunId || isActive);
    setDisabledLike(resumeApplyTemplateBtn, !String(resumePresetSelect?.value || "").trim() || isActive);
    setDisabledLike(resumeSaveTemplateBtn, !hasYaml || isActive);
    setDisabledLike(resumePresetSelect, isActive);
    setDisabledLike($("monitor-resume-preset-dir"), isActive);
    setDisabledLike($("monitor-resume-preset-dir-browse"), isActive);
    setDisabledLike($("monitor-resume-preset-dir-reload"), isActive);
    setDisabledLike($("monitor-resume-config-revision"), isActive);
    setDisabledLike(resumeEditor, isActive);
    setMonitorResumeInfo(
      showHistoryResumeHint
        ? t(
            "ui.message.resume_info_history_bge_requires_artifacts",
            "Hinweis fuer History-Resume: Der Run verwendet seine vorhandenen Artefakte. Resume ab BGE berechnet BGE nur neu, wenn passende Local-Metrics- und BGE-Grid-Artefakte im Run vorhanden sind. Fehlen sie, wird BGE uebersprungen.",
          )
        : "",
    );
  };
  document.querySelectorAll(".ps-phase-row").forEach((row) => {
    row.addEventListener("click", () => {
      if (String(row.dataset.resumeAllowed || "") !== "1") {
        setFooter("Resume ist aktuell nur ab ASTROMETRY, BGE oder PCC unterstützt.", true);
        return;
      }
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
  resumePresetSelect?.addEventListener("change", updateResumeEnabled);
  resumeEditor?.addEventListener("input", updateResumeEnabled);

  if (resumePresetSelect) {
    await bindPresetDirectoryControl({
      inputId: "monitor-resume-preset-dir",
      browseId: "monitor-resume-preset-dir-browse",
      reloadId: "monitor-resume-preset-dir-reload",
      selectId: "monitor-resume-preset-select",
    });
    await populatePresetSelect("monitor-resume-preset-select", true);
    restoreUnifiedPresetSelectValue("monitor-resume-preset-select");
    bindUnifiedPresetSelect("monitor-resume-preset-select");
  }

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
    const status = await api.get(API_ENDPOINTS.runs.statsStatus(uiState.currentRunId, uiState.currentRunDir)).catch(() => null);
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
    uiState.runProcessStatus = isActive ? "running" : "idle";
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
      row.classList.remove("done", "running", "error", "skipped", "is-selected");
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
    uiState.runProcessStatus = "";
    resetPhaseRows();
    renderArtifacts([]);
    setMonitorReportAvailable(false);
    setRunMonitorConfigEditor("", {});
    setRunMonitorConfigStatus("");
    const revisionSelect = $("monitor-resume-config-revision");
    if (revisionSelect) revisionSelect.innerHTML = "";
    const logBox = runMonitorLogBox();
    if (logBox) {
      logBox.textContent = "";
      scrollLogToEnd(logBox);
    }
    if (sub) sub.textContent = text;
  };
  const refreshCurrentRunMonitorState = async ({ reconnectSocket = false } = {}) => {
    if (!uiState.currentRunId) return null;
    const status = await loadRunStatus(uiState.currentRunId);
    uiState.runProcessStatus = String(status?.status || "").trim().toLowerCase();
    await loadRunRevisions();
    if (!String(runMonitorConfigEditorValue() || "").trim()) {
      await loadRunMonitorCurrentConfig().catch(() => {});
    }
    await refreshArtifacts();
    await refreshStatsActions();
    const isActive = isRunActiveStatus(status?.status || "");
    setMonitorActionState(isActive);
    if (reconnectSocket) {
      if (isActive) {
        connectRunMonitorStream(uiState.currentRunId);
      } else if (uiState.runSocket) {
        uiState.runSocket.close();
        uiState.runSocket = null;
      }
    }
    updateResumeEnabled();
    return status;
  };
  document.addEventListener("gui2:run-monitor-terminal", (event) => {
    const detail = event?.detail || {};
    if (String(detail.runId || "").trim() && String(detail.runId || "").trim() !== String(uiState.currentRunId || "").trim()) {
      return;
    }
    void refreshCurrentRunMonitorState({ reconnectSocket: true });
  });

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
      clearCurrentRunHistoryMark();
      setMonitorStartValidationMessage("");
      setFooter(`Run gestartet (Job ${accepted?.job_id || "-"}).`);
      await refreshCurrentRunMonitorState({ reconnectSocket: true });
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

  resumeLoadCurrentBtn?.addEventListener("click", async () => {
    try {
      await loadRunMonitorCurrentConfig();
      setFooter("Run-Config in den Resume-Editor geladen.");
      updateResumeEnabled();
    } catch (err) {
      setFooter(`Run-Config laden fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  resumeApplyTemplateBtn?.addEventListener("click", async () => {
    try {
      const path = String(resumePresetSelect?.value || "").trim();
      if (!path) {
        setFooter("Kein Template ausgewaehlt.", true);
        return;
      }
      syncUnifiedPresetSelection(path);
      const applied = await api.post(API_ENDPOINTS.config.applyPreset, { path });
      setRunMonitorConfigEditor(String(applied?.config || ""), {
        source: path,
        revisionId: "",
      });
      setFooter("Template in den Resume-Editor geladen.");
      updateResumeEnabled();
    } catch (err) {
      setFooter(`Template laden fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  resumeSaveTemplateBtn?.addEventListener("click", async () => {
    try {
      const yaml = runMonitorConfigEditorValue();
      if (!String(yaml || "").trim()) {
        setFooter("Keine Resume-Config zum Speichern vorhanden.", true);
        return;
      }
      const targetPath = await chooseRunMonitorTemplateSavePath();
      if (!targetPath) return;
      const saved = await api.post(API_ENDPOINTS.config.save, {
        yaml,
        path: targetPath,
      });
      await populatePresetSelect("monitor-resume-preset-select", false);
      restoreUnifiedPresetSelectValue("monitor-resume-preset-select");
      setFooter(`Template gespeichert unter ${saved?.path || targetPath}.`);
    } catch (err) {
      setFooter(`Template speichern fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-resume")?.addEventListener("click", async () => {
    const phase = runMonitorSelectedPhase();
    if (!phase) {
      setFooter("Bitte Zielphase waehlen.", true);
      return;
    }
    try {
      const yaml = runMonitorConfigEditorValue();
      if (!String(yaml || "").trim()) {
        setFooter("Bitte zuerst eine Resume-Config laden oder eingeben.", true);
        return;
      }
      const accepted = await api.post(API_ENDPOINTS.runs.resume(uiState.currentRunId), {
        from_phase: phase,
        config_yaml: yaml,
        run_dir: uiState.currentRunDir || undefined,
        filter_context: runMonitorSelectedFilter() || undefined,
      });
      setConfigDraft(yaml);
      setFooter(`Resume gestartet (Job ${accepted.job_id}).`);
      await refreshCurrentRunMonitorState({ reconnectSocket: true });
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
      await loadRunMonitorSelectedRevision();
      setFooter(`Revision ${revisionId} in den Resume-Editor geladen.`);
      updateResumeEnabled();
    } catch (err) {
      setFooter(`Revision laden fehlgeschlagen: ${errorText(err)}`, true);
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
      const status = await api.get(API_ENDPOINTS.runs.statsStatus(uiState.currentRunId, uiState.currentRunDir));
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
    const appConstants = await api.get(API_ENDPOINTS.app.constants).catch(() => null);
    applyRunMonitorResumePhaseAvailability(appConstants?.resume_from);
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
    uiState.runProcessStatus = String(status?.status || "").trim().toLowerCase();
    await loadRunRevisions();
    await loadRunMonitorCurrentConfig().catch(() => {});
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
    let runStatus;
    try {
      runStatus = await api.get(API_ENDPOINTS.runs.status(runId));
    } catch (err) {
      if (Number(err?.status) === 404) {
        uiState.missingHistoryRunIds.add(String(runId));
        return null;
      }
      throw err;
    }
    uiState.missingHistoryRunIds.delete(String(runId));
    const runDir = String(runStatus?.run_dir || "-");
    const [statsStatus, artifactResult] = await Promise.all([
      api.get(API_ENDPOINTS.runs.statsStatus(runId, runDir)).catch(() => ({ report_path: "", output_dir: "", state: "unknown" })),
      api.get(API_ENDPOINTS.runs.artifacts(runId)).catch(() => ({ items: [] })),
    ]);
    const artifacts = Array.isArray(artifactResult?.items) ? artifactResult.items : [];
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
      persistHistorySelectionState();
      return null;
    }
    const snapshot = await loadRunSnapshot(uiState.selectedHistoryRunId);
    if (!snapshot) {
      uiState.selectedHistoryRunId = "";
      selectedSnapshotCache = null;
      clearHistoryDetails(selectedRefs);
      updateHistoryActionState(null);
      persistHistorySelectionState();
      return null;
    }
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
    persistHistorySelectionState();
  };
  const renderCompareDetails = async (selectedSnapshot) => {
    if (!uiState.compareHistoryRunId || uiState.compareHistoryRunId === uiState.selectedHistoryRunId) {
      clearHistoryDetails(compareRefs, "Vergleichs-Run wählen");
      if (compareRunSelect) compareRunSelect.value = "";
      persistHistorySelectionState();
      return;
    }
    const snapshot = await loadRunSnapshot(uiState.compareHistoryRunId);
    if (!snapshot) {
      uiState.compareHistoryRunId = "";
      clearHistoryDetails(compareRefs, "Vergleichs-Run wählen");
      if (compareRunSelect) compareRunSelect.value = "";
      persistHistorySelectionState();
      return;
    }
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
    const items = (Array.isArray(runs?.items) ? runs.items : [])
      .filter((item) => !uiState.missingHistoryRunIds.has(String(item?.run_id || "")));
    if (items.length === 0) {
      list.innerHTML = "<li><button>Keine Runs gefunden</button></li>";
      clearHistoryDetails(selectedRefs);
      clearHistoryDetails(compareRefs, "Vergleichs-Run wählen");
      if (compareRunSelect) compareRunSelect.innerHTML = '<option value="">-</option>';
      selectedSnapshotCache = null;
      updateHistoryActionState(null);
      uiState.selectedHistoryRunId = "";
      uiState.compareHistoryRunId = "";
      persistHistorySelectionState();
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
    persistHistorySelectionState();
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
        persistHistorySelectionState();
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
      markCurrentRunFromHistory(uiState.selectedHistoryRunId);
      setFooter(`Current Run gesetzt: ${uiState.selectedHistoryRunId}`);
      window.location.href = "run-monitor.html";
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
      setFooter(historyDeleteStartedMessage(runId));
      await api.post(API_ENDPOINTS.runs.delete(runId), {});
      if (uiState.currentRunId === runId) clearCurrentRunId();
      if (uiState.compareHistoryRunId === runId) uiState.compareHistoryRunId = "";
      if (uiState.selectedHistoryRunId === runId) uiState.selectedHistoryRunId = "";
      persistHistorySelectionState();
      setFooter(historyDeleteDoneMessage(runId));
      await render();
    } catch (err) {
      setFooter(historyDeleteFailedMessage(err), true);
    }
  });

  compareRunSelect?.addEventListener("change", () => {
    uiState.compareHistoryRunId = String(compareRunSelect.value || "").trim();
    persistHistorySelectionState();
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
    persistHistorySelectionState();
    render().catch((err) => {
      setFooter(`History laden fehlgeschlagen: ${errorText(err)}`, true);
    });
  });

  $("history-compare-clear")?.addEventListener("click", () => {
    uiState.compareHistoryRunId = "";
    persistHistorySelectionState();
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
  if (logBox) logBox.textContent = "";

  const raField = $("tools-astrometry-ra");
  const decField = $("tools-astrometry-dec");
  const pixelScaleField = $("tools-astrometry-pixel-scale");
  const rotationField = $("tools-astrometry-rotation");
  const fovField = $("tools-astrometry-fov");
  const binaryInput = $("tools-astrometry-bin");
  const dataDirInput = $("tools-astrometry-data-dir");
  let autoResolving = false;

  bindStoredField("tools-astrometry-bin", UI_STORAGE_KEYS.astrometryBinary, { absolute: true });
  bindStoredField("tools-astrometry-data-dir", UI_STORAGE_KEYS.astrometryDataDir, { absolute: true });
  bindStoredField("tools-astrometry-file", UI_STORAGE_KEYS.astrometryFile, { absolute: true });
  bindStoredField("tools-astrometry-catalog", UI_STORAGE_KEYS.astrometryCatalog, { overwrite: true });

  const append = (msg) => appendStructuredLog(logBox, msg, { suppressRunStatus: true });
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
      persistTextValue(UI_STORAGE_KEYS.astrometryLastWcs, uiState.lastAstrometryWcs, { absolute: true });
    }
    persistJsonValue(UI_STORAGE_KEYS.astrometryLastResult, payload);
  };

  const storedAstrometryWcs = storedTextValue(UI_STORAGE_KEYS.astrometryLastWcs, { absolute: true });
  if (storedAstrometryWcs) {
    uiState.lastAstrometryWcs = storedAstrometryWcs;
  }
  const storedAstrometryResult = storedJsonValue(UI_STORAGE_KEYS.astrometryLastResult, null);
  if (storedAstrometryResult && typeof storedAstrometryResult === "object") {
    applyAstrometryResult(storedAstrometryResult);
  }

  async function detect({ logResult = true } = {}) {
    const selectedBinary = String(binaryInput?.value || "").trim();
    const payload = {
      astap_cli: selectedBinary,
      astap_data_dir: dataDirInput?.value || "",
    };
    const result = await withPathGrantRetry(
      () => api.post(API_ENDPOINTS.astrometry.detect, payload),
      { fallbackPath: payload.astap_cli || payload.astap_data_dir },
    );
    if (statusChip) statusChip.textContent = result.installed ? "Installed" : "Missing";
    if (binaryInput && result.binary && !shouldKeepAstapSelection(selectedBinary, result.binary)) {
      binaryInput.value = String(result.binary);
      persistTextValue(UI_STORAGE_KEYS.astrometryBinary, binaryInput.value, { absolute: true });
    }
    if (dataDirInput && result.data_dir) {
      dataDirInput.value = String(result.data_dir);
      persistTextValue(UI_STORAGE_KEYS.astrometryDataDir, dataDirInput.value, { absolute: true });
    }
    if (logResult) append(result);
    return result;
  }

  async function autoResolveSelection(origin) {
    if (autoResolving) return;
    autoResolving = true;
    try {
      const result = await detect({ logResult: true });
      if (result.installed) {
        const location = origin === "data-dir"
          ? String(result.binary || result.data_dir || "")
          : String(result.binary || "");
        setFooter(location ? `ASTAP erkannt: ${location}` : "ASTAP erkannt.");
      } else {
        setFooter("ASTAP im ausgewaehlten Pfad nicht gefunden.", true);
      }
    } catch (err) {
      setFooter(`ASTAP-Pfadauflosung fehlgeschlagen: ${errorText(err)}`, true);
    } finally {
      autoResolving = false;
    }
  }

  document.querySelector("[data-control='tools.astrometry.detect']")?.addEventListener("click", async () => {
    try {
      const result = await detect();
      setFooter(result.installed ? `ASTAP gefunden: ${result.binary || "-"}` : "ASTAP nicht gefunden.", !result.installed);
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
      await detect({ logResult: false });
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
      if (uiState.lastAstrometryWcs) {
        persistTextValue(UI_STORAGE_KEYS.astrometryLastWcs, uiState.lastAstrometryWcs, { absolute: true });
      }
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

  [binaryInput, dataDirInput].forEach((input) => {
    input?.addEventListener("input", (event) => {
      if (event.isTrusted || autoResolving) return;
      void autoResolveSelection(input === dataDirInput ? "data-dir" : "binary");
    });
  });

  try {
    await detect({ logResult: false });
  } catch {
    if (statusChip) statusChip.textContent = "Missing";
  }
}

async function bindPccPage() {
  if (!$("tools-pcc-rgb")) return;
  const logBox = findLogBoxBySectionTitle("Result + Log");
  const statusField = document.querySelector("[data-control='tools.pcc.siril_status']");
  if (logBox) logBox.textContent = "";

  [
    ["tools-pcc-rgb", UI_STORAGE_KEYS.pccRgb, true],
    ["tools-pcc-wcs", UI_STORAGE_KEYS.pccWcs, true],
    ["tools-pcc-source", UI_STORAGE_KEYS.pccSource, false],
    ["tools-pcc-catalog-dir", UI_STORAGE_KEYS.pccCatalogDir, true],
    ["tools-pcc-mag-limit", UI_STORAGE_KEYS.pccMagLimit, false],
    ["tools-pcc-mag-bright", UI_STORAGE_KEYS.pccMagBrightLimit, false],
    ["tools-pcc-min-stars", UI_STORAGE_KEYS.pccMinStars, false],
    ["tools-pcc-sigma", UI_STORAGE_KEYS.pccSigma, false],
    ["tools-pcc-aperture", UI_STORAGE_KEYS.pccAperture, false],
    ["tools-pcc-annulus-in", UI_STORAGE_KEYS.pccAnnulusInner, false],
    ["tools-pcc-annulus-out", UI_STORAGE_KEYS.pccAnnulusOuter, false],
  ].forEach(([id, key, absolute]) => bindStoredField(id, key, { absolute, overwrite: id === "tools-pcc-source" }));

  const missingField = $("tools-pcc-missing-chunks");
  const starsMatchedField = $("tools-pcc-stars-matched");
  const starsUsedField = $("tools-pcc-stars-used");
  const residualField = $("tools-pcc-residual-rms");
  const matrixField = $("tools-pcc-matrix");

  const append = (msg) => appendStructuredLog(logBox, msg, { suppressRunStatus: true });
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
      persistTextValue(UI_STORAGE_KEYS.pccLastOutput, uiState.lastPccOutput, { absolute: true });
    }
    if (Array.isArray(payload.output_channels)) {
      uiState.lastPccChannels = payload.output_channels.map((item) => String(item));
      persistJsonValue(UI_STORAGE_KEYS.pccLastChannels, uiState.lastPccChannels);
    }
    uiState.lastPccResult = payload;
    persistJsonValue(UI_STORAGE_KEYS.pccLastResult, payload);
  };

  const storedPccOutput = storedTextValue(UI_STORAGE_KEYS.pccLastOutput, { absolute: true });
  if (storedPccOutput) {
    uiState.lastPccOutput = storedPccOutput;
  }
  const storedPccChannels = storedJsonValue(UI_STORAGE_KEYS.pccLastChannels, []);
  if (Array.isArray(storedPccChannels)) {
    uiState.lastPccChannels = storedPccChannels.map((item) => String(item));
  }
  const storedPccResult = storedJsonValue(UI_STORAGE_KEYS.pccLastResult, null);
  if (storedPccResult && typeof storedPccResult === "object") {
    uiState.lastPccResult = storedPccResult;
    applyPccResult(storedPccResult);
  }

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
      persistTextValue(UI_STORAGE_KEYS.pccCatalogDir, status.catalog_dir, { absolute: true });
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

  const escapeHtml = (text) =>
    String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  function render() {
    const lines = uiState.liveLines.filter((item) => uiState.liveFilter === "all" || item.level === uiState.liveFilter);
    const paletteByLevel = {
      info: { text: "#e5edf6", badgeBg: "#334155", badgeFg: "#dbeafe", lineBg: "transparent", border: "#334155" },
      warning: { text: "#fdba74", badgeBg: "#7c2d12", badgeFg: "#ffedd5", lineBg: "rgba(249, 115, 22, 0.08)", border: "#f59e0b" },
      error: { text: "#fca5a5", badgeBg: "#7f1d1d", badgeFg: "#fee2e2", lineBg: "rgba(239, 68, 68, 0.1)", border: "#ef4444" },
    };
    box.innerHTML = lines
      .map((item) => {
        const level = String(item.level || "info");
        const palette = paletteByLevel[level] || paletteByLevel.info;
        return `<div style="display:flex;gap:10px;align-items:flex-start;padding:4px 0 4px 10px;border-left:3px solid ${palette.border};background:${palette.lineBg};color:${palette.text};white-space:pre-wrap;"><span style="flex:0 0 auto;display:inline-block;min-width:42px;padding:1px 6px;border-radius:999px;background:${palette.badgeBg};color:${palette.badgeFg};font-size:11px;font-weight:700;line-height:1.5;text-align:center;">${liveLogTag(level)}</span><span style="display:block;min-width:0;">${escapeHtml(item.line)}</span></div>`;
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
      persistTextValue(UI_STORAGE_KEYS.liveFilter, uiState.liveFilter);
      render();
    });
  });

  if (!["all", "info", "warning", "error"].includes(String(uiState.liveFilter || "").toLowerCase())) {
    uiState.liveFilter = "all";
    persistTextValue(UI_STORAGE_KEYS.liveFilter, uiState.liveFilter);
  }

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
    uiState.liveLines = (logs.lines || [])
      .map((line) => ({
        line: formatStructuredLogLine(line, { suppressRunStatus: true }) || String(line || "").trim(),
        level: detectStructuredLogLevel(line),
      }))
      .filter((item) => item.line)
      .map((item) => ({ line: item.line, level: item.level }))
      .filter(Boolean);
    render();
    if (uiState.liveSocket) uiState.liveSocket.close();
    uiState.liveSocket = api.ws(
      API_ENDPOINTS.ws.run(runId),
      (event) => {
        const line = formatStructuredLogLine(event, { suppressRunStatus: true });
        if (!line) return;
        uiState.livePendingLines.push({ line, level: detectStructuredLogLevel(event) });
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
  return normalizeDetectedColorMode(readServerUiStateValue(LAST_SCAN_COLOR_MODE_KEY) || "");
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
  if (sec) sec.style.display = isMono ? "" : "none";
  setRunMonitorFilterVisibility(selectedModeRaw);
}

function collectQueueRows() {
  const out = [];
  for (const row of collectQueueDraftRows()) {
    const filter = String(row.filter || "").trim();
    const inputDir = String(row.input_dir || "").trim();
    const pattern = String(row.pattern || "").trim();
    const runLabel = String(row.run_id || "").trim();
    const isOn = row.enabled !== false;
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

function currentValidationStateForYaml(yamlText) {
  const validation = getConfigValidationState();
  if (!validation) return null;
  return String(validation.yaml || "") === String(yamlText || "") ? validation : null;
}

function updateDashboardRunStartState(validationState, guardrailStatus = uiState.dashboardGuardrailStatus) {
  const runStart = $("dashboard-run-start");
  if (!runStart) return;
  const guardrailError = String(guardrailStatus || "").trim().toLowerCase() === "error";
  const validationOk = Boolean(validationState?.ok);
  setDisabledLike(runStart, guardrailError || !validationOk);
  if (guardrailError) {
    runStart.title = "Run/Queue starten ist blockiert: Guardrail-Status ist ERROR.";
  } else if (!validationState) {
    runStart.title = "Run/Queue starten ist blockiert: zuerst Validieren.";
  } else if (!validationOk) {
    runStart.title = "Run/Queue starten ist blockiert: Validierung hat Fehler.";
  } else {
    runStart.title = "Run/Queue starten.";
  }
}

function dashboardPipelineStepElements() {
  return Array.from(document.querySelectorAll("#dashboard-pipeline-preview [data-pipeline-step]"));
}

function setDashboardPipelineStepVisual(el, state, pct = 0) {
  if (!el) return;
  const normalized = String(state || "pending").trim().toLowerCase();
  const label = String(el.getAttribute("data-pipeline-step") || el.textContent || "").trim();
  let background = "#f0f0f0";
  let color = "#64748b";
  if (normalized === "done") {
    background = "#d1f4e0";
    color = "#15808d";
  } else if (normalized === "running") {
    background = "#dbeafe";
    color = "#1d4ed8";
  } else if (normalized === "error") {
    background = "#fee2e2";
    color = "#b91c1c";
  }
  el.textContent = label;
  el.style.background = background;
  el.style.color = color;
  el.title = pct > 0 ? `${label} (${Math.round(pct)}%)` : label;
}

function normalizePipelinePhaseState(status) {
  const normalized = String(status || "pending").trim().toLowerCase();
  if (["ok", "completed", "done", "finished", "skipped"].includes(normalized)) return "done";
  if (["running", "active", "started"].includes(normalized)) return "running";
  if (["error", "failed", "aborted", "cancelled"].includes(normalized)) return "error";
  return "pending";
}

function summarizeDashboardPipelineGroup(group, phaseEntries, runStatus, currentPhase) {
  if (group.key === "DONE") {
    const normalizedRunStatus = String(runStatus || "").trim().toLowerCase();
    if (["completed", "done", "finished"].includes(normalizedRunStatus)) return { state: "done", pct: 100 };
    if (["failed", "error", "aborted", "cancelled"].includes(normalizedRunStatus)) return { state: "error", pct: 0 };
    return { state: "pending", pct: 0 };
  }

  const phaseStates = group.phases.map((phase) => {
    const entry = phaseEntries.get(phase);
    const state = normalizePipelinePhaseState(entry?.status);
    let pct = Number(entry?.pct || 0);
    if (Number.isFinite(pct) && pct <= 1.0) pct *= 100.0;
    if (!Number.isFinite(pct)) pct = 0;
    if (state === "done") pct = 100;
    pct = Math.max(0, Math.min(100, pct));
    return { state, pct };
  });

  if (phaseStates.some((item) => item.state === "error")) {
    return { state: "error", pct: Math.max(...phaseStates.map((item) => item.pct), 0) };
  }

  if (phaseStates.length > 0 && phaseStates.every((item) => item.state === "done")) {
    return { state: "done", pct: 100 };
  }

  const currentInGroup = group.phases.includes(currentPhase);
  const anyRunning = currentInGroup || phaseStates.some((item) => item.state === "running");
  const anyStarted = phaseStates.some((item) => item.state === "done" || item.state === "running" || item.pct > 0);
  const pct = phaseStates.length > 0
    ? phaseStates.reduce((sum, item) => sum + item.pct, 0) / phaseStates.length
    : 0;

  if (anyRunning || anyStarted) return { state: "running", pct };
  return { state: "pending", pct: 0 };
}

async function renderDashboardPipelinePreview(appState) {
  const stepEls = dashboardPipelineStepElements();
  if (stepEls.length === 0) return;

  stepEls.forEach((el) => setDashboardPipelineStepVisual(el, "pending", 0));

  const runId = String(appState?.run?.current?.run_id || "").trim();
  if (!runId) return;

  let status = null;
  try {
    status = await api.get(API_ENDPOINTS.runs.status(runId));
  } catch {
    status = {
      status: String(appState?.run?.current?.status || "unknown"),
      current_phase: String(appState?.run?.current?.current_phase || ""),
      phases: [],
    };
  }

  const phaseEntries = new Map();
  if (Array.isArray(status?.phases)) {
    status.phases.forEach((entry) => {
      const phase = String(entry?.phase || "").trim().toUpperCase();
      if (phase) phaseEntries.set(phase, entry);
    });
  }

  const currentPhase = String(status?.current_phase || "").trim().toUpperCase();
  if (phaseEntries.size === 0 && currentPhase) {
    const currentGroupIndex = DASHBOARD_PIPELINE_GROUPS.findIndex((group) => group.phases.includes(currentPhase));
    stepEls.forEach((el, index) => {
      const step = String(el.getAttribute("data-pipeline-step") || "").trim().toUpperCase();
      if (step === "DONE") {
        setDashboardPipelineStepVisual(el, summarizeDashboardPipelineGroup({ key: "DONE", phases: [] }, phaseEntries, status?.status, currentPhase).state, 0);
        return;
      }
      if (currentGroupIndex >= 0) {
        if (index < currentGroupIndex) setDashboardPipelineStepVisual(el, "done", 100);
        else if (index === currentGroupIndex) setDashboardPipelineStepVisual(el, "running", 0);
      }
    });
    return;
  }

  DASHBOARD_PIPELINE_GROUPS.forEach((group) => {
    const el = stepEls.find((node) => String(node.getAttribute("data-pipeline-step") || "").trim().toUpperCase() === group.key);
    if (!el) return;
    const summary = summarizeDashboardPipelineGroup(group, phaseEntries, status?.status, currentPhase);
    setDashboardPipelineStepVisual(el, summary.state, summary.pct);
  });
}

async function renderDashboardDerivedGuardrails(appState) {
  let yaml = "";
  try {
    yaml = await ensureConfigYaml();
  } catch {
    yaml = "";
  }

  const configValidation = currentValidationStateForYaml(yaml);
  const validationErrors = Array.isArray(configValidation?.errors) ? configValidation.errors.length : 0;
  const validationWarnings = Array.isArray(configValidation?.warnings) ? configValidation.warnings.length : 0;
  const validationHasErrors = Boolean(configValidation) && (!configValidation.ok || validationErrors > 0);
  const currentRunId = String(appState?.run?.current?.run_id || "").trim();

  renderGuardrailRow(
    $("dashboard-guardrail-config-valid"),
    !configValidation ? "check" : validationHasErrors ? "error" : validationWarnings > 0 ? "check" : "ok",
    !configValidation
      ? "Config nicht geprüft"
      : validationHasErrors
        ? (validationErrors > 0 ? `Config mit ${validationErrors} Fehlern` : "Config mit Fehlern")
        : validationWarnings > 0
          ? `Config validiert (${validationWarnings} Warnungen)`
          : "Config validiert",
  );
  setDashboardValidateStatus(configValidation, "Validierung: nicht geprüft");
  setDashboardValidateDetails(configValidation);
  renderGuardrailRow($("dashboard-guardrail-calibration-paths"), "check", "Kalibrierpfade nicht separat geprüft");
  renderGuardrailRow(
    $("dashboard-guardrail-bge-pcc"),
    "check",
    currentRunId ? "BGE/PCC nicht automatisch bewertet" : "BGE/PCC nicht geprüft (kein Run)",
  );
  updateDashboardRunStartState(configValidation);
}

async function bindDashboard() {
  if (!$("dashboard-kpi-scan-quality")) return;
  setDisabledLike($("dashboard-run-start"), true);
  setDashboardValidateStatus(null, "Validierung: nicht geprüft");
  setDashboardValidateDetails(null);
  bindInputDirMemory("dashboard-input-dirs");
  bindStoredField("dashboard-run-runs-dir", UI_STORAGE_KEYS.dashboardRunsDir, { absolute: true });
  bindStoredField("dashboard-run-name", UI_STORAGE_KEYS.dashboardRunName, { normalize: sanitizeRunName });
  bindQueueDraftPersistence(UI_STORAGE_KEYS.dashboardQueue);
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
    uiState.dashboardGuardrailStatus = String(guardrails?.status || "");
    setRunReady(guardrails?.status || "check", appState?.run?.current?.status || "");
    const summary = summarizeScanResult(
      latestScan?.has_scan ? latestScan : quality?.scan || {},
      String($("dashboard-input-dirs")?.value || "").trim(),
    );
    renderDashboardScanKpis(summary, quality?.score ?? 0);
    renderDashboardLastRunKpi(appState);
    await renderDashboardPipelinePreview(appState);
    renderScanSummary("dashboard-scan", summary);
    applyDetectedColorModeToSelect($("dashboard-color-mode"), summary);
    applyDetectedColorModeToSelect($("inp-colormode"), summary);
    const mergedInputText = summary.input_dirs?.length > 0 ? summary.input_dirs.join(", ") : summary.input_path;
    if (mergedInputText) {
      $("dashboard-input-dirs") && ($("dashboard-input-dirs").value = mergedInputText);
      persistLastInputDirs(mergedInputText);
      restoreLastInputDirs("dashboard-input-dirs");
    }
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

    await bindPresetDirectoryControl({
      inputId: "dashboard-preset-dir",
      browseId: "dashboard-preset-dir-browse",
      reloadId: "dashboard-preset-dir-reload",
      selectId: "dashboard-preset",
    });
    await populatePresetSelect("dashboard-preset", false);
    restoreUnifiedPresetSelectValue("dashboard-preset");
    bindUnifiedPresetSelect("dashboard-preset");

    const preview = () => {
      const runsDir = String($("dashboard-run-runs-dir")?.value || "").trim();
      const rawRunName = String($("dashboard-run-name")?.value || "");
      const sanitizedRunName = sanitizeRunName(rawRunName);
      const runName = sanitizedRunName || preferredRunName({
        inputId: "dashboard-run-name",
        storageKey: UI_STORAGE_KEYS.dashboardRunName,
        fallbackDirs: parseInputDirs($("dashboard-input-dirs")?.value || ""),
      });
      if ($("dashboard-run-name") && sanitizedRunName) $("dashboard-run-name").value = sanitizedRunName;
      persistTextValue(UI_STORAGE_KEYS.dashboardRunsDir, runsDir, { absolute: true });
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
    $("dashboard-input-dirs")?.addEventListener("change", preview);
    $("dashboard-input-dirs")?.addEventListener("input", preview);
    preview();

    $("dashboard-preset")?.addEventListener("change", async () => {
      try {
        const path = String($("dashboard-preset")?.value || "").trim();
        if (!path) return;
        syncUnifiedPresetSelection(path);
        const applied = await api.post(API_ENDPOINTS.config.applyPreset, { path });
        setConfigDraft(String(applied?.config || ""));
        uiState.parameterDirty = {};
        clearParameterDirtyState();
        preview();
        clearConfigValidationState();
        const appStateNow = await api.get(API_ENDPOINTS.app.state).catch(() => appState);
        await renderDashboardDerivedGuardrails(appStateNow);
        setFooter("Preset fuer Guided Run aktualisiert.");
      } catch (err) {
        setFooter(`Preset-Laden fehlgeschlagen: ${errorText(err)}`, true);
      }
    });

    $("dashboard-validate")?.addEventListener("click", async () => {
      const validateButton = $("dashboard-validate");
      try {
        setDisabledLike(validateButton, true);
        setDashboardValidateStatus(null, "Validierung läuft...");
        setDashboardValidateDetails(null);
        const yaml = await ensureConfigYaml();
        const result = await api.post(API_ENDPOINTS.config.validate, { yaml });
        setConfigValidationState({
          yaml,
          ok: Boolean(result?.ok),
          errors: Array.isArray(result?.errors) ? result.errors : [],
          warnings: Array.isArray(result?.warnings) ? result.warnings : [],
        });
        const appStateNow = await api.get(API_ENDPOINTS.app.state).catch(() => appState);
        await renderDashboardDerivedGuardrails(appStateNow);
        setFooter(result?.ok ? "Validierung OK." : "Validierung hat Fehler.");
      } catch (err) {
        clearConfigValidationState();
        const appStateNow = await api.get(API_ENDPOINTS.app.state).catch(() => appState);
        await renderDashboardDerivedGuardrails(appStateNow);
        setDashboardValidateStatus(null, "Validierung: fehlgeschlagen");
        setDashboardValidateDetails(null);
        setFooter(`Validierung fehlgeschlagen: ${errorText(err)}`, true);
      } finally {
        setDisabledLike(validateButton, false);
      }
    });

    $("dashboard-run-start")?.addEventListener("click", async (ev) => {
      ev.preventDefault();
      const runStartButton = $("dashboard-run-start");
      try {
        setDisabledLike(runStartButton, true);
        const latestGuardrails = await api.get(API_ENDPOINTS.guardrails.root);
        uiState.dashboardGuardrailStatus = String(latestGuardrails?.status || "");
        const yaml = await ensureConfigYaml();
        const validation = currentValidationStateForYaml(yaml);
        if (String(latestGuardrails?.status || "").toLowerCase() === "error") {
          updateDashboardRunStartState(validation, latestGuardrails?.status || "");
          setFooter("Run blockiert: Guardrail-Status ist ERROR.", true);
          return;
        }
        if (!validation) {
          updateDashboardRunStartState(null, latestGuardrails?.status || "");
          setFooter("Run blockiert: zuerst Validieren.", true);
          return;
        }
        if (!validation.ok) {
          updateDashboardRunStartState(validation, latestGuardrails?.status || "");
          setFooter("Run blockiert: Validierung hat Fehler.", true);
          return;
        }
        const accepted = await startRunFromCurrentForm({ source: "dashboard" });
        setCurrentRunId(accepted?.run_id || uiState.currentRunId);
        clearCurrentRunHistoryMark();
        setRunReady(latestGuardrails?.status || "check", "running");
        setFooter(`Run gestartet (Job ${accepted?.job_id || "-"}).`);
        window.location.href = "run-monitor.html";
      } catch (err) {
        const yaml = await ensureConfigYaml().catch(() => "");
        updateDashboardRunStartState(currentValidationStateForYaml(yaml), uiState.dashboardGuardrailStatus);
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
        uiState.dashboardGuardrailStatus = String(guardrails2?.status || "");
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
        await renderDashboardPipelinePreview(appState2);
        await renderDashboardDerivedGuardrails(appState2);
        if (String(job?.state) === "missing") {
          setFooter(
            "Scan-Status war kurzzeitig nicht abrufbar (Backend-Reload). Letztes Scan-Ergebnis wurde geladen.",
            true,
          );
        } else {
          if (job?.state === "ok") {
            setFooter("Scan abgeschlossen.");
          } else {
            const detail = scanErrorFromResult(job?.data?.result || {});
            setFooter(
              detail ? `Scan fehlgeschlagen: ${detail}` : `Scan beendet mit Status: ${job?.state || "unknown"}`,
              true,
            );
          }
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
  updateWizardStartState(null);
  setWizardValidationResult(null, "Validierung ausstehend.");
  bindStoredField("wizard-runs-dir", UI_STORAGE_KEYS.wizardRunsDir, { absolute: true });
  bindStoredField("wizard-run-name", UI_STORAGE_KEYS.wizardRunName, { normalize: sanitizeRunName });
  bindQueueDraftPersistence(UI_STORAGE_KEYS.wizardQueue);
  const wizardRunsDir = $("wizard-runs-dir");
  if (wizardRunsDir && !String(wizardRunsDir.value || "").trim() && uiState.projectRunsDir) {
    wizardRunsDir.value = uiState.projectRunsDir;
  }
  const applyWizardValidationState = (validationState, fallbackText = "Validierung ausstehend.") => {
    if (validationState) {
      setWizardValidationResult({
        ok: Boolean(validationState.ok),
        errors: Array.isArray(validationState.errors) ? validationState.errors : [],
        warnings: Array.isArray(validationState.warnings) ? validationState.warnings : [],
      });
    } else {
      setWizardValidationResult(null, fallbackText);
    }
    updateWizardStartState(validationState);
  };
  const validateWizardYaml = async (yamlText, { quiet = false, pendingText = "Validierung läuft..." } = {}) => {
    const yaml = String(yamlText || "");
    applyWizardValidationState(null, pendingText);
    try {
      const result = await api.post(API_ENDPOINTS.config.validate, { yaml });
      setConfigValidationState({
        yaml,
        ok: Boolean(result?.ok),
        errors: Array.isArray(result?.errors) ? result.errors : [],
        warnings: Array.isArray(result?.warnings) ? result.warnings : [],
      });
      applyWizardValidationState(currentValidationStateForYaml(yaml));
      return result;
    } catch (err) {
      clearConfigValidationState();
      applyWizardValidationState(null, "Validierung fehlgeschlagen.");
      if (!quiet) {
        setFooter(`Wizard-Validierung fehlgeschlagen: ${errorText(err)}`, true);
      }
      throw err;
    }
  };
  const updateWizardPreview = () => {
    const runsDir = String($("wizard-runs-dir")?.value || "").trim();
    const dirs = parseInputDirs(String($("inp-dirs")?.value || ""));
    const rawRunName = String($("wizard-run-name")?.value || "");
    const sanitizedRunName = sanitizeRunName(rawRunName);
    const suggested = sanitizedRunName || preferredRunName({
      inputId: "wizard-run-name",
      storageKey: UI_STORAGE_KEYS.wizardRunName,
      fallbackDirs: dirs,
    });
    if ($("wizard-run-name") && sanitizedRunName) $("wizard-run-name").value = sanitizedRunName;
    persistTextValue(UI_STORAGE_KEYS.wizardRunsDir, runsDir, { absolute: true });
    const previewEl = $("wizard-run-path-preview");
    if (!previewEl) return;
    if (!runsDir || !suggested) {
      previewEl.value = "";
      return;
    }
    previewEl.value = `${runsDir}/${suggested}_${timestampSuffix()}`;
  };
  try {
    await bindPresetDirectoryControl({
      inputId: "wizard-preset-dir",
      browseId: "wizard-preset-dir-browse",
      reloadId: "wizard-preset-dir-reload",
      selectId: "wizard-preset-select",
    });
    await populatePresetSelect("wizard-preset-select", true);
    restoreUnifiedPresetSelectValue("wizard-preset-select");
    bindUnifiedPresetSelect("wizard-preset-select");
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
      syncUnifiedPresetSelection(path);
      const applied = await api.post(API_ENDPOINTS.config.applyPreset, { path });
      const yaml = String(applied?.config || "");
      setConfigDraft(yaml);
      uiState.parameterDirty = {};
      clearParameterDirtyState();
      updateWizardPreview();
      const v = await validateWizardYaml(yaml, { quiet: true });
      setFooter(v.ok ? "Wizard-Preset angewendet. Validierung OK." : "Wizard-Preset angewendet. Validierung hat Fehler.", !v.ok);
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
      const v = await validateWizardYaml(patched?.config_yaml || "", { quiet: true });
      setFooter(
        v.ok
          ? `Wizard-Szenario angewendet (${updates.length} Deltas). Validierung OK.`
          : `Wizard-Szenario angewendet (${updates.length} Deltas). Validierung hat Fehler.`,
        !v.ok,
      );
    } catch (err) {
      setFooter(`Wizard-Szenario fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("wizard-start")?.addEventListener("click", async (ev) => {
    ev.preventDefault();
    try {
      const yaml = await ensureConfigYaml();
      const validation = currentValidationStateForYaml(yaml);
      if (!validation) {
        updateWizardStartState(null);
        setFooter("Wizard-Run blockiert: zuerst erfolgreiche Validierung abwarten.", true);
        return;
      }
      if (!validation.ok) {
        updateWizardStartState(validation);
        setFooter("Wizard-Run blockiert: Validierung hat Fehler.", true);
        return;
      }
      const accepted = await startRunFromCurrentForm({ source: "wizard" });
      setCurrentRunId(accepted?.run_id || uiState.currentRunId);
      clearCurrentRunHistoryMark();
      setFooter(`Wizard-Run gestartet (Job ${accepted?.job_id || "-"}).`);
      window.location.href = "run-monitor.html";
    } catch (err) {
      setFooter(`Wizard-Runstart fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  try {
    const initialYaml = await ensureConfigYaml();
    const existingValidation = currentValidationStateForYaml(initialYaml);
    if (existingValidation) {
      applyWizardValidationState(existingValidation);
    } else {
      await validateWizardYaml(initialYaml, { quiet: true, pendingText: "Validierung läuft..." });
    }
  } catch (err) {
    applyWizardValidationState(null, "Validierung fehlgeschlagen.");
    setFooter(`Wizard-Validierung konnte nicht initialisiert werden: ${errorText(err)}`, true);
  }
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
  await initGlobalState();
  bindLocaleControls();
  bindRunMonitorFilterSync();
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
