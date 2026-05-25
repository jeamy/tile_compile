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
const CALIBRATION_PATH_CACHE_KEY = "gui2.calibrationPathCache";
const LAST_SCAN_COLOR_MODE_KEY = "gui2.lastScanColorMode";
const ASTROMETRY_LAST_RESULT_KEY = "gui2.tools.astrometry.lastResult";
const ASTROMETRY_LAST_WCS_KEY = "gui2.tools.astrometry.lastWcs";
const ASTROMETRY_INSTALL_JOB_KEY = "gui2.tools.astrometry.installJob";
const ASTROMETRY_CATALOG_JOB_KEY = "gui2.tools.astrometry.catalogJob";
const PCC_LAST_OUTPUT_KEY = "gui2.tools.pcc.lastOutput";
const PCC_LAST_CHANNELS_KEY = "gui2.tools.pcc.lastChannels";
const PCC_LAST_RESULT_KEY = "gui2.tools.pcc.lastResult";
const PCC_DOWNLOAD_JOB_KEY = "gui2.tools.pcc.downloadJob";
const PCC_TEMP_OUTPUT_KEY = "gui2.tools.pcc.tempOutput";
const PCC_TEMP_CHANNELS_KEY = "gui2.tools.pcc.tempChannels";
const PCC_TEMP_JOB_KEY = "gui2.tools.pcc.tempJob";
const PREPROCESSING_JOB_KEY = "gui2.preprocessing.jobId";
const PREPROCESSING_RUN_ID_KEY = "gui2.preprocessing.runId";
const PREPROCESSING_RUN_DIR_KEY = "gui2.preprocessing.runDir";
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
  pccApplyAttenuation: "gui2.tools.pcc.applyAttenuation",
  pccChromaStrength: "gui2.tools.pcc.chromaStrength",
  pccKMax: "gui2.tools.pcc.kMax",
  pccBgNeutralizationMode: "gui2.tools.pcc.bgNeutralizationMode",
  astrometryLastResult: ASTROMETRY_LAST_RESULT_KEY,
  astrometryLastWcs: ASTROMETRY_LAST_WCS_KEY,
  astrometryInstallJob: ASTROMETRY_INSTALL_JOB_KEY,
  astrometryCatalogJob: ASTROMETRY_CATALOG_JOB_KEY,
  pccLastOutput: PCC_LAST_OUTPUT_KEY,
  pccLastChannels: PCC_LAST_CHANNELS_KEY,
  pccLastResult: PCC_LAST_RESULT_KEY,
  pccDownloadJob: PCC_DOWNLOAD_JOB_KEY,
  pccTempOutput: PCC_TEMP_OUTPUT_KEY,
  pccTempChannels: PCC_TEMP_CHANNELS_KEY,
  pccTempJob: PCC_TEMP_JOB_KEY,
};

const QUEUE_FILTER_PRESETS = ["", "OSC", "L", "R", "G", "B", "Ha", "OIII", "SII"];

function activeQueueStorageKey() {
  return pageName() === "wizard.html" ? UI_STORAGE_KEYS.wizardQueue : UI_STORAGE_KEYS.dashboardQueue;
}

const uiState = {
  currentRunId: "",
  currentRunDir: "",
  currentRunQueue: [],
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
  currentPccTempOutput: "",
  currentPccTempChannels: [],
  currentPccTempJobId: "",
  locale: "de",
  projectRunsDir: "",
  projectPresetsDir: "",
  monitorStatsStatus: null,
  monitorStatsRunId: "",
  monitorStatsRunDir: "",
  dashboardGuardrailStatus: "",
  runReadyStatus: "check",
  runProcessStatus: "",
  configSchemaPaths: null,
  runMonitorSwitchHandler: null,
  runMonitorResumePhases: ["ASTROMETRY", "BGE", "PCC", "HYPERMETRIC_STRETCH"],
  runPhaseSnapshots: {},
  runMonitorSelectedBatchKey: "",
};

const appRuntime = {
  tempRoot: "/tmp",
};

let serverUiState = {};
let serverUiStateLoaded = false;
let serverUiStateSaveTimer = null;
let serverUiStateSavePromise = Promise.resolve();
let configPatchRequestSeq = 0;
let pendingScanConfigSync = Promise.resolve();
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
  { key: "HMS", phases: ["HYPERMETRIC_STRETCH"] },
  { key: "DONE", phases: [] },
];

const RUN_MONITOR_PHASE_ORDER = [
  "SCAN_INPUT",
  "CHANNEL_SPLIT",
  "NORMALIZATION",
  "GLOBAL_METRICS",
  "TILE_GRID",
  "REGISTRATION",
  "PREWARP",
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

const PARAM_CONTROL_PATHS = {
  "parameter.registration.engine": "registration.engine",
  "parameter.registration.allow_rotation": "registration.allow_rotation",
  "parameter.registration.transform_model": "registration.transform_model",
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
  "parameter-bge-sample-estimator": "bge.sample_estimator",
  "parameter-bge-min-sample-bg-value": "bge.min_sample_bg_value",
  "parameter-bge-min-tiles": "bge.min_tiles_per_cell",
  "parameter-pcc-min-stars": "pcc.min_stars",
  "parameter-input-pattern": "input.pattern",
  "parameter-input-max-frames": "input.max_frames",
  "parameter-data-bayer": "data.bayer_pattern",
  "parameter-runtime-workers": "runtime_limits.parallel_workers",
  "parameter-runtime-memory": "runtime_limits.memory_budget",
  "parameter-runtime-hard-abort": "runtime_limits.hard_abort_hours",
  "parameter-cal-use-dark": "calibration.use_dark",
  "parameter-cal-dark-bias-corrected": "calibration.dark_already_bias_corrected",
  "parameter-cal-darks-dir": "calibration.darks_dir",
  "parameter-cal-use-flat": "calibration.use_flat",
  "parameter-cal-flats-dir": "calibration.flats_dir",
  "parameter-ass-frames-min": "assumptions.frames_min",
  "parameter-ass-frames-reduced-threshold": "assumptions.frames_reduced_threshold",
  "parameter-ass-skip-cluster": "assumptions.reduced_mode_skip_clustering",
  "parameter-ass-cluster-range": "assumptions.reduced_mode_cluster_range",
};

const ASSUMPTION_ID_PATHS = {
  "asmpt-min": "assumptions.frames_min",
  "asmpt-reduced": "assumptions.frames_reduced_threshold",
  "asmpt-skip-cluster": "assumptions.reduced_mode_skip_clustering",
  "asmpt-cluster-range": "assumptions.reduced_mode_cluster_range",
};

const SCAN_CALIBRATION_BINDINGS = [
  {
    storageKey: "bias",
    toggleId: "cal-bias",
    sourceId: "cal-bias-source",
    inputId: "cal-bias-dir",
    usePath: "calibration.use_bias",
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
    toggleId: "cal-dark",
    sourceId: "cal-dark-source",
    inputId: "cal-dark-dir",
    usePath: "calibration.use_dark",
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
    toggleId: "cal-flat",
    sourceId: "cal-flat-source",
    inputId: "cal-flat-dir",
    usePath: "calibration.use_flat",
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
    ["registration.transform_model", "affine"],
    ["registration.star_topk", 180],
    ["registration.reject_shift_px_min", 120],
    ["registration.reject_shift_median_multiplier", 5.0],
  ],
  rotation: [
    ["registration.engine", "robust_phase_ecc"],
    ["registration.allow_rotation", true],
    ["registration.transform_model", "affine"],
    ["registration.star_inlier_tol_px", 4.0],
    ["registration.reject_cc_min_abs", 0.25],
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
    ["bge.sample_estimator", "quantile"],
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
  const message = err?.payload?.detail?.error?.message || err?.payload?.error?.message || err?.message || String(err);
  const detail = err?.payload?.detail?.error?.details?.detail || err?.payload?.error?.details?.detail || "";
  if (detail && !String(message).includes(detail)) {
    return `${message} (${detail})`;
  }
  return message;
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
  const validationState = currentRunReadyValidationState();
  let normalized = String(status || "check").toLowerCase();
  if (normalized !== "error" && (!validationState || !validationState.ok)) {
    normalized = "error";
  }
  const statusText = normalized === "ok"
    ? t("ui.status.run_ready_ok", "Status: ready to run")
    : normalized === "error"
      ? t("ui.status.run_ready_blocked", "Status: blocked")
      : t("ui.status.run_ready_check", "Status: check");
  applyChip(chip, normalized === "ok" ? "ok" : normalized === "error" ? "error" : "check", statusText);
}

function refreshRunReadyIndicators() {
  setRunReady(uiState.runReadyStatus, uiState.runProcessStatus);
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

function isActiveJobState(state) {
  return ["pending", "queued", "starting", "running"].includes(String(state || "").trim().toLowerCase());
}

function isTerminalJobState(state) {
  return ["ok", "done", "completed", "finished", "error", "failed", "cancelled", "aborted", "missing"]
    .includes(String(state || "").trim().toLowerCase());
}

async function fetchJobSnapshot(jobId, { allowMissing = false } = {}) {
  try {
    return await api.get(API_ENDPOINTS.jobs.byId(jobId));
  } catch (err) {
    if (allowMissing && Number(err?.status) === 404) {
      return { job_id: jobId, state: "missing", data: {} };
    }
    throw err;
  }
}

function jobRecencyValue(job) {
  return String(job?.updated_at || job?.started_at || job?.created_at || "");
}

async function resolveTrackedJob(storageKey, allowedTypes = []) {
  const matchesType = (job) => allowedTypes.length === 0 || allowedTypes.includes(String(job?.type || ""));
  const trackedJobId = storedTextValue(storageKey);
  if (trackedJobId) {
    const trackedJob = await fetchJobSnapshot(trackedJobId, { allowMissing: true });
    if (trackedJob && matchesType(trackedJob)) return trackedJob;
  }
  const listed = await api.get(`${API_ENDPOINTS.jobs.list}?limit=100`).catch(() => null);
  const jobs = Array.isArray(listed?.items) ? listed.items : [];
  const active = jobs
    .filter((job) => matchesType(job) && isActiveJobState(job?.state))
    .sort((a, b) => jobRecencyValue(b).localeCompare(jobRecencyValue(a)));
  return active[0] || null;
}

function trackedJobProgressPayload(job) {
  return {
    state: job?.state,
    current_chunk: job?.data?.current_chunk ?? null,
    progress: job?.data?.progress ?? job?.progress ?? null,
    stage: job?.data?.stage ?? null,
  };
}

async function resumeTrackedJob({
  storageKey,
  jobTypes = [],
  statusChip,
  labels = {},
  append = null,
  onTerminal = null,
} = {}) {
  const job = await resolveTrackedJob(storageKey, jobTypes);
  if (!job) {
    persistTextValue(storageKey, "");
    return null;
  }
  persistTextValue(storageKey, job.job_id || "");
  updateTransferStatusChip(statusChip, job, labels);
  append?.(trackedJobProgressPayload(job));
  if (!isActiveJobState(job.state)) {
    persistTextValue(storageKey, "");
    await onTerminal?.(job);
    return job;
  }
  const finalJob = await waitForJob(job.job_id, {
    allowMissing: true,
    onTick: (snapshot) => {
      updateTransferStatusChip(statusChip, snapshot, labels);
      append?.(trackedJobProgressPayload(snapshot));
    },
  });
  updateTransferStatusChip(statusChip, finalJob, labels);
  if (isTerminalJobState(finalJob.state)) persistTextValue(storageKey, "");
  await onTerminal?.(finalJob);
  return finalJob;
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
  const clamped = Math.max(0, Math.min(100, pct));
  return `${clamped.toFixed(clamped >= 10 ? 0 : 1)}%`;
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
  } else if (phase === "HYPERMETRIC_STRETCH") {
    const logD = formatLogNumber(payload.log_d, 3);
    const anchor = formatLogNumber(payload.anchor, 6);
    const starPressure = formatLogNumber(payload.star_pressure, 3);
    const blackClip = formatLogNumber(payload.black_clip_percent, 3);
    const whiteClip = formatLogNumber(payload.white_clip_percent, 3);
    if (logD) parts.push(`LogD ${logD}`);
    if (anchor) parts.push(`anchor ${anchor}`);
    if (starPressure) parts.push(`star ${starPressure}`);
    if (blackClip) parts.push(`black ${blackClip}%`);
    if (whiteClip) parts.push(`white ${whiteClip}%`);
    pushLogPart(parts, payload.profile);
    pushLogPart(parts, payload.output_rgb);
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
    const stateLabel = humanizeLogToken(
      payload.state || entry.status || (entry.success === false ? "failed" : entry.success === true ? "completed" : ""),
    );
    const parts = ["Run beendet"];
    if (stateLabel) parts.push(stateLabel);
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

function parseEventTimestampMs(tsRaw) {
  const ts = String(tsRaw || "").trim();
  if (!ts) return Number.NaN;
  const ms = Date.parse(ts);
  return Number.isFinite(ms) ? ms : Number.NaN;
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
  refreshRunReadyIndicators();
}

function getConfigValidationState() {
  try {
    const raw = readServerUiStateValue(CONFIG_VALIDATION_STATE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function currentRunReadyValidationState() {
  const yamlText = String(uiState.configYaml || getConfigDraft() || "");
  if (!yamlText.trim()) return getConfigValidationState();
  return currentValidationStateForYaml(yamlText);
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
  refreshRunReadyIndicators();
}

function clearConfigValidationState() {
  writeServerUiStateValue(CONFIG_VALIDATION_STATE_KEY, "");
  refreshRunReadyIndicators();
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

function currentDraftValueForPath(path) {
  const normalized = String(path || "").trim();
  if (!normalized) return undefined;
  if (uiState.parameterDirty && typeof uiState.parameterDirty === "object"
      && Object.prototype.hasOwnProperty.call(uiState.parameterDirty, normalized)) {
    return uiState.parameterDirty[normalized];
  }
  return getByPath(uiState.configObject, normalized);
}

function rememberParameterDirtyUpdates(updates = []) {
  const next = {
    ...(uiState.parameterDirty && typeof uiState.parameterDirty === "object" ? uiState.parameterDirty : {}),
  };
  let changed = false;
  for (const entry of Array.isArray(updates) ? updates : []) {
    const path = String(entry?.path || "").trim();
    if (!path || !isKnownConfigSchemaPath(path)) continue;
    next[path] = entry?.value;
    changed = true;
  }
  if (!changed) return;
  uiState.parameterDirty = next;
  setParameterDirtyState(next);
}

function enqueueScanConfigSync(task) {
  const next = pendingScanConfigSync
    .catch(() => {})
    .then(() => task());
  pendingScanConfigSync = next.catch(() => {});
  return next;
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

function dateStamp(now = new Date()) {
  const pad = (n) => String(n).padStart(2, "0");
  const yyyy = String(now.getFullYear());
  const mm = pad(now.getMonth() + 1);
  const dd = pad(now.getDate());
  const hh = pad(now.getHours());
  const mi = pad(now.getMinutes());
  return `${yyyy}${mm}${dd}_${hh}${mi}`;
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
  return SCAN_CALIBRATION_BINDINGS.find((binding) => (
    binding.toggleId === id
    || binding.sourceId === id
    || binding.inputId === id
  )) || null;
}

function scanCalibrationUseMaster(binding) {
  return readFieldValue($(binding?.sourceId)) === true;
}

function scanCalibrationEnabled(binding) {
  return readFieldValue($(binding?.toggleId)) === true;
}

function scanCalibrationActivePath(binding, useMaster = scanCalibrationUseMaster(binding)) {
  return useMaster ? binding.masterPath : binding.dirPath;
}

function scanCalibrationStoredPaths(binding, config = null) {
  const readValue = (path) => {
    const raw = config ? getByPath(config, path) : currentDraftValueForPath(path);
    return raw === undefined || raw === null ? "" : String(raw).trim();
  };
  return {
    [binding.dirPath]: readValue(binding.dirPath),
    [binding.masterPath]: readValue(binding.masterPath),
  };
}

function calibrationPathCacheState() {
  const raw = storedJsonValue(CALIBRATION_PATH_CACHE_KEY, {});
  return raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
}

function calibrationPathCacheEntry(binding) {
  if (!binding?.storageKey) return {};
  const cache = calibrationPathCacheState();
  const entry = cache[binding.storageKey];
  return entry && typeof entry === "object" && !Array.isArray(entry) ? entry : {};
}

function persistCalibrationPathCache(binding, values = {}) {
  if (!binding?.storageKey) return;
  const paths = [binding.dirPath, binding.masterPath];
  const explicit = paths.filter((path) => Object.prototype.hasOwnProperty.call(values, path));
  if (explicit.length === 0) return;
  const cache = calibrationPathCacheState();
  const current = calibrationPathCacheEntry(binding);
  const next = {
    ...current,
  };
  let changed = false;
  explicit.forEach((path) => {
    const normalized = String(values[path] ?? "").trim();
    if (String(next[path] ?? "") === normalized) return;
    next[path] = normalized;
    changed = true;
  });
  if (!changed) return;
  cache[binding.storageKey] = next;
  persistJsonValue(CALIBRATION_PATH_CACHE_KEY, cache);
}

function scanCalibrationRestoreUpdates(binding) {
  const cached = calibrationPathCacheEntry(binding);
  return [binding.dirPath, binding.masterPath]
    .filter((path) => Object.prototype.hasOwnProperty.call(cached, path))
    .map((path) => ({ path, value: String(cached[path] ?? "").trim() }));
}

function syncScanCalibrationInputPresentation(binding, useMaster) {
  const input = $(binding?.inputId);
  if (!input) return;
  input.placeholder = useMaster ? binding.masterPlaceholder : binding.dirPlaceholder;
  input.title = useMaster ? binding.masterTitle : binding.dirTitle;
}

function syncScanCalibrationUiFromConfig(config) {
  SCAN_CALIBRATION_BINDINGS.forEach((binding) => {
    const toggleEl = $(binding.toggleId);
    const sourceEl = $(binding.sourceId);
    const inputEl = $(binding.inputId);
    if (!sourceEl || !inputEl) return;
    const enabled = Boolean(getByPath(config, binding.usePath));
    const useMaster = Boolean(getByPath(config, binding.useMasterPath));
    const pathValues = scanCalibrationStoredPaths(binding, config);
    if (Object.values(pathValues).some(Boolean)) {
      persistCalibrationPathCache(binding, pathValues);
    }
    if (toggleEl) writeFieldValue(toggleEl, enabled);
    writeFieldValue(sourceEl, useMaster);
    syncScanCalibrationInputPresentation(binding, useMaster);
    const activeValue = enabled ? getByPath(config, scanCalibrationActivePath(binding, useMaster)) : "";
    inputEl.value =
      activeValue === undefined || activeValue === null ? "" : String(activeValue);
  });
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

function canonicalInputDirsText(rawValue) {
  return parseInputDirs(rawValue).join(", ");
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

function clearUnifiedRunName() {
  persistTextValue(UI_STORAGE_KEYS.dashboardRunName, "");
  persistTextValue(UI_STORAGE_KEYS.wizardRunName, "");
  ["dashboard-run-name", "wizard-run-name", "scan-run-name"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    if (String(el.value || "") === "") return;
    el.value = "";
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

function maybeResetUnifiedRunNameOnInputDirsChange(previousValue, rawValue) {
  const next = canonicalInputDirsText(rawValue);
  if (!next) return;
  const prev = canonicalInputDirsText(previousValue);
  const hasRunName = Boolean(preferredStoredRunName());
  if (prev === next || !hasRunName) return;
  clearUnifiedRunName();
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

function canonicalQueueFilterLabel(raw) {
  const normalized = normalizeMonitorFilterName(raw);
  if (!normalized) return "";
  if (normalized === "HA") return "Ha";
  if (normalized === "OSC") return "OSC";
  if (normalized === "OIII") return "OIII";
  if (normalized === "SII") return "SII";
  if (["L", "R", "G", "B"].includes(normalized)) return normalized;
  return String(raw || "").trim();
}

function splitQueueFilterValue(raw) {
  const canonical = canonicalQueueFilterLabel(raw);
  if (!canonical) return { preset: "", custom: "" };
  return QUEUE_FILTER_PRESETS.includes(canonical)
    ? { preset: canonical, custom: "" }
    : { preset: "", custom: canonical };
}

function queueBlock(scope = document) {
  return scope.querySelector(".ps-queue-block");
}

function ensureQueueBody(scope = document) {
  const block = queueBlock(scope);
  if (!block) return null;
  let body = block.querySelector(".ps-queue-body");
  if (!body) {
    body = document.createElement("div");
    body.className = "ps-queue-body";
    Array.from(block.children)
      .filter((child) => child instanceof Element && child.classList.contains("ps-queue-row"))
      .forEach((child) => body.appendChild(child));
    block.appendChild(body);
  }
  return body;
}

function updateQueueRowToggleLabel(row) {
  if (!row) return;
  const checkbox = row.querySelector("[data-queue-field='enabled']");
  const label = row.querySelector(".ps-queue-toggle span");
  if (label) label.textContent = checkbox?.checked ? "on" : "off";
}

function createQueueRowElement(item = {}) {
  const row = document.createElement("div");
  row.className = "ps-queue-row";
  const filterParts = splitQueueFilterValue(item.filter || "");
  const inputDir = String(item.input_dir || "").trim();
  const pattern = String(item.pattern || "").trim();
  const runId = String(item.run_id || "").trim();
  const enabled = item.enabled !== false;
  row.innerHTML = `
    <div class="ps-queue-cell">
      <div class="ps-queue-filter-stack">
        <select class="ps-select" data-queue-field="filter-select" title="Vordefinierten Filter fuer Queue-Eintrag setzen.">
          ${QUEUE_FILTER_PRESETS.map((value) => `<option value="${value}">${value || "-"}</option>`).join("")}
        </select>
        <input class="ps-input" data-queue-field="filter-custom" type="text" value="" placeholder="frei" title="Beliebigen Filternamen fuer Queue-Eintrag setzen.">
      </div>
    </div>
    <div class="ps-queue-cell">
      <div class="ps-inline-cluster">
        <input class="ps-input" data-queue-field="input_dir" type="text" value="" placeholder="Verzeichnis waehlen" title="Input-Ordner fuer Queue-Eintrag setzen.">
        <button class="ps-btn ps-btn-secondary ps-btn-compact" data-queue-action="browse" type="button" title="Ordnerdialog fuer Queue-Eintrag oeffnen.">Browse…</button>
      </div>
    </div>
    <div class="ps-queue-cell"><input class="ps-input" data-queue-field="pattern" type="text" value="" placeholder="optional, z. B. *.fits" title="Optionales Pattern fuer Queue-Eintrag."></div>
    <div class="ps-queue-cell"><input class="ps-input" data-queue-field="run_id" type="text" value="" placeholder="optional" title="Optionales Run-Label/Subfolder fuer Queue-Eintrag."></div>
    <div class="ps-queue-toggle"><input type="checkbox" data-queue-field="enabled" title="Queue-Eintrag aktiv/inaktiv setzen."><span>on</span></div>
    <div class="ps-queue-actions">
      <button class="ps-btn ps-btn-secondary ps-btn-compact ps-queue-remove-btn" data-queue-action="remove" type="button" title="Queue-Eintrag entfernen.">-</button>
    </div>
  `;
  const select = row.querySelector("[data-queue-field='filter-select']");
  const custom = row.querySelector("[data-queue-field='filter-custom']");
  const input = row.querySelector("[data-queue-field='input_dir']");
  const patternInput = row.querySelector("[data-queue-field='pattern']");
  const runIdInput = row.querySelector("[data-queue-field='run_id']");
  const checkbox = row.querySelector("[data-queue-field='enabled']");
  if (select) select.value = filterParts.preset;
  if (custom) custom.value = filterParts.custom;
  if (input) input.value = inputDir;
  if (patternInput) patternInput.value = pattern;
  if (runIdInput) runIdInput.value = runId;
  if (checkbox) checkbox.checked = enabled;
  checkbox?.addEventListener("change", () => updateQueueRowToggleLabel(row));
  updateQueueRowToggleLabel(row);
  return row;
}

function renderQueueRows(items = [], scope = document) {
  const body = ensureQueueBody(scope);
  if (!body) return [];
  body.innerHTML = "";
  const rows = Array.isArray(items) && items.length > 0 ? items : [{}];
  rows.forEach((item) => body.appendChild(createQueueRowElement(item)));
  return Array.from(body.querySelectorAll(".ps-queue-row"));
}

function collectQueueDraftRows(scope = document) {
  const body = ensureQueueBody(scope);
  const rows = body ? Array.from(body.querySelectorAll(".ps-queue-row")) : [];
  return rows.map((row) => {
    const select = row.querySelector("[data-queue-field='filter-select']");
    const custom = row.querySelector("[data-queue-field='filter-custom']");
    const inputDir = row.querySelector("[data-queue-field='input_dir']");
    const pattern = row.querySelector("[data-queue-field='pattern']");
    const runId = row.querySelector("[data-queue-field='run_id']");
    const enabled = row.querySelector("[data-queue-field='enabled']");
    return {
      filter: String(custom?.value || "").trim() || canonicalQueueFilterLabel(select?.value || ""),
      input_dir: String(inputDir?.value || "").trim(),
      pattern: String(pattern?.value || "").trim(),
      run_id: String(runId?.value || "").trim(),
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
    renderQueueRows([], scope);
    return [];
  }
  renderQueueRows(rows, scope);
  return rows;
}

function persistQueueDraftRows(key, scope = document) {
  const items = collectQueueDraftRows(scope);
  const hasContent = items.some((item) => !item.enabled || item.filter || item.input_dir || item.pattern || item.run_id);
  if (!hasContent) {
    writeServerUiStateValue(key, "");
    return;
  }
  writeServerUiStateValue(key, JSON.stringify(items));
}

function bindQueueDraftPersistence(key, scope = document) {
  const block = queueBlock(scope);
  if (!block) return;
  restoreQueueDraftRows(key, scope);
  const persist = () => persistQueueDraftRows(key, scope);
  if (block.dataset.queuePersistenceBound === "1") {
    persist();
    return;
  }
  block.dataset.queuePersistenceBound = "1";
  block.addEventListener("input", persist);
  block.addEventListener("change", persist);
  block.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const removeBtn = target.closest("[data-queue-action='remove']");
    if (removeBtn) {
      const row = removeBtn.closest(".ps-queue-row");
      if (!row) return;
      row.remove();
      const remainingRows = Array.from(block.querySelectorAll(".ps-queue-row"));
      if (remainingRows.length === 0) renderQueueRows([], scope);
      persist();
      document.dispatchEvent(new CustomEvent("gui2:queue-changed"));
      return;
    }
    const browseBtn = target.closest("[data-queue-action='browse']");
    if (!browseBtn) return;
    const row = browseBtn.closest(".ps-queue-row");
    const input = row?.querySelector("[data-queue-field='input_dir']");
    if (!(input instanceof HTMLInputElement)) return;
    try {
      const chosen = await pickDirectoryPath(String(input.value || "").trim() || firstNonEmptyText(readServerUiStateValue(LAST_INPUT_DIRS_KEY), uiState.projectRunsDir, ""));
      if (!chosen) return;
      input.value = chosen;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    } catch (err) {
      setFooter(`Queue-Ordner konnte nicht gesetzt werden: ${errorText(err)}`, true);
    }
  });
  persist();
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

function deriveQueueLeafNames(queueItems = []) {
  const counts = new Map();
  return (Array.isArray(queueItems) ? queueItems : []).map((item, index) => {
    let leaf = sanitizeRunName(item?.filter || "");
    if (!leaf) leaf = `item-${index + 1}`;
    const nextCount = (counts.get(leaf) || 0) + 1;
    counts.set(leaf, nextCount);
    return nextCount > 1 ? `${leaf}-${nextCount}` : leaf;
  });
}

function buildRunPathPreview({
  runsDir = "",
  explicitRunName = "",
  fallbackRunName = "",
  queueItems = [],
} = {}) {
  const baseDir = String(runsDir || "").trim().replace(/\/+$/g, "");
  if (!baseDir) return "";

  const explicit = sanitizeRunName(explicitRunName);
  const queueLeaves = deriveQueueLeafNames(queueItems);
  if (queueLeaves.length > 0) {
    const root = explicit ? `${explicit}_${timestampSuffix()}` : dateStamp();
    if (queueLeaves.length === 1) return `${baseDir}/${root}/${queueLeaves[0]}`;
    return `${baseDir}/${root}/{${queueLeaves.join(", ")}}`;
  }

  const runName = explicit || sanitizeRunName(fallbackRunName);
  if (!runName) return "";
  return `${baseDir}/${runName}_${timestampSuffix()}`;
}

async function resolveConfigYamlForRun() {
  await pendingScanConfigSync.catch(() => {});
  const updates = collectParameterDirtyUpdates();
  if (updates.length === 0) {
    return await ensureConfigYaml();
  }
  const patched = await patchConfig({ updates, persist: false });
  const resolvedYaml = String(patched?.config_yaml || "");
  if (resolvedYaml) {
    uiState.configYaml = resolvedYaml;
    setConfigDraft(resolvedYaml);
    uiState.parameterDirty = {};
    clearParameterDirtyState();
    return resolvedYaml;
  }
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
  const queue = queueRowsForRunStart(normalizedSource);
  if (queue.length === 0 && inputDirs.length === 0) {
    throw new Error("Bitte mindestens einen Eingabeordner setzen.");
  }
  if (inputDirs.length > 0) {
    persistLastInputDirs(inputDirsText);
  }
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
  const useQueueNaming = queue.length > 0;
  const explicitRunName = useDashboardFields
    ? explicitRunNameValue("dashboard-run-name")
    : useWizardFields
      ? explicitRunNameValue("wizard-run-name")
      : sanitizeRunName(runNameEl?.value || "");
  const runName = useQueueNaming
    ? explicitRunName
    : useDashboardFields
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

  const astapBin = storedTextValue(UI_STORAGE_KEYS.astrometryBinary, { absolute: true });
  const astapDataDir = storedTextValue(UI_STORAGE_KEYS.astrometryDataDir, { absolute: true });
  
  const payload = {
    color_mode: colorMode,
    run_name: runName || undefined,
    runs_dir: runsDir || undefined,
    config_yaml: configYaml,
    astap_bin: astapBin || undefined,
    astap_data_dir: astapDataDir || undefined,
  };
  if (queue.length > 0) {
    payload.queue = queue;
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
  const perDirResults = Array.isArray(src.per_dir_results) ? src.per_dir_results : [];
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
    frames: Array.isArray(src.frames) ? src.frames : [],
    frames_total: Number(src.frames_total || src.frames_detected || 0),
    frames_truncated: Boolean(src.frames_truncated),
    color_mode: colorMode,
    color_mode_candidates: candidates,
    image_width: Number.isFinite(width) ? width : 0,
    image_height: Number.isFinite(height) ? height : 0,
    bayer_pattern: src.bayer_pattern ?? null,
    requires_user_confirmation: Boolean(src.requires_user_confirmation),
    errors,
    warnings,
    per_dir_results: perDirResults,
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

function scanSummaryInputDirs(summary) {
  const src = summary && typeof summary === "object" ? summary : {};
  const dirs = Array.isArray(src.input_dirs) ? src.input_dirs.map((x) => String(x || "").trim()).filter(Boolean) : [];
  if (dirs.length > 0) return dirs;
  const single = String(src.input_path || "").trim();
  return single ? [single] : [];
}

function sameInputDirSet(left, right) {
  const a = canonicalInputDirsText(Array.isArray(left) ? left.join(", ") : "");
  const b = canonicalInputDirsText(Array.isArray(right) ? right.join(", ") : "");
  return Boolean(a) && a === b;
}

function collectNonEmptyQueueItems(scope = document) {
  return collectQueueDraftRows(scope).filter((item) => {
    const inputDir = String(item?.input_dir || "").trim();
    const filter = String(item?.filter || "").trim();
    const pattern = String(item?.pattern || "").trim();
    const runId = String(item?.run_id || "").trim();
    return Boolean(inputDir || filter || pattern || runId || item?.enabled === false);
  });
}

function queueItemsForInputDirs(inputDirs, { selectedColorMode = "", summary = null } = {}) {
  const dirs = Array.isArray(inputDirs) ? inputDirs.map((x) => String(x || "").trim()).filter(Boolean) : [];
  if (dirs.length === 0) return [];
  const summaryMatches = summary && sameInputDirSet(dirs, scanSummaryInputDirs(summary));
  const detectedByDir = new Map();
  if (summaryMatches) {
    const perDir = Array.isArray(summary.per_dir_results) ? summary.per_dir_results : [];
    if (perDir.length > 0) {
      perDir.forEach((item) => {
        const inputDir = String(item?.input_path || item?.input_dir || "").trim();
        const mode = normalizeDetectedColorMode(item?.color_mode || "");
        if (inputDir && mode) detectedByDir.set(inputDir, mode);
      });
    } else {
      const mode = normalizeDetectedColorMode(summary.color_mode || "");
      if (mode) dirs.forEach((dir) => detectedByDir.set(dir, mode));
    }
  }
  const fallbackMode = normalizeDetectedColorMode(selectedColorMode);
  return dirs.map((dir) => {
    const detectedMode = detectedByDir.get(dir) || fallbackMode;
    return {
      filter: detectedMode === "OSC" ? "OSC" : "",
      input_dir: dir,
      pattern: "",
      run_id: "",
      enabled: true,
    };
  });
}

function mergeQueueItems(existingItems, additions) {
  const merged = [];
  const byDir = new Map();
  (Array.isArray(existingItems) ? existingItems : []).forEach((item) => {
    const normalized = {
      filter: String(item?.filter || "").trim(),
      input_dir: String(item?.input_dir || "").trim(),
      pattern: String(item?.pattern || "").trim(),
      run_id: String(item?.run_id || "").trim(),
      enabled: item?.enabled !== false,
    };
    if (!normalized.input_dir && !normalized.filter && !normalized.pattern && !normalized.run_id) return;
    merged.push(normalized);
    if (normalized.input_dir) byDir.set(normalized.input_dir, normalized);
  });
  (Array.isArray(additions) ? additions : []).forEach((item) => {
    const inputDir = String(item?.input_dir || "").trim();
    if (!inputDir) return;
    const existing = byDir.get(inputDir);
    if (existing) {
      if (!existing.filter && item.filter) existing.filter = String(item.filter);
      return;
    }
    const normalized = {
      filter: String(item?.filter || "").trim(),
      input_dir: inputDir,
      pattern: String(item?.pattern || "").trim(),
      run_id: String(item?.run_id || "").trim(),
      enabled: item?.enabled !== false,
    };
    merged.push(normalized);
    byDir.set(inputDir, normalized);
  });
  return merged;
}

async function addCurrentInputDirsToQueue({
  inputId,
  colorModeId,
  storageKey = "",
  scope = document,
} = {}) {
  const input = $(inputId);
  const dirs = parseInputDirs(String(input?.value || ""));
  if (dirs.length === 0) {
    setFooter("Bitte zuerst einen Eingabeordner waehlen und scannen.", true);
    return;
  }
  let summary = null;
  try {
    const latest = await api.get(API_ENDPOINTS.scan.latest);
    const parsed = summarizeScanResult(latest, dirs[0] || "");
    if (parsed?.has_scan) summary = parsed;
  } catch {
    summary = null;
  }
  const additions = queueItemsForInputDirs(dirs, {
    selectedColorMode: String($(colorModeId)?.value || "").trim(),
    summary,
  });
  const merged = mergeQueueItems(collectNonEmptyQueueItems(scope), additions);
  renderQueueRows(merged, scope);
  if (storageKey) persistQueueDraftRows(storageKey, scope);
  document.dispatchEvent(new CustomEvent("gui2:queue-changed"));
  setRunQueueVisible(String($(colorModeId)?.value || "").trim());
  setFooter(`${additions.length} Eingabeordner in die Run-Queue uebernommen.`);
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

function derivePccTempOutputPath(inputPath) {
  const stem = pathBaseName(inputPath).replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_") || "pcc";
  const base = String(appRuntime.tempRoot || "/tmp").trim().replace(/\\/g, "/").replace(/\/+$/, "");
  return `${base}/tile_compile_gui2/pcc/${stem}_${Date.now()}.fits`;
}

function derivePccTempStem(inputPath) {
  return pathBaseName(inputPath).replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_") || "pcc";
}

function isPccTempOutputPath(pathValue) {
  const normalized = String(pathValue || "").trim().replace(/\\/g, "/");
  const base = String(appRuntime.tempRoot || "/tmp").trim().replace(/\\/g, "/").replace(/\/+$/, "");
  return normalized.startsWith(`${base}/tile_compile_gui2/pcc/`) && /\.(fit|fits|fts)$/i.test(normalized);
}

function derivePccTempChannelPaths(outputPath) {
  const normalized = String(outputPath || "").trim();
  if (!isPccTempOutputPath(normalized)) return [];
  const dotIndex = normalized.lastIndexOf(".");
  const stem = dotIndex > 0 ? normalized.slice(0, dotIndex) : normalized;
  return [`${stem}_R.fit`, `${stem}_G.fit`, `${stem}_B.fit`];
}

function setCurrentPccTempArtifact({ outputRgb = "", outputChannels = [], jobId = "" } = {}) {
  const normalizedOutput = String(outputRgb || "").trim();
  if (!isPccTempOutputPath(normalizedOutput)) return false;
  const normalizedChannels = (Array.isArray(outputChannels) ? outputChannels : derivePccTempChannelPaths(normalizedOutput))
    .map((item) => String(item || "").trim())
    .filter(Boolean);
  uiState.currentPccTempOutput = normalizedOutput;
  uiState.currentPccTempChannels = normalizedChannels;
  uiState.currentPccTempJobId = String(jobId || "").trim();
  persistTextValue(UI_STORAGE_KEYS.pccTempOutput, normalizedOutput, { absolute: true });
  persistJsonValue(UI_STORAGE_KEYS.pccTempChannels, normalizedChannels);
  persistTextValue(UI_STORAGE_KEYS.pccTempJob, uiState.currentPccTempJobId);
  return true;
}

function clearCurrentPccTempArtifact() {
  uiState.currentPccTempOutput = "";
  uiState.currentPccTempChannels = [];
  uiState.currentPccTempJobId = "";
  persistTextValue(UI_STORAGE_KEYS.pccTempOutput, "");
  persistJsonValue(UI_STORAGE_KEYS.pccTempChannels, null);
  persistTextValue(UI_STORAGE_KEYS.pccTempJob, "");
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

async function pickFilePath(initialPath, filter = "") {
  if (typeof window.gui2PickPathValue === "function") {
    return window.gui2PickPathValue(initialPath, "file", filter);
  }
  const typed = window.prompt("Dateipfad eingeben", initialPath || "");
  return String(typed || "").trim() || null;
}

async function bindInputDirectoryPicker({ inputId, browseId }) {
  const input = $(inputId);
  if (!input) return;
  const setValue = (value) => {
    input.value = String(value || "").trim();
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const pick = async () => {
    const currentDirs = parseInputDirs(String(input.value || ""));
    const initial = currentDirs[0] || "";
    const chosen = await pickDirectoryPath(initial || firstNonEmptyText(readServerUiStateValue(LAST_INPUT_DIRS_KEY), ""));
    if (!chosen) return;
    setValue(chosen);
  };
  $(browseId)?.addEventListener("click", async () => {
    try {
      await pick();
    } catch (err) {
      setFooter(`Input-Ordner konnte nicht gesetzt werden: ${errorText(err)}`, true);
    }
  });
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

async function chooseFitsSavePath(defaultPath, { label = "Datei speichern" } = {}) {
  const normalizedDefault = String(defaultPath || "").trim();
  const defaultName = pathBaseName(normalizedDefault) || "output.fits";
  if (typeof window.gui2PickPathValue === "function") {
    const pickedPath = await window.gui2PickPathValue(normalizedDefault, {
      mode: "save-file",
      defaultFileName: defaultName,
      label,
    });
    return String(pickedPath || "").trim() || null;
  }
  const typedPath = window.prompt(label, normalizedDefault);
  return String(typedPath || "").trim() || null;
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
    const [guardrails, appState, appConstants] = await Promise.all([
      api.get(API_ENDPOINTS.guardrails.root),
      api.get(API_ENDPOINTS.app.state),
      api.get(API_ENDPOINTS.app.constants).catch(() => null),
    ]);
    const tempRoot = String(appConstants?.temp_root || "").trim();
    if (tempRoot) appRuntime.tempRoot = tempRoot;
    hydrateServerUiState(appState?.ui_state || {});
    setRunReady(guardrails?.status || "check", appState?.run?.current?.status || "");
    const rid = String(appState?.project?.current_run_id || "").trim();
    if (rid) setCurrentRunId(rid);
    else if (!String(uiState.currentRunId || "").trim()) clearCurrentRunId();
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

function bindUiStateNavigationFlush() {
  if (document.body?.dataset?.uiStateNavFlushBound === "1") return;
  if (document.body) document.body.dataset.uiStateNavFlushBound = "1";
  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const target = event.target;
    if (!(target instanceof Element)) return;
    const anchor = target.closest("a[href]");
    if (!(anchor instanceof HTMLAnchorElement)) return;
    if (anchor.target && anchor.target !== "_self") return;
    const href = String(anchor.getAttribute("href") || "").trim();
    if (!href || href.startsWith("#")) return;
    let url;
    try {
      url = new URL(anchor.href, window.location.href);
    } catch {
      return;
    }
    if (url.origin !== window.location.origin) return;
    const current = new URL(window.location.href);
    if (url.pathname === current.pathname && url.search === current.search && url.hash) return;
    if (!/\.html$/i.test(url.pathname.split("/").pop() || "")) return;
    event.preventDefault();
    void (async () => {
      try {
        await pendingScanConfigSync.catch(() => {});
        await flushServerUiState();
      } catch {
        // best effort before navigation
      }
      window.location.href = url.href;
    })();
  }, true);
}

document.addEventListener("gui2:locale-changed", () => {
  refreshRunReadyIndicators();
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
    el.dataset.lastCommittedInputDirs = canonicalInputDirsText(el.value);
    el.addEventListener("input", () => {
      maybeResetUnifiedRunNameOnInputDirsChange(el.dataset.lastCommittedInputDirs || "", el.value);
      persistLastInputDirs(el.value);
    });
    el.addEventListener("change", () => {
      maybeResetUnifiedRunNameOnInputDirsChange(el.dataset.lastCommittedInputDirs || "", el.value);
      persistLastInputDirs(el.value);
      el.dataset.lastCommittedInputDirs = canonicalInputDirsText(el.value);
    });
  });
}

function bindScanPages() {
  const queueStorageKey = activeQueueStorageKey();
  const queueColorModeId = pageName() === "wizard.html" ? "inp-colormode" : "inp-colormode";
  bindInputDirMemory("inp-dirs");
  bindQueueDraftPersistence(queueStorageKey);
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
  void bindInputDirectoryPicker({
    inputId: "inp-dirs",
    browseId: "inp-dirs-browse-btn",
  });
  void bindInputDirectoryPicker({
    inputId: "inp-dirs",
    browseId: "wizard-inp-dirs-browse-btn",
  });

  // Browse-Handler für Calibration-Felder
  SCAN_CALIBRATION_BINDINGS.forEach((binding) => {
    const browseId = `${binding.inputId.replace("-dir", "")}-browse`;
    const browseBtn = $(browseId);
    if (!browseBtn) return;
    browseBtn.addEventListener("click", async () => {
      try {
        const inputEl = $(binding.inputId);
        if (!inputEl) return;
        const useMaster = scanCalibrationUseMaster(binding);
        const current = String(inputEl.value || "").trim();
        const chosen = useMaster
          ? await pickFilePath(current, "*.fits;*.fit;*.fts;*.fits.fz")
          : await pickDirectoryPath(current);
        if (!chosen) return;
        inputEl.value = chosen;
        inputEl.dispatchEvent(new Event("input", { bubbles: true }));
        inputEl.dispatchEvent(new Event("change", { bubbles: true }));
      } catch (err) {
        setFooter(`Kalibrierpfad konnte nicht gesetzt werden: ${errorText(err)}`, true);
      }
    });
  });
  $("inp-dirs-add-btn")?.addEventListener("click", async () => {
    try {
      await addCurrentInputDirsToQueue({
        inputId: "inp-dirs",
        colorModeId: queueColorModeId,
        storageKey: queueStorageKey,
      });
    } catch (err) {
      setFooter(`Run-Queue konnte nicht erweitert werden: ${errorText(err)}`, true);
    }
  });
  $("wizard-inp-dirs-add-btn")?.addEventListener("click", async () => {
    try {
      await addCurrentInputDirsToQueue({
        inputId: "inp-dirs",
        colorModeId: queueColorModeId,
        storageKey: queueStorageKey,
      });
    } catch (err) {
      setFooter(`Run-Queue konnte nicht erweitert werden: ${errorText(err)}`, true);
    }
  });
  window.runScan = () => {
    void executeScanFlow();
  };
  const syncScanConfigField = async (el) => {
    const calibrationBinding = scanCalibrationBindingForElement(el);
    try {
      if (calibrationBinding) {
        const toggleChanged = String(el.id || "") === calibrationBinding.toggleId;
        const sourceChanged = String(el.id || "") === calibrationBinding.sourceId;
        const updates = [];
        if (toggleChanged) {
          const enabled = Boolean(readFieldValue(el));
          if (!enabled) {
            const currentPaths = scanCalibrationStoredPaths(calibrationBinding);
            persistCalibrationPathCache(calibrationBinding, currentPaths);
            updates.push({ path: calibrationBinding.usePath, value: false });
            updates.push({ path: calibrationBinding.dirPath, value: "" });
            updates.push({ path: calibrationBinding.masterPath, value: "" });
            const inputEl = $(calibrationBinding.inputId);
            if (inputEl) inputEl.value = "";
          } else {
            updates.push({ path: calibrationBinding.usePath, value: true });
            updates.push(...scanCalibrationRestoreUpdates(calibrationBinding));
          }
        } else if (sourceChanged) {
          const inputEl = $(calibrationBinding.inputId);
          const nextUseMaster = Boolean(readFieldValue(el));
          updates.push({
            path: calibrationBinding.useMasterPath,
            value: nextUseMaster,
          });
          syncScanCalibrationInputPresentation(calibrationBinding, nextUseMaster);
          if (inputEl) {
            const nextValue = scanCalibrationEnabled(calibrationBinding)
              ? currentDraftValueForPath(scanCalibrationActivePath(calibrationBinding, nextUseMaster))
              : "";
            inputEl.value = nextValue === undefined || nextValue === null ? "" : String(nextValue);
          }
        } else if (String(el.id || "") === calibrationBinding.inputId) {
          const newValue = String(readFieldValue(el) || "").trim();
          // Wenn eine einzelne FITS-Datei eingegeben wird, automatisch auf Master-Modus umschalten
          const isFitsFile = /\.(fits?|fts)(\.fz)?$/i.test(newValue);
          if (isFitsFile && !scanCalibrationUseMaster(calibrationBinding)) {
            const sourceEl = $(calibrationBinding.sourceId);
            if (sourceEl) writeFieldValue(sourceEl, true);
            syncScanCalibrationInputPresentation(calibrationBinding, true);
            updates.push({ path: calibrationBinding.useMasterPath, value: true });
          }
          const useMaster = isFitsFile ? true : scanCalibrationUseMaster(calibrationBinding);
          const activePath = useMaster ? calibrationBinding.masterPath : calibrationBinding.dirPath;
          updates.push({
            path: activePath,
            value: newValue,
          });
          persistCalibrationPathCache(calibrationBinding, {
            [activePath]: newValue,
          });
          // use_* automatisch setzen: true wenn Pfad gesetzt, false wenn leer
          if (calibrationBinding.usePath) {
            updates.push({
              path: calibrationBinding.usePath,
              value: Boolean(newValue),
            });
          }
        }
        if (updates.length === 0) return;
        rememberParameterDirtyUpdates(updates);
        const patched = await patchConfig({ updates, persist: false });
        if (patched?.config) {
          syncScanCalibrationUiFromConfig(patched.config);
        } else {
          syncScanCalibrationInputPresentation(
            calibrationBinding,
            scanCalibrationUseMaster(calibrationBinding),
          );
        }
        return;
      }
      const path = parameterPathFromElement(el);
      if (!path) return;
      const updates = [{ path, value: readFieldValue(el) }];
      rememberParameterDirtyUpdates(updates);
      await patchConfig({ updates, persist: false });
    } catch (err) {
      setFooter(`Input-Config-Update fehlgeschlagen: ${errorText(err)}`, true);
    }
  };
  document.querySelectorAll(".app-content [data-control]").forEach((el) => {
    const path = parameterPathFromElement(el);
    const calibrationBinding = scanCalibrationBindingForElement(el);
    if (!path && !calibrationBinding) return;
    el.addEventListener("input", () => {
      void enqueueScanConfigSync(() => syncScanConfigField(el));
    });
    el.addEventListener("change", () => {
      void enqueueScanConfigSync(() => syncScanConfigField(el));
    });
  });
  const colorModeEl = $("inp-colormode");
  if (colorModeEl) {
    const updateQueue = () => setRunQueueVisible(colorModeEl.value || "");
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
          persistLastInputDirs(mergedInputText);
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

function setStatusChip(el, text = "", tone = "check") {
  if (!el) return;
  const message = String(text || "").trim();
  el.textContent = message;
  el.style.display = message ? "inline-flex" : "none";
  if (!message) return;
  const variant = tone === "ok" ? "ok" : tone === "error" ? "error" : tone === "running" ? "running" : "check";
  el.className = `shell-status-chip shell-status-chip-${variant}`;
}

function updateTransferStatusChip(el, job, labels = {}) {
  if (!el) return;
  const {
    running = "Download läuft",
    extracting = "Entpacke",
    ok = "Download OK",
    cancelled = "Abgebrochen",
    error = "Download nicht OK",
  } = labels;
  const state = String(job?.state || "").trim().toLowerCase();
  const stage = String(job?.data?.stage || "").trim().toLowerCase();
  const pct = formatLogPercent(job?.data?.progress ?? job?.progress);
  if (["ok", "done", "completed", "finished"].includes(state)) {
    setStatusChip(el, ok, "ok");
    return;
  }
  if (["cancelled", "aborted"].includes(state)) {
    setStatusChip(el, cancelled, "check");
    return;
  }
  if (["error", "failed"].includes(state)) {
    setStatusChip(el, error, "error");
    return;
  }
  if (stage === "extract") {
    setStatusChip(el, extracting, "running");
    return;
  }
  const runningText = pct ? `${running} ${pct}` : running;
  setStatusChip(el, runningText, "running");
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

function formatI18n(key, fallback = "", replacements = {}) {
  let text = t(key, fallback);
  Object.entries(replacements || {}).forEach(([name, value]) => {
    text = text.replaceAll(`{${name}}`, String(value ?? ""));
  });
  return text;
}

function localizedRunMonitorState(stateRaw) {
  const state = String(stateRaw || "").trim().toLowerCase();
  if (state === "running") return t("ui.status.run_monitor_state_running", "Läuft");
  if (["ok", "completed", "done", "finished"].includes(state)) return t("ui.status.run_monitor_state_done", "Fertig");
  if (["error", "failed", "aborted"].includes(state)) return t("ui.status.run_monitor_state_error", "Fehler");
  if (state === "cancelled") return t("ui.status.run_monitor_state_cancelled", "Abgebrochen");
  if (state === "pending") return t("ui.status.run_monitor_state_pending", "Ausstehend");
  return t("ui.status.run_monitor_state_unknown", "Unbekannt");
}

function localizedRunMonitorPhaseName(phaseRaw) {
  const phase = String(phaseRaw || "").trim().toUpperCase();
  if (!phase) return "";
  return t(`phase.${phase.toLowerCase()}`, phase);
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
  const requestSeq = ++configPatchRequestSeq;
  const baseYaml = yamlText !== undefined ? String(yamlText || "") : await ensureConfigYaml();
  const result = await api.post(API_ENDPOINTS.config.patch, {
    yaml: baseYaml,
    updates,
    parse_values: true,
    persist,
  });
  const isLatestRequest = requestSeq === configPatchRequestSeq;
  if (isLatestRequest && result?.config_yaml) {
    setConfigDraft(result.config_yaml);
  }
  if (isLatestRequest && result?.config && typeof result.config === "object") {
    uiState.configObject = result.config;
  }
  if (!isLatestRequest) {
    return {
      ...result,
      stale: true,
      config_yaml: String(uiState.configYaml || result?.config_yaml || ""),
      config:
        uiState.configObject && typeof uiState.configObject === "object"
          ? uiState.configObject
          : result?.config,
    };
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

function splitSchemaYamlTopLevel(text, delimiter = ",") {
  const parts = [];
  let current = "";
  let depthCurly = 0;
  let depthSquare = 0;
  let inQuote = false;
  let quoteChar = "";
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const prev = i > 0 ? text[i - 1] : "";
    if ((ch === '"' || ch === "'") && prev !== "\\") {
      if (!inQuote) {
        inQuote = true;
        quoteChar = ch;
      } else if (quoteChar === ch) {
        inQuote = false;
        quoteChar = "";
      }
      current += ch;
      continue;
    }
    if (!inQuote) {
      if (ch === "{") depthCurly += 1;
      else if (ch === "}") depthCurly = Math.max(0, depthCurly - 1);
      else if (ch === "[") depthSquare += 1;
      else if (ch === "]") depthSquare = Math.max(0, depthSquare - 1);
      else if (ch === delimiter && depthCurly === 0 && depthSquare === 0) {
        if (current.trim()) parts.push(current.trim());
        current = "";
        continue;
      }
    }
    current += ch;
  }
  if (current.trim()) parts.push(current.trim());
  return parts;
}

function splitSchemaYamlKeyValue(text) {
  let depthCurly = 0;
  let depthSquare = 0;
  let inQuote = false;
  let quoteChar = "";
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const prev = i > 0 ? text[i - 1] : "";
    if ((ch === '"' || ch === "'") && prev !== "\\") {
      if (!inQuote) {
        inQuote = true;
        quoteChar = ch;
      } else if (quoteChar === ch) {
        inQuote = false;
        quoteChar = "";
      }
      continue;
    }
    if (inQuote) continue;
    if (ch === "{") depthCurly += 1;
    else if (ch === "}") depthCurly = Math.max(0, depthCurly - 1);
    else if (ch === "[") depthSquare += 1;
    else if (ch === "]") depthSquare = Math.max(0, depthSquare - 1);
    else if (ch === ":" && depthCurly === 0 && depthSquare === 0) {
      return [text.slice(0, i).trim(), text.slice(i + 1).trim()];
    }
  }
  return [text.trim(), ""];
}

function parseSchemaYamlScalar(rawValue) {
  const trimmed = String(rawValue || "").trim();
  if (!trimmed) return "";
  if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
    return trimmed.slice(1, -1);
  }
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (trimmed === "null") return null;
  if (/^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$/.test(trimmed)) return Number(trimmed);
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    const inner = trimmed.slice(1, -1).trim();
    if (!inner) return [];
    return splitSchemaYamlTopLevel(inner).map((part) => parseSchemaYamlScalar(part));
  }
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    const inner = trimmed.slice(1, -1).trim();
    const out = {};
    if (!inner) return out;
    splitSchemaYamlTopLevel(inner).forEach((part) => {
      const [key, value] = splitSchemaYamlKeyValue(part);
      if (!key) return;
      out[key] = parseSchemaYamlScalar(value);
    });
    return out;
  }
  return trimmed;
}

function parseSchemaYamlObject(lines, startIndex = 0, parentIndent = -1) {
  const out = {};
  let index = startIndex;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim() || /^\s*#/.test(line)) {
      index += 1;
      continue;
    }
    const indent = line.match(/^(\s*)/)[1].length;
    if (indent <= parentIndent) break;
    const match = line.match(/^\s*([A-Za-z0-9_]+):(?:\s*(.*))?$/);
    if (!match) {
      index += 1;
      continue;
    }
    const key = match[1];
    const rawRest = match[2] || "";
    if (rawRest.trim()) {
      out[key] = parseSchemaYamlScalar(rawRest);
      index += 1;
      continue;
    }
    const [child, nextIndex] = parseSchemaYamlObject(lines, index + 1, indent);
    out[key] = child;
    index = nextIndex;
  }
  return [out, index];
}

function parseConfigSchemaYaml(text) {
  try {
    const [parsed] = parseSchemaYamlObject(String(text || "").split(/\r?\n/), 0, -1);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

async function ensureConfigSchemaPaths() {
  if (uiState.configSchemaPaths instanceof Set) return uiState.configSchemaPaths;
  try {
    const [apiSchema, localSchema, localSchemaYamlText] = await Promise.all([
      api.get(API_ENDPOINTS.config.schema).catch(() => ({})),
      fetch("../tile_compile_cpp/tile_compile.schema.json")
        .then((response) => (response.ok ? response.json() : {}))
        .catch(() => ({})),
      fetch("../tile_compile_cpp/tile_compile.schema.yaml")
        .then((response) => (response.ok ? response.text() : ""))
        .catch(() => ""),
    ]);
    const localSchemaYaml = parseConfigSchemaYaml(localSchemaYamlText);
    const merged = new Set([
      ...flattenConfigSchemaPaths(apiSchema),
      ...flattenConfigSchemaPaths(localSchema),
      ...flattenConfigSchemaPaths(localSchemaYaml),
    ]);
    uiState.configSchemaPaths = merged.size > 0 ? merged : null;
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
  const staticRow = el.closest(".ps-row[data-path]:not(.ps-dyn-row)");
  if (staticRow) {
    const path = String(staticRow.getAttribute("data-path") || "");
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
  document.querySelectorAll(".ps-row[data-path]:not(.ps-dyn-row)").forEach((row) => {
    const path = String(row.getAttribute("data-path") || "");
    if (!path || !isKnownConfigSchemaPath(path)) return;
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
  const syncRenderedFields = () => {
    if (uiState.configObject && typeof uiState.configObject === "object") {
      syncParameterFieldsFromConfig(uiState.configObject);
    }
  };
  document.addEventListener("gui2:parameter-studio-rendered", syncRenderedFields);

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
  const selected = document.querySelector(".ps-phase-row.is-selected");
  return selected ? normalizeRunMonitorPhaseName(selected.dataset.phaseName || "") : "";
}

function createEmptyRunPhaseSnapshot() {
  const snapshot = {};
  RUN_MONITOR_PHASE_ORDER.forEach((phase) => {
    snapshot[phase] = { phase, status: "pending", pct: 0 };
  });
  return snapshot;
}

function normalizeRunMonitorPhaseName(raw) {
  return String(raw || "").trim().toUpperCase();
}

function normalizeRunMonitorPct(pctRaw) {
  let pct = Number(pctRaw || 0);
  if (Number.isFinite(pct) && pct <= 1.0) pct *= 100.0;
  if (!Number.isFinite(pct)) pct = 0;
  return Math.max(0, Math.min(100, pct));
}

function normalizedRunPhaseStatus(statusRaw) {
  const normalized = String(statusRaw || "pending").trim().toLowerCase();
  if (normalized === "ok" || normalized === "completed" || normalized === "done") return "done";
  if (normalized === "running") return "running";
  if (normalized === "skipped") return "skipped";
  if (normalized === "error" || normalized === "failed" || normalized === "aborted" || normalized === "cancelled") return "error";
  return "pending";
}

function phaseStateBadgeText(statusRaw) {
  const normalized = normalizedRunPhaseStatus(statusRaw);
  if (normalized === "done") return "OK";
  if (normalized === "running") return "RUN";
  if (normalized === "skipped") return "SKIP";
  if (normalized === "error") return "ERR";
  return "P";
}

function orderedRunPhaseEntries(snapshotRaw) {
  const snapshot = snapshotRaw && typeof snapshotRaw === "object" ? snapshotRaw : {};
  return RUN_MONITOR_PHASE_ORDER.map((phase) => {
    const entry = snapshot[phase];
    return {
      phase,
      status: entry?.status || "pending",
      pct: Number.isFinite(Number(entry?.pct)) ? Number(entry.pct) : 0,
    };
  });
}

function ensureRunPhaseSnapshot(runIdRaw) {
  const runId = normalizeRunIdPath(runIdRaw);
  if (!runId) return createEmptyRunPhaseSnapshot();
  if (!uiState.runPhaseSnapshots[runId]) {
    uiState.runPhaseSnapshots[runId] = createEmptyRunPhaseSnapshot();
  }
  return uiState.runPhaseSnapshots[runId];
}

function resetRunPhaseSnapshot(runIdRaw) {
  const runId = normalizeRunIdPath(runIdRaw);
  if (!runId) return createEmptyRunPhaseSnapshot();
  uiState.runPhaseSnapshots[runId] = createEmptyRunPhaseSnapshot();
  return uiState.runPhaseSnapshots[runId];
}

function setRunPhaseSnapshot(runIdRaw, phases) {
  const runId = normalizeRunIdPath(runIdRaw);
  if (!runId) return;
  const snapshot = resetRunPhaseSnapshot(runId);
  (Array.isArray(phases) ? phases : []).forEach((entry) => {
    const phaseName = normalizeRunMonitorPhaseName(entry?.phase);
    if (!phaseName || !snapshot[phaseName]) return;
    snapshot[phaseName] = {
      phase: phaseName,
      status: String(entry?.status || "pending").trim().toLowerCase() || "pending",
      pct: normalizeRunMonitorPct(entry?.pct ?? entry?.progress ?? 0),
    };
  });
}

function updateRunPhaseSnapshot(runIdRaw, phaseNameRaw, status, pctRaw) {
  const runId = normalizeRunIdPath(runIdRaw);
  const phaseName = normalizeRunMonitorPhaseName(phaseNameRaw);
  if (!runId || !phaseName) return;
  const snapshot = ensureRunPhaseSnapshot(runId);
  if (!snapshot[phaseName]) return;
  snapshot[phaseName] = {
    phase: phaseName,
    status: String(status || "pending").trim().toLowerCase() || "pending",
    pct: normalizeRunMonitorPct(pctRaw),
  };
}

function syntheticRunPhaseEntries(stateRaw) {
  const state = String(stateRaw || "pending").trim().toLowerCase();
  if (["ok", "completed", "done", "finished"].includes(state)) {
    return RUN_MONITOR_PHASE_ORDER.map((phase) => ({ phase, status: "done", pct: 100 }));
  }
  return RUN_MONITOR_PHASE_ORDER.map((phase) => ({ phase, status: "pending", pct: 0 }));
}

function findRunMonitorPhaseRow(runIdRaw, phaseNameRaw) {
  const runId = normalizeRunIdPath(runIdRaw);
  const phaseName = normalizeRunMonitorPhaseName(phaseNameRaw);
  return Array.from(document.querySelectorAll(".ps-phase-row")).find((row) => (
    normalizeRunIdPath(row.dataset.runId || "") === runId
    && normalizeRunMonitorPhaseName(row.dataset.phaseName || row.querySelector(".phase-name")?.textContent || "") === phaseName
  )) || null;
}

function applyPhaseRowState(row, status, pctRaw) {
  if (!row) return;
  row.classList.remove("done", "running", "pending", "error", "skipped");
  row.classList.add(normalizedRunPhaseStatus(status));
  const stateEl = row.querySelector(".state");
  if (stateEl) stateEl.textContent = phaseStateBadgeText(status);
  const pctEl = row.querySelector(".phase-progress");
  if (pctEl) pctEl.textContent = `${normalizeRunMonitorPct(pctRaw).toFixed(0)}%`;
}

function renderRunMonitorPhaseLists(selectedPhaseRaw = runMonitorSelectedPhase()) {
  const host = $("monitor-phase-lists");
  if (!host) return;
  const selectedPhase = normalizeRunMonitorPhaseName(selectedPhaseRaw);
  const queueItems = Array.isArray(uiState.currentRunQueue) ? uiState.currentRunQueue.filter((item) => item && typeof item === "object") : [];
  const currentRunId = normalizeRunIdPath(uiState.currentRunId);
  const groups = [];

  if (queueItems.length > 0) {
    queueItems.forEach((item, index) => {
      const runId = normalizeRunIdPath(item?.run_id || "");
      const label = canonicalQueueFilterLabel(item?.filter || "") || runIdLeaf(runId) || `${t("page.run_monitor.batch", "Batch")} ${index + 1}`;
      const state = String(item?.state || (runId === currentRunId ? uiState.runProcessStatus : "pending")).trim().toLowerCase() || "pending";
      const entries = runId && uiState.runPhaseSnapshots[runId]
        ? orderedRunPhaseEntries(uiState.runPhaseSnapshots[runId])
        : syntheticRunPhaseEntries(state);
      groups.push({
        key: runId || `batch-${index + 1}`,
        runId,
        label,
        meta: String(item?.input_dir || "").trim(),
        state,
        active: runId === currentRunId,
        entries,
      });
    });
  } else if (currentRunId || String(uiState.currentRunDir || "").trim() || String(uiState.runProcessStatus || "").trim()) {
    const entries = currentRunId && uiState.runPhaseSnapshots[currentRunId]
      ? orderedRunPhaseEntries(uiState.runPhaseSnapshots[currentRunId])
      : syntheticRunPhaseEntries(uiState.runProcessStatus || "pending");
    groups.push({
      key: currentRunId || "single-run",
      runId: currentRunId,
      label: currentRunId ? runIdLeaf(currentRunId) || currentRunId : t("page.run_monitor.current_run", "Aktueller Run"),
      meta: String(uiState.currentRunDir || "").trim(),
      state: String(uiState.runProcessStatus || "pending").trim().toLowerCase() || "pending",
      active: true,
      entries,
    });
  }

  if (groups.length === 0) {
    uiState.runMonitorSelectedBatchKey = "";
    host.innerHTML = `<div class="ps-note">${escapeRunMonitorHtml(t("ui.message.monitor_no_run_loaded", "Kein Run geladen."))}</div>`;
    return;
  }

  const selectedGroup = groups.find((group) => group.key === uiState.runMonitorSelectedBatchKey)
    || groups.find((group) => group.active)
    || groups[0];
  uiState.runMonitorSelectedBatchKey = selectedGroup?.key || "";

  const tabsHtml = groups.map((group) => {
    const activeTabClass = group.key === uiState.runMonitorSelectedBatchKey ? " active" : "";
    const currentRunClass = group.active ? " current-run" : "";
    const title = group.meta || group.runId || group.label;
    return `
      <button
        type="button"
        class="ps-phase-tab${activeTabClass}${currentRunClass}"
        data-batch-key="${escapeRunMonitorAttr(group.key)}"
        title="${escapeRunMonitorAttr(title)}"
      >
        <span class="ps-phase-tab-label">${escapeRunMonitorHtml(group.label)}</span>
      </button>
    `;
  }).join("");

  const batchStateClass = queueItemStateClass(selectedGroup.state);
  const rowsHtml = selectedGroup.entries.map((entry) => {
    const rowStateClass = normalizedRunPhaseStatus(entry.status);
    const selectedClass = selectedGroup.active && selectedPhase && entry.phase === selectedPhase ? " is-selected" : "";
    const readonlyClass = selectedGroup.active ? "" : " is-readonly";
    const title = selectedGroup.active
      ? formatI18n("ui.message.monitor_resume_from_phase", "Resume ab {phase} starten.", { phase: localizedRunMonitorPhaseName(entry.phase) || entry.phase })
      : t("ui.message.monitor_batch_progress", "Fortschritt dieses Batches.");
    return `
      <button
        type="button"
        class="ps-phase-row ${rowStateClass}${selectedClass}${readonlyClass}"
        data-phase-name="${escapeRunMonitorAttr(entry.phase)}"
        data-run-id="${escapeRunMonitorAttr(selectedGroup.runId || "")}"
        data-active-batch="${selectedGroup.active ? "1" : "0"}"
        title="${escapeRunMonitorAttr(title)}"
      >
        <span class="state">${phaseStateBadgeText(entry.status)}</span>
        <span class="phase-name">${escapeRunMonitorHtml(localizedRunMonitorPhaseName(entry.phase) || entry.phase)}</span>
        <span class="phase-progress" data-control="monitor.phase.progress_pct" title="${escapeRunMonitorAttr(t("ui.tooltip.monitor.phase_progress_pct", "Prozentwert für den aktuellen Fortschritt dieser Phase"))}">${normalizeRunMonitorPct(entry.pct).toFixed(0)}%</span>
      </button>
    `;
  }).join("");

  host.innerHTML = `
    <div class="ps-phase-tabs" role="tablist" aria-label="${escapeRunMonitorAttr(t("page.run_monitor.batch_selection", "Batch-Auswahl"))}">
      ${tabsHtml}
    </div>
    <section class="ps-phase-batch${selectedGroup.active ? " active" : ""}" data-run-id="${escapeRunMonitorAttr(selectedGroup.runId || "")}">
      <div class="ps-phase-batch-header">
        <div class="ps-phase-batch-title">
          <div class="ps-phase-batch-name">${escapeRunMonitorHtml(selectedGroup.label)}</div>
          <div class="ps-phase-batch-meta">${escapeRunMonitorHtml(selectedGroup.meta || selectedGroup.runId || "-")}</div>
        </div>
        <span class="ps-phase-batch-state ${batchStateClass}">${escapeRunMonitorHtml(localizedRunMonitorState(selectedGroup.state || "pending"))}</span>
      </div>
      <div class="ps-phase-list">${rowsHtml}</div>
    </section>
  `;
  applyRunMonitorResumePhaseAvailability(uiState.runMonitorResumePhases);
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
      : ["ASTROMETRY", "BGE", "PCC", "HYPERMETRIC_STRETCH"],
  );
  uiState.runMonitorResumePhases = Array.from(allowed);
  document.querySelectorAll(".ps-phase-row").forEach((row) => {
    const phaseName = normalizeRunMonitorPhaseName(row.dataset.phaseName || "");
    if (String(row.dataset.activeBatch || "") !== "1") {
      row.dataset.resumeAllowed = "0";
      row.classList.remove("is-selected");
      row.classList.add("is-readonly");
      row.style.opacity = "";
      row.style.cursor = "default";
      row.title = t("ui.message.monitor_batch_progress", "Fortschritt dieses Batches.");
      return;
    }
    const resumable = allowed.has(phaseName);
    row.dataset.resumeAllowed = resumable ? "1" : "0";
    row.classList.remove("is-readonly");
    if (!resumable) {
      row.classList.remove("is-selected");
      row.style.opacity = "0.6";
      row.style.cursor = "not-allowed";
      row.title = formatI18n(
        "ui.message.monitor_resume_supported_from",
        "Resume aktuell nur ab {phases} unterstützt.",
        { phases: Array.from(allowed).map((phase) => localizedRunMonitorPhaseName(phase) || phase).join(", ") },
      );
      return;
    }
    row.style.opacity = "";
    row.style.cursor = "pointer";
  });
}

function runMonitorSelectedFilter() {
  const chipRow = $("monitor-filter-row");
  if (chipRow && chipRow.style.display === "none") return "";
  const selected = chipRow?.querySelector(".ps-chip-btn.active");
  if (!selected) return "";
  return String(selected.textContent || "").trim().toUpperCase();
}

function runMonitorFilterButtons() {
  const chipRow = $("monitor-filter-row");
  return chipRow ? Array.from(chipRow.querySelectorAll(".ps-chip-btn")) : [];
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

function collectRunMonitorFilterEntries(queueItemsRaw = null) {
  const source = Array.isArray(queueItemsRaw) ? queueItemsRaw : collectQueueRows();
  const grouped = new Map();
  for (const item of source) {
    const rawFilter = typeof item === "string" ? item : item?.filter || item?.filter_name || "";
    const token = normalizeMonitorFilterName(rawFilter);
    if (!token) continue;
    const label = canonicalQueueFilterLabel(rawFilter) || token;
    if (!grouped.has(token)) {
      grouped.set(token, {
        token,
        label,
        total: 0,
        done: 0,
        running: 0,
        pending: 0,
        error: 0,
        cancelled: 0,
      });
    }
    const entry = grouped.get(token);
    if (typeof item === "string") {
      entry.total += 1;
      entry.pending += 1;
      continue;
    }
    const explicitTotal = Number(item?.total);
    const explicitDone = Number(item?.done);
    if (Number.isFinite(explicitTotal) && explicitTotal > 0) {
      entry.total += explicitTotal;
      if (Number.isFinite(explicitDone) && explicitDone >= 0) {
        entry.done += explicitDone;
      }
      const state = String(item?.state || "pending").trim().toLowerCase();
      if (state === "ok" || state === "completed" || state === "done") entry.done += 0;
      else if (state === "running") entry.running += 1;
      else if (state === "error" || state === "failed") entry.error += 1;
      else if (state === "cancelled") entry.cancelled += 1;
      else entry.pending += 1;
      continue;
    }
    entry.total += 1;
    const state = String(item?.state || "pending").trim().toLowerCase();
    if (state === "ok" || state === "completed" || state === "done") entry.done += 1;
    else if (state === "running") entry.running += 1;
    else if (state === "error" || state === "failed") entry.error += 1;
    else if (state === "cancelled") entry.cancelled += 1;
    else entry.pending += 1;
  }
  return Array.from(grouped.values()).map((entry) => {
    let state = "pending";
    if (entry.error > 0) state = "error";
    else if (entry.cancelled > 0) state = "cancelled";
    else if (entry.done > 0 && entry.done >= entry.total && entry.total > 0) state = "done";
    else if (entry.running > 0 || entry.done > 0) state = "running";
    return {
      filter: entry.label,
      state,
      done: entry.done,
      total: entry.total,
    };
  });
}

function ensureRunMonitorFilterButtons(filters) {
  const chipRow = $("monitor-filter-row");
  if (!chipRow) return [];
  const previous = normalizeMonitorFilterName(runMonitorSelectedFilter());
  const normalizedFilters = [];
  const seen = new Set();
  for (const raw of Array.isArray(filters) ? filters : []) {
    const rawFilter = typeof raw === "string" ? raw : raw?.filter || raw?.filter_name || "";
    const token = normalizeMonitorFilterName(rawFilter);
    if (!token || seen.has(token)) continue;
    seen.add(token);
    normalizedFilters.push({
      filter: canonicalQueueFilterLabel(rawFilter) || token,
      state: typeof raw === "string" ? "pending" : String(raw?.state || "pending").trim().toLowerCase(),
    });
  }
  chipRow.innerHTML = "";
  normalizedFilters.forEach((entry, index) => {
    const filter = entry.filter;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ps-chip-btn";
    btn.id = `monitor-filter-${normalizeMonitorFilterName(filter) || index}`;
    btn.dataset.control = `monitor.filter.${filter}`;
    btn.dataset.state = entry.state || "pending";
    btn.title = formatI18n("ui.message.monitor_filter_context_select", "Filterkontext {filter} wählen.", { filter });
    btn.textContent = filter;
    if (entry.state === "ok" || entry.state === "completed" || entry.state === "done") {
      btn.classList.add("done");
    } else if (entry.state === "running") {
      btn.classList.add("running");
    } else if (entry.state === "error" || entry.state === "failed") {
      btn.classList.add("error");
    }
    if ((previous && normalizeMonitorFilterName(filter) === previous) || (!previous && index === 0)) {
      btn.classList.add("active");
    }
    chipRow.appendChild(btn);
  });
  return runMonitorFilterButtons();
}

function setRunMonitorFilterVisibility(colorModeRaw, queueItemsRaw = null) {
  const chipRow = $("monitor-filter-row");
  if (!chipRow) return;
  const hasActiveRun = Boolean(String(uiState.currentRunId || "").trim());
  if (!hasActiveRun && !Array.isArray(queueItemsRaw)) {
    chipRow.innerHTML = "";
    chipRow.style.display = "none";
    return;
  }
  const filterEntries = collectRunMonitorFilterEntries(queueItemsRaw);
  const chipButtons = ensureRunMonitorFilterButtons(filterEntries);
  const hideFilters = filterEntries.length === 0;
  chipRow.style.display = hideFilters ? "none" : "";
  if (hideFilters) {
    return;
  }
  if (!chipRow.querySelector(".ps-chip-btn.active")) {
    chipButtons[0]?.classList.add("active");
  }
}

function escapeRunMonitorHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeRunMonitorAttr(text) {
  return escapeRunMonitorHtml(text).replaceAll("\"", "&quot;");
}

function normalizeRunIdPath(raw) {
  return String(raw || "")
    .trim()
    .replace(/\\/g, "/")
    .replace(/^\/+|\/+$/g, "");
}

function normalizeFsPath(raw) {
  return String(raw || "")
    .trim()
    .replace(/\\/g, "/")
    .replace(/\/+$/g, "");
}

function queueItemStateClass(stateRaw) {
  const state = String(stateRaw || "pending").trim().toLowerCase();
  if (["ok", "completed", "done"].includes(state)) return "done";
  if (["error", "failed", "aborted"].includes(state)) return "error";
  if (state === "running") return "running";
  return "pending";
}

function basenameSafe(pathRaw) {
  const normalized = String(pathRaw || "").trim().replace(/\\/g, "/").replace(/\/+$/g, "");
  if (!normalized) return "";
  const parts = normalized.split("/").filter(Boolean);
  return parts[parts.length - 1] || normalized;
}

function runIdParentPath(runIdRaw) {
  const normalized = normalizeRunIdPath(runIdRaw);
  if (!normalized || !normalized.includes("/")) return "";
  return normalized.split("/").slice(0, -1).join("/");
}

function runIdLeaf(runIdRaw) {
  const normalized = normalizeRunIdPath(runIdRaw);
  if (!normalized) return "";
  const parts = normalized.split("/").filter(Boolean);
  return parts[parts.length - 1] || normalized;
}

function selectedRunMonitorBatchItem() {
  const queueItems = Array.isArray(uiState.currentRunQueue) ? uiState.currentRunQueue.filter((item) => item && typeof item === "object") : [];
  if (queueItems.length === 0) return null;
  const selectedKey = String(uiState.runMonitorSelectedBatchKey || "").trim();
  if (selectedKey) {
    for (let index = 0; index < queueItems.length; index += 1) {
      const item = queueItems[index];
      const runId = normalizeRunIdPath(item?.run_id || "");
      const key = runId || `batch-${index + 1}`;
      if (key === selectedKey) return item;
    }
  }
  const currentRunId = normalizeRunIdPath(uiState.currentRunId);
  if (currentRunId) {
    const activeItem = queueItems.find((item) => normalizeRunIdPath(item?.run_id || "") === currentRunId);
    if (activeItem) return activeItem;
  }
  return queueItems[0] || null;
}

function runMonitorTargetContext() {
  const batchItem = selectedRunMonitorBatchItem();
  if (batchItem) {
    const runId = normalizeRunIdPath(batchItem?.run_id || "");
    return {
      runId,
      runDir: runDirForRunId(uiState.currentRunDir, uiState.currentRunId, runId),
      state: String(batchItem?.state || (runId === normalizeRunIdPath(uiState.currentRunId) ? uiState.runProcessStatus : "pending")).trim().toLowerCase() || "pending",
      isSelectedBatch: true,
      isActiveRun: runId === normalizeRunIdPath(uiState.currentRunId),
    };
  }
  return {
    runId: normalizeRunIdPath(uiState.currentRunId),
    runDir: normalizeFsPath(uiState.currentRunDir),
    state: String(uiState.runProcessStatus || "").trim().toLowerCase(),
    isSelectedBatch: false,
    isActiveRun: true,
  };
}

function isTerminalRunStatus(statusRaw) {
  const status = String(statusRaw || "").trim().toLowerCase();
  return ["ok", "done", "completed", "finished", "error", "failed", "cancelled", "aborted"].includes(status);
}

function runDirForRunId(runDirRaw, currentRunIdRaw, targetRunIdRaw) {
  const runDir = normalizeFsPath(runDirRaw);
  const currentRunId = normalizeRunIdPath(currentRunIdRaw);
  const targetRunId = normalizeRunIdPath(targetRunIdRaw);
  if (!runDir || !targetRunId) return "";
  if (!currentRunId) return runDir;
  const suffix = `/${currentRunId}`;
  if (runDir.endsWith(suffix)) return `${runDir.slice(0, -suffix.length)}/${targetRunId}`;
  return runDir;
}

function queueRootRunId(queueItemsRaw = [], fallbackRunId = "") {
  const parents = Array.from(new Set(
    (Array.isArray(queueItemsRaw) ? queueItemsRaw : [])
      .map((item) => runIdParentPath(item?.run_id || ""))
      .filter(Boolean),
  ));
  if (parents.length === 1) return parents[0];
  const fallback = normalizeRunIdPath(fallbackRunId);
  if (parents.includes(fallback)) return fallback;
  const fallbackParent = runIdParentPath(fallback);
  if (fallbackParent && parents.includes(fallbackParent)) return fallbackParent;
  return parents[0] || fallbackParent || fallback;
}

function queueActiveItem(queueItemsRaw = [], { currentIndex = -1, fallbackRunId = "" } = {}) {
  const queueItems = Array.isArray(queueItemsRaw) ? queueItemsRaw.filter((item) => item && typeof item === "object") : [];
  if (!queueItems.length) return null;
  const runningItem = queueItems.find((item) => String(item?.state || "").trim().toLowerCase() === "running");
  if (runningItem) return runningItem;
  if (Number.isInteger(currentIndex) && currentIndex >= 0 && currentIndex < queueItems.length) return queueItems[currentIndex];
  const normalizedFallback = normalizeRunIdPath(fallbackRunId);
  if (normalizedFallback) {
    const exactItem = queueItems.find((item) => normalizeRunIdPath(item?.run_id || "") === normalizedFallback);
    if (exactItem) return exactItem;
  }
  return queueItems[0] || null;
}

function queueActiveRunId(queueItemsRaw = [], { currentIndex = -1, fallbackRunId = "" } = {}) {
  return String(queueActiveItem(queueItemsRaw, { currentIndex, fallbackRunId })?.run_id || "").trim();
}

function renderRunMonitorSummary(runId, runStatus, queueItemsRaw = null, runDirRaw = "") {
  const section = $("monitor-run-summary");
  const metaEl = $("monitor-run-summary-meta");
  const structureEl = $("monitor-run-structure");
  if (!section || !metaEl || !structureEl) return;

  const queueItems = Array.isArray(queueItemsRaw) ? queueItemsRaw.filter((item) => item && typeof item === "object") : [];
  const normalizedRunId = normalizeRunIdPath(runId);
  const hasSingleRun = Boolean(normalizedRunId || String(runDirRaw || "").trim() || String(runStatus || "").trim());
  if (!queueItems.length && !hasSingleRun) {
    section.hidden = true;
    metaEl.innerHTML = "";
    structureEl.innerHTML = "";
    return;
  }

  section.hidden = false;
  if (!queueItems.length) {
    const runDir = String(runDirRaw || "").trim();
    metaEl.innerHTML = `
      <strong>${escapeRunMonitorHtml(t("ui.nav.run_monitor", "Run Monitor"))}</strong><span><code>${escapeRunMonitorHtml(normalizedRunId || "-")}</code></span>
      <strong>${escapeRunMonitorHtml(t("ui.field.status", "Status"))}</strong><span><code>${escapeRunMonitorHtml(localizedRunMonitorState(runStatus || "unknown"))}</code></span>
      <strong>${escapeRunMonitorHtml(t("page.run_monitor.directory", "Verzeichnis"))}</strong><span><code>${escapeRunMonitorHtml(runDir || "-")}</code></span>
      <strong>${escapeRunMonitorHtml(t("page.run_monitor.batch_progress", "Batch-Fortschritt"))}</strong><span>1/1</span>
      <strong>${escapeRunMonitorHtml(t("queue.filter.title", "Filter"))}</strong><span>-</span>
      <strong>${escapeRunMonitorHtml(t("page.run_monitor.source", "Quelle"))}</strong><span>-</span>
    `;
    structureEl.innerHTML = `
      <div class="ps-monitor-run-node"><strong>${escapeRunMonitorHtml(t("ui.nav.run_monitor", "Run Monitor"))}</strong><code>${escapeRunMonitorHtml(normalizedRunId || "-")}</code></div>
      <div class="ps-monitor-run-node"><strong>${escapeRunMonitorHtml(t("ui.field.status", "Status"))}</strong><code>${escapeRunMonitorHtml(localizedRunMonitorState(runStatus || "unknown"))}</code></div>
      <div class="ps-monitor-run-node"><strong>${escapeRunMonitorHtml(t("page.run_monitor.run_path", "Run-Pfad"))}</strong><code>${escapeRunMonitorHtml(runDir || "-")}</code></div>
    `;
    return;
  }

  const activeItem = queueActiveItem(queueItems, { fallbackRunId: normalizedRunId });
  const activeItemRunId = normalizeRunIdPath(activeItem?.run_id || normalizedRunId);
  const doneCount = queueItems.filter((item) => ["ok", "completed", "done"].includes(String(item?.state || "").trim().toLowerCase())).length;
  const rootRunId = queueRootRunId(queueItems, activeItemRunId || normalizedRunId);
  const activeFilter = canonicalQueueFilterLabel(activeItem?.filter || "") || runIdLeaf(activeItemRunId || normalizedRunId) || "-";
  const activeInputDir = String(activeItem?.input_dir || "").trim();
  const activeRunDir = runDirForRunId(runDirRaw, normalizedRunId || activeItemRunId, activeItemRunId || normalizedRunId);
  const rootRunDir = rootRunId ? runDirForRunId(runDirRaw, normalizedRunId || activeItemRunId, rootRunId) : "";

  metaEl.innerHTML = `
    <strong>${escapeRunMonitorHtml(t("page.run_monitor.root", "Root"))}</strong><span><code>${escapeRunMonitorHtml(rootRunId || "-")}</code></span>
    <strong>${escapeRunMonitorHtml(t("page.run_monitor.root_dir", "Root Dir"))}</strong><span><code>${escapeRunMonitorHtml(rootRunDir || "-")}</code></span>
    <strong>${escapeRunMonitorHtml(t("page.run_monitor.active", "Aktiv"))}</strong><span><code>${escapeRunMonitorHtml(activeItemRunId || normalizedRunId || "-")}</code></span>
    <strong>${escapeRunMonitorHtml(t("ui.field.status", "Status"))}</strong><span><code>${escapeRunMonitorHtml(localizedRunMonitorState(runStatus || "unknown"))}</code></span>
    <strong>${escapeRunMonitorHtml(t("page.run_monitor.batch_progress", "Batch-Fortschritt"))}</strong><span>${doneCount}/${queueItems.length}</span>
    <strong>${escapeRunMonitorHtml(t("queue.filter.title", "Filter"))}</strong><span>${escapeRunMonitorHtml(activeFilter || "-")}</span>
    <strong>${escapeRunMonitorHtml(t("page.run_monitor.source", "Quelle"))}</strong><span>${escapeRunMonitorHtml(activeInputDir || "-")}</span>
  `;

  const structureNodes = [];
  if (rootRunId) {
    structureNodes.push(`<div class="ps-monitor-run-node"><strong>${escapeRunMonitorHtml(t("page.run_monitor.root", "Root"))}</strong><code>${escapeRunMonitorHtml(rootRunId)}</code></div>`);
  }
  if (rootRunDir) {
    structureNodes.push(`<div class="ps-monitor-run-node"><strong>${escapeRunMonitorHtml(t("page.run_monitor.root_path", "Root-Pfad"))}</strong><code>${escapeRunMonitorHtml(rootRunDir)}</code></div>`);
  }
  queueItems.forEach((item, index) => {
    const itemRunId = normalizeRunIdPath(item?.run_id || "");
    const itemLabel = canonicalQueueFilterLabel(item?.filter || "") || runIdLeaf(itemRunId) || `${t("page.run_monitor.batch", "Batch")} ${index + 1}`;
    const stateLabel = localizedRunMonitorState(String(item?.state || "pending").trim() || "pending");
    const nodeLabel = itemRunId === activeItemRunId
      ? formatI18n("page.run_monitor.active_batch_label", "Aktiv {label}", { label: itemLabel })
      : formatI18n("page.run_monitor.batch_state_label", "{label} ({state})", { label: itemLabel, state: stateLabel });
    structureNodes.push(
      `<div class="ps-monitor-run-node"><strong>${escapeRunMonitorHtml(nodeLabel)}</strong><code>${escapeRunMonitorHtml(itemRunId || "-")}</code></div>`,
    );
  });
  if (activeRunDir) {
    structureNodes.push(`<div class="ps-monitor-run-node"><strong>${escapeRunMonitorHtml(t("page.run_monitor.active_path", "Aktiver Pfad"))}</strong><code>${escapeRunMonitorHtml(activeRunDir)}</code></div>`);
  }
  structureEl.innerHTML = structureNodes.join("");
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

function setRunMonitorLogLines(lines = []) {
  uiState.runLogPending = [];
  uiState.runLogLines = (Array.isArray(lines) ? lines : [])
    .map((line) => String(line || "").trim())
    .filter(Boolean)
    .slice(-300);
  const logBox = runMonitorLogBox();
  if (!logBox) return;
  logBox.textContent = uiState.runLogLines.join("\n");
  scrollLogToEnd(logBox);
}

function structuredRunMonitorLogLines(entries) {
  return (Array.isArray(entries) ? entries : [])
    .map((entry) => formatStructuredLogLine(entry, { suppressRunStatus: true }) || String(entry || "").trim())
    .filter(Boolean)
    .slice(-300);
}

async function loadRunMonitorLogs(runId, fallbackEntries = []) {
  if (!runId) {
    setRunMonitorLogLines([]);
    return;
  }
  let lines = [];
  try {
    const logs = await api.get(API_ENDPOINTS.runs.logs(runId, 250));
    lines = structuredRunMonitorLogLines(logs?.lines || []);
  } catch {
    lines = [];
  }
  if (lines.length === 0) {
    lines = structuredRunMonitorLogLines(fallbackEntries);
  }
  setRunMonitorLogLines(lines);
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
  if (!targetPath) throw new Error(t("ui.message.monitor_path_missing", "Pfad fehlt."));
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
    throw new Error(t("ui.message.monitor_report_unavailable", "Report nicht verfuegbar."));
  }
  const reportUrl = api.httpUrl(API_ENDPOINTS.runs.artifactRaw(runId, artifactPath));
  const targetWindow = window.open(reportUrl, "_blank");
  if (!targetWindow) {
    throw new Error(t("ui.message.monitor_report_open_failed", "Report konnte nicht in neuem Tab geoeffnet werden."));
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
  updateRunPhaseSnapshot(uiState.currentRunId, phaseName, status, pctRaw);
  const row = findRunMonitorPhaseRow(uiState.currentRunId, phaseName);
  if (!row) return;
  applyPhaseRowState(row, status, pctRaw);
}

function updateRunMonitorSubtitle(runId, runStatus, currentPhase) {
  const sub = document.querySelector(".app-content .ps-sub");
  if (!sub) return;
  sub.innerHTML = formatI18n(
    "page.run_monitor.sub_status",
    "Run-ID <code>{run_id}</code>, Status <code>{status}</code>, Phase <code>{phase}</code>.",
    {
      run_id: escapeRunMonitorHtml(runId || "-"),
      status: escapeRunMonitorHtml(localizedRunMonitorState(runStatus || "unknown")),
      phase: escapeRunMonitorHtml(localizedRunMonitorPhaseName(currentPhase || "") || currentPhase || "-"),
    },
  );
}

async function loadRunStatus(runId) {
  const status = await api.get(API_ENDPOINTS.runs.status(runId));
  uiState.currentRunDir = String(status?.run_dir || "");
  uiState.currentRunQueue = Array.isArray(status?.queue) ? status.queue : [];
  uiState.runProcessStatus = String(status?.status || "").trim().toLowerCase();
  const fallbackScanColorMode = String(readServerUiStateValue(LAST_SCAN_COLOR_MODE_KEY) || "").trim().toUpperCase();
  const effectiveColorMode = String(status?.color_mode || "").trim().toUpperCase() || fallbackScanColorMode;
  uiState.currentRunColorMode = effectiveColorMode;
  setRunMonitorFilterVisibility(effectiveColorMode, Array.isArray(status?.queue_filters) ? status.queue_filters : null);
  renderRunMonitorSummary(runId, status?.status || "unknown", uiState.currentRunQueue, uiState.currentRunDir);
  setRunPhaseSnapshot(runId, status?.phases);
  renderRunMonitorPhaseLists();
  updateRunMonitorSubtitle(runId, status.status, status.current_phase);
  if (!isRunActiveStatus(status?.status || "")) {
    await loadRunMonitorLogs(runId, status?.events || []);
  }
  return status;
}

function preprocessingMonitorJobId() {
  const params = new URLSearchParams(window.location.search || "");
  return String(params.get("preprocessing_job_id") || "").trim();
}

function preprocessingRunIdFromStatus(status, fallbackJobId = "") {
  return normalizeRunIdPath(
    status?.job?.data?.run_id ||
    status?.job?.run_id ||
    localStorage.getItem(PREPROCESSING_RUN_ID_KEY) ||
    fallbackJobId ||
    "",
  );
}

function preprocessingRunDirFromStatus(status) {
  return String(
    status?.job?.data?.run_dir ||
    localStorage.getItem(PREPROCESSING_RUN_DIR_KEY) ||
    "",
  ).trim();
}

async function loadPreprocessingMonitorLogs(runId) {
  if (!runId) return;
  try {
    const payload = await api.get(API_ENDPOINTS.runs.artifactView(runId, "artifacts/preprocess/events.jsonl"));
    const text = String(payload?.text || "");
    const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line) => {
      try {
        return formatStructuredLogLine(JSON.parse(line), { suppressRunStatus: true }) || line;
      } catch {
        return line;
      }
    });
    setRunMonitorLogLines(lines);
  } catch {
    // The event artifact can appear after the runner created the run dir.
  }
}

async function loadPreprocessingMonitorStatus(jobId) {
  const status = await api.get(API_ENDPOINTS.preprocessing.status(jobId));
  const runId = preprocessingRunIdFromStatus(status, jobId);
  const runDir = preprocessingRunDirFromStatus(status);
  if (runId) setCurrentRunId(runId);
  uiState.currentRunDir = runDir;
  uiState.currentRunQueue = [];
  uiState.currentRunColorMode = "";
  uiState.runProcessStatus = String(status?.status || "unknown").trim().toLowerCase();
  setRunMonitorFilterVisibility("", []);
  renderRunMonitorSummary(runId, status?.status || "unknown", [], runDir);
  setRunPhaseSnapshot(runId, status?.phases);
  renderRunMonitorPhaseLists();
  updateRunMonitorSubtitle(runId, status?.status || "unknown", status?.current_phase || "");
  await loadPreprocessingMonitorLogs(runId);
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
  // Only update if value actually changed to avoid input event loops
  if (editor.value !== value) {
    editor.value = value;
  }
  editor.dataset.source = String(source || "");
  editor.dataset.revisionId = String(revisionId || "");
  const parts = [];
  if (source) parts.push(formatI18n("page.run_monitor.config_status_source", "Quelle: {source}", { source }));
  if (revisionId) parts.push(formatI18n("page.run_monitor.config_status_revision", "Revision: {revision}", { revision: revisionId }));
  parts.push(formatI18n("page.run_monitor.config_status_lines", "Zeilen: {count}", { count: value ? value.split(/\r?\n/).length : 0 }));
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
  let terminalDispatched = false; // pro Socket-Instanz: Terminal-Event nur einmal dispatchen
  const streamOpenedAtMs = Date.now();
  uiState.runSocket = api.ws(
    API_ENDPOINTS.ws.run(runId),
    (event) => {
      const eventType = String(event?.type || "").trim().toLowerCase();
      const eventTsMs = parseEventTimestampMs(event?.ts);
      const isStaleTerminalReplay =
        (eventType === "run_end" || eventType === "resume_end")
        && Number.isFinite(eventTsMs)
        && eventTsMs < (streamOpenedAtMs - 1000);
      if (isStaleTerminalReplay) {
        return;
      }
      const streamRunId = String(runId || "").trim();
      const currentRunId = String(uiState.currentRunId || "").trim();
      if (currentRunId && streamRunId && currentRunId !== streamRunId) return;
      const line = formatStructuredLogLine(event, { suppressRunStatus: true });
      if (line) enqueueRunMonitorLogLine(line);
      const payload = event?.payload || {};
      const eventPhase =
        payload.phase_name ||
        payload.phase ||
        event?.phase_name ||
        event?.phase ||
        "";
      const eventStatus =
        payload.status ||
        event?.status ||
        (event.type === "phase_start" ? "running" : event.type === "phase_end" ? "ok" : "running");
      const eventPct =
        payload.progress ??
        payload.pct ??
        event?.progress ??
        event?.pct ??
        0;
      if (event?.type === "phase_progress" || event?.type === "phase_end" || event?.type === "phase_start") {
        if (eventPhase) setPhaseRow(eventPhase, eventStatus, eventPct);
      }
      if (eventType === "resume_start") {
        const resumePhase =
          payload.from_phase ||
          event?.from_phase ||
          "";
        if (resumePhase) {
          setPhaseRow(resumePhase, "running", 0);
          updateRunMonitorSubtitle(runId, "running", resumePhase);
        }
      }
      if (event?.type === "run_status" && event?.payload?.phases) {
        uiState.runProcessStatus = String(event?.payload?.status || event?.state || "").trim().toLowerCase();
        setRunPhaseSnapshot(runId, event.payload.phases);
        if (Array.isArray(event?.payload?.queue)) uiState.currentRunQueue = event.payload.queue;
        if (event?.payload?.run_dir) uiState.currentRunDir = String(event.payload.run_dir || uiState.currentRunDir || "");
        renderRunMonitorSummary(
          runId,
          event?.payload?.status || event?.state || "unknown",
          uiState.currentRunQueue,
          uiState.currentRunDir,
        );
        renderRunMonitorPhaseLists();
        updateRunMonitorSubtitle(
          runId,
          event?.payload?.status || event?.state || "unknown",
          event?.payload?.current_phase || event?.phase || "",
        );
      }
      if (eventType === "queue_progress") {
        uiState.runProcessStatus = String(event?.payload?.status || event?.state || uiState.runProcessStatus || "").trim().toLowerCase();
        if (Array.isArray(event?.payload?.queue)) uiState.currentRunQueue = event.payload.queue;
        const currentIndex = Number.isInteger(event?.payload?.current_index) ? event.payload.current_index : Number(event?.payload?.current_index ?? -1);
        const activeQueueRunId = queueActiveRunId(event?.payload?.queue, {
          currentIndex: Number.isFinite(currentIndex) ? currentIndex : -1,
          fallbackRunId: runId,
        }) || streamRunId;
        if (event?.payload?.runs_dir && activeQueueRunId) {
          uiState.currentRunDir = `${String(event.payload.runs_dir).replace(/\/+$/, "")}/${normalizeRunIdPath(activeQueueRunId)}`;
        }
        setRunMonitorFilterVisibility(uiState.currentRunColorMode, Array.isArray(event?.payload?.queue) ? event.payload.queue : null);
        renderRunMonitorSummary(activeQueueRunId || streamRunId, uiState.runProcessStatus || event?.state || "unknown", uiState.currentRunQueue, uiState.currentRunDir);
        renderRunMonitorPhaseLists();
        if (activeQueueRunId && activeQueueRunId !== currentRunId) {
          void uiState.runMonitorSwitchHandler?.(activeQueueRunId, {
            queue: Array.isArray(event?.payload?.queue) ? event.payload.queue : null,
            runsDir: String(event?.payload?.runs_dir || "").trim(),
          });
          return;
        }
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
          if (terminalDispatched) return;
          terminalDispatched = true;
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

  const preprocessingJobId = preprocessingMonitorJobId();
  const isPreprocessingMonitor = Boolean(preprocessingJobId);
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
    if (isPreprocessingMonitor) {
      setDisabledLike($("monitor-resume"), true);
      setDisabledLike($("monitor-resume-restore-revision"), true);
      setDisabledLike(resumeLoadCurrentBtn, true);
      setDisabledLike(resumeApplyTemplateBtn, true);
      setDisabledLike(resumeSaveTemplateBtn, true);
      setDisabledLike(resumePresetSelect, true);
      setDisabledLike($("monitor-resume-preset-dir"), true);
      setDisabledLike($("monitor-resume-preset-dir-browse"), true);
      setDisabledLike($("monitor-resume-preset-dir-reload"), true);
      setDisabledLike($("monitor-resume-config-revision"), true);
      setDisabledLike(resumeEditor, true);
      setMonitorResumeInfo(t("page.raw_stack.title", "Raw Stack"));
      return;
    }
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
  $("monitor-phase-lists")?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const tab = target.closest(".ps-phase-tab");
    if (tab) {
      const nextBatchKey = String(tab.dataset.batchKey || "").trim();
      if (nextBatchKey && nextBatchKey !== uiState.runMonitorSelectedBatchKey) {
        uiState.runMonitorSelectedBatchKey = nextBatchKey;
        renderRunMonitorPhaseLists();
        updateResumeEnabled();
        void refreshStatsActions().then(() => {
          setMonitorActionState(isRunActiveStatus(uiState.runProcessStatus || ""));
        });
      }
      return;
    }
    const row = target.closest(".ps-phase-row");
    if (!row) return;
    if (String(row.dataset.activeBatch || "") !== "1") return;
    if (normalizeRunIdPath(row.dataset.runId || "") !== normalizeRunIdPath(uiState.currentRunId)) return;
    if (String(row.dataset.resumeAllowed || "") !== "1") {
      setFooter(
        formatI18n("ui.message.monitor_resume_supported_from", "Resume aktuell nur ab {phases} unterstützt.", {
          phases: ["ASTROMETRY", "BGE", "PCC", "HYPERMETRIC_STRETCH"].map((phase) => localizedRunMonitorPhaseName(phase) || phase).join(", "),
        }),
        true,
      );
      return;
    }
    document.querySelectorAll(".ps-phase-row").forEach((x) => x.classList.remove("is-selected"));
    row.classList.add("is-selected");
    updateResumeEnabled();
  });
  $("monitor-filter-row")?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const btn = target.closest(".ps-chip-btn");
    if (!btn) return;
    runMonitorFilterButtons().forEach((x) => x.classList.remove("active"));
    btn.classList.add("active");
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
    return title && String(title.textContent || "").trim() === t("page.run_monitor.artifacts", "Artefakte");
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
    if (artifactViewerTitle) artifactViewerTitle.textContent = title || t("page.run_monitor.artifact", "Artefakt");
    artifactViewerBody.textContent = t("ui.message.monitor_artifact_loading", "Lade Artefakt ...");
    try {
      const payload = await api.get(API_ENDPOINTS.runs.artifactView(uiState.currentRunId, path));
      if (artifactViewerTitle) artifactViewerTitle.textContent = String(payload?.filename || title || t("page.run_monitor.artifact", "Artefakt"));
      artifactViewerBody.textContent = formatArtifactContent(payload);
    } catch (err) {
      artifactViewerBody.textContent = formatI18n("ui.message.monitor_artifact_load_failed", "Artefakt konnte nicht geladen werden:\n{error}", { error: errorText(err) });
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
      artifactList.innerHTML = `<li><button>${escapeRunMonitorHtml(t("ui.message.monitor_no_artifacts", "Keine Artefakte gefunden"))}</button></li>`;
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
          btn.textContent || btn.getAttribute("title") || t("page.run_monitor.artifact", "Artefakt"),
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
    if (isPreprocessingMonitor) {
      setDisabledLike(statsGenerateBtn, true);
      setDisabledLike(statsOpenFolderBtn, true);
      setMonitorReportAvailable(Boolean(uiState.currentRunId));
      setInlineAsyncStatus(statsStatusEl, "");
      return null;
    }
    const target = runMonitorTargetContext();
    if (!target.runId) {
      uiState.monitorStatsStatus = null;
      uiState.monitorStatsRunId = "";
      uiState.monitorStatsRunDir = "";
      setMonitorReportAvailable(false);
      setInlineAsyncStatus(statsStatusEl, "");
      return null;
    }
    const status = await api.get(API_ENDPOINTS.runs.statsStatus(target.runId, target.runDir)).catch(() => null);
    uiState.monitorStatsStatus = status;
    uiState.monitorStatsRunId = target.runId;
    uiState.monitorStatsRunDir = target.runDir;
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
    const target = runMonitorTargetContext();
    const hasRun = Boolean(String(target.runId || "").trim());
    const statsTargetMatches = normalizeRunIdPath(uiState.monitorStatsRunId || "") === normalizeRunIdPath(target.runId || "");
    const canGenerateStats = hasRun && isTerminalRunStatus(target.state) && !(statsTargetMatches && String(uiState.monitorStatsStatus?.state || "").toLowerCase() === "running");
    const hasStatsOutput = statsTargetMatches && Boolean(String(uiState.monitorStatsStatus?.output_dir || "").trim());
    setDisabledLike(startBtn, isPreprocessingMonitor || isActive);
    setDisabledLike(stopBtn, !isActive);
    setDisabledLike(statsGenerateBtn, !canGenerateStats);
    setDisabledLike(statsOpenFolderBtn, !hasStatsOutput);
    if (isActive || !hasRun) {
      setMonitorReportAvailable(false);
      if (!hasRun) setInlineAsyncStatus(statsStatusEl, "");
    }
    renderRunMonitorPhaseLists();
  };
  const resetPhaseRows = () => {
    if (uiState.currentRunId) resetRunPhaseSnapshot(uiState.currentRunId);
    renderRunMonitorPhaseLists();
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
    uiState.currentRunQueue = [];
    uiState.runPhaseSnapshots = {};
    uiState.runMonitorSelectedBatchKey = "";
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
  let runMonitorSwitchPromise = Promise.resolve(null);
  const switchRunMonitorToRunId = (nextRunId, { queue = null, runsDir = "" } = {}) => {
    const normalizedNextRunId = normalizeRunIdPath(nextRunId);
    if (!normalizedNextRunId) return Promise.resolve(null);
    if (normalizedNextRunId === normalizeRunIdPath(uiState.currentRunId)) return Promise.resolve(null);
    runMonitorSwitchPromise = runMonitorSwitchPromise.then(async () => {
      if (normalizedNextRunId === normalizeRunIdPath(uiState.currentRunId)) return null;
      const previousRunId = normalizeRunIdPath(uiState.currentRunId);
      const previousRunDir = String(uiState.currentRunDir || "").trim();
      setCurrentRunId(normalizedNextRunId);
      clearCurrentRunHistoryMark();
      if (Array.isArray(queue)) uiState.currentRunQueue = queue;
      if (runsDir) {
        uiState.currentRunDir = `${String(runsDir).replace(/\/+$/g, "")}/${normalizedNextRunId}`;
      } else if (previousRunDir && previousRunId) {
        uiState.currentRunDir = runDirForRunId(previousRunDir, previousRunId, normalizedNextRunId);
      }
      if (uiState.runSocket) {
        uiState.runSocket.close();
        uiState.runSocket = null;
      }
      const status = await loadRunStatus(normalizedNextRunId);
      uiState.runProcessStatus = String(status?.status || "").trim().toLowerCase();
      await loadRunRevisions();
      await loadRunMonitorCurrentConfig().catch(() => {});
      await refreshArtifacts();
      await refreshStatsActions();
      const isActive = isRunActiveStatus(status?.status || "");
      setMonitorActionState(isActive);
      if (isActive) {
        setRunMonitorLogLines([]);
        connectRunMonitorStream(uiState.currentRunId);
      }
      updateResumeEnabled();
      return status;
    }).catch((err) => {
      setFooter(formatI18n("ui.message.monitor_active_batch_load_failed", "Aktiver Queue-Batch konnte nicht geladen werden: {error}", { error: errorText(err) }), true);
      return null;
    });
    return runMonitorSwitchPromise;
  };
  uiState.runMonitorSwitchHandler = switchRunMonitorToRunId;
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
    // Socket schließen wenn Run nicht mehr aktiv — nur wenn reconnectSocket aktiv,
    // damit ein manuell geöffneter Socket (z.B. beim Resume) nicht sofort wieder
    // geschlossen wird bevor der Backend-Job als "running" sichtbar ist.
    if (reconnectSocket) {
      if (isActive) {
        setRunMonitorLogLines([]);
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
    // Socket sofort schließen — der Run ist beendet.
    if (uiState.runSocket) {
      uiState.runSocket.close();
      uiState.runSocket = null;
    }
    // Finalen Status laden. Retry bis der Backend-Status nicht mehr "running" ist,
    // da das Backend den Status nach run_end/resume_end kurz verzögert schreibt.
    const pollFinalStatus = async (attemptsLeft) => {
      await refreshCurrentRunMonitorState({ reconnectSocket: false });
      if (attemptsLeft > 0 && isRunActiveStatus(uiState.runProcessStatus || "")) {
        window.setTimeout(() => pollFinalStatus(attemptsLeft - 1), 600);
      }
    };
    window.setTimeout(() => pollFinalStatus(5), 300);
  });

  updateResumeEnabled();
  void refreshRunMonitorValidationMessage();

  $("monitor-start")?.addEventListener("click", async () => {
    setDisabledLike(startBtn, true);
    try {
      const validationMessage = await refreshRunMonitorValidationMessage();
      if (validationMessage) {
        setMonitorActionState(isRunActiveStatus(uiState.runProcessStatus || ""));
        setFooter(validationMessage, true);
        return;
      }
      const appState = await api.get(API_ENDPOINTS.app.state).catch(() => ({ run: { current: {} }, project: {} }));
      const currentStatus = String(appState?.run?.current?.status || "").trim().toLowerCase();
      if (currentStatus === "running") {
        uiState.runProcessStatus = currentStatus;
        setMonitorActionState(true);
        setFooter(t("ui.message.monitor_run_already_active", "Es läuft bereits ein aktiver Run."), true);
        return;
      }
      const latestGuardrails = await api.get(API_ENDPOINTS.guardrails.root);
      if (String(latestGuardrails?.status || "").toLowerCase() === "error") {
        setMonitorActionState(isRunActiveStatus(uiState.runProcessStatus || ""));
        setFooter(t("ui.message.monitor_guardrail_blocked", "Run blockiert: Guardrail-Status ist ERROR."), true);
        return;
      }
      uiState.runProcessStatus = "running";
      setMonitorActionState(true);
      const accepted = await startRunFromCurrentForm({ source: "monitor" });
      setCurrentRunId(accepted?.run_id || uiState.currentRunId);
      clearCurrentRunHistoryMark();
      setMonitorStartValidationMessage("");
      setFooter(formatI18n("ui.message.monitor_run_started", "Run gestartet (Job {job_id}).", { job_id: accepted?.job_id || "-" }));
      await refreshCurrentRunMonitorState({ reconnectSocket: true });
    } catch (err) {
      const appState = await api.get(API_ENDPOINTS.app.state).catch(() => ({ run: { current: {} } }));
      uiState.runProcessStatus = String(appState?.run?.current?.status || "").trim().toLowerCase();
      setMonitorActionState(isRunActiveStatus(uiState.runProcessStatus || ""));
      setFooter(formatI18n("ui.message.monitor_run_start_failed", "Run-Start fehlgeschlagen: {error}", { error: errorText(err) }), true);
    }
  });

  $("monitor-stop")?.addEventListener("click", async () => {
    if (!uiState.currentRunId) return;
    try {
      if (isPreprocessingMonitor) {
        const result = await api.post(API_ENDPOINTS.preprocessing.cancel, { job_id: preprocessingJobId });
        setFooter(result?.ok ? "Raw-Stack-Stop gesendet." : "Raw-Stack-Stop nicht bestaetigt.", !result?.ok);
        const status = await loadPreprocessingMonitorStatus(preprocessingJobId);
        setMonitorActionState(isRunActiveStatus(status?.status || ""));
        return;
      }
      const result = await api.post(API_ENDPOINTS.runs.stop(uiState.currentRunId), {});
      if (result.ok) {
        const stoppedJobs = Array.isArray(result.cancelled_jobs) ? result.cancelled_jobs.length : 0;
        const killedPids = Array.isArray(result.killed_pids) ? result.killed_pids.length : 0;
        setFooter(formatI18n("ui.message.monitor_stop_sent", "Stop gesendet. Jobs beendet: {jobs}, verwaiste Prozesse beendet: {pids}.", { jobs: stoppedJobs, pids: killedPids }));
      } else {
        setFooter(t("ui.message.monitor_stop_not_found", "Kein laufender Job/Prozess fuer diesen Run gefunden."), true);
      }
      const status = await loadRunStatus(uiState.currentRunId);
      setMonitorActionState(String(status?.status || "").toLowerCase() === "running");
    } catch (err) {
      setFooter(formatI18n("ui.message.monitor_stop_failed", "Stop fehlgeschlagen: {error}", { error: errorText(err) }), true);
    }
  });

  resumeLoadCurrentBtn?.addEventListener("click", async () => {
    try {
      await loadRunMonitorCurrentConfig();
      setFooter(t("ui.message.monitor_run_config_loaded", "Run-Config in den Resume-Editor geladen."));
      updateResumeEnabled();
    } catch (err) {
      setFooter(formatI18n("ui.message.monitor_run_config_load_failed", "Run-Config laden fehlgeschlagen: {error}", { error: errorText(err) }), true);
    }
  });

  resumeApplyTemplateBtn?.addEventListener("click", async () => {
    try {
      const path = String(resumePresetSelect?.value || "").trim();
      if (!path) {
        setFooter(t("ui.message.monitor_no_template_selected", "Kein Template ausgewaehlt."), true);
        return;
      }
      syncUnifiedPresetSelection(path);
      const applied = await api.post(API_ENDPOINTS.config.applyPreset, { path });
      setRunMonitorConfigEditor(String(applied?.config || ""), {
        source: path,
        revisionId: "",
      });
      setFooter(t("ui.message.monitor_template_loaded", "Template in den Resume-Editor geladen."));
      updateResumeEnabled();
    } catch (err) {
      setFooter(formatI18n("ui.message.monitor_template_load_failed", "Template laden fehlgeschlagen: {error}", { error: errorText(err) }), true);
    }
  });

  resumeSaveTemplateBtn?.addEventListener("click", async () => {
    try {
      const yaml = runMonitorConfigEditorValue();
      if (!String(yaml || "").trim()) {
        setFooter(t("ui.message.monitor_no_resume_config_to_save", "Keine Resume-Config zum Speichern vorhanden."), true);
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
      setFooter(formatI18n("ui.message.monitor_template_saved", "Template gespeichert unter {path}.", { path: saved?.path || targetPath }));
    } catch (err) {
      setFooter(formatI18n("ui.message.monitor_template_save_failed", "Template speichern fehlgeschlagen: {error}", { error: errorText(err) }), true);
    }
  });

  $("monitor-resume")?.addEventListener("click", async () => {
    const phase = runMonitorSelectedPhase();
    if (!phase) {
      setFooter(t("ui.message.monitor_choose_target_phase", "Bitte Zielphase waehlen."), true);
      return;
    }
    try {
      const yaml = runMonitorConfigEditorValue();
      if (!String(yaml || "").trim()) {
        setFooter(t("ui.message.monitor_enter_resume_config", "Bitte zuerst eine Resume-Config laden oder eingeben."), true);
        return;
      }
      const accepted = await api.post(API_ENDPOINTS.runs.resume(uiState.currentRunId), {
        from_phase: phase,
        config_yaml: yaml,
        run_dir: uiState.currentRunDir || undefined,
        filter_context: runMonitorSelectedFilter() || undefined,
      });
      setConfigDraft(yaml);
      setPhaseRow(phase, "running", 0);
      updateRunMonitorSubtitle(uiState.currentRunId, "running", phase);
      setMonitorActionState(true);
      setFooter(formatI18n("ui.message.monitor_resume_started", "Resume gestartet (Job {job_id}).", { job_id: accepted.job_id }));
      // Socket sofort öffnen — nicht auf loadRunStatus warten, da der Backend-Job
      // möglicherweise noch nicht als "running" sichtbar ist wenn wir abfragen.
      // Der Socket liefert run_status Events sobald der Job aktiv ist.
      setRunMonitorLogLines([]);
      connectRunMonitorStream(uiState.currentRunId);
      // Parallel den State aktualisieren (Artifacts, Revisions etc.) ohne Socket-Reconnect
      await refreshCurrentRunMonitorState({ reconnectSocket: false });
    } catch (err) {
      setFooter(formatI18n("ui.message.monitor_resume_failed", "Resume fehlgeschlagen: {error}", { error: errorText(err) }), true);
    }
  });

  $("monitor-resume-restore-revision")?.addEventListener("click", async () => {
    const revisionId = $("monitor-resume-config-revision")?.value || "";
    if (!revisionId) {
      setFooter(t("ui.message.monitor_choose_config_revision", "Bitte Config-Revision waehlen."), true);
      return;
    }
    try {
      await loadRunMonitorSelectedRevision();
      setFooter(formatI18n("ui.message.monitor_revision_loaded", "Revision {revision_id} in den Resume-Editor geladen.", { revision_id: revisionId }));
      updateResumeEnabled();
    } catch (err) {
      setFooter(formatI18n("ui.message.monitor_revision_load_failed", "Revision laden fehlgeschlagen: {error}", { error: errorText(err) }), true);
    }
  });

  $("monitor-stats-generate")?.addEventListener("click", async () => {
    try {
      const target = runMonitorTargetContext();
      if (!target.runId || !isTerminalRunStatus(target.state)) {
        setMonitorActionState(isRunActiveStatus(uiState.runProcessStatus || ""));
        setFooter(t("ui.message.monitor_stats_after_batch", "Stats erst nach beendetem Batch verfuegbar."), true);
        return;
      }
      const accepted = await api.post(API_ENDPOINTS.runs.stats(target.runId), {
        run_dir: target.runDir || undefined,
      });
      uiState.monitorStatsRunId = target.runId;
      uiState.monitorStatsRunDir = target.runDir;
      uiState.monitorStatsStatus = { ...(uiState.monitorStatsStatus || {}), state: "running" };
      setInlineAsyncStatus(statsStatusEl, t("ui.status.stats_running", "Stats laeuft"), "running");
      setFooter(statsStartedMessage(accepted.job_id));
      await waitForJob(accepted.job_id);
      await refreshArtifacts();
      await refreshStatsActions();
      setMonitorActionState(isRunActiveStatus(uiState.runProcessStatus || ""));
      setFooter(t("ui.message.stats_completed", "Stats-Generierung beendet."));
    } catch (err) {
      setFooter(statsFailedMessage(err), true);
    }
  });

  $("monitor-stats-open-folder")?.addEventListener("click", async () => {
    try {
      const target = runMonitorTargetContext();
      const status = await api.get(API_ENDPOINTS.runs.statsStatus(target.runId, target.runDir));
      const targetDir = String(status.output_dir || "").trim();
      if (!targetDir) {
        setFooter(t("ui.message.monitor_stats_folder_unavailable", "Stats-Ordner nicht verfuegbar."), true);
        return;
      }
      await openExternalPath(targetDir);
      setFooter(formatI18n("ui.message.monitor_stats_folder", "Stats-Ordner: {path}", { path: targetDir }));
    } catch (err) {
      setFooter(formatI18n("ui.message.monitor_stats_status_failed", "Stats-Status fehlgeschlagen: {error}", { error: errorText(err) }), true);
    }
  });

  document.addEventListener("gui2:locale-changed", () => {
    void refreshRunMonitorValidationMessage();
    renderRunMonitorPhaseLists();
    renderRunMonitorSummary(uiState.currentRunId, uiState.runProcessStatus || "unknown", uiState.currentRunQueue, uiState.currentRunDir);
    updateRunMonitorSubtitle(uiState.currentRunId, uiState.runProcessStatus || "unknown", runMonitorSelectedPhase());
    if (String(runMonitorConfigEditorValue() || "").trim()) {
      setRunMonitorConfigEditor(runMonitorConfigEditorValue(), {
        source: String($("monitor-resume-config-editor")?.dataset?.source || ""),
        revisionId: String($("monitor-resume-config-editor")?.dataset?.revisionId || ""),
      });
    }
  });

  $("monitor-report")?.addEventListener("click", async () => {
    try {
      if (isPreprocessingMonitor) {
        const report = await api.get(API_ENDPOINTS.preprocessing.report(preprocessingJobId));
        const targetRunId = normalizeRunIdPath(report?.run_id || uiState.currentRunId || localStorage.getItem(PREPROCESSING_RUN_ID_KEY) || "");
        if (!targetRunId) {
          setFooter("Preprocessing-Report nicht verfuegbar.", true);
          return;
        }
        const artifactPath = "artifacts/preprocess/preprocessing_report.html";
        const targetWindow = window.open(api.httpUrl(API_ENDPOINTS.runs.artifactRaw(targetRunId, artifactPath)), "_blank");
        if (!targetWindow) setFooter("Preprocessing-Report konnte nicht geoeffnet werden.", true);
        else setFooter(formatI18n("ui.message.monitor_report_path", "Report: {path}", { path: report.report_html || artifactPath }));
        return;
      }
      const target = runMonitorTargetContext();
      const status = await api.get(API_ENDPOINTS.runs.statsStatus(target.runId, target.runDir)).catch(() => uiState.monitorStatsStatus);
      if (!status?.report_path) {
        setFooter(t("ui.message.monitor_report_after_stats", "Report erst nach Generate Stats verfuegbar."), true);
        return;
      }
      const targetRunId = target.runId || normalizeRunIdPath(uiState.monitorStatsRunId || "");
      const targetRunDir = target.runDir || normalizeFsPath(uiState.monitorStatsRunDir || "");
      const runStatus = targetRunDir ? { run_dir: targetRunDir } : await ensureCurrentRunStatus();
      const { artifactPath } = openRunReportInNewTab(
        targetRunId,
        runStatus?.run_dir || targetRunDir,
        status.report_path,
      );
      setFooter(formatI18n("ui.message.monitor_report_path", "Report: {path}", { path: status.report_path || artifactPath }));
    } catch (err) {
      setFooter(formatI18n("ui.message.monitor_report_status_failed", "Report-Status fehlgeschlagen: {error}", { error: errorText(err) }), true);
    }
  });

  $("monitor-open-run-folder")?.addEventListener("click", async () => {
    try {
      const runStatus = await ensureCurrentRunStatus();
      const runDir = String(runStatus?.run_dir || uiState.currentRunDir || "").trim();
      if (!runDir) {
        setFooter(t("ui.message.monitor_run_folder_unavailable", "Run-Ordner nicht verfuegbar."), true);
        return;
      }
      await openExternalPath(runDir);
      setFooter(formatI18n("ui.message.monitor_run_folder", "Run-Ordner: {path}", { path: runDir }));
    } catch (err) {
      setFooter(formatI18n("ui.message.monitor_run_folder_open_failed", "Run-Ordner konnte nicht geoeffnet werden: {error}", { error: errorText(err) }), true);
    }
  });

  try {
    const appConstants = await api.get(API_ENDPOINTS.app.constants).catch(() => null);
    applyRunMonitorResumePhaseAvailability(appConstants?.resume_from);
    if (isPreprocessingMonitor) {
      const status = await loadPreprocessingMonitorStatus(preprocessingJobId);
      await refreshArtifacts();
      await refreshStatsActions();
      const isActive = isRunActiveStatus(status?.status || "");
      setMonitorActionState(isActive);
      if (isActive) {
        setRunMonitorLogLines([]);
        const pollPreprocessing = async () => {
          const latest = await loadPreprocessingMonitorStatus(preprocessingJobId).catch(() => null);
          await refreshArtifacts().catch(() => {});
          const active = isRunActiveStatus(latest?.status || "");
          setMonitorActionState(active);
          if (active) window.setTimeout(pollPreprocessing, 1500);
        };
        window.setTimeout(pollPreprocessing, 1500);
      }
      updateResumeEnabled();
      return;
    }
    await loadRunRevisions();
    const appState = await api.get(API_ENDPOINTS.app.state).catch(() => ({ project: {}, run: { current: {} } }));
    const currentRunId = String(appState?.project?.current_run_id || "").trim();
    if (currentRunId) setCurrentRunId(currentRunId);
    const hintedRunId = currentRunId || ensureRunIdFromHeader();
    if (!currentRunId && !hintedRunId) {
      clearCurrentRunId();
      setMonitorActionState(false);
      renderNoRunState(t("ui.message.monitor_no_active_run", "Kein aktiver Run. Start über Run starten."));
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
      setRunMonitorLogLines([]);
      connectRunMonitorStream(uiState.currentRunId);
    } else if (uiState.runSocket) {
      uiState.runSocket.close();
      uiState.runSocket = null;
    }
    updateResumeEnabled();
  } catch (err) {
    setFooter(formatI18n("ui.message.monitor_init_failed", "Run-Monitor Initialisierung fehlgeschlagen: {error}", { error: errorText(err) }), true);
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
  const detectStatusChip = $("tools-astrometry-detect-status");
  const installStatusChip = $("tools-astrometry-install-status");
  const catalogStatusChip = $("tools-astrometry-catalog-status");
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
    const selectedDataDir = String(dataDirInput?.value || "").trim();
    const payload = {
      astap_cli: selectedBinary,
      astap_data_dir: selectedDataDir,
    };
    const result = await withPathGrantRetry(
      () => api.post(API_ENDPOINTS.astrometry.detect, payload),
      { fallbackPath: payload.astap_cli || payload.astap_data_dir },
    );
    if (statusChip) statusChip.textContent = result.installed ? "Installed" : "Missing";
    setStatusChip(detectStatusChip, result.installed ? "ASTAP gefunden" : "ASTAP nicht gefunden", result.installed ? "ok" : "error");
    if (binaryInput && result.binary && !shouldKeepAstapSelection(selectedBinary, result.binary)) {
      binaryInput.value = String(result.binary);
      persistTextValue(UI_STORAGE_KEYS.astrometryBinary, binaryInput.value, { absolute: true });
    }
    if (dataDirInput && result.data_dir && !shouldKeepAstapSelection(selectedDataDir, result.data_dir)) {
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
      setStatusChip(detectStatusChip, "Prüfe...", "running");
      const result = await detect();
      setFooter(result.installed ? `ASTAP gefunden: ${result.binary || "-"}` : "ASTAP nicht gefunden.", !result.installed);
    } catch (err) {
      setStatusChip(detectStatusChip, "ASTAP nicht gefunden", "error");
      setFooter(`Astrometry detect fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.astrometry.install_cli']")?.addEventListener("click", async () => {
    try {
      const astapDataDir = $("tools-astrometry-data-dir")?.value || "";
      setStatusChip(installStatusChip, "Download gestartet", "running");
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.astrometry.installCli, { astap_data_dir: astapDataDir }),
        { fallbackPath: astapDataDir },
      );
      persistTextValue(UI_STORAGE_KEYS.astrometryInstallJob, accepted.job_id || "");
      append(accepted);
      const job = await waitForJob(accepted.job_id, {
        onTick: (j) => {
          updateTransferStatusChip(installStatusChip, j, {
            running: "Download läuft",
            extracting: "Entpacke",
            ok: "Install OK",
            error: "Install nicht OK",
          });
          append({ state: j.state, progress: j.data?.progress ?? null, stage: j.data?.stage ?? null });
        },
      });
      persistTextValue(UI_STORAGE_KEYS.astrometryInstallJob, "");
      updateTransferStatusChip(installStatusChip, job, {
        running: "Download läuft",
        extracting: "Entpacke",
        ok: "Install OK",
        error: "Install nicht OK",
      });
      append(job);
      await detect({ logResult: false });
    } catch (err) {
      persistTextValue(UI_STORAGE_KEYS.astrometryInstallJob, "");
      setStatusChip(installStatusChip, "Install nicht OK", "error");
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
      setStatusChip(catalogStatusChip, "Download gestartet", "running");
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.astrometry.downloadCatalog, {
          catalog_id: catalogId,
          astap_data_dir: astapDataDir,
        }),
        { fallbackPath: astapDataDir },
      );
      persistTextValue(UI_STORAGE_KEYS.astrometryCatalogJob, accepted.job_id || "");
      append(accepted);
      const job = await waitForJob(accepted.job_id, {
        onTick: (j) => {
          updateTransferStatusChip(catalogStatusChip, j, {
            running: "Download läuft",
            extracting: "Entpacke",
            ok: "Download OK",
            error: "Download nicht OK",
          });
          append({ state: j.state, current_chunk: j.data?.current_chunk, progress: j.data?.progress ?? null, stage: j.data?.stage ?? null });
        },
      });
      persistTextValue(UI_STORAGE_KEYS.astrometryCatalogJob, "");
      updateTransferStatusChip(catalogStatusChip, job, {
        running: "Download läuft",
        extracting: "Entpacke",
        ok: "Download OK",
        error: "Download nicht OK",
      });
      append(job);
    } catch (err) {
      persistTextValue(UI_STORAGE_KEYS.astrometryCatalogJob, "");
      setStatusChip(catalogStatusChip, "Download nicht OK", "error");
      setFooter(`Catalog-Download fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.astrometry.cancel_download']")?.addEventListener("click", async () => {
    try {
      const result = await api.post(API_ENDPOINTS.astrometry.cancelDownload, {});
      persistTextValue(UI_STORAGE_KEYS.astrometryCatalogJob, "");
      setStatusChip(catalogStatusChip, "Abgebrochen", "check");
      append(result);
    } catch (err) {
      setStatusChip(catalogStatusChip, "Cancel nicht OK", "error");
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
      const output = await chooseFitsSavePath(defaultOutput, { label: "Astrometry-Ergebnis speichern" });
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
    setStatusChip(detectStatusChip, "ASTAP nicht gefunden", "error");
  }
  await resumeTrackedJob({
    storageKey: UI_STORAGE_KEYS.astrometryInstallJob,
    jobTypes: ["astrometry_install_cli"],
    statusChip: installStatusChip,
    labels: {
      running: "Download läuft",
      extracting: "Entpacke",
      ok: "Install OK",
      error: "Install nicht OK",
    },
    append,
    onTerminal: async (job) => {
      if (String(job?.state || "").toLowerCase() === "ok") {
        await detect({ logResult: false }).catch(() => {});
      }
    },
  }).catch(() => {});
  await resumeTrackedJob({
    storageKey: UI_STORAGE_KEYS.astrometryCatalogJob,
    jobTypes: ["astrometry_catalog_download"],
    statusChip: catalogStatusChip,
    labels: {
      running: "Download läuft",
      extracting: "Entpacke",
      ok: "Download OK",
      error: "Download nicht OK",
    },
    append,
  }).catch(() => {});
}

async function bindPccPage() {
  if (!$("tools-pcc-rgb")) return;
  const logBox = findLogBoxBySectionTitle("Result + Log");
  const statusField = document.querySelector("[data-control='tools.pcc.siril_status']");
  const downloadStatusChip = $("tools-pcc-download-status");
  const onlineStatusChip = $("tools-pcc-online-status");
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
    ["tools-pcc-apply-attenuation", UI_STORAGE_KEYS.pccApplyAttenuation, false],
    ["tools-pcc-chroma-strength", UI_STORAGE_KEYS.pccChromaStrength, false],
    ["tools-pcc-k-max", UI_STORAGE_KEYS.pccKMax, false],
    ["tools-pcc-bg-neutralization", UI_STORAGE_KEYS.pccBgNeutralizationMode, false],
  ].forEach(([id, key, absolute]) => bindStoredField(id, key, { absolute, overwrite: id === "tools-pcc-source" }));

  const missingField = $("tools-pcc-missing-chunks");
  const starsMatchedField = $("tools-pcc-stars-matched");
  const starsUsedField = $("tools-pcc-stars-used");
  const residualField = $("tools-pcc-residual-rms");
  const matrixField = $("tools-pcc-matrix");
  const rgbInput = $("tools-pcc-rgb");
  const wcsInput = $("tools-pcc-wcs");
  const configNote = $("tools-pcc-config-note");
  const inputHint = $("tools-pcc-input-hint");
  let importedPccExtras = {};

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
  const pathStem = (pathValue) => {
    const baseName = pathBaseName(pathValue);
    return baseName.replace(/\.[^.]+$/, "");
  };
  const dedupePaths = (items) => {
    const seen = new Set();
    return items.filter((item) => {
      const normalized = String(item || "").trim();
      if (!normalized || seen.has(normalized)) return false;
      seen.add(normalized);
      return true;
    });
  };
  const filteredObject = (value) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    return Object.fromEntries(
      Object.entries(value).filter(([, item]) => item !== undefined && item !== null && item !== ""),
    );
  };
  const refreshPccInputHint = () => {
    if (!inputHint) return;
    const notes = [];
    if (lastImportedPccConfigPath) {
      notes.push("solve.fits mit passender WCS ist der richtige PCC-Input. Manuelle Tool-Laeufe koennen trotzdem vom Pipeline-Run abweichen, weil z. B. canvas_mask und interne Seeing-Schaetzungen nicht 1:1 aus dem Run uebernommen werden.");
    }
    inputHint.textContent = notes.join(" ");
    inputHint.style.display = notes.length ? "" : "none";
  };
  const inferRunConfigCandidates = (pathValue) => {
    const normalized = String(pathValue || "").trim().replace(/\\/g, "/");
    if (!normalized || !isAbsolutePath(normalized)) return [];
    const candidates = [];
    const markers = ["/outputs/", "/artifacts/", "/registered/", "/logs/"];
    for (const marker of markers) {
      const idx = normalized.lastIndexOf(marker);
      if (idx > 0) {
        candidates.push(`${normalized.slice(0, idx)}/config.yaml`);
      }
    }
    const directParent = parentDirOfPath(normalized);
    if (directParent) {
      candidates.push(joinPath(directParent, "config.yaml"));
      const parent = parentDirOfPath(directParent);
      if (parent) candidates.push(joinPath(parent, "config.yaml"));
      const grandParent = parentDirOfPath(parent);
      if (grandParent) candidates.push(joinPath(grandParent, "config.yaml"));
    }
    return dedupePaths(candidates);
  };
  const applyPccConfigToUi = (configObject, sourcePath) => {
    const pcc = configObject?.pcc;
    if (!pcc || typeof pcc !== "object") return false;
    importedPccExtras = filteredObject({
      radii_mode: pcc.radii_mode,
      aperture_fwhm_mult: pcc.aperture_fwhm_mult,
      annulus_inner_fwhm_mult: pcc.annulus_inner_fwhm_mult,
      annulus_outer_fwhm_mult: pcc.annulus_outer_fwhm_mult,
      min_aperture_px: pcc.min_aperture_px,
      background_model: pcc.background_model,
      max_condition_number: pcc.max_condition_number,
      max_residual_rms: pcc.max_residual_rms,
    });
    const fieldBindings = [
      ["tools-pcc-source", UI_STORAGE_KEYS.pccSource, pcc.source],
      ["tools-pcc-catalog-dir", UI_STORAGE_KEYS.pccCatalogDir, pcc.siril_catalog_dir],
      ["tools-pcc-mag-limit", UI_STORAGE_KEYS.pccMagLimit, pcc.mag_limit],
      ["tools-pcc-mag-bright", UI_STORAGE_KEYS.pccMagBrightLimit, pcc.mag_bright_limit],
      ["tools-pcc-min-stars", UI_STORAGE_KEYS.pccMinStars, pcc.min_stars],
      ["tools-pcc-sigma", UI_STORAGE_KEYS.pccSigma, pcc.sigma_clip],
      ["tools-pcc-aperture", UI_STORAGE_KEYS.pccAperture, pcc.aperture_radius_px],
      ["tools-pcc-annulus-in", UI_STORAGE_KEYS.pccAnnulusInner, pcc.annulus_inner_px],
      ["tools-pcc-annulus-out", UI_STORAGE_KEYS.pccAnnulusOuter, pcc.annulus_outer_px],
      ["tools-pcc-apply-attenuation", UI_STORAGE_KEYS.pccApplyAttenuation, pcc.apply_attenuation],
      ["tools-pcc-chroma-strength", UI_STORAGE_KEYS.pccChromaStrength, pcc.chroma_strength],
      ["tools-pcc-k-max", UI_STORAGE_KEYS.pccKMax, pcc.k_max],
      ["tools-pcc-bg-neutralization", UI_STORAGE_KEYS.pccBgNeutralizationMode, pcc.background_neutralization_mode],
    ];
    fieldBindings.forEach(([id, storageKey, value]) => {
      const el = $(id);
      if (value === undefined || value === null || !el) return;
      writeFieldValue(el, value);
      if (storageKey === UI_STORAGE_KEYS.pccCatalogDir) {
        persistTextValue(storageKey, String(el.value || "").trim(), { absolute: true });
      } else {
        persistTextValue(storageKey, String(el.value || "").trim());
      }
      el.dispatchEvent(new Event("change", { bubbles: true }));
    });
    if (configNote) {
      configNote.textContent = `PCC-Parameter automatisch aus ${pathBaseName(sourcePath)} übernommen.`;
    }
    refreshPccInputHint();
    append(`PCC config geladen | ${sourcePath}`);
    return true;
  };
  const loadPccConfigObjectFromPath = async (configPath) => {
    const loaded = await withPathGrantRetry(
      () => api.get(`${API_ENDPOINTS.config.current}?path=${encodeURIComponent(configPath)}`),
      { fallbackPath: configPath },
    );
    if (!loaded?.config) return null;
    const parsed = await api.post(API_ENDPOINTS.config.patch, {
      yaml: String(loaded.config || ""),
      updates: [],
      persist: false,
    });
    return parsed?.config && typeof parsed.config === "object" ? parsed.config : null;
  };
  let lastImportedPccConfigPath = "";
  let pccConfigImportTimer = null;
  const maybeImportPccConfigFromPaths = async () => {
    const candidates = dedupePaths([
      ...inferRunConfigCandidates(String(rgbInput?.value || "")),
      ...inferRunConfigCandidates(String(wcsInput?.value || "")),
    ]);
    if (!candidates.length) {
      lastImportedPccConfigPath = "";
      importedPccExtras = {};
      if (configNote) {
        configNote.textContent = "Wenn RGB/WCS aus einem Run stammen, werden PCC-Parameter automatisch aus der zugehörigen config.yaml übernommen.";
      }
      refreshPccInputHint();
      return;
    }
    for (const configPath of candidates) {
      if (configPath === lastImportedPccConfigPath) return;
      const configObject = await loadPccConfigObjectFromPath(configPath).catch(() => null);
      if (!configObject) continue;
      if (applyPccConfigToUi(configObject, configPath)) {
        lastImportedPccConfigPath = configPath;
        return;
      }
    }
    lastImportedPccConfigPath = "";
    importedPccExtras = {};
    refreshPccInputHint();
  };
  const schedulePccConfigImport = () => {
    if (pccConfigImportTimer) window.clearTimeout(pccConfigImportTimer);
    pccConfigImportTimer = window.setTimeout(() => {
      pccConfigImportTimer = null;
      void maybeImportPccConfigFromPaths();
    }, 160);
  };
  let lastAutoDetectedWcs = "";
  let lastRgbAutoLookup = "";
  let rgbAutoLookupTimer = null;
  const setWcsPath = (pathValue, { autoDetected = false } = {}) => {
    const nextValue = String(pathValue || "").trim();
    setInputValue(wcsInput, nextValue);
    persistTextValue(UI_STORAGE_KEYS.pccWcs, nextValue, { absolute: true });
    lastAutoDetectedWcs = autoDetected ? nextValue : "";
  };
  const findMatchingWcsPath = async (rgbPath) => {
    const trimmedRgbPath = String(rgbPath || "").trim();
    const dir = parentDirOfPath(trimmedRgbPath);
    const stem = pathStem(trimmedRgbPath);
    if (!dir || !stem) return "";
    const expectedNames = new Set([
      `${stem}.wcs`.toLowerCase(),
      `${pathBaseName(trimmedRgbPath)}.wcs`.toLowerCase(),
    ]);
    const listing = await withPathGrantRetry(
      () => api.get(`/api/fs/list?path=${encodeURIComponent(dir)}&include_files=1`),
      { fallbackPath: dir },
    );
    const items = Array.isArray(listing?.items) ? listing.items : [];
    const match = items.find((item) => {
      if (String(item?.type || "") !== "file") return false;
      return expectedNames.has(String(item?.name || "").toLowerCase());
    });
    return String(match?.path || "").trim();
  };
  const maybeAutoLoadPccWcs = async () => {
    const rgbPath = String(rgbInput?.value || "").trim();
    if (!rgbPath || !isAbsolutePath(rgbPath) || !/\.(fit|fits|fts)$/i.test(rgbPath)) return;
    if (rgbPath === lastRgbAutoLookup) return;
    const currentWcs = String(wcsInput?.value || "").trim();
    if (currentWcs && currentWcs !== lastAutoDetectedWcs) return;
    lastRgbAutoLookup = rgbPath;
    const matchedWcs = await findMatchingWcsPath(rgbPath).catch(() => "");
    if (!matchedWcs) return;
    if (matchedWcs === currentWcs) {
      lastAutoDetectedWcs = matchedWcs;
      return;
    }
    setWcsPath(matchedWcs, { autoDetected: true });
  };
  const schedulePccWcsAutoLoad = () => {
    if (rgbAutoLookupTimer) window.clearTimeout(rgbAutoLookupTimer);
    rgbAutoLookupTimer = window.setTimeout(() => {
      rgbAutoLookupTimer = null;
      void maybeAutoLoadPccWcs();
    }, 120);
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
  const fileExistsPath = async (pathValue) => {
    const absolutePath = String(pathValue || "").trim();
    if (!absolutePath || !isAbsolutePath(absolutePath)) return false;
    const dir = parentDirOfPath(absolutePath);
    if (!dir) return false;
    const listing = await withPathGrantRetry(
      () => api.get(`/api/fs/list?path=${encodeURIComponent(dir)}&include_files=1`),
      { fallbackPath: dir },
    ).catch(() => null);
    const items = Array.isArray(listing?.items) ? listing.items : [];
    return items.some((item) => String(item?.type || "") === "file" && String(item?.path || "").trim() === absolutePath);
  };
  const ensureCurrentPccTempArtifact = async () => {
    const candidates = [];
    if (isPccTempOutputPath(uiState.currentPccTempOutput)) {
      candidates.push({
        outputRgb: uiState.currentPccTempOutput,
        outputChannels: uiState.currentPccTempChannels,
        jobId: uiState.currentPccTempJobId,
      });
    }
    const resultOutput = String(uiState.lastPccResult?.output_rgb || "").trim();
    if (isPccTempOutputPath(resultOutput)) {
      candidates.push({
        outputRgb: resultOutput,
        outputChannels: Array.isArray(uiState.lastPccResult?.output_channels) ? uiState.lastPccResult.output_channels : [],
        jobId: uiState.currentPccTempJobId,
      });
    }
    const storedTempOutput = storedTextValue(UI_STORAGE_KEYS.pccTempOutput, { absolute: true });
    if (isPccTempOutputPath(storedTempOutput)) {
      candidates.push({
        outputRgb: storedTempOutput,
        outputChannels: storedJsonValue(UI_STORAGE_KEYS.pccTempChannels, []),
        jobId: storedTextValue(UI_STORAGE_KEYS.pccTempJob) || "",
      });
    }

    for (const candidate of candidates) {
      if (await fileExistsPath(candidate.outputRgb)) {
        setCurrentPccTempArtifact(candidate);
        return candidate;
      }
    }

    const currentInput = String(rgbInput?.value || "").trim();
    const currentWcs = String(wcsInput?.value || "").trim();
    const jobsResponse = await api.get(`${API_ENDPOINTS.jobs.list}?limit=100`).catch(() => null);
    const jobs = Array.isArray(jobsResponse?.items) ? jobsResponse.items : [];
    const scored = jobs
      .filter((job) => String(job?.type || "") === "pcc_run" && String(job?.state || "") === "ok")
      .map((job) => {
        const result = job?.data?.result;
        const outputRgb = String(result?.output_rgb || "").trim();
        if (!isPccTempOutputPath(outputRgb)) return null;
        let score = 0;
        if (currentInput && String(job?.data?.input_rgb || "").trim() === currentInput) score += 4;
        if (currentWcs && String(job?.data?.wcs_file || "").trim() === currentWcs) score += 2;
        if (String(job?.job_id || "").trim() === String(uiState.currentPccTempJobId || "").trim()) score += 1;
        return {
          jobId: String(job?.job_id || "").trim(),
          outputRgb,
          outputChannels: Array.isArray(result?.output_channels) ? result.output_channels : derivePccTempChannelPaths(outputRgb),
          score,
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.score - a.score);

    for (const candidate of scored) {
      if (await fileExistsPath(candidate.outputRgb)) {
        setCurrentPccTempArtifact(candidate);
        return candidate;
      }
    }

    const tempDir = `${String(appRuntime.tempRoot || "/tmp").trim().replace(/\\/g, "/").replace(/\/+$/, "")}/tile_compile_gui2/pcc`;
    const inputStem = derivePccTempStem(currentInput);
    const tempListing = await withPathGrantRetry(
      () => api.get(`/api/fs/list?path=${encodeURIComponent(tempDir)}&include_files=1`),
      { fallbackPath: tempDir },
    ).catch(() => null);
    const tempItems = Array.isArray(tempListing?.items) ? tempListing.items : [];
    const fallbackCandidates = tempItems
      .filter((item) => String(item?.type || "") === "file")
      .map((item) => {
        const name = String(item?.name || "");
        const match = name.match(new RegExp(`^${inputStem.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}_([0-9]+)\\.(fit|fits|fts)$`, "i"));
        if (!match) return null;
        return {
          outputRgb: String(item?.path || "").trim(),
          outputChannels: derivePccTempChannelPaths(String(item?.path || "").trim()),
          jobId: "",
          ts: Number(match[1]) || 0,
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.ts - a.ts);

    for (const candidate of fallbackCandidates) {
      if (await fileExistsPath(candidate.outputRgb)) {
        setCurrentPccTempArtifact(candidate);
        return candidate;
      }
    }

    clearCurrentPccTempArtifact();
    return null;
  };

  const storedPccOutput = storedTextValue(UI_STORAGE_KEYS.pccLastOutput, { absolute: true });
  if (storedPccOutput) {
    uiState.lastPccOutput = storedPccOutput;
  }
  const storedPccChannels = storedJsonValue(UI_STORAGE_KEYS.pccLastChannels, []);
  if (Array.isArray(storedPccChannels)) {
    uiState.lastPccChannels = storedPccChannels.map((item) => String(item));
  }
  const storedTempOutput = storedTextValue(UI_STORAGE_KEYS.pccTempOutput, { absolute: true });
  if (storedTempOutput) {
    uiState.currentPccTempOutput = storedTempOutput;
  }
  const storedTempChannels = storedJsonValue(UI_STORAGE_KEYS.pccTempChannels, []);
  if (Array.isArray(storedTempChannels)) {
    uiState.currentPccTempChannels = storedTempChannels.map((item) => String(item));
  }
  uiState.currentPccTempJobId = storedTextValue(UI_STORAGE_KEYS.pccTempJob) || "";
  const storedPccResult = storedJsonValue(UI_STORAGE_KEYS.pccLastResult, null);
  if (storedPccResult && typeof storedPccResult === "object") {
    uiState.lastPccResult = storedPccResult;
    applyPccResult(storedPccResult);
  } else {
    uiState.lastPccResult = null;
    uiState.lastPccOutput = "";
    uiState.lastPccChannels = [];
    persistTextValue(UI_STORAGE_KEYS.pccLastOutput, "");
    persistJsonValue(UI_STORAGE_KEYS.pccLastChannels, null);
  }
  rgbInput?.addEventListener("input", schedulePccWcsAutoLoad);
  rgbInput?.addEventListener("change", schedulePccWcsAutoLoad);
  rgbInput?.addEventListener("input", schedulePccConfigImport);
  rgbInput?.addEventListener("change", schedulePccConfigImport);
  wcsInput?.addEventListener("input", schedulePccConfigImport);
  wcsInput?.addEventListener("change", schedulePccConfigImport);
  wcsInput?.addEventListener("input", () => {
    if (String(wcsInput.value || "").trim() !== lastAutoDetectedWcs) {
      lastAutoDetectedWcs = "";
      lastRgbAutoLookup = "";
    }
  });
  schedulePccWcsAutoLoad();
  schedulePccConfigImport();
  await ensureCurrentPccTempArtifact().catch(() => {});

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
      setStatusChip(downloadStatusChip, "Download gestartet", "running");
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.pcc.downloadMissing, { catalog_dir: catalogDir }),
        { fallbackPath: catalogDir },
      );
      persistTextValue(UI_STORAGE_KEYS.pccDownloadJob, accepted.job_id || "");
      append(accepted);
      const job = await waitForJob(accepted.job_id, {
        onTick: (j) => {
          updateTransferStatusChip(downloadStatusChip, j, {
            running: "Download läuft",
            extracting: "Entpacke",
            ok: "Download OK",
            error: "Download nicht OK",
          });
          append({ state: j.state, current_chunk: j.data?.current_chunk, progress: j.data?.progress ?? null, stage: j.data?.stage ?? null });
        },
      });
      persistTextValue(UI_STORAGE_KEYS.pccDownloadJob, "");
      updateTransferStatusChip(downloadStatusChip, job, {
        running: "Download läuft",
        extracting: "Entpacke",
        ok: "Download OK",
        error: "Download nicht OK",
      });
      append(job);
      await refreshStatus();
    } catch (err) {
      persistTextValue(UI_STORAGE_KEYS.pccDownloadJob, "");
      setStatusChip(downloadStatusChip, "Download nicht OK", "error");
      setFooter(`PCC Download fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.pcc.cancel_download']")?.addEventListener("click", async () => {
    try {
      const result = await api.post(API_ENDPOINTS.pcc.cancelDownload, {});
      persistTextValue(UI_STORAGE_KEYS.pccDownloadJob, "");
      setStatusChip(downloadStatusChip, "Abgebrochen", "check");
      append(result);
    } catch (err) {
      setStatusChip(downloadStatusChip, "Cancel nicht OK", "error");
      setFooter(`PCC Cancel fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.pcc.check_online']")?.addEventListener("click", async () => {
    try {
      setStatusChip(onlineStatusChip, "Prüfe...", "running");
      const result = await api.post(API_ENDPOINTS.pcc.checkOnline, {});
      setStatusChip(
        onlineStatusChip,
        result.ok ? `OK${Number.isFinite(Number(result.latency_ms)) ? ` ${Math.round(Number(result.latency_ms))} ms` : ""}` : "nicht OK",
        result.ok ? "ok" : "error",
      );
      append(result);
      setFooter(result.ok ? `Online source OK (${result.latency_ms} ms)` : "Online source nicht erreichbar.", !result.ok);
    } catch (err) {
      setStatusChip(onlineStatusChip, "nicht OK", "error");
      setFooter(`Online-Check fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.pcc.run']")?.addEventListener("click", async () => {
    try {
      const input = $("tools-pcc-rgb")?.value || "";
      const output = derivePccTempOutputPath(input);
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
        apply_attenuation: readFieldValue($("tools-pcc-apply-attenuation")),
        chroma_strength: readNumber("tools-pcc-chroma-strength"),
        k_max: readNumber("tools-pcc-k-max"),
        background_neutralization_mode: $("tools-pcc-bg-neutralization")?.value || undefined,
        ...importedPccExtras,
      };
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.pcc.run, payload),
        { fallbackPath: input || payload.wcs_file || payload.catalog_dir || output },
      );
      append(accepted);
      const job = await waitForJob(accepted.job_id);
      const jobResult = job?.data?.result;
      if (jobResult) {
        applyPccResult(jobResult);
        setCurrentPccTempArtifact({
          outputRgb: jobResult.output_rgb,
          outputChannels: jobResult.output_channels,
          jobId: accepted.job_id || job?.job_id || "",
        });
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
      const currentTemp = await ensureCurrentPccTempArtifact();
      const sourceOutput = String(currentTemp?.outputRgb || "").trim();
      let sourceChannels = Array.isArray(currentTemp?.outputChannels)
        ? currentTemp.outputChannels.map((item) => String(item))
        : [];
      if (!sourceOutput) {
        throw new Error("Kein aktuelles PCC-Temp-Ergebnis zum Speichern vorhanden. Bitte Run PCC erneut ausführen.");
      }
      if (!sourceChannels.length || sourceChannels.some((value) => !String(value || "").trim())) {
        sourceChannels = derivePccTempChannelPaths(sourceOutput);
      }
      const defaultOutput = deriveOutputPath($("tools-pcc-rgb")?.value || sourceOutput, "_pcc");
      const output = await chooseFitsSavePath(defaultOutput, { label: "Korrigiertes PCC-Bild speichern" });
      if (!output) return;
      const result = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.pcc.saveCorrected, {
          source_output_rgb: sourceOutput,
          source_output_channels: sourceChannels,
          output_rgb: output,
          wcs_file: $("tools-pcc-wcs")?.value || "",
        }),
        { fallbackPath: output || sourceOutput || $("tools-pcc-wcs")?.value || "" },
      );
      append(result);
      setFooter(`Save Corrected: ${result.output_rgb || "-"}${Number.isFinite(Number(result?.size_bytes)) ? ` (${Math.round(Number(result.size_bytes) / 1024 / 1024)} MiB)` : ""}`);
    } catch (err) {
      setFooter(`Save Corrected fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  await resumeTrackedJob({
    storageKey: UI_STORAGE_KEYS.pccDownloadJob,
    jobTypes: ["pcc_siril_download"],
    statusChip: downloadStatusChip,
    labels: {
      running: "Download läuft",
      extracting: "Entpacke",
      ok: "Download OK",
      error: "Download nicht OK",
    },
    append,
    onTerminal: async (job) => {
      if (String(job?.state || "").toLowerCase() === "ok") {
        await refreshStatus().catch(() => {});
      }
    },
  }).catch(() => {});
}

function rawStackBool(id) {
  return String($(id)?.value || "false") === "true";
}

function rawStackNumber(id, fallback) {
  const value = Number($(id)?.value);
  return Number.isFinite(value) ? value : fallback;
}

function rawStackInputDir() {
  return parseInputDirs(String($("inp-dirs")?.value || ""))[0] || "";
}

function rawStackInputMode() {
  const mode = String($("inp-colormode")?.value || "").trim().toUpperCase();
  if (mode === "OSC") return "cfa_osc";
  if (mode === "MONO") return "mono";
  return "auto";
}

function rawStackBayerPattern() {
  const raw = String($("inp-bayer")?.value || "").trim();
  if (!raw || raw.toLowerCase().startsWith("auto")) return "auto";
  return raw.toUpperCase();
}

function rawStackCalibrationInput(kind) {
  const enabled = Boolean($(`cal-${kind}`)?.checked);
  const useMaster = String($(`cal-${kind}-source`)?.value || "false") === "true";
  const path = String($(`cal-${kind}-dir`)?.value || "").trim();
  return { enabled, useMaster, path };
}

function rawStackSetCheckbox(id, value) {
  const el = $(id);
  if (el) el.checked = Boolean(value);
}

function rawStackSetScanInputMode(value) {
  const mode = String(value || "auto").toLowerCase();
  if (mode === "cfa_osc" || mode === "osc") {
    rawStackSetSelect("inp-colormode", "OSC");
    return;
  }
  if (mode === "mono") {
    rawStackSetSelect("inp-colormode", "MONO");
  }
}

function rawStackSetScanBayer(value) {
  const raw = String(value || "auto").trim();
  rawStackSetSelect("inp-bayer", raw.toLowerCase() === "auto" ? "auto (aus FITS-Header)" : raw.toUpperCase());
}

function rawStackSetCalibrationInput(kind, enabled, useMaster, dirPath, masterPath) {
  rawStackSetCheckbox(`cal-${kind}`, enabled);
  rawStackSetSelect(`cal-${kind}-source`, Boolean(useMaster));
  const el = $(`cal-${kind}-dir`);
  if (el) el.value = useMaster ? (masterPath || "") : (dirPath || "");
}

function rawStackDefaultHmsConfig() {
  return {
    require_successful_pcc: true,
    mode: "ready_to_use",
    sensor_profile: "rec709",
    fallback_profile: "rec709",
    adaptive_anchor: true,
    target_bg: 0.15,
    protect_b: 6.0,
    convergence_power: 3.5,
    log_d_mode: "auto",
    fixed_log_d: 2.0,
    color_strategy: "fixed",
    fixed_color_strategy: 0.0,
    color_grip: 1.0,
    shadow_convergence: 0.0,
    linear_expansion: 0.0,
    write_channels: false,
    output_rgb: "stacked_rgb_hms.fits",
  };
}

function rawStackNormalizeDefaultsConfig(config) {
  const normalized = config && typeof config === "object" && !Array.isArray(config)
    ? { ...config }
    : {};
  const postprocess = normalized.postprocess && typeof normalized.postprocess === "object" && !Array.isArray(normalized.postprocess)
    ? { ...normalized.postprocess }
    : {};
  postprocess.hypermetric_stretch = true;
  normalized.postprocess = postprocess;
  normalized.hypermetric_stretch = { ...rawStackDefaultHmsConfig(), ...(normalized.hypermetric_stretch || {}) };
  normalized.rejection = {
    method: "sigma",
    low: 3,
    high: 3,
    max_iters: 3,
    min_fraction: 0.4,
    ...(normalized.rejection || {}),
  };
  normalized.stacking = {
    normalization: "addscale",
    weighting: "quality",
    cosmetic_correction: false,
    cosmetic_correction_sigma: 5,
    per_frame_cosmetic_correction: false,
    per_frame_cosmetic_correction_sigma: 5,
    ...(normalized.stacking || {}),
  };
  normalized.runtime_limits = {
    parallel_workers: 4,
    memory_budget: 512,
    ...(normalized.runtime_limits || {}),
  };
  return normalized;
}

async function rawStackConfigFromLoadedTileConfig() {
  try {
    const yamlText = String(await resolveConfigYamlForRun() || "");
    if (!yamlText.trim()) return null;
    const parsed = await patchConfig({ yamlText, updates: [] });
    const config = parsed?.config && typeof parsed.config === "object" ? parsed.config : null;
    if (!config) return null;
    const out = {};
    const runtime = config.runtime_limits;
    if (runtime && typeof runtime === "object" && !Array.isArray(runtime)) {
      out.runtime_limits = {};
      if (Number(runtime.parallel_workers) >= 1) out.runtime_limits.parallel_workers = Number(runtime.parallel_workers);
      if (Number(runtime.memory_budget) >= 1) out.runtime_limits.memory_budget = Number(runtime.memory_budget);
      if (Object.keys(out.runtime_limits).length === 0) delete out.runtime_limits;
    }
    const normalization = config.normalization;
    if (normalization && typeof normalization === "object" && !Array.isArray(normalization)) {
      const mode = String(normalization.mode || "").trim();
      if (["background", "median", "addscale", "none"].includes(mode)) {
        out.stacking = { ...(out.stacking || {}), normalization: mode };
      }
    }
    const stacking = config.stacking;
    if (stacking && typeof stacking === "object" && !Array.isArray(stacking)) {
      const nextStacking = { ...(out.stacking || {}) };
      const nextRejection = {};
      if (typeof stacking.cosmetic_correction === "boolean") nextStacking.cosmetic_correction = stacking.cosmetic_correction;
      if (Number(stacking.cosmetic_correction_sigma) > 0) nextStacking.cosmetic_correction_sigma = Number(stacking.cosmetic_correction_sigma);
      if (typeof stacking.per_frame_cosmetic_correction === "boolean") nextStacking.per_frame_cosmetic_correction = stacking.per_frame_cosmetic_correction;
      if (Number(stacking.per_frame_cosmetic_correction_sigma) > 0) nextStacking.per_frame_cosmetic_correction_sigma = Number(stacking.per_frame_cosmetic_correction_sigma);
      if (["quality", "uniform"].includes(String(stacking.weighting || ""))) nextStacking.weighting = String(stacking.weighting);
      const sigma = stacking.sigma_clip;
      if (sigma && typeof sigma === "object" && !Array.isArray(sigma)) {
        if (Number(sigma.sigma_low) > 0) nextRejection.low = Number(sigma.sigma_low);
        if (Number(sigma.sigma_high) > 0) nextRejection.high = Number(sigma.sigma_high);
        if (Number(sigma.max_iters) >= 1) nextRejection.max_iters = Number(sigma.max_iters);
        if (Number(sigma.min_fraction) >= 0 && Number(sigma.min_fraction) <= 1) nextRejection.min_fraction = Number(sigma.min_fraction);
      }
      if (Object.keys(nextStacking).length > 0) out.stacking = nextStacking;
      if (Object.keys(nextRejection).length > 0) out.rejection = nextRejection;
    }
    const nextPostprocess = {};
    const astrometry = config.astrometry;
    if (astrometry && typeof astrometry === "object" && !Array.isArray(astrometry) && typeof astrometry.enabled === "boolean") {
      nextPostprocess.astrometry = astrometry.enabled;
      out.astrometry = { ...astrometry };
    }
    const bge = config.bge;
    if (bge && typeof bge === "object" && !Array.isArray(bge)) {
      if (typeof bge.enabled === "boolean") nextPostprocess.bge = bge.enabled;
      out.bge = { ...bge };
    }
    const pcc = config.pcc;
    if (pcc && typeof pcc === "object" && !Array.isArray(pcc) && typeof pcc.enabled === "boolean") {
      nextPostprocess.pcc = pcc.enabled;
      out.pcc = { ...pcc };
    }
    const hms = config.hypermetric_stretch;
    if (hms && typeof hms === "object" && !Array.isArray(hms)) {
      out.hypermetric_stretch = { ...hms };
      if (typeof hms.enabled === "boolean") nextPostprocess.hypermetric_stretch = hms.enabled;
    }
    if (Object.keys(nextPostprocess).length > 0) out.postprocess = nextPostprocess;
    return Object.keys(out).length > 0 ? out : null;
  } catch {
    return null;
  }
}

function rawStackMergeConfigPatch(config, patch) {
  if (!patch || typeof patch !== "object") return config;
  const next = { ...config };
  ["runtime_limits", "stacking", "rejection", "postprocess", "astrometry", "bge", "pcc", "hypermetric_stretch"].forEach((key) => {
    if (patch[key] && typeof patch[key] === "object" && !Array.isArray(patch[key])) {
      next[key] = { ...(next[key] || {}), ...patch[key] };
    }
  });
  return next;
}

function rawStackParameterOptions(path) {
  const options = {
    "input_mode": ["auto", "cfa_osc", "mono"],
    "bayer_pattern": ["auto", "RGGB", "BGGR", "GRBG", "GBRG", "UNKNOWN"],
    "mono_mode": ["auto", "mono"],
    "registration_reference": ["best_quality"],
    "rejection.method": ["sigma", "median", "winsor"],
    "quality_filter.mode": ["auto", "strict", "relaxed", "off"],
    "stacking.normalization": ["background", "median", "addscale", "none"],
    "stacking.weighting": ["quality", "uniform"],
    "hypermetric_stretch.mode": ["ready_to_use", "scientific"],
    "hypermetric_stretch.sensor_profile": ["rec709", "Sony IMX415"],
    "hypermetric_stretch.fallback_profile": ["rec709", "Sony IMX415"],
    "hypermetric_stretch.log_d_mode": ["auto", "fixed"],
    "hypermetric_stretch.color_strategy": ["auto", "fixed"],
  };
  return options[path] || null;
}

function rawStackEditorConfig() {
  const raw = String($("raw-stack-config-json")?.value || "").trim();
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function rawStackReadConfig() {
  const bias = rawStackCalibrationInput("bias");
  const dark = rawStackCalibrationInput("dark");
  const flat = rawStackCalibrationInput("flat");
  const editorConfig = rawStackEditorConfig();
  const editorCalibration = editorConfig.calibration && typeof editorConfig.calibration === "object" && !Array.isArray(editorConfig.calibration)
    ? editorConfig.calibration
    : {};
  const editorRejection = editorConfig.rejection && typeof editorConfig.rejection === "object" && !Array.isArray(editorConfig.rejection)
    ? editorConfig.rejection
    : {};
  const editorQuality = editorConfig.quality_filter && typeof editorConfig.quality_filter === "object" && !Array.isArray(editorConfig.quality_filter)
    ? editorConfig.quality_filter
    : {};
  const editorStacking = editorConfig.stacking && typeof editorConfig.stacking === "object" && !Array.isArray(editorConfig.stacking)
    ? editorConfig.stacking
    : {};
  const editorReport = editorConfig.report && typeof editorConfig.report === "object" && !Array.isArray(editorConfig.report)
    ? editorConfig.report
    : {};
  const editorHms = editorConfig.hypermetric_stretch && typeof editorConfig.hypermetric_stretch === "object" && !Array.isArray(editorConfig.hypermetric_stretch)
    ? editorConfig.hypermetric_stretch
    : {};
  return {
    ...editorConfig,
    mode: "linear_prestack",
    lights_dir: rawStackInputDir(),
    bias_dir: bias.useMaster ? "" : bias.path,
    darks_dir: dark.useMaster ? "" : dark.path,
    flats_dir: flat.useMaster ? "" : flat.path,
    input_mode: rawStackInputMode(),
    raw_formats: "tile_compile",
    bayer_pattern: rawStackBayerPattern(),
    cfa_mode: "tile_compile",
    mono_mode: editorConfig.mono_mode || "auto",
    registration_reference: editorConfig.registration_reference || "best_quality",
    calibration: {
      ...editorCalibration,
      use_bias: bias.enabled,
      use_dark: dark.enabled,
      use_flat: flat.enabled,
      bias_master: bias.useMaster ? bias.path : "",
      dark_master: dark.useMaster ? dark.path : "",
      flat_master: flat.useMaster ? flat.path : "",
      dark_auto_select: editorCalibration.dark_auto_select !== false,
      dark_match_use_temp: Boolean(editorCalibration.dark_match_use_temp),
      pattern: editorCalibration.pattern || "*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz",
    },
    rejection: {
      ...editorRejection,
      method: $("raw-stack-rejection")?.value || editorRejection.method || "sigma",
      low: rawStackNumber("raw-stack-rej-low", 3.0),
      high: rawStackNumber("raw-stack-rej-high", 3.0),
      max_iters: Number(editorRejection.max_iters) >= 1 ? Number(editorRejection.max_iters) : 3,
      min_fraction: Number(editorRejection.min_fraction) >= 0 ? Number(editorRejection.min_fraction) : 0.4,
    },
    quality_filter: {
      ...editorQuality,
      mode: $("raw-stack-quality-mode")?.value || editorQuality.mode || "auto",
      min_stars: rawStackNumber("raw-stack-min-stars", 30),
      min_correlation: rawStackNumber("raw-stack-min-corr", 0.75),
      max_fwhm_sigma: rawStackNumber("raw-stack-fwhm-sigma", 2.0),
      max_eccentricity: rawStackNumber("raw-stack-ecc", 0.65),
    },
    stacking: {
      ...editorStacking,
      normalization: $("raw-stack-normalization")?.value || editorStacking.normalization || "addscale",
      weighting: $("raw-stack-weighting")?.value || editorStacking.weighting || "quality",
      cosmetic_correction: Boolean(editorStacking.cosmetic_correction),
      cosmetic_correction_sigma: Number(editorStacking.cosmetic_correction_sigma) > 0 ? Number(editorStacking.cosmetic_correction_sigma) : 5,
      per_frame_cosmetic_correction: Boolean(editorStacking.per_frame_cosmetic_correction),
      per_frame_cosmetic_correction_sigma: Number(editorStacking.per_frame_cosmetic_correction_sigma) > 0 ? Number(editorStacking.per_frame_cosmetic_correction_sigma) : 5,
    },
    postprocess: {
      astrometry: rawStackBool("raw-stack-astrometry"),
      bge: rawStackBool("raw-stack-bge"),
      pcc: rawStackBool("raw-stack-pcc"),
      hypermetric_stretch: rawStackBool("raw-stack-hms"),
    },
    hypermetric_stretch: { ...rawStackDefaultHmsConfig(), ...editorHms },
    report: {
      ...editorReport,
      detailed: true,
      formats: Array.isArray(editorReport.formats) ? editorReport.formats : ["json", "markdown", "html"],
    },
  };
}

function rawStackSetSelect(id, value) {
  const el = $(id);
  if (el) el.value = String(value);
}

function rawStackApplyConfig(config) {
  const c = config || {};
  if ($("inp-dirs")) $("inp-dirs").value = c.lights_dir || "";
  rawStackSetScanInputMode(c.input_mode || "auto");
  rawStackSetScanBayer(c.bayer_pattern || "auto");
  const cal = c.calibration || {};
  rawStackSetCalibrationInput("bias", cal.use_bias, Boolean(cal.bias_master), c.bias_dir, cal.bias_master);
  rawStackSetCalibrationInput("dark", cal.use_dark, Boolean(cal.dark_master), c.darks_dir, cal.dark_master);
  rawStackSetCalibrationInput("flat", cal.use_flat, Boolean(cal.flat_master), c.flats_dir, cal.flat_master);
  const q = c.quality_filter || {};
  rawStackSetSelect("raw-stack-quality-mode", q.mode || "auto");
  if ($("raw-stack-min-stars")) $("raw-stack-min-stars").value = q.min_stars ?? 30;
  if ($("raw-stack-min-corr")) $("raw-stack-min-corr").value = q.min_correlation ?? 0.75;
  if ($("raw-stack-fwhm-sigma")) $("raw-stack-fwhm-sigma").value = q.max_fwhm_sigma ?? 2.0;
  if ($("raw-stack-ecc")) $("raw-stack-ecc").value = q.max_eccentricity ?? 0.65;
  const r = c.rejection || {};
  rawStackSetSelect("raw-stack-rejection", r.method || "sigma");
  if ($("raw-stack-rej-low")) $("raw-stack-rej-low").value = r.low ?? 3.0;
  if ($("raw-stack-rej-high")) $("raw-stack-rej-high").value = r.high ?? 3.0;
  const s = c.stacking || {};
  rawStackSetSelect("raw-stack-normalization", s.normalization || "addscale");
  rawStackSetSelect("raw-stack-weighting", s.weighting || "quality");
  const p = c.postprocess || {};
  rawStackSetSelect("raw-stack-astrometry", p.astrometry !== false);
  rawStackSetSelect("raw-stack-bge", p.bge !== false);
  rawStackSetSelect("raw-stack-pcc", p.pcc !== false);
  rawStackSetSelect("raw-stack-hms", p.hypermetric_stretch !== false);
  if ($("raw-stack-config-json")) {
    $("raw-stack-config-json").value = JSON.stringify(c, null, 2);
  }
  rawStackUpdateJson();
}

function rawStackUpdateJson() {
  const el = $("raw-stack-config-json");
  if (!el) return;
  el.value = JSON.stringify(rawStackReadConfig(), null, 2);
}

function rawStackValueAt(config, path) {
  return String(path || "").split(".").reduce((acc, part) => (
    acc && typeof acc === "object" ? acc[part] : undefined
  ), config);
}

function rawStackSetValueAt(config, path, value) {
  const parts = String(path || "").split(".").filter(Boolean);
  if (parts.length === 0) return config;
  let target = config;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const key = parts[i];
    if (!target[key] || typeof target[key] !== "object" || Array.isArray(target[key])) {
      target[key] = {};
    }
    target = target[key];
  }
  target[parts[parts.length - 1]] = value;
  return config;
}

function rawStackParseParameterValue(raw, previousValue) {
  const text = String(raw ?? "");
  if (typeof previousValue === "boolean") return text === "true";
  if (typeof previousValue === "number") {
    const n = Number(text);
    return Number.isFinite(n) ? n : previousValue;
  }
  if (previousValue && typeof previousValue === "object") {
    try {
      return JSON.parse(text);
    } catch {
      return previousValue;
    }
  }
  return text;
}

function rawStackParameterControl(path, value) {
  const data = `data-raw-stack-param="1" data-path="${escapeRunMonitorAttr(path)}"`;
  const enumOptions = rawStackParameterOptions(path);
  if (Array.isArray(enumOptions) && enumOptions.length > 0) {
    const current = String(value ?? "");
    const options = enumOptions.map((item) => {
      const v = String(item);
      return `<option value="${escapeRunMonitorAttr(v)}" ${v === current ? "selected" : ""}>${escapeRunMonitorHtml(v)}</option>`;
    }).join("");
    return `<select class="ps-input raw-stack-param-input" ${data}>${options}</select>`;
  }
  if (typeof value === "boolean") {
    return `<select class="ps-input raw-stack-param-input" ${data}>
      <option value="true" ${value ? "selected" : ""}>true</option>
      <option value="false" ${!value ? "selected" : ""}>false</option>
    </select>`;
  }
  if (typeof value === "number") {
    return `<input class="ps-input raw-stack-param-input" type="number" step="any" value="${escapeRunMonitorAttr(value)}" ${data}>`;
  }
  if (value && typeof value === "object") {
    return `<textarea class="ps-input raw-stack-param-input raw-stack-param-json" rows="3" spellcheck="false" ${data}>${escapeRunMonitorHtml(JSON.stringify(value))}</textarea>`;
  }
  return `<input class="ps-input raw-stack-param-input" type="text" value="${escapeRunMonitorAttr(value ?? "")}" ${data}>`;
}

function rawStackCommitParameterEdit(el) {
  const path = String(el?.dataset?.path || "");
  if (!path) return;
  const current = rawStackReadConfig();
  const previousValue = rawStackValueAt(current, path);
  const nextValue = rawStackParseParameterValue(el.value, previousValue);
  rawStackSetValueAt(current, path, nextValue);
  if ($("raw-stack-config-json")) {
    $("raw-stack-config-json").value = JSON.stringify(current, null, 2);
  }
  rawStackApplyConfig(current);
}

function rawStackRenderParameterGroups(groups, config) {
  const host = $("raw-stack-parameter-groups");
  if (!host) return;
  const visibleGroups = Array.isArray(groups) ? groups : [];
  host.innerHTML = visibleGroups.map((group) => {
    const paths = Array.isArray(group.paths) ? group.paths : [];
    const rows = paths.map((path) => {
      const value = rawStackValueAt(config, path);
      return `<div class="ps-row raw-stack-param-row">
        <label>${escapeRunMonitorHtml(path)}</label>
        ${rawStackParameterControl(path, value)}
      </div>`;
    }).join("");
    return `<details class="ps-card raw-stack-parameter-group">
      <summary class="ps-section-title">${escapeRunMonitorHtml(group.label || group.id || "Group")}</summary>
      ${rows}
    </details>`;
  }).join("");
  host.querySelectorAll("[data-raw-stack-param]").forEach((el) => {
    const eventName = el.tagName === "SELECT" ? "change" : "change";
    el.addEventListener(eventName, () => rawStackCommitParameterEdit(el));
  });
}

function rawStackParseCsv(text) {
  const lines = String(text || "").split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    header.forEach((key, idx) => { row[key] = cells[idx] ?? ""; });
    return row;
  });
}

function rawStackSetManualOverride(index, filename, include) {
  const editor = rawStackEditorConfig();
  const quality = editor.quality_filter && typeof editor.quality_filter === "object" && !Array.isArray(editor.quality_filter)
    ? { ...editor.quality_filter }
    : {};
  const overrides = quality.manual_overrides && typeof quality.manual_overrides === "object" && !Array.isArray(quality.manual_overrides)
    ? { ...quality.manual_overrides }
    : {};
  overrides[String(index)] = { index: Number(index), filename: String(filename || ""), include: Boolean(include) };
  quality.manual_overrides = overrides;
  const next = { ...rawStackReadConfig(), quality_filter: { ...(rawStackReadConfig().quality_filter || {}), ...quality } };
  const el = $("raw-stack-config-json");
  if (el) el.value = JSON.stringify(next, null, 2);
}

function rawStackRenderFrameTable(rows) {
  const host = $("raw-stack-frame-table");
  if (!host) return;
  if (!Array.isArray(rows) || rows.length === 0) {
    host.innerHTML = "";
    return;
  }
  host.innerHTML = `<div class="ps-section-title" style="margin-bottom:6px;">Verwendete Frames (${rows.length})</div><table class="ps-table" style="min-width:820px;">
    <thead><tr><th>Use</th><th>Index</th><th>Frame</th><th>Stars</th><th>FWHM</th><th>ECC</th><th>CC</th><th>Score</th><th>Reason</th></tr></thead>
    <tbody>${rows.map((row) => {
      const included = String(row.included || "0") === "1";
      const index = String(row.index || "0");
      const filename = String(row.filename || "");
      return `<tr>
        <td><input type="checkbox" data-raw-stack-frame-override="1" data-index="${escapeRunMonitorAttr(index)}" data-filename="${escapeRunMonitorAttr(filename)}" ${included ? "checked" : ""}></td>
        <td>${escapeRunMonitorHtml(index)}</td>
        <td>${escapeRunMonitorHtml(filename)}</td>
        <td>${escapeRunMonitorHtml(row.star_count || "-")}</td>
        <td>${escapeRunMonitorHtml(row.fwhm || "-")}</td>
        <td>${escapeRunMonitorHtml(row.eccentricity || "-")}</td>
        <td>${escapeRunMonitorHtml(row.registration_cc || "-")}</td>
        <td>${escapeRunMonitorHtml(row.quality_score || "-")}</td>
        <td>${escapeRunMonitorHtml(row.exclusion_reason || "-")}</td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
  host.querySelectorAll("[data-raw-stack-frame-override]").forEach((el) => {
    el.addEventListener("change", () => {
      rawStackSetManualOverride(el.dataset.index || "0", el.dataset.filename || "", el.checked);
      rawStackUpdateJson();
    });
  });
}

async function rawStackLoadFrameQuality(runId) {
  const target = String(runId || "").trim();
  if (!target) return;
  const payload = await api.get(API_ENDPOINTS.runs.artifactView(target, "artifacts/preprocess/frame_quality.csv"));
  rawStackRenderFrameTable(rawStackParseCsv(payload?.text || ""));
}

const RAW_STACK_PHASE_ORDER = [
  "INPUT_SCAN","CALIBRATION","CFA_CHANNEL_PREP","REFERENCE_SELECTION",
  "REGISTRATION","QUALITY_ANALYSIS","FRAME_FILTERING","STACKING",
  "ASTROMETRY","BGE","PCC","HYPERMETRIC_STRETCH","REPORT",
];

function rawStackRenderPhases(status, livePhase = "", livePct = 0) {
  const list = $("raw-stack-phase-list");
  if (!list) return;
  let phases = Array.isArray(status?.phases) ? status.phases : [];
  // Wenn Backend noch kein phases-Array liefert, synthetisch aus livePhase aufbauen
  if (phases.length === 0 && livePhase) {
    let found = false;
    phases = RAW_STACK_PHASE_ORDER.map((name) => {
      if (name === livePhase) { found = true; return { phase: name, status: "running", pct: livePct / 100 }; }
      return { phase: name, status: found ? "pending" : "ok", pct: found ? 0 : 1 };
    });
  }
  list.innerHTML = phases.map((p) => {
    const phase = String(p.phase || "-");
    const state = String(p.status || "pending");
    const tone = state === "ok" ? "ok" : state === "failed" || state === "error" ? "error" : state === "running" ? "running" : "check";
    const pct = formatLogPercent(p.pct ?? p.progress ?? 0);
    return `<div class="ps-row"><label>${phase}</label><span class="shell-status-chip shell-status-chip-${tone}">${state}${pct ? ` ${pct}` : ""}</span></div>`;
  }).join("");
}

async function bindRawStackPage() {
  if (pageName() !== "raw-stack.html") return;
  const chip = $("raw-stack-status-chip");
  const logBox = $("raw-stack-log");
  let currentJobId = localStorage.getItem(PREPROCESSING_JOB_KEY) || "";

  const currentConfig = () => rawStackReadConfig();

  async function loadDefaults(reset = false) {
    const parameters = reset
      ? await api.get(API_ENDPOINTS.preprocessing.defaults)
      : await api.get(API_ENDPOINTS.preprocessing.parameters);
    let config = rawStackNormalizeDefaultsConfig(parameters?.config || {});
    config = rawStackMergeConfigPatch(config, await rawStackConfigFromLoadedTileConfig());
    rawStackApplyConfig(config);
    rawStackRenderParameterGroups(parameters?.groups || [], rawStackReadConfig());
  }

  // Liest events.jsonl und leitet livePhase/livePct ab; gibt { livePhase, livePct, logLines } zurück
  async function loadEventsLog(runId) {
    if (!runId) return { livePhase: "", livePct: 0, logLines: [] };
    try {
      const payload = await api.get(API_ENDPOINTS.runs.artifactView(runId, "artifacts/preprocess/events.jsonl"));
      const text = String(payload?.text || "");
      if (!text.trim()) return { livePhase: "", livePct: 0, logLines: [] };
      const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
      let livePhase = "";
      let livePct = 0;
      const logLines = [];
      for (const l of lines) {
        try {
          const ev = JSON.parse(l);
          const evType = String(ev.type || "").toLowerCase();
          const evPhase = String(ev.phase_name || ev.payload?.phase_name || ev.phase || ev.payload?.phase || "").trim();
          if (evType === "phase_start" && evPhase) { livePhase = evPhase; livePct = 0; }
          if (evType === "phase_progress" && evPhase) {
            livePhase = evPhase;
            livePct = Math.round((ev.pct ?? ev.progress ?? ev.payload?.pct ?? ev.payload?.progress ?? 0) * 100);
          }
          if (evType === "phase_end") { livePhase = ""; livePct = 0; }
          const formatted = formatStructuredLogLine(ev, { suppressRunStatus: true });
          if (formatted) logLines.push(formatted);
        } catch { logLines.push(l); }
      }
      return { livePhase, livePct, logLines };
    } catch { return { livePhase: "", livePct: 0, logLines: [] }; }
  }

  async function refreshStatus() {
    if (!currentJobId) return null;
    const status = await api.get(API_ENDPOINTS.preprocessing.status(currentJobId));
    const s = String(status.status || "unknown").toLowerCase();
    const isRunning = isActiveJobState(s);
    console.log("[raw-stack] refreshStatus", s, "phase:", status.current_phase, "phases:", status.phases?.length);

    // Run-ID aus Status oder localStorage
    const runId = String(
      status.job?.data?.run_id || status.run_id ||
      localStorage.getItem(PREPROCESSING_RUN_ID_KEY) || ""
    ).trim();

    // events.jsonl: Log + livePhase
    const { livePhase, livePct, logLines } = await loadEventsLog(runId);

    // Log-Box befüllen
    if (logBox && logLines.length > 0) {
      logBox.textContent = logLines.join("\n");
      scrollLogToEnd(logBox);
    }

    // Phasen-Chips: Backend-Array bevorzugen, sonst aus livePhase ableiten
    const backendPhase = String(status.current_phase || "").trim();
    if (isRunning && !backendPhase && livePhase) {
      rawStackRenderPhases(status, livePhase, livePct);
    } else {
      rawStackRenderPhases(status);
    }

    // Status-Chip
    const effectivePhase = backendPhase || (isRunning ? livePhase : "");
    const pctStr = isRunning && livePct > 0 ? ` ${livePct}%` : "";
    if (s === "ok") {
      setStatusChip(chip, "OK", "ok");
    } else if (s === "error" || s === "failed") {
      setStatusChip(chip, "error", "error");
    } else if (s === "cancelled") {
      setStatusChip(chip, "cancelled", "check");
    } else if (isRunning) {
      setStatusChip(chip, effectivePhase ? `running: ${effectivePhase}${pctStr}` : "running…", "running");
    } else {
      setStatusChip(chip, s || "idle", "check");
    }

    // frame_quality.csv nach Laufende laden
    if (!isRunning && runId) {
      rawStackLoadFrameQuality(runId).catch(() => {});
    }

    return status;
  }

  // Poll-Schleife solange aktiv (wie bindRunMonitor)
  async function startPolling() {
    const poll = async () => {
      const status = await refreshStatus().catch(() => null);
      if (isActiveJobState(String(status?.status || ""))) {
        window.setTimeout(poll, 2000);
      }
    };
    window.setTimeout(poll, 1000);
  }

  // --- Event-Handler ---

  document.querySelectorAll("#inp-dirs,#inp-colormode,#inp-bayer,#cal-bias,#cal-bias-source,#cal-bias-dir,#cal-dark,#cal-dark-source,#cal-dark-dir,#cal-flat,#cal-flat-source,#cal-flat-dir,#raw-stack-quality-mode,#raw-stack-min-stars,#raw-stack-min-corr,#raw-stack-fwhm-sigma,#raw-stack-ecc,#raw-stack-rejection,#raw-stack-rej-low,#raw-stack-rej-high,#raw-stack-normalization,#raw-stack-weighting,#raw-stack-astrometry,#raw-stack-bge,#raw-stack-pcc,#raw-stack-hms").forEach((el) => {
    el.addEventListener("input", rawStackUpdateJson);
    el.addEventListener("change", rawStackUpdateJson);
  });

  $("raw-stack-load-defaults")?.addEventListener("click", () => void loadDefaults(false));
  $("raw-stack-reset-defaults")?.addEventListener("click", () => void loadDefaults(true));

  $("raw-stack-validate-params")?.addEventListener("click", async () => {
    const statusEl = $("raw-stack-validate-status");
    const detailsEl = $("raw-stack-validate-details");
    const setVS = (text, color) => { if (statusEl) { statusEl.textContent = text; statusEl.style.color = color; } };
    const clearDetails = () => { if (detailsEl) { detailsEl.innerHTML = ""; detailsEl.style.display = "none"; } };
    try {
      clearDetails();
      setStatusChip(chip, "validating", "running");
      rawStackRenderPhases({ phases: [{ phase: "CONFIG_VALIDATION", status: "running", pct: 0 }] });
      setVS("Validierung läuft…", "#475569");
      const result = await api.patch(API_ENDPOINTS.preprocessing.parameters, { config: currentConfig() });
      rawStackApplyConfig(result.config || {});
      rawStackRenderParameterGroups(result.groups || [], rawStackReadConfig());
      const validation = result.validation || {};
      if (detailsEl) {
        const checks = Array.isArray(validation.checks) ? validation.checks : [];
        detailsEl.innerHTML = "";
        const summary = document.createElement("div");
        summary.textContent = "Scope: configuration. Bilddaten, Stack-Farbkanäle und Postprocess-Outputs werden erst beim Lauf geprüft.";
        detailsEl.appendChild(summary);
        if (checks.length > 0) {
          const ul = document.createElement("ul");
          ul.style.cssText = "margin:4px 0 0 18px;padding:0;";
          checks.forEach((check) => {
            const li = document.createElement("li");
            li.textContent = String(check);
            ul.appendChild(li);
          });
          detailsEl.appendChild(ul);
        }
        detailsEl.style.display = "block";
      }
      setStatusChip(chip, "validated", "ok");
      rawStackRenderPhases({ phases: [{ phase: "CONFIG_VALIDATION", status: "ok", pct: 1 }] });
      setVS("Validierung: OK", "#166534");
      setFooter("Raw-Stack-Parameter validiert.");
    } catch (err) {
      const msg = errorText(err);
      setStatusChip(chip, "validation error", "error");
      rawStackRenderPhases({ phases: [{ phase: "CONFIG_VALIDATION", status: "error", pct: 1 }] });
      setVS(`Validierung: ERROR – ${msg}`, "#b91c1c");
      if (detailsEl) {
        detailsEl.innerHTML = "";
        const title = document.createElement("div");
        title.textContent = "Fehler (1)";
        title.style.cssText = "margin-top:6px;font-weight:600;color:#b91c1c;";
        const ul = document.createElement("ul");
        ul.style.cssText = "margin:4px 0 0 18px;padding:0;";
        const li = document.createElement("li");
        li.textContent = msg;
        ul.appendChild(li);
        detailsEl.appendChild(title);
        detailsEl.appendChild(ul);
        detailsEl.style.display = "block";
      }
      setFooter(`Raw-Stack-Validierung fehlgeschlagen: ${msg}`, true);
    }
  });

  $("raw-stack-start")?.addEventListener("click", async () => {
    try {
      const cfg = currentConfig();
      const explicitName = sanitizeRunName($("scan-run-name")?.value || "");
      const runName = explicitName || ("rs_" + (sanitizeRunName(suggestRunNameFromInputs([cfg.lights_dir])) || "run"));
      const cfgWithName = { ...cfg, run_name: runName };
      setStatusChip(chip, "starting…", "running");
      const accepted = await withPathGrantRetry(
        () => api.post(API_ENDPOINTS.preprocessing.run, cfgWithName),
        { fallbackPath: cfg.lights_dir },
      );
      currentJobId = accepted.job_id || "";
      if (currentJobId) localStorage.setItem(PREPROCESSING_JOB_KEY, currentJobId);
      if (accepted.run_id) localStorage.setItem(PREPROCESSING_RUN_ID_KEY, String(accepted.run_id));
      if (accepted.run_dir) localStorage.setItem(PREPROCESSING_RUN_DIR_KEY, String(accepted.run_dir));
      await refreshStatus().catch(() => {});
      startPolling();
    } catch (err) {
      setStatusChip(chip, "error", "error");
      setFooter(`Raw-Stack-Lauf fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("raw-stack-cancel")?.addEventListener("click", async () => {
    if (!currentJobId) return;
    try {
      await api.post(API_ENDPOINTS.preprocessing.cancel, { job_id: currentJobId });
      setStatusChip(chip, "cancelled", "check");
      setFooter("Raw-Stack-Lauf abgebrochen.");
    } catch (err) {
      setFooter(`Raw-Stack-Cancel fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("raw-stack-open-report")?.addEventListener("click", async () => {
    const jobId = currentJobId || localStorage.getItem(PREPROCESSING_JOB_KEY) || "";
    if (!jobId) return;
    const report = await api.get(API_ENDPOINTS.preprocessing.report(jobId)).catch(() => null);
    if ($("raw-stack-artifacts") && report) {
      $("raw-stack-artifacts").textContent = `JSON: ${report.report_json || "-"} | HTML: ${report.report_html || "-"} | Manifest: ${report.manifest || "-"}`;
    }
  });

  // --- Init: Defaults laden + letzten Job-Status wiederherstellen ---
  await loadDefaults().catch((err) => setFooter(`Raw-Stack-Defaults nicht geladen: ${errorText(err)}`, true));
  if (currentJobId) {
    const status = await refreshStatus().catch(() => null);
    if (isActiveJobState(String(status?.status || ""))) {
      startPolling();
    }
  }
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

function findRunQueueSection() {
  return document.querySelector("[data-queue-section='run']");
}

function getPersistedDetectedColorMode() {
  return normalizeDetectedColorMode(readServerUiStateValue(LAST_SCAN_COLOR_MODE_KEY) || "");
}

function setRunQueueVisible(selectedModeRaw) {
  const dashboardAdvancedBtn = $("dashboard-guided-mode-advanced");
  const dashboardHasModeToggle = Boolean($("dashboard-guided-mode-simple") || dashboardAdvancedBtn);
  const dashboardAdvancedVisible = !dashboardHasModeToggle || Boolean(dashboardAdvancedBtn?.classList.contains("active"));
  document.querySelectorAll(".guided-queue-only").forEach((el) => {
    el.style.display = dashboardAdvancedVisible ? "" : "none";
  });
  const sec = findRunQueueSection();
  if (sec) sec.style.display = dashboardAdvancedVisible ? "" : "none";
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

function normalizePersistedQueueItems(items) {
  return (Array.isArray(items) ? items : [])
    .map((item) => {
      const filter = String(item?.filter || "").trim();
      const inputDir = String(item?.input_dir || item?.input_path || "").trim();
      const pattern = String(item?.pattern || "").trim();
      const runLabel = String(item?.run_id || "").trim();
      const enabled = item?.enabled !== false;
      if (!enabled || !inputDir) return null;
      const normalized = { filter, input_dir: inputDir };
      if (pattern) normalized.pattern = pattern;
      if (runLabel) normalized.run_id = runLabel;
      return normalized;
    })
    .filter(Boolean);
}

function persistedQueueRowsForMonitor() {
  const preferred = normalizePersistedQueueItems(storedJsonValue(activeQueueStorageKey(), []));
  if (preferred.length > 0) return preferred;
  const dashboard = normalizePersistedQueueItems(storedJsonValue(UI_STORAGE_KEYS.dashboardQueue, []));
  if (dashboard.length > 0) return dashboard;
  return normalizePersistedQueueItems(storedJsonValue(UI_STORAGE_KEYS.wizardQueue, []));
}

function queueRowsForRunStart(source = "") {
  const normalizedSource = String(source || "").trim().toLowerCase();
  if (normalizedSource === "dashboard" || normalizedSource === "wizard") {
    return collectQueueRows();
  }
  const currentPageRows = collectQueueRows();
  if (currentPageRows.length > 0) return currentPageRows;
  return persistedQueueRowsForMonitor();
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
  void bindInputDirectoryPicker({
    inputId: "dashboard-input-dirs",
    browseId: "dashboard-input-dirs-browse-btn",
  });
  $("dashboard-input-dirs-add-btn")?.addEventListener("click", async () => {
    try {
      await addCurrentInputDirsToQueue({
        inputId: "dashboard-input-dirs",
        colorModeId: "dashboard-color-mode",
        storageKey: UI_STORAGE_KEYS.dashboardQueue,
      });
    } catch (err) {
      setFooter(`Run-Queue konnte nicht erweitert werden: ${errorText(err)}`, true);
    }
  });
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
      const fallbackRunName = preferredRunName({
        inputId: "dashboard-run-name",
        storageKey: UI_STORAGE_KEYS.dashboardRunName,
        fallbackDirs: parseInputDirs($("dashboard-input-dirs")?.value || ""),
      });
      const queueItems = collectQueueRows();
      if ($("dashboard-run-name") && sanitizedRunName) $("dashboard-run-name").value = sanitizedRunName;
      persistTextValue(UI_STORAGE_KEYS.dashboardRunsDir, runsDir, { absolute: true });
      if (!$("dashboard-run-path-preview")) return;
      $("dashboard-run-path-preview").value = buildRunPathPreview({
        runsDir,
        explicitRunName: sanitizedRunName,
        fallbackRunName,
        queueItems,
      });
    };
    $("dashboard-run-runs-dir")?.addEventListener("input", preview);
    $("dashboard-run-name")?.addEventListener("input", preview);
    findRunQueueSection()?.addEventListener("input", preview);
    findRunQueueSection()?.addEventListener("change", preview);
    document.addEventListener("gui2:queue-changed", preview);
    preview();

    $("dashboard-color-mode")?.addEventListener("change", () => {
      setRunQueueVisible($("dashboard-color-mode")?.value || "");
    });
    setRunQueueVisible($("dashboard-color-mode")?.value || "");
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
    const fallbackRunName = preferredRunName({
      inputId: "wizard-run-name",
      storageKey: UI_STORAGE_KEYS.wizardRunName,
      fallbackDirs: dirs,
    });
    const queueItems = collectQueueRows();
    if ($("wizard-run-name") && sanitizedRunName) $("wizard-run-name").value = sanitizedRunName;
    persistTextValue(UI_STORAGE_KEYS.wizardRunsDir, runsDir, { absolute: true });
    const previewEl = $("wizard-run-path-preview");
    if (!previewEl) return;
    previewEl.value = buildRunPathPreview({
      runsDir,
      explicitRunName: sanitizedRunName,
      fallbackRunName,
      queueItems,
    });
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
    setRunQueueVisible($("inp-colormode")?.value || "");
  });
  setRunQueueVisible($("inp-colormode")?.value || "");
  findRunQueueSection()?.addEventListener("input", updateWizardPreview);
  findRunQueueSection()?.addEventListener("change", updateWizardPreview);
  document.addEventListener("gui2:queue-changed", updateWizardPreview);
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
  bindUiStateNavigationFlush();
  bindLocaleControls();
  bindRunMonitorFilterSync();
  bindScanPages();
  await bindParameterStudio();
  await bindRunMonitor();
  await bindHistoryPage();
  await bindAstrometryPage();
  await bindPccPage();
  await bindRawStackPage();
  await bindLiveLogPage();
  await bindDashboard();
  await bindWizard();
  await bindAssumptions();
}

document.addEventListener("DOMContentLoaded", () => {
  void init();
});
