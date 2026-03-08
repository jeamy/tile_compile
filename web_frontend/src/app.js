import { ApiClient } from "./api.js";
import { API_ENDPOINTS } from "./constants.js";

const api = new ApiClient(localStorage.getItem("gui2.backendBase") || "");
const CONFIG_DRAFT_KEY = "gui2.configYamlDraft";
const LOCALE_KEY = "gui2.locale";
const LAST_INPUT_DIRS_KEY = "gui2.lastInputDirs";

const uiState = {
  currentRunId: localStorage.getItem("gui2.currentRunId") || "",
  currentRunDir: "",
  selectedHistoryRunId: "",
  configYaml: "",
  configObject: null,
  parameterDirty: {},
  runSocket: null,
  liveSocket: null,
  liveLines: [],
  liveFilter: "all",
  lastAstrometryWcs: "",
  lastPccOutput: "",
  lastPccChannels: [],
  lastPccResult: null,
  locale: localStorage.getItem(LOCALE_KEY) || "de",
  projectRunsDir: "",
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
    await api.post("/api/fs/grant-root", { path: candidatePath });
    return fn();
  }
}

function setCurrentRunId(runId) {
  if (!runId) return;
  uiState.currentRunId = String(runId);
  localStorage.setItem("gui2.currentRunId", uiState.currentRunId);
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

function setRunReady(status) {
  const chip = $("status-run-ready");
  if (!chip) return;
  const normalized = String(status || "check").toLowerCase();
  chip.textContent = normalized === "ok" ? "Run Ready" : normalized === "error" ? "Run Blocked" : "Run Check";
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
      job = await api.get(`/api/jobs/${encodeURIComponent(jobId)}`);
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
  return {
    has_scan: hasScan,
    ok,
    input_path: String(src.input_path || fallbackInputPath || ""),
    input_dirs: inputDirs,
    frames_detected: Number.isFinite(framesDetected) ? framesDetected : 0,
    color_mode: String(src.color_mode || "UNKNOWN"),
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
  const status = !data.has_scan ? "Kein Scan" : data.ok ? "OK" : data.errors.length > 0 ? "ERROR" : "CHECK";
  const sizeText =
    data.image_width > 0 && data.image_height > 0 ? `${data.image_width} x ${data.image_height}` : "unbekannt";
  const candidates = data.color_mode_candidates.length > 0 ? data.color_mode_candidates.join(", ") : "-";
  setText($(`${prefix}-status`), status);
  setText($(`${prefix}-input-path`), data.input_path || "-");
  setText($(`${prefix}-frames`), String(data.frames_detected));
  setText($(`${prefix}-color-mode`), data.color_mode || "UNKNOWN");
  setText($(`${prefix}-candidates`), candidates);
  setText($(`${prefix}-size`), sizeText);
  setText($(`${prefix}-bayer`), data.bayer_pattern || "-");
  setText($(`${prefix}-confirm`), data.requires_user_confirmation ? "ja" : "nein");
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
      api.get("/api/guardrails"),
      api.get("/api/app/state"),
    ]);
    setRunReady(guardrails?.status || "check");
    const rid = appState?.project?.current_run_id;
    if (rid) setCurrentRunId(rid);
    const runsDir = String(appState?.project?.runs_dir || "").trim();
    if (runsDir) uiState.projectRunsDir = runsDir;
    const scanPath = String(appState?.scan?.last_input_path || "").trim();
    if (scanPath) persistLastInputDirs(scanPath);
  } catch (err) {
    setFooter(`Backend nicht erreichbar: ${errorText(err)}`, true);
  }
}

function applyLocale(localeRaw) {
  const locale = String(localeRaw || "de").toLowerCase() === "en" ? "en" : "de";
  uiState.locale = locale;
  localStorage.setItem(LOCALE_KEY, locale);
  document.documentElement.setAttribute("lang", locale);
  $("locale-de")?.classList.toggle("active", locale === "de");
  $("locale-en")?.classList.toggle("active", locale === "en");
}

function bindLocaleControls() {
  applyLocale(uiState.locale);
  $("locale-de")?.addEventListener("click", () => applyLocale("de"));
  $("locale-en")?.addEventListener("click", () => applyLocale("en"));
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
    const accepted = await withPathGrantRetry(() => api.post("/api/scan", payload), {
      fallbackPath: dirs[0] || "",
    });
    if (resultPanel) resultPanel.style.display = "block";
    renderScanSummary(summaryPrefix, { has_scan: true, input_path: payload.input_path, color_mode: "UNKNOWN" });
    setText(resultBody, { state: accepted.state, message: "Scan gestartet..." });
    const job = await waitForJob(accepted.job_id, { allowMissing: true });
    if (String(job?.state) === "missing") {
      const latest = await api.get("/api/scan/latest");
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
      const latest = await api.get("/api/scan/latest");
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
  const colorModeEl = $("inp-colormode");
  if (colorModeEl) {
    const updateQueue = () => setMonoQueueVisible(String(colorModeEl.value || "").toUpperCase() === "MONO");
    colorModeEl.addEventListener("change", updateQueue);
    updateQueue();
  }
  void (async () => {
    try {
      const latest = await api.get("/api/scan/latest");
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

function setParameterPreview(value) {
  const box = parameterDiffBox();
  if (!box) return;
  setText(box, value);
}

async function ensureConfigYaml() {
  if (uiState.configYaml) return uiState.configYaml;
  const draft = getConfigDraft();
  if (draft) {
    uiState.configYaml = draft;
    return draft;
  }
  const current = await api.get("/api/config/current");
  uiState.configYaml = String(current?.config || "");
  setConfigDraft(uiState.configYaml);
  return uiState.configYaml;
}

async function patchConfig({ updates = [], persist = false, yamlText } = {}) {
  const baseYaml = yamlText !== undefined ? String(yamlText || "") : await ensureConfigYaml();
  const result = await api.post("/api/config/patch", {
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
    const presets = await api.get("/api/config/presets");
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
  } catch (err) {
    setFooter(`Preset-Liste konnte nicht geladen werden: ${errorText(err)}`, true);
  }

  $("parameter-yaml-sync")?.addEventListener("click", async () => {
    try {
      const current = await api.get("/api/config/current");
      uiState.configYaml = String(current?.config || "");
      setConfigDraft(uiState.configYaml);
      uiState.parameterDirty = {};
      const parsed = await patchConfig({ yamlText: uiState.configYaml, updates: [] });
      if (parsed?.config) syncParameterFieldsFromConfig(parsed.config);
      setParameterPreview(uiState.configYaml);
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
      const applied = await api.post("/api/config/presets/apply", { path });
      uiState.configYaml = String(applied?.config || "");
      setConfigDraft(uiState.configYaml);
      uiState.parameterDirty = {};
      const parsed = await patchConfig({ yamlText: uiState.configYaml, updates: [] });
      if (parsed?.config) syncParameterFieldsFromConfig(parsed.config);
      setParameterPreview(String(parsed?.config_yaml || uiState.configYaml));
      setFooter("Preset angewendet.");
    } catch (err) {
      setFooter(`Preset anwenden fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-validate")?.addEventListener("click", async () => {
    try {
      const patched = await applyPreview({ persist: false });
      const result = await api.post("/api/config/validate", { yaml: patched?.config_yaml || "" });
      setParameterPreview(result);
      setFooter(result.ok ? "Validierung OK." : "Validierung hat Fehler.");
    } catch (err) {
      setFooter(`Validierung fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-save")?.addEventListener("click", async () => {
    try {
      const result = await applyPreview({ persist: true });
      setParameterPreview(result);
      setFooter(`Config gespeichert. Revision: ${result?.revision_id || "-"}`);
    } catch (err) {
      setFooter(`Speichern fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-review-changes")?.addEventListener("click", async () => {
    try {
      const result = await applyPreview({ persist: false });
      setFooter(`YAML-Vorschau aktualisiert (${result?.applied?.length || 0} Aenderungen).`);
    } catch (err) {
      setFooter(`Vorschau fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("parameter-reset-default")?.addEventListener("click", async () => {
    try {
      const current = await api.get("/api/config/current");
      uiState.parameterDirty = {};
      uiState.configYaml = String(current?.config || "");
      setConfigDraft(uiState.configYaml);
      const parsed = await patchConfig({ yamlText: uiState.configYaml, updates: [] });
      if (parsed?.config) syncParameterFieldsFromConfig(parsed.config);
      setParameterPreview(uiState.configYaml);
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
      setFooter(`Situation angewendet (${scenarioUpdates.length} Deltas).`);
    } catch (err) {
      setFooter(`Situation anwenden fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelectorAll("#parameter-category-list button[data-category]").forEach((btn) => {
    btn.addEventListener("click", () => {
      window.setTimeout(() => {
        if (uiState.configObject) syncParameterFieldsFromConfig(uiState.configObject);
      }, 0);
    });
  });
}

function runMonitorSelectedPhase() {
  const selected = document.querySelector(".ps-phase-row.is-selected .phase-name");
  return selected ? String(selected.textContent || "").trim().toUpperCase() : "";
}

function runMonitorSelectedFilter() {
  const selected = document.querySelector(".ps-chip-btn.active[id^='monitor-filter-']");
  if (!selected) return "";
  return String(selected.textContent || "").trim().toUpperCase();
}

function runMonitorLogBox() {
  return findLogBoxBySectionTitle("Live Log");
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
  const status = await api.get(`/api/runs/${encodeURIComponent(runId)}/status`);
  uiState.currentRunDir = String(status?.run_dir || "");
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
  const revisions = await api.get("/api/config/revisions");
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
  uiState.runSocket = api.ws(
    `/api/ws/runs/${encodeURIComponent(runId)}`,
    (event) => {
      appendLine(logBox, JSON.stringify(event));
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
      appendLine(logBox, `ws_error: ${String(err)}`);
    },
  );
}

async function bindRunMonitor() {
  if (!$("monitor-stop")) return;

  let runId = ensureRunIdFromHeader();
  if (!runId) {
    try {
      const appState = await api.get("/api/app/state");
      runId = appState?.project?.current_run_id || "";
    } catch {
      runId = "";
    }
  }
  if (runId) setCurrentRunId(runId);
  if (!uiState.currentRunId) {
    setFooter("Kein aktueller Run gesetzt. Bitte in History einen Run als Current markieren.", true);
    return;
  }

  try {
    await loadRunRevisions();
    await loadRunStatus(uiState.currentRunId);
    connectRunMonitorStream(uiState.currentRunId);
  } catch (err) {
    setFooter(`Run-Monitor Initialisierung fehlgeschlagen: ${errorText(err)}`, true);
  }

  const updateResumeEnabled = () => {
    const phase = runMonitorSelectedPhase();
    const revisionId = $("monitor-resume-config-revision")?.value || "";
    setDisabledLike($("monitor-resume"), !phase || !revisionId);
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
  updateResumeEnabled();

  $("monitor-stop")?.addEventListener("click", async () => {
    try {
      const result = await api.post(`/api/runs/${encodeURIComponent(uiState.currentRunId)}/stop`, {});
      if (result.ok) {
        const stoppedJobs = Array.isArray(result.cancelled_jobs) ? result.cancelled_jobs.length : 0;
        const killedPids = Array.isArray(result.killed_pids) ? result.killed_pids.length : 0;
        setFooter(`Stop gesendet. Jobs beendet: ${stoppedJobs}, verwaiste Prozesse beendet: ${killedPids}.`);
      } else {
        setFooter("Kein laufender Job/Prozess fuer diesen Run gefunden.", true);
      }
      await loadRunStatus(uiState.currentRunId);
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
      const accepted = await api.post(`/api/runs/${encodeURIComponent(uiState.currentRunId)}/resume`, {
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
      await api.post(`/api/runs/${encodeURIComponent(uiState.currentRunId)}/config-revisions/${encodeURIComponent(revisionId)}/restore`, {});
      const current = await api.get("/api/config/current");
      setConfigDraft(String(current?.config || ""));
      setFooter(`Revision ${revisionId} wiederhergestellt.`);
    } catch (err) {
      setFooter(`Revision-Restore fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-stats-generate")?.addEventListener("click", async () => {
    try {
      const accepted = await api.post(`/api/runs/${encodeURIComponent(uiState.currentRunId)}/stats`, {
        run_dir: uiState.currentRunDir || undefined,
      });
      setFooter(`Stats-Generierung gestartet (Job ${accepted.job_id}).`);
      await waitForJob(accepted.job_id);
    } catch (err) {
      setFooter(`Stats-Generierung fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-stats-open-folder")?.addEventListener("click", async () => {
    try {
      const status = await api.get(`/api/runs/${encodeURIComponent(uiState.currentRunId)}/stats/status`);
      setFooter(`Stats-Ordner: ${status.output_dir || "-"}`);
    } catch (err) {
      setFooter(`Stats-Status fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-report")?.addEventListener("click", async () => {
    try {
      const status = await api.get(`/api/runs/${encodeURIComponent(uiState.currentRunId)}/stats/status`);
      setFooter(`Report: ${status.report_path || "-"}`);
    } catch (err) {
      setFooter(`Report-Status fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("monitor-open-run-folder")?.addEventListener("click", () => {
    setFooter(`Run-Ordner: ${uiState.currentRunDir || "-"}`);
  });
}

async function bindHistoryPage() {
  const list = document.querySelector(".ps-section ul.ps-list");
  if (!list || !$("history-refresh")) return;

  const render = async () => {
    const runs = await api.get("/api/runs");
    const items = runs.items || [];
    if (items.length === 0) {
      list.innerHTML = "<li><button>Keine Runs gefunden</button></li>";
      return;
    }
    if (!uiState.selectedHistoryRunId) {
      uiState.selectedHistoryRunId = uiState.currentRunId || items[0].run_id;
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
        render().catch(() => {});
      });
    });
  };

  $("history-refresh").addEventListener("click", () => void render());
  $("history-set-current")?.addEventListener("click", async () => {
    if (!uiState.selectedHistoryRunId) return;
    try {
      await api.post(`/api/runs/${encodeURIComponent(uiState.selectedHistoryRunId)}/set-current`, {});
      setCurrentRunId(uiState.selectedHistoryRunId);
      setFooter(`Current Run gesetzt: ${uiState.selectedHistoryRunId}`);
    } catch (err) {
      setFooter(`Set Current fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  $("history-open-report")?.addEventListener("click", async () => {
    if (!uiState.selectedHistoryRunId) return;
    try {
      const status = await api.get(`/api/runs/${encodeURIComponent(uiState.selectedHistoryRunId)}/stats/status`);
      setFooter(`Report: ${status.report_path || "-"}`);
    } catch (err) {
      setFooter(`Report-Status fehlgeschlagen: ${errorText(err)}`, true);
    }
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

  const append = (msg) => appendLine(logBox, typeof msg === "string" ? msg : JSON.stringify(msg));

  async function detect() {
    const payload = {
      astap_cli: $("tools-astrometry-bin")?.value || "",
      astap_data_dir: $("tools-astrometry-data-dir")?.value || "",
    };
    const result = await api.post("/api/tools/astrometry/detect", payload);
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
      const accepted = await api.post("/api/tools/astrometry/install-cli", {
        astap_data_dir: $("tools-astrometry-data-dir")?.value || "",
      });
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
      const accepted = await api.post("/api/tools/astrometry/catalog/download", {
        catalog_id: catalogId,
        astap_data_dir: $("tools-astrometry-data-dir")?.value || "",
      });
      append(accepted);
      const job = await waitForJob(accepted.job_id, { onTick: (j) => append({ state: j.state, progress: j.data?.progress ?? null }) });
      append(job);
    } catch (err) {
      setFooter(`Catalog-Download fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.astrometry.cancel_download']")?.addEventListener("click", async () => {
    try {
      const result = await api.post("/api/tools/astrometry/catalog/cancel", {});
      append(result);
    } catch (err) {
      setFooter(`Catalog-Cancel fehlgeschlagen: ${errorText(err)}`, true);
    }
  });

  document.querySelector("[data-control='tools.astrometry.solve']")?.addEventListener("click", async () => {
    try {
      const accepted = await api.post("/api/tools/astrometry/solve", {
        solve_file: $("tools-astrometry-file")?.value || "",
        astap_cli: $("tools-astrometry-bin")?.value || "",
        astap_data_dir: $("tools-astrometry-data-dir")?.value || "",
      });
      append(accepted);
      const job = await waitForJob(accepted.job_id);
      uiState.lastAstrometryWcs = String(job?.data?.wcs_path || "");
      append(job);
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
      const result = await api.post("/api/tools/astrometry/save-solved", {
        input_path: input,
        output_path: output,
        wcs_path: uiState.lastAstrometryWcs || undefined,
      });
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

  function render() {
    const lines = uiState.liveLines.filter((item) => uiState.liveFilter === "all" || item.level === uiState.liveFilter);
    const escapeHtml = (text) =>
      String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
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
  }

  levelButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const t = String(btn.textContent || "").trim().toLowerCase();
      if (t === "clear") {
        uiState.liveLines = [];
        render();
        return;
      }
      uiState.liveFilter = t;
      render();
    });
  });

  const runId = uiState.currentRunId || ensureRunIdFromHeader();
  if (!runId) {
    setFooter("Kein aktueller Run gesetzt. Bitte in History einen Run als Current markieren.", true);
    return;
  }
  try {
    const logs = await api.get(`/api/runs/${encodeURIComponent(runId)}/logs?tail=250`);
    uiState.liveLines = (logs.lines || []).map((line) => ({ line, level: detectLevel(line) }));
    render();
    if (uiState.liveSocket) uiState.liveSocket.close();
    uiState.liveSocket = api.ws(
      `/api/ws/runs/${encodeURIComponent(runId)}`,
      (event) => {
        const line = typeof event === "string" ? event : JSON.stringify(event);
        uiState.liveLines.push({ line, level: detectLevel(line) });
        if (uiState.liveLines.length > 600) uiState.liveLines = uiState.liveLines.slice(-600);
        render();
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
    return title && String(title.textContent || "").toLowerCase().includes("mono filter-queue");
  });
}

function setMonoQueueVisible(isMono) {
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

async function populatePresetSelect(selectId, keepCurrent = true) {
  const select = $(selectId);
  if (!select) return;
  const old = select.value;
  const presets = await api.get("/api/config/presets");
  if (!Array.isArray(presets?.items) || presets.items.length === 0) return;
  select.innerHTML = "";
  presets.items.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.path;
    opt.textContent = item.name;
    select.appendChild(opt);
  });
  if (keepCurrent && old) select.value = old;
}

function sanitizeRunName(raw) {
  let value = String(raw || "").trim();
  if (!value) return "";
  value = value.replace(/[\\/]+/g, "_").replace(/\s+/g, "_");
  value = value.replace(/[^A-Za-z0-9._-]+/g, "_");
  value = value.replace(/_+/g, "_");
  return value.replace(/^[._-]+|[._-]+$/g, "");
}

function suggestRunNameFromInputs(inputDirs) {
  const first = String(inputDirs?.[0] || "").trim();
  if (!first) return `run_${timestampSuffix()}`;
  const trimmed = first.endsWith("/") ? first.slice(0, -1) : first;
  const base = trimmed.split(/[\\/]/).filter(Boolean).pop() || "run";
  const safe = sanitizeRunName(base);
  return safe || `run_${timestampSuffix()}`;
}

function collectInputCalibrationUpdates() {
  const updates = [];
  const push = (selector, path) => {
    const el = selector.startsWith("#")
      ? document.getElementById(selector.slice(1))
      : document.querySelector(selector);
    if (!el) return;
    updates.push({ path, value: readFieldValue(el) });
  };
  push("#inp-pattern", "input.pattern");
  push("#inp-maxframes", "input.max_frames");
  push("#inp-sort", "input.sort");
  push("#inp-bayer", "data.bayer_pattern");
  push("#cal-bias", "calibration.use_bias");
  push("#cal-bias-dir", "calibration.bias_dir");
  push("#cal-dark", "calibration.use_dark");
  push("#cal-dark-dir", "calibration.darks_dir");
  push("#cal-flat", "calibration.use_flat");
  push("#cal-flat-dir", "calibration.flats_dir");
  return updates;
}

async function resolveConfigYamlForRun({ source }) {
  const updates = source === "wizard" ? collectInputCalibrationUpdates() : [];
  if (updates.length === 0) return ensureConfigYaml();
  const patched = await patchConfig({ updates, persist: false });
  return String(patched?.config_yaml || "");
}

async function startRunFromCurrentForm({ source }) {
  const isDashboard = source === "dashboard";
  const inputDirsEl = isDashboard ? $("dashboard-input-dirs") : $("inp-dirs");
  const colorModeEl = isDashboard ? $("dashboard-color-mode") : $("inp-colormode");
  const runsDirEl = isDashboard ? $("dashboard-run-runs-dir") : $("wizard-runs-dir");
  const runNameEl = isDashboard ? $("dashboard-run-name") : $("wizard-run-name");

  const inputDirs = parseInputDirs(inputDirsEl?.value || "");
  if (inputDirs.length > 0) {
    persistLastInputDirs(inputDirs.join(", "));
  }
  const colorMode = String(colorModeEl?.value || "OSC").toUpperCase();
  const queue = colorMode === "MONO" ? collectQueueRows() : [];
  let runName = sanitizeRunName(runNameEl?.value || "");
  if (!runName) runName = suggestRunNameFromInputs(inputDirs);
  if (runNameEl) runNameEl.value = runName;
  if (runName) {
    queue.forEach((item, idx) => {
      if (!item.run_id) {
        const suffix = item.filter || `q${idx + 1}`;
        item.run_id = sanitizeRunName(`${runName}_${suffix}`) || `${runName}_${suffix}`;
      }
    });
  }

  const payload = {
    color_mode: colorMode,
    run_name: runName || undefined,
    runs_dir: String(runsDirEl?.value || "").trim() || undefined,
    config_yaml: await resolveConfigYamlForRun({ source }),
  };
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
  return withPathGrantRetry(() => api.post("/api/runs/start", payload), {
    fallbackPath: String(payload.runs_dir || inputDirs[0] || ""),
  });
}

async function bindDashboard() {
  if (!$("dashboard-kpi-scan-quality")) return;
  bindInputDirMemory("dashboard-input-dirs");
  const runsDirInput = $("dashboard-run-runs-dir");
  if (runsDirInput && !String(runsDirInput.value || "").trim() && uiState.projectRunsDir) {
    runsDirInput.value = uiState.projectRunsDir;
  }
  try {
    const [quality, guardrails, latestScan] = await Promise.all([
      api.get("/api/scan/quality"),
      api.get("/api/guardrails"),
      api.get("/api/scan/latest"),
    ]);
    setRunReady(guardrails?.status || "check");
    const summary = summarizeScanResult(
      latestScan?.has_scan ? latestScan : quality?.scan || {},
      String($("dashboard-input-dirs")?.value || "").trim(),
    );
    renderDashboardScanKpis(summary, quality?.score ?? 0);
    renderScanSummary("dashboard-scan", summary);
    applyDetectedColorModeToSelect($("dashboard-color-mode"), summary);
    applyDetectedColorModeToSelect($("inp-colormode"), summary);
    const mergedInputText = summary.input_dirs?.length > 0 ? summary.input_dirs.join(", ") : summary.input_path;
    if (mergedInputText) {
      $("dashboard-input-dirs") && ($("dashboard-input-dirs").value = mergedInputText);
      persistLastInputDirs(summary.input_path);
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
      setMonoQueueVisible(String($("dashboard-color-mode")?.value || "").toUpperCase() === "MONO");
    });
    setMonoQueueVisible(String($("dashboard-color-mode")?.value || "").toUpperCase() === "MONO");
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
        const applied = await api.post("/api/config/presets/apply", { path });
        setConfigDraft(String(applied?.config || ""));
        setFooter("Preset fuer Guided Run aktualisiert.");
      } catch (err) {
        setFooter(`Preset-Laden fehlgeschlagen: ${errorText(err)}`, true);
      }
    });

    $("dashboard-run-start")?.addEventListener("click", async (ev) => {
      ev.preventDefault();
      try {
        const latestGuardrails = await api.get("/api/guardrails");
        if (String(latestGuardrails?.status || "").toLowerCase() === "error") {
          setFooter("Run blockiert: Guardrail-Status ist ERROR.", true);
          return;
        }
        const accepted = await startRunFromCurrentForm({ source: "dashboard" });
        setCurrentRunId(accepted?.run_id || uiState.currentRunId);
        setFooter(`Run gestartet (Job ${accepted?.job_id || "-"}).`);
        window.location.href = "run-monitor.html";
      } catch (err) {
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
              "/api/scan",
              buildScanPayloadFromDirs(
                dirs,
                1,
                false,
              ),
            ),
          { fallbackPath: dirs[0] || "" },
        );
        setFooter(`Scan gestartet (Job ${accepted.job_id}).`);
        await waitForJob(accepted.job_id, { allowMissing: true });
        const [quality2, guardrails2, latest2] = await Promise.all([
          api.get("/api/scan/quality"),
          api.get("/api/guardrails"),
          api.get("/api/scan/latest"),
        ]);
        const summary2 = summarizeScanResult(latest2?.has_scan ? latest2 : quality2?.scan || {}, dirs[0] || "");
        renderDashboardScanKpis(summary2, quality2?.score ?? 0);
        renderScanSummary("dashboard-scan", summary2);
        applyDetectedColorModeToSelect($("dashboard-color-mode"), summary2);
        applyDetectedColorModeToSelect($("inp-colormode"), summary2);
        setRunReady(guardrails2?.status || "check");
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
      } catch (err) {
        setFooter(`Scan neu fehlgeschlagen: ${errorText(err)}`, true);
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
    setMonoQueueVisible(String($("inp-colormode")?.value || "").toUpperCase() === "MONO");
  });
  setMonoQueueVisible(String($("inp-colormode")?.value || "").toUpperCase() === "MONO");
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
      const applied = await api.post("/api/config/presets/apply", { path });
      setConfigDraft(String(applied?.config || ""));
      const v = await api.post("/api/config/validate", { yaml: String(applied?.config || "") });
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
      const v = await api.post("/api/config/validate", { yaml: patched?.config_yaml || "" });
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
