import { escapeHtml, getActiveLocale, getStorageJson, setStorageJson, STORAGE_KEYS } from "./src/utils.js";

(function () {
  const LOCALE_KEY = STORAGE_KEYS.locale;
  const PARAMETER_UI_KEYS = {
    category: "gui2.parameterStudio.category",
    search: "gui2.parameterStudio.search",
    scenarios: "gui2.parameterStudio.scenarios",
  };
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
  const PHASE_MAP = {
    assumptions: "ASSUMPTIONS",
    aqmh: "AQMH_QUALITY_MAPS",
    astrometry: "ASTROMETRY",
    bge: "BGE",
    calibration: "CALIBRATION",
    chroma_denoise: "CHROMA_DENOISE",
    data: "DATA",
    debayer: "DEBAYER",
    dithering: "DITHERING",
    global_metrics: "GLOBAL_METRICS",
    input: "INPUT",
    linearity: "LINEARITY",
    local_metrics: "LOCAL_METRICS",
    normalization: "NORMALIZATION",
    output: "OUTPUT",
    pcc: "PCC",
    pipeline: "PIPELINE",
    registration: "REGISTRATION",
    runtime_limits: "RUNTIME_LIMITS",
    stacking: "STACKING",
    synthetic: "SYNTHETIC",
    tile: "TILE",
    tile_denoise: "TILE_DENOISE",
    validation: "VALIDATION",
    run_dir: "SYSTEM",
    log_level: "SYSTEM",
  };
  const AQMH_CLASSIC_ONLY_CATEGORIES = new Set(["local_metrics", "synthetic"]);

  const categoryListEl = document.getElementById("parameter-category-list");
  let categoryButtons = Array.from(document.querySelectorAll("#parameter-category-list button[data-category]"));
  const parameterGroups = Array.from(document.querySelectorAll(".ps-parameter-group"));
  const editorGroup = document.getElementById("parameter-full-editor-group");
  const editorTitleEl = document.getElementById("parameter-full-editor-title");
  const editorNoteEl = document.getElementById("parameter-full-editor-note");
  const editorMetaEl = document.getElementById("parameter-editor-meta");
  const editorFieldsEl = document.getElementById("parameter-editor-fields");
  const searchInput = document.getElementById("parameter-search");
  const searchSummaryEl = document.getElementById("parameter-search-summary");
  const searchResultsEl = document.getElementById("parameter-search-results");
  const legacyParamEditorIndex = Array.isArray(window.PARAM_EDITOR_INDEX) ? window.PARAM_EDITOR_INDEX : [];
  let paramEditorIndex = [];
  const explainIndex = new Map();
  const validSchemaPaths = new Set();
  const staticRows = Array.from(document.querySelectorAll(".ps-section.ps-parameter-group .ps-row"));
  let localeMessages = {};
  let schemaCategoryOrder = [];
  let activeCategory = "aqmh";
  let activeExplainPath = "registration.star_topk";

  function refreshCategoryButtons() {
    categoryButtons = Array.from(document.querySelectorAll("#parameter-category-list button[data-category]"));
    return categoryButtons;
  }

  function humanizeCategory(category) {
    const normalized = String(category || "").trim();
    if (!normalized) return "";
    if (normalized === "all") return textFor("page.parameter_studio.category.all", "All");
    return normalized
      .split("_")
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }

  function persistCategory(category) {
    const value = String(category || "").trim();
    if (!value) {
      localStorage.removeItem(PARAMETER_UI_KEYS.category);
      return;
    }
    localStorage.setItem(PARAMETER_UI_KEYS.category, value);
  }

  function persistSearch(rawValue) {
    const value = String(rawValue || "");
    if (!value.trim()) {
      localStorage.removeItem(PARAMETER_UI_KEYS.search);
      return;
    }
    localStorage.setItem(PARAMETER_UI_KEYS.search, value);
  }

  function persistScenarioState() {
    const active = Array.from(document.querySelectorAll(".ps-chip-btn.active"))
      .map((el) => String(el.dataset.scenario || "").trim())
      .filter(Boolean);
    localStorage.setItem(PARAMETER_UI_KEYS.scenarios, JSON.stringify(active));
  }

  function textFor(key, fallback) {
    return localeMessages[key] ?? fallback;
  }

  function isGermanLocale() {
    return getLocale() !== "en";
  }

  const scenarioNames = {
    altaz: () => textFor("page.parameter_studio.scenario.altaz", "Alt/Az"),
    rotation: () => textFor("page.parameter_studio.scenario.rotation", "Starke Rotation"),
    bright_stars: () => textFor("page.parameter_studio.scenario.bright_stars", "Helle Sterne"),
    few_frames: () => textFor("page.parameter_studio.scenario.few_frames", "Wenige Frames"),
    gradient: () => textFor("page.parameter_studio.scenario.gradient", "Starker Gradient"),
  };

  function categoryLabel(category) {
    refreshCategoryButtons();
    const button = categoryButtons.find((item) => item.dataset.category === category);
    return String(button?.textContent || humanizeCategory(category) || category || "").trim();
  }

  function orderedCategories(entries) {
    refreshCategoryButtons();
    const fromButtons = categoryButtons
      .filter((button) => !button.hidden)
      .map((button) => String(button.dataset.category || "").trim())
      .filter((category) => category && category !== "all");
    const entryCategories = Array.from(
      new Set(entries.map((entry) => String(entry.category || "").trim()).filter(Boolean)),
    );
    const preferred = schemaCategoryOrder.filter((category) => category && category !== "all");
    const seen = new Set();
    return [...fromButtons, ...preferred, ...entryCategories].filter((category) => {
      if (seen.has(category)) return false;
      seen.add(category);
      return true;
    });
  }

  const scenarioDeltas = {
    altaz: [
      ["registration.allow_rotation", "true", "allow_rotation"],
      ["registration.transform_model", "affine", "affine_model_for_rotating_sessions"],
      ["registration.star_topk", "180", "more_star_candidates"],
      ["registration.reject_shift_px_min", "120", "tolerate_large_natural_shifts"],
      ["registration.reject_shift_median_multiplier", "5.0", "wider_shift_distribution"],
    ],
    rotation: [
      ["registration.engine", "triangle_star_matching", "star_matching_required_for_rotation"],
      ["registration.auto_engine", "true", "auto_engine_detects_rotation"],
      ["registration.allow_rotation", "true", "required_for_rotation"],
      ["registration.transform_model", "affine", "affine_model_for_rotating_sessions"],
      ["registration.star_inlier_tol_px", "4.0", "more_tolerant_inlier_condition"],
      ["registration.reject_cc_min_abs", "0.25", "avoid_too_strict_cc_limits"],
    ],
    bright_stars: [
      ["pcc.mag_bright_limit", "6", "limit_very_bright_stars"],
      ["pcc.k_max", "2.4", "limit_extreme_color_gains"],
      ["pcc.sigma_clip", "2.7", "more_robust_outlier_suppression"],
      ["bge.mask.star_dilate_px", "6", "mask_star_surroundings_stronger"],
    ],
    few_frames: [
      ["assumptions.frames_reduced_threshold", "200", "switch_earlier_to_reduced_mode"],
      ["assumptions.reduced_mode_skip_clustering", "true", "avoid_unstable_clustering"],
      ["synthetic.frames_min", "4", "ensure_minimum_synthetic_base"],
      ["synthetic.clustering.cluster_count_range", "[3,10]", "smaller_cluster_count"],
    ],
    gradient: [
      ["bge.enabled", "true", "model_gradient_explicitly"],
      ["bge.fit.method", "rbf", "flexible_gradient_model"],
      ["bge.fit.rbf_lambda", "1e-2", "regularization_against_overshoot"],
      ["bge.sample_estimator", "quantile", "robust_background_samples"],
      ["bge.sample_quantile", "0.15", "robust_background_samples"],
      ["bge.structure_thresh_percentile", "0.80", "separate_structure_from_background"],
    ],
  };

  function getLocale() {
    return getActiveLocale();
  }

  function hasOwn(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj, key);
  }

  function formatValue(value) {
    if (value === null || value === undefined || value === "") return "-";
    if (Array.isArray(value) || (value && typeof value === "object")) return JSON.stringify(value);
    return String(value);
  }

  function formatEditorValue(entry, value) {
    if (value === null || value === undefined || value === "") return value;
    return value;
  }

  function computeRange(entry) {
    if (entry.range) return String(entry.range);
    const hints = [];
    if (entry?.minimum !== undefined && entry.minimum !== null) hints.push(`>= ${entry.minimum}`);
    if (entry?.exclusiveMinimum !== undefined && entry.exclusiveMinimum !== null) hints.push(`> ${entry.exclusiveMinimum}`);
    if (entry?.maximum !== undefined && entry.maximum !== null) hints.push(`<= ${entry.maximum}`);
    if (entry?.exclusiveMaximum !== undefined && entry.exclusiveMaximum !== null) hints.push(`< ${entry.exclusiveMaximum}`);
    if (Array.isArray(entry.enum) && entry.enum.length > 0) hints.push(entry.enum.map((item) => String(item)).join(" | "));
    return hints.join(", ");
  }

  function labelForPath(path) {
    return localeMessages[`param.${path}.label`] || path;
  }

  function topLevelKeyForPath(path) {
    return String(path || "").split(".")[0] || "";
  }

  function categoryForPath(path, editorEntry = {}) {
    const fallback = topLevelKeyForPath(path);
    return String(editorEntry?.category || fallback || "").trim() || fallback;
  }

  function shortHelpForPath(path, fallback) {
    return localeMessages[`param.${path}.short_help`] || fallback || "-";
  }

  function normalizeExplainText(value) {
    return String(value || "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function looksGermanText(value) {
    const text = normalizeExplainText(value).toLowerCase();
    if (!text) return false;
    if (/[äöüß]/.test(text)) return true;
    return /\b(und|oder|fuer|für|mit|ohne|bei|nur|mehr|wenig|wert|werte|bereich|hintergrund|sterne|aktivieren|deaktivieren|schwelle|mindest|maximale|minimale|anzahl|pfad|bild|stark|robust|feld|zulaessig|zulässig|empfohlen|bearbeiten|erklaer|erklär)\b/.test(text);
  }

  function firstNonEmpty(values) {
    return values.map((value) => normalizeExplainText(value)).find(Boolean) || "";
  }

  function chooseLocalizedExplainText({ schemaDescription, katalogShortExplanation, refDePurpose, refEnPurpose, editorDescription }) {
    const schemaText = normalizeExplainText(schemaDescription);
    const katalogText = normalizeExplainText(katalogShortExplanation);
    const refDeText = normalizeExplainText(refDePurpose);
    const refEnText = normalizeExplainText(refEnPurpose);
    const editorText = normalizeExplainText(editorDescription);
    if (isGermanLocale()) {
      return firstNonEmpty([
        katalogText,
        editorText,
        refDeText,
        schemaText,
        refEnText,
      ]);
    }
    return firstNonEmpty([
      schemaText,
      refEnText,
      looksGermanText(katalogText) ? "" : katalogText,
      looksGermanText(editorText) ? "" : editorText,
      refDeText,
      editorText,
      katalogText,
    ]);
  }

  function sameExplainText(a, b) {
    return normalizeExplainText(a).toLowerCase() === normalizeExplainText(b).toLowerCase();
  }

  function typeLabelForEntry(entry) {
    const type = String(entry?.type || "").trim();
    if (!type) return "-";
    if (Array.isArray(entry?.enum) && entry.enum.length > 0) {
      return `${type} (${entry.enum.map((item) => String(item)).join(" | ")})`;
    }
    return type;
  }

  function parseParameterKatalog(text) {
    const map = new Map();
    String(text || "").split(/\r?\n/).forEach((line) => {
      if (!line.startsWith("| `")) return;
      const parts = line.split("|").slice(1, -1).map((part) => part.trim());
      if (parts.length < 6) return;
      map.set(parts[0].replaceAll("`", ""), {
        katalogType: parts[1].replaceAll("`", ""),
        katalogDefault: parts[2].replaceAll("`", ""),
        shortExplanation: parts[3],
        scenarioHint: parts[4],
        guiTarget: parts[5],
      });
    });
    return map;
  }

  function parseReferenceMarkdown(text) {
    const map = new Map();
    const lines = String(text || "").split(/\r?\n/);
    let currentPath = "";
    let buffer = [];
    const flush = () => {
      if (!currentPath) return;
      const entry = {};
      buffer.forEach((raw) => {
        const line = raw.trim();
        const tableMatch = line.match(/^\| \*\*(.+?)\*\* \| (.+?) \|$/);
        if (tableMatch) {
          entry[tableMatch[1].trim().toLowerCase().replace(/\s+/g, "_")] = tableMatch[2].trim();
        }
        const purposeMatch = line.match(/^\*\*(Purpose|Zweck):\*\*\s*(.+)$/);
        if (purposeMatch) {
          entry.purpose = purposeMatch[2].trim();
        }
      });
      map.set(currentPath, entry);
    };
    lines.forEach((line) => {
      const headerMatch = line.match(/^### `([^`]+)`$/);
      if (headerMatch) {
        flush();
        currentPath = headerMatch[1];
        buffer = [];
        return;
      }
      if (currentPath && line.startsWith("## ")) {
        flush();
        currentPath = "";
        buffer = [];
        return;
      }
      if (currentPath) {
        buffer.push(line);
      }
    });
    flush();
    return map;
  }

  function splitTopLevel(text, delimiter = ",") {
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

  function splitKeyValueTopLevel(text) {
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

  function parseYamlScalar(rawValue) {
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
      return splitTopLevel(inner).map((part) => parseYamlScalar(part));
    }
    if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
      const inner = trimmed.slice(1, -1).trim();
      const out = {};
      if (!inner) return out;
      splitTopLevel(inner).forEach((part) => {
        const [key, value] = splitKeyValueTopLevel(part);
        if (!key) return;
        out[key] = parseYamlScalar(value);
      });
      return out;
    }
    return trimmed;
  }

  function parseYamlObject(lines, startIndex = 0, parentIndent = -1) {
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
        out[key] = parseYamlScalar(rawRest);
        index += 1;
        continue;
      }
      const [child, nextIndex] = parseYamlObject(lines, index + 1, indent);
      out[key] = child;
      index = nextIndex;
    }
    return [out, index];
  }

  function parseYamlSchema(text) {
    try {
      const [parsed] = parseYamlObject(String(text || "").split(/\r?\n/), 0, -1);
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      return {};
    }
  }

  function flattenSchema(node, prefix = [], out = new Map()) {
    if (!node || typeof node !== "object") return out;
    const properties = node.properties;
    if (!properties || typeof properties !== "object") return out;
    Object.entries(properties).forEach(([key, value]) => {
      const path = [...prefix, key];
      if (value && typeof value === "object" && value.type === "object" && value.properties) {
        flattenSchema(value, path, out);
        return;
      }
      out.set(path.join("."), {
        type: value?.type,
        enum: Array.isArray(value?.enum) ? value.enum.slice() : undefined,
        minimum: value?.minimum,
        maximum: value?.maximum,
        exclusiveMinimum: value?.exclusiveMinimum,
        exclusiveMaximum: value?.exclusiveMaximum,
        description: value?.description,
        deprecated: Boolean(value?.deprecated),
      });
    });
    return out;
  }

  function flattenConfigValues(node, prefix = [], out = new Map()) {
    if (Array.isArray(node)) {
      out.set(prefix.join("."), node.slice());
      return out;
    }
    if (!node || typeof node !== "object") {
      out.set(prefix.join("."), node);
      return out;
    }
    Object.entries(node).forEach(([key, value]) => {
      flattenConfigValues(value, [...prefix, key], out);
    });
    return out;
  }

  function deriveRisk(entry) {
    const isGerman = getLocale() === "de";
    if (entry.deprecated) {
      return textFor(
        "page.parameter_studio.explain.risk.deprecated",
        isGerman
          ? "Deprecated: Feld nur fuer Rueckwaertskompatibilitaet verwenden."
          : "Deprecated: keep this field for backward compatibility only.",
      );
    }
    if (entry.range) {
      return textFor(
        "page.parameter_studio.explain.risk.out_of_range",
        isGerman
          ? "Ausserhalb des erlaubten Bereichs drohen Validierungsfehler oder instabiles Verhalten."
          : "Out-of-range values may cause validation errors or unstable behavior.",
      );
    }
    if (entry.scenarioHint && entry.scenarioHint !== "-") {
      return textFor(
        "page.parameter_studio.explain.risk.scenario",
        isGerman ? "Im Szenario '{scenario}' sorgfaeltig abstimmen." : "Tune carefully for scenario '{scenario}'.",
      )
        .replace("{scenario}", entry.scenarioHint);
    }
    return textFor(
      "page.parameter_studio.explain.risk.none",
      isGerman
        ? "Kein expliziter Risiko-Hinweis in den Quellen."
        : "No explicit risk hint in the sources.",
    );
  }

  function buildExplainEntry(path, schemaEntry, katalogEntry, refDeEntry, refEnEntry, editorEntry) {
    const firstKey = topLevelKeyForPath(path);
    const range = computeRange({ ...(schemaEntry || {}), ...(editorEntry || {}) });
    const description = chooseLocalizedExplainText({
      schemaDescription: schemaEntry?.description,
      katalogShortExplanation: katalogEntry?.shortExplanation,
      refDePurpose: refDeEntry?.purpose,
      refEnPurpose: refEnEntry?.purpose,
      editorDescription: editorEntry?.description,
    });
    const category = categoryForPath(path, editorEntry);
    return {
      path,
      label: labelForPath(path),
      category,
      phase: PHASE_MAP[firstKey] || String(firstKey || "").toUpperCase(),
      defaultValue: editorEntry?.yaml_default ?? katalogEntry?.katalogDefault ?? "",
      range,
      description,
      shortExplanation: chooseLocalizedExplainText({
        schemaDescription: schemaEntry?.description,
        katalogShortExplanation: katalogEntry?.shortExplanation,
        refDePurpose: refDeEntry?.purpose,
        refEnPurpose: refEnEntry?.purpose,
        editorDescription: editorEntry?.description,
      }) || description,
      guiTarget: katalogEntry?.guiTarget || category,
      deprecated: Boolean(schemaEntry?.deprecated || editorEntry?.deprecated),
      type: editorEntry?.type || schemaEntry?.type || katalogEntry?.katalogType || "",
      enum: editorEntry?.enum || schemaEntry?.enum || [],
      minimum: editorEntry?.minimum ?? schemaEntry?.minimum,
      maximum: editorEntry?.maximum ?? schemaEntry?.maximum,
      exclusiveMinimum: editorEntry?.exclusiveMinimum ?? schemaEntry?.exclusiveMinimum,
      exclusiveMaximum: editorEntry?.exclusiveMaximum ?? schemaEntry?.exclusiveMaximum,
      source: editorEntry?.source || "schema",
    };
  }

  async function fetchJson(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${path}`);
    return response.json();
  }

  async function fetchText(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${path}`);
    return response.text();
  }

  async function loadLocaleMessages() {
    try {
      localeMessages = await fetchJson(`i18n/${getLocale()}.json`);
    } catch {
      localeMessages = {};
    }
  }

  async function buildExplainIndex() {
    const [katalogText, apiSchemaJson, localSchemaJson, localSchemaYamlText, defaultsJson, refDeText, refEnText] = await Promise.all([
      fetchText("../doc/gui2/parameter_katalog.md")
        .catch(() => fetchText("../doc/gui2/attic/parameter_katalog.md"))
        .catch(() => ""),
      fetchJson("../api/config/schema").catch(() => ({})),
      fetchJson("../tile_compile_cpp/tile_compile.schema.json").catch(() => ({})),
      fetchText("../tile_compile_cpp/tile_compile.schema.yaml").catch(() => ""),
      fetchJson("../api/config/defaults").catch(() => ({ config: {} })),
      fetchText("../doc/v3/configuration_reference.md").catch(() => ""),
      fetchText("../doc/v3/configuration_reference_en.md").catch(() => ""),
    ]);
    const katalogMap = parseParameterKatalog(katalogText);
    const apiSchemaMap = flattenSchema(apiSchemaJson);
    const localSchemaMap = flattenSchema(localSchemaJson);
    const yamlSchemaMap = flattenSchema(parseYamlSchema(localSchemaYamlText));
    const schemaMap = new Map([...apiSchemaMap, ...localSchemaMap, ...yamlSchemaMap]);
    const defaultsMap = flattenConfigValues(defaultsJson?.config || {});
    const refDeMap = parseReferenceMarkdown(refDeText);
    const refEnMap = parseReferenceMarkdown(refEnText);
    const legacyByPath = new Map(
      legacyParamEditorIndex
        .filter((entry) => entry && typeof entry === "object" && entry.path)
        .map((entry) => [String(entry.path), entry]),
    );

    explainIndex.clear();
    validSchemaPaths.clear();
    const allPaths = Array.from(schemaMap.keys());
    paramEditorIndex = allPaths
      .map((path) => {
        const schemaEntry = schemaMap.get(path) || {};
        validSchemaPaths.add(path);
        const legacyEntry = legacyByPath.get(path) || {};
        return {
          path,
          category: categoryForPath(path),
          source: "schema",
          type: schemaEntry?.type || legacyEntry?.type || "",
          enum: schemaEntry?.enum || legacyEntry?.enum || [],
          minimum: schemaEntry?.minimum ?? legacyEntry?.minimum,
          maximum: schemaEntry?.maximum ?? legacyEntry?.maximum,
          exclusiveMinimum: schemaEntry?.exclusiveMinimum ?? legacyEntry?.exclusiveMinimum,
          exclusiveMaximum: schemaEntry?.exclusiveMaximum ?? legacyEntry?.exclusiveMaximum,
          description: schemaEntry?.description || legacyEntry?.description || "",
          deprecated: Boolean(schemaEntry?.deprecated || legacyEntry?.deprecated),
          yaml_default: defaultsMap.has(path) ? defaultsMap.get(path) : legacyEntry?.yaml_default,
        };
      })
      .sort((a, b) => String(a.path || "").localeCompare(String(b.path || "")));
    schemaCategoryOrder = Array.from(
      new Set(allPaths.map((path) => categoryForPath(path)).filter(Boolean)),
    );

    paramEditorIndex.forEach((editorEntry) => {
      const path = String(editorEntry.path || "").trim();
      if (!path) return;
      explainIndex.set(
        path,
        buildExplainEntry(
          path,
          schemaMap.get(path) || {},
          katalogMap.get(path) || {},
          refDeMap.get(path) || {},
          refEnMap.get(path) || {},
          editorEntry,
        ),
      );
    });
  }

  function syncStaticGroupsToSchema() {
    refreshCategoryButtons();
    parameterGroups.forEach((group) => {
      if (group === editorGroup) return;
      const rows = Array.from(group.querySelectorAll(".ps-row"));
      if (rows.length === 0) {
        group.dataset.schemaVisible = "0";
        return;
      }
      let visibleCount = 0;
      rows.forEach((row) => {
        const path = resolvePathFromElement(row) || String(row.querySelector("label")?.textContent || "").trim();
        const visible = path ? validSchemaPaths.has(path) : false;
        row.style.display = visible ? "" : "none";
        if (visible) visibleCount += 1;
      });
      group.dataset.schemaVisible = visibleCount > 0 ? "1" : "0";
    });

    const dynamicCategories = new Set(paramEditorIndex.map((entry) => String(entry.category || "").trim()).filter(Boolean));
    categoryButtons.forEach((button) => {
      const category = String(button.dataset.category || "").trim();
      if (!category || category === "all") {
        button.hidden = false;
        return;
      }
      if (!categoryAllowedForCurrentMode(category)) {
        button.hidden = true;
        return;
      }
      const hasDynamicEntries = dynamicCategories.has(category);
      const hasStaticEntries = Array.from(document.querySelectorAll(`.ps-parameter-group[data-category="${category}"]`))
        .some((group) => group !== editorGroup && group.dataset.schemaVisible === "1");
      button.hidden = !(hasDynamicEntries || hasStaticEntries);
    });

    const activeButton = categoryButtons.find((button) => button.dataset.category === activeCategory && !button.hidden);
    if (!activeButton) {
      const fallback = categoryButtons.find((button) => (button.dataset.category || "") !== "all" && !button.hidden);
      if (fallback) activeCategory = String(fallback.dataset.category || "all");
    }
  }

  function syncCategoryButtonsToSchema() {
    if (!categoryListEl) return;
    const existing = new Map(
      refreshCategoryButtons().map((button) => [String(button.dataset.category || "").trim(), button]),
    );
    schemaCategoryOrder.forEach((category) => {
      const normalized = String(category || "").trim();
      if (!normalized || normalized === "all" || existing.has(normalized)) return;
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.category = normalized;
      button.dataset.control = `parameter.category.${normalized}`;
      button.dataset.tooltip = `${humanizeCategory(normalized)} ${textFor(
        "page.parameter_studio.category.tooltip_suffix",
        isGermanLocale() ? "Parameter bearbeiten." : "Edit parameters.",
      )}`;
      button.textContent = humanizeCategory(normalized);
      categoryListEl.appendChild(button);
    });
    refreshCategoryButtons();
  }

  function bindCategoryNavigation() {
    if (!categoryListEl || categoryListEl.dataset.bound === "1") return;
    categoryListEl.dataset.bound = "1";
    categoryListEl.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-category]");
      if (!button || button.hidden) return;
      setCategory(button.dataset.category || "all");
    });
  }

  function setExplainField(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = formatValue(value);
  }

  function updateExplainPanel(path) {
    const normalizedPath = String(path || "").trim();
    if (!normalizedPath) return;
    const entry = explainIndex.get(normalizedPath) || {
      path: normalizedPath,
      label: labelForPath(normalizedPath),
      category: normalizedPath.split(".")[0] || "-",
      phase: PHASE_MAP[normalizedPath.split(".")[0]] || "-",
      shortExplanation: "-",
      defaultValue: "-",
      range: "-",
      guiTarget: "-",
      type: "-",
    };
    activeExplainPath = normalizedPath;
    const shortHelp = shortHelpForPath(normalizedPath, entry.shortExplanation || entry.description || "-");
    setExplainField("parameter-explain-label", entry.label || normalizedPath);
    setExplainField("parameter-explain-path", normalizedPath);
    setExplainField("parameter-explain-category", entry.category || "-");
    setExplainField("parameter-explain-type", typeLabelForEntry(entry));
    setExplainField("parameter-explain-short", shortHelp);
    setExplainField("parameter-explain-default", entry.defaultValue);
    setExplainField("parameter-explain-range", entry.range || "-");
    setExplainField("parameter-explain-phase", entry.phase || "-");
    setExplainField("parameter-explain-target", entry.guiTarget || "-");
  }

  function localizeCategoryButtons() {
    refreshCategoryButtons().forEach((button) => {
      const category = String(button.dataset.category || "").trim();
      if (!category) return;
      if (category === "all") button.textContent = humanizeCategory(category);
      const tooltip = `${humanizeCategory(category) || String(button.textContent || "").trim()} ${textFor(
        "page.parameter_studio.category.tooltip_suffix",
        isGermanLocale() ? "Parameter bearbeiten." : "Edit parameters.",
      )}`.trim();
      button.dataset.tooltip = tooltip;
      button.setAttribute("title", tooltip);
    });
  }

  function localizeStaticParameterRows() {
    staticRows.forEach((row) => {
      const path = resolvePathFromElement(row) || String(row.querySelector("label")?.textContent || "").trim();
      if (!path || !validSchemaPaths.has(path)) return;
      const entry = explainIndex.get(path) || {
        path,
        shortExplanation: "",
        description: "",
      };
      const tooltip = tooltipForEntry(entry, "");
      if (tooltip && tooltip !== "-") {
        row.setAttribute("title", tooltip);
        row.querySelectorAll("label, input, select, textarea").forEach((el) => {
          el.setAttribute("title", tooltip);
        });
      }
      const hintEl = row.querySelector(".ps-hint");
      if (hintEl) {
        const shortHelp = shortHelpForPath(path, entry.shortExplanation || entry.description || "");
        if (shortHelp && shortHelp !== "-") hintEl.textContent = shortHelp;
      }
    });
  }

  function resolvePathFromElement(el) {
    const dynamicRow = el.closest(".ps-dyn-row");
    if (dynamicRow?.dataset.path) return dynamicRow.dataset.path;
    const staticRow = el.closest(".ps-row[data-path]:not(.ps-dyn-row)");
    if (staticRow?.dataset.path) return staticRow.dataset.path;
    const controlPath = el.getAttribute("data-control");
    if (controlPath && PARAM_CONTROL_PATHS[controlPath]) return PARAM_CONTROL_PATHS[controlPath];
    if (el.id && PARAM_ID_PATHS[el.id]) return PARAM_ID_PATHS[el.id];
    if (el.matches("label")) {
      const labelText = String(el.textContent || "").trim();
      if (labelText.includes(".")) return labelText;
      const targetId = el.getAttribute("for");
      if (targetId && PARAM_ID_PATHS[targetId]) return PARAM_ID_PATHS[targetId];
    }
    const row = el.closest(".ps-row");
    const labelText = String(row?.querySelector("label")?.textContent || "").trim();
    if (labelText.includes(".")) return labelText;
    const field = row?.querySelector("input, select, textarea");
    if (field?.id && PARAM_ID_PATHS[field.id]) return PARAM_ID_PATHS[field.id];
    const fieldControl = field?.getAttribute("data-control");
    if (fieldControl && PARAM_CONTROL_PATHS[fieldControl]) return PARAM_CONTROL_PATHS[fieldControl];
    return "";
  }

  function staticGroupsForCategory(category) {
    if (!categoryAllowedForCurrentMode(category)) return [];
    return Array.from(document.querySelectorAll(`.ps-parameter-group[data-category="${category}"]`))
      .filter((group) => group !== editorGroup && group.dataset.schemaVisible === "1");
  }

  function currentParameterMethod() {
    const hash = new URLSearchParams(String(window.location.hash || "").replace(/^#/, ""));
    const fromHash = hash.get("method");
    if (fromHash === "aqmh" || fromHash === "classic_tile_compile") return fromHash;
    const fromStorage = localStorage.getItem("tileCompile.method");
    if (fromStorage === "aqmh" || fromStorage === "classic_tile_compile") return fromStorage;
    return "aqmh";
  }

  function categoryAllowedForCurrentMode(category) {
    const normalized = String(category || "").trim();
    return !(currentParameterMethod() === "aqmh" && AQMH_CLASSIC_ONLY_CATEGORIES.has(normalized));
  }

  function clearDynamicCategoryExtensions() {
    document.querySelectorAll(".ps-dyn-extension").forEach((el) => el.remove());
  }

  function setGroupTitle(group, title) {
    const titleEl = group?.querySelector(".ps-section-title");
    if (!titleEl) return;
    titleEl.textContent = String(title || "").trim();
  }

  function collectStaticPathsForCategory(category) {
    const paths = new Set();
    staticGroupsForCategory(category).forEach((group) => {
      group.querySelectorAll(".ps-row").forEach((row) => {
        const path = resolvePathFromElement(row) || String(row.querySelector("label")?.textContent || "").trim();
        if (path && validSchemaPaths.has(path)) paths.add(path);
      });
    });
    return paths;
  }

  function findRenderedRowByPath(path) {
    const normalized = String(path || "").trim();
    if (!normalized) return null;
    return Array.from(document.querySelectorAll(".ps-row"))
      .find((row) => {
        if (row.offsetParent === null) return false;
        const rowPath = resolvePathFromElement(row) || String(row.querySelector("label")?.textContent || "").trim();
        return rowPath === normalized;
      }) || null;
  }

  function bindExplainInteractions(root = document) {
    root.querySelectorAll(".ps-row label, .ps-row input, .ps-row select, .ps-row textarea").forEach((el) => {
      if (el.dataset.explainBound === "1") return;
      el.dataset.explainBound = "1";
      const handler = () => {
        const path = resolvePathFromElement(el);
        if (path) updateExplainPanel(path);
      };
      el.addEventListener("click", handler);
      el.addEventListener("focus", handler);
    });
  }

  function tooltipForEntry(entry, value) {
    const path = String(entry?.path || "").trim();
    const explainEntry = explainIndex.get(path) || entry || {};
    const localized = path ? localeMessages[`param.${path}.short_help`] : "";
    if (localized) return localized;
    if (explainEntry.shortExplanation) return explainEntry.shortExplanation;
    if (explainEntry.description) return explainEntry.description;
    if (isGermanLocale()) return entry?.shortExplanation || entry?.description || "-";
    const englishSafeFallback = [entry?.shortExplanation, entry?.description]
      .map((candidate) => normalizeExplainText(candidate))
      .find((candidate) => candidate && !looksGermanText(candidate));
    return englishSafeFallback || "-";
  }

  function inputControlHtml(entry, value, fieldId) {
    const safeTitle = escapeHtml(tooltipForEntry(entry, value));
    if (Array.isArray(entry.enum) && entry.enum.length > 0) {
      const current = String(value);
      const options = entry.enum.map((opt) => {
        const selected = String(opt) === current ? " selected" : "";
        return `<option${selected}>${escapeHtml(opt)}</option>`;
      }).join("");
      return `<select id="${fieldId}" class="ps-select" title="${safeTitle}">${options}</select>`;
    }
    if (entry.type === "boolean") {
      const boolValue = String(value).toLowerCase() === "true";
      return `<select id="${fieldId}" class="ps-select" title="${safeTitle}"><option${boolValue ? " selected" : ""}>true</option><option${!boolValue ? " selected" : ""}>false</option></select>`;
    }
    if (entry.type === "integer" || entry.type === "number") {
      const step = entry.type === "integer" ? "1" : "any";
      return `<input id="${fieldId}" class="ps-input ps-short" type="number" step="${step}" value="${escapeHtml(value)}" title="${safeTitle}">`;
    }
    return `<input id="${fieldId}" class="ps-input ps-wide" type="text" value="${escapeHtml(value)}" title="${safeTitle}">`;
  }

  function renderDynamicRows(entries) {
    return entries.map((entry) => {
      const fieldId = `param-edit-${entry.path.replace(/[^a-zA-Z0-9_]+/g, "_")}`;
      const rawValue = hasOwn(entry, "yaml_default") ? entry.yaml_default : "";
      const value = formatValue(formatEditorValue(entry, rawValue));
      const tooltip = escapeHtml(tooltipForEntry(entry, value));
      const hints = [
        entry.type || textFor("page.parameter_studio.hint.any", "any"),
        entry.source === "yaml_only"
          ? textFor("page.parameter_studio.hint.yaml_only", "yaml-only")
          : textFor("page.parameter_studio.hint.schema", "schema"),
      ];
      const range = computeRange(entry);
      if (range) hints.push(range);
      if (entry.deprecated) hints.push(textFor("page.parameter_studio.hint.deprecated", "deprecated"));
      return `<div class="ps-row ps-dyn-row" data-path="${escapeHtml(entry.path)}" title="${tooltip}"><label for="${fieldId}" title="${tooltip}">${escapeHtml(entry.path)}</label>${inputControlHtml(entry, value, fieldId)}<span class="ps-hint">${escapeHtml(hints.join(" | "))}</span></div>`;
    }).join("");
  }

  function renderDynamicEditor(category, options = {}) {
    if (!editorMetaEl || !editorFieldsEl) return { entryCount: 0, staticCount: 0 };
    const targetStaticGroups = Array.isArray(options.targetStaticGroups) ? options.targetStaticGroups : [];
    const staticPaths = category === "all" || targetStaticGroups.length === 0
      ? new Set()
      : collectStaticPathsForCategory(category);
    const entries = paramEditorIndex
      .filter((entry) => categoryAllowedForCurrentMode(entry.category))
      .filter((entry) => (category === "all" || entry.category === category) && !staticPaths.has(entry.path))
      .sort((a, b) => String(a.path).localeCompare(String(b.path)));
    const staticCount = staticPaths.size;
    if (editorTitleEl) {
      editorTitleEl.textContent =
        category === "all"
          ? textFor("page.parameter_studio.category.all", "All")
          : String(category);
    }
    if (editorNoteEl) {
      editorNoteEl.style.display = category === "all" ? "" : "none";
    }
    editorMetaEl.style.display = category === "all" ? "" : "none";
    editorMetaEl.innerHTML =
      category === "all"
        ? `<b>${escapeHtml(textFor("page.parameter_studio.editor.all", "All"))}</b> - ${entries.length} ${escapeHtml(textFor("page.parameter_studio.editor.editable_count", "editable parameters, grouped by category"))}`
        : "";
    if (entries.length === 0) {
      if (targetStaticGroups.length > 0) {
        editorFieldsEl.innerHTML = "";
      } else {
        editorFieldsEl.innerHTML = staticCount > 0
          ? ""
          : `<div class="ps-note">${escapeHtml(textFor("page.parameter_studio.editor.none_in_category", "No parameters in this category."))}</div>`;
      }
      return { entryCount: 0, staticCount };
    }
    if (category === "all") {
      const grouped = orderedCategories(entries)
        .map((groupCategory) => {
          const groupEntries = entries.filter((entry) => entry.category === groupCategory);
          if (groupEntries.length === 0) return "";
          return [
            `<div class="ps-section-title" style="margin-top:18px;">${escapeHtml(categoryLabel(groupCategory))}</div>`,
            renderDynamicRows(groupEntries),
          ].join("");
        })
        .filter(Boolean)
        .join("");
      editorFieldsEl.innerHTML = grouped;
      bindExplainInteractions(editorFieldsEl);
      return { entryCount: entries.length, staticCount };
    }

    if (targetStaticGroups.length > 0) {
      const targetGroup = targetStaticGroups[targetStaticGroups.length - 1];
      const extension = document.createElement("div");
      extension.className = "ps-dyn-extension";
      extension.innerHTML = renderDynamicRows(entries);
      targetGroup.appendChild(extension);
      bindExplainInteractions(extension);
    } else {
      editorFieldsEl.innerHTML = renderDynamicRows(entries);
      bindExplainInteractions(editorFieldsEl);
    }
    return { entryCount: entries.length, staticCount };
  }

  function setCategory(category) {
    refreshCategoryButtons();
    const requested = categoryButtons.find((btn) => btn.dataset.category === category && !btn.hidden)
      ? category
      : (categoryButtons.find((btn) => (btn.dataset.category || "") !== "all" && !btn.hidden)?.dataset.category || "all");
    activeCategory = requested;
    persistCategory(requested);
    categoryButtons.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.category === requested);
    });
    clearDynamicCategoryExtensions();
    parameterGroups.forEach((group) => {
      group.style.display = "none";
    });
    if (requested === "all") {
      if (editorGroup) editorGroup.style.display = "";
      renderDynamicEditor("all");
      bindExplainInteractions(document);
      document.dispatchEvent(new CustomEvent("gui2:parameter-studio-rendered", {
        detail: { category: requested },
      }));
      return;
    }
    const visibleStaticGroups = staticGroupsForCategory(requested);
    visibleStaticGroups.forEach((group) => {
      group.style.display = "";
      setGroupTitle(group, requested);
    });
    const editorState = renderDynamicEditor(requested, {
      targetStaticGroups: visibleStaticGroups,
    });
    if (editorGroup) {
      editorGroup.style.display = visibleStaticGroups.length > 0
        ? "none"
        : (editorState.entryCount > 0 ? "" : "none");
    }
    bindExplainInteractions(document);
    document.dispatchEvent(new CustomEvent("gui2:parameter-studio-rendered", {
      detail: { category: requested },
    }));
  }

  function clearSearchHits() {
    document.querySelectorAll(".ps-search-hit").forEach((row) => row.classList.remove("ps-search-hit"));
  }

  function jumpToPath(path) {
    const entry = paramEditorIndex.find((item) => item.path === path);
    if (!entry) {
      updateExplainPanel(path);
      return;
    }
    if (!categoryAllowedForCurrentMode(entry.category)) return;
    setCategory(entry.category || "all");
    clearSearchHits();
    const row = findRenderedRowByPath(path);
    if (row) {
      row.classList.add("ps-search-hit");
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      window.setTimeout(() => row.classList.remove("ps-search-hit"), 1400);
    }
    updateExplainPanel(path);
  }

  function renderSearchResults() {
    if (!searchInput || !searchSummaryEl || !searchResultsEl) return;
    const queryRaw = String(searchInput.value || "");
    const query = queryRaw.trim().toLowerCase();
    if (!query) {
      searchSummaryEl.textContent = textFor("page.parameter_studio.search.none_active", "No active search.");
      searchResultsEl.innerHTML = "";
      clearSearchHits();
      return;
    }
    const matches = paramEditorIndex
      .filter((entry) => categoryAllowedForCurrentMode(entry.category))
      .filter((entry) => String(entry.path || "").toLowerCase().includes(query));
    searchSummaryEl.innerHTML = `<b>${matches.length}</b> ${escapeHtml(textFor("page.parameter_studio.search.hits_for", "hits for"))} <code>${escapeHtml(queryRaw.trim())}</code>.`;
    const lines = matches.slice(0, 40).map((entry) => {
      const source = entry.source === "yaml_only"
        ? textFor("page.parameter_studio.hint.yaml_only", "yaml-only")
        : textFor("page.parameter_studio.hint.schema", "schema");
      return `<button class="ps-search-item is-form" type="button" data-path="${escapeHtml(entry.path)}"><code>${escapeHtml(entry.path)}</code><span>${escapeHtml(entry.category + " | " + source)}</span></button>`;
    });
    if (matches.length > 40) lines.push(`<div class="ps-note">${escapeHtml(textFor("page.parameter_studio.search.more_hidden", "More hits hidden ..."))}</div>`);
    if (lines.length === 0) lines.push(`<div class="ps-note">${escapeHtml(textFor("page.parameter_studio.search.none", "No hits."))}</div>`);
    searchResultsEl.innerHTML = lines.join("");
    searchResultsEl.querySelectorAll(".ps-search-item.is-form").forEach((button) => {
      button.addEventListener("click", () => {
        jumpToPath(button.getAttribute("data-path") || "");
      });
    });
  }

  function renderSituationDeltas() {
    const summaryEl = document.getElementById("parameter-situation-summary");
    const deltasEl = document.getElementById("parameter-situation-deltas");
    const chipLabels = [
      ["parameter-situation-altaz", "altaz"],
      ["parameter-situation-rotation", "rotation"],
      ["parameter-situation-bright", "bright_stars"],
      ["parameter-situation-few", "few_frames"],
      ["parameter-situation-gradient", "gradient"],
    ];
    chipLabels.forEach(([id, key]) => {
      const el = document.getElementById(id);
      if (el) el.textContent = scenarioNames[key]?.() || key;
    });
    const activeScenarios = Array.from(document.querySelectorAll(".ps-chip-btn.active"))
      .map((el) => el.dataset.scenario)
      .filter(Boolean);
    if (!summaryEl || !deltasEl) return;
    if (activeScenarios.length === 0) {
      summaryEl.textContent = textFor("page.parameter_studio.situation_none", "Keine Situation aktiv.");
      deltasEl.textContent = textFor("page.parameter_studio.situation_no_deltas", "Keine empfohlenen Deltas.");
      return;
    }
    summaryEl.innerHTML = `${textFor("page.parameter_studio.situation_active", "Aktive Situationen")}: <b>${activeScenarios.map((key) => scenarioNames[key]?.() || key).join(", ")}</b>`;
    const merged = new Map();
    activeScenarios.forEach((scenarioKey) => {
      (scenarioDeltas[scenarioKey] || []).forEach(([path, value, reason]) => {
        if (!validSchemaPaths.has(path)) return;
        if (!merged.has(path)) merged.set(path, { values: new Set(), reasons: [] });
        const info = merged.get(path);
        info.values.add(value);
        info.reasons.push(reason);
      });
    });
    deltasEl.innerHTML = Array.from(merged.entries()).map(([path, info]) => {
      const values = Array.from(info.values);
      const valueText = values.length > 1
        ? `${values.join(" | ")} (${textFor("page.parameter_studio.conflict", "Conflict")})`
        : values[0];
      const reasonKey = `page.parameter_studio.delta_reason.${info.reasons[0] || ""}`;
      const reasonText = textFor(reasonKey, info.reasons[0] || "");
      return `<div><code>${escapeHtml(path)}=${escapeHtml(valueText)}</code> - ${escapeHtml(reasonText)}</div>`;
    }).join("") || textFor("page.parameter_studio.situation_no_deltas", "Keine empfohlenen Deltas.");
  }

  async function refreshLocaleSensitiveUi() {
    await loadLocaleMessages();
    await buildExplainIndex();
    localizeCategoryButtons();
    localizeStaticParameterRows();
    if (activeExplainPath) updateExplainPanel(activeExplainPath);
    setCategory(activeCategory);
    renderSearchResults();
    renderSituationDeltas();
  }

  async function init() {
    await loadLocaleMessages();
    await buildExplainIndex();
    syncCategoryButtonsToSchema();
    syncStaticGroupsToSchema();
    localizeCategoryButtons();
    localizeStaticParameterRows();
    bindCategoryNavigation();
    document.querySelectorAll(".ps-chip-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        btn.classList.toggle("active");
        persistScenarioState();
        renderSituationDeltas();
      });
    });
    if (searchInput) {
      const storedSearch = String(localStorage.getItem(PARAMETER_UI_KEYS.search) || "");
      if (!String(searchInput.value || "").trim() && storedSearch.trim()) {
        searchInput.value = storedSearch;
      }
      searchInput.addEventListener("input", renderSearchResults);
      searchInput.addEventListener("input", () => persistSearch(searchInput.value));
      searchInput.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        const first = paramEditorIndex.find((entry) => String(entry.path || "").toLowerCase().includes(String(searchInput.value || "").trim().toLowerCase()));
        if (first) jumpToPath(first.path);
      });
    }
    document.addEventListener("gui2:locale-changed", () => {
      window.setTimeout(() => void refreshLocaleSensitiveUi(), 0);
    });
    const storedCategory = String(localStorage.getItem(PARAMETER_UI_KEYS.category) || "").trim();
    refreshCategoryButtons();
    if (storedCategory && categoryButtons.some((btn) => btn.dataset.category === storedCategory)) {
      activeCategory = storedCategory;
    }
    try {
      const storedRaw = localStorage.getItem(PARAMETER_UI_KEYS.scenarios);
      if (storedRaw !== null) {
        const storedScenarios = JSON.parse(String(storedRaw || "[]"));
        if (Array.isArray(storedScenarios)) {
          const activeSet = new Set(storedScenarios.map((value) => String(value || "").trim()).filter(Boolean));
          document.querySelectorAll(".ps-chip-btn[data-scenario]").forEach((btn) => {
            btn.classList.toggle("active", activeSet.has(String(btn.dataset.scenario || "").trim()));
          });
        }
      }
    } catch {
      localStorage.removeItem(PARAMETER_UI_KEYS.scenarios);
    }
    setCategory(activeCategory);
    bindExplainInteractions(document);
    updateExplainPanel(activeExplainPath);
    renderSearchResults();
    renderSituationDeltas();
    staticRows.forEach((row) => {
      const path = resolvePathFromElement(row) || String(row.querySelector("label")?.textContent || "").trim();
      if (path && validSchemaPaths.has(path) && !explainIndex.has(path)) {
        explainIndex.set(path, buildExplainEntry(path, {}, {}, {}, {}, { path, category: path.split(".")[0], type: "string", source: "manual" }));
      }
    });
    localizeStaticParameterRows();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      void init();
    }, { once: true });
  } else {
    void init();
  }
})();
