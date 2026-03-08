(function () {
  const LOCALE_KEY = "gui2.locale";
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
  const PHASE_MAP = {
    assumptions: "ASSUMPTIONS",
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

  const categoryButtons = Array.from(document.querySelectorAll("#parameter-category-list button[data-category]"));
  const parameterGroups = Array.from(document.querySelectorAll(".ps-parameter-group"));
  const editorGroup = document.getElementById("parameter-full-editor-group");
  const editorMetaEl = document.getElementById("parameter-editor-meta");
  const editorFieldsEl = document.getElementById("parameter-editor-fields");
  const searchInput = document.getElementById("parameter-search");
  const searchSummaryEl = document.getElementById("parameter-search-summary");
  const searchResultsEl = document.getElementById("parameter-search-results");
  const paramEditorIndex = Array.isArray(window.PARAM_EDITOR_INDEX) ? window.PARAM_EDITOR_INDEX : [];
  const explainIndex = new Map();
  const staticRows = Array.from(document.querySelectorAll(".ps-section.ps-parameter-group .ps-row"));
  let localeMessages = {};
  let activeCategory = "registration";
  let activeExplainPath = "registration.star_topk";

  const scenarioNames = {
    altaz: "Alt/Az",
    rotation: "Starke Rotation",
    bright_stars: "Helle Sterne",
    few_frames: "Wenige Frames",
    gradient: "Starker Gradient",
  };

  const scenarioDeltas = {
    altaz: [
      ["registration.allow_rotation", "true", "Rotation im Modell erlauben"],
      ["registration.star_topk", "180", "mehr Sternkandidaten"],
      ["registration.reject_shift_px_min", "120", "grosse natuerliche Shifts tolerieren"],
      ["registration.reject_shift_median_multiplier", "5.0", "breite Shift-Verteilung"],
    ],
    rotation: [
      ["registration.engine", "robust_phase_ecc", "robuster bei Feldrotation"],
      ["registration.allow_rotation", "true", "zwingend bei Rotation"],
      ["registration.star_inlier_tol_px", "4.0", "tolerantere Inlier-Bedingung"],
      ["registration.reject_cc_min_abs", "0.30", "zu harte CC-Grenzen vermeiden"],
    ],
    bright_stars: [
      ["pcc.mag_bright_limit", "6", "sehr helle Sterne begrenzen"],
      ["pcc.k_max", "2.4", "extreme Farbgains begrenzen"],
      ["pcc.sigma_clip", "2.7", "robustere Ausreisserunterdrueckung"],
      ["bge.mask.star_dilate_px", "6", "Sternumgebung staerker maskieren"],
    ],
    few_frames: [
      ["assumptions.frames_reduced_threshold", "200", "frueher in Reduced-Mode wechseln"],
      ["assumptions.reduced_mode_skip_clustering", "true", "instabile Clusterbildung vermeiden"],
      ["synthetic.frames_min", "4", "minimale Synthetic-Basis sichern"],
      ["synthetic.clustering.cluster_count_range", "[3,10]", "kleinere Clusterzahl"],
    ],
    gradient: [
      ["bge.enabled", "true", "Gradient aktiv modellieren"],
      ["bge.fit.method", "rbf", "flexibles Gradientenmodell"],
      ["bge.fit.rbf_lambda", "1e-2", "Regularisierung gegen Ueberschwingen"],
      ["bge.sample_quantile", "0.15", "robuste Hintergrundsamples"],
      ["bge.structure_thresh_percentile", "0.80", "Struktur vom Hintergrund trennen"],
    ],
  };

  function getLocale() {
    return String(localStorage.getItem(LOCALE_KEY) || document.documentElement.lang || "de").toLowerCase() === "en" ? "en" : "de";
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function hasOwn(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj, key);
  }

  function formatValue(value) {
    if (value === null || value === undefined || value === "") return "-";
    if (Array.isArray(value) || (value && typeof value === "object")) return JSON.stringify(value);
    return String(value);
  }

  function computeRange(entry) {
    if (entry.range) return String(entry.range);
    const hints = [];
    if (hasOwn(entry, "minimum")) hints.push(`>= ${entry.minimum}`);
    if (hasOwn(entry, "exclusiveMinimum")) hints.push(`> ${entry.exclusiveMinimum}`);
    if (hasOwn(entry, "maximum")) hints.push(`<= ${entry.maximum}`);
    if (hasOwn(entry, "exclusiveMaximum")) hints.push(`< ${entry.exclusiveMaximum}`);
    if (Array.isArray(entry.enum) && entry.enum.length > 0) hints.push(entry.enum.map((item) => String(item)).join(" | "));
    return hints.join(", ");
  }

  function labelForPath(path) {
    return localeMessages[`param.${path}.label`] || path;
  }

  function shortHelpForPath(path, fallback) {
    return localeMessages[`param.${path}.short_help`] || fallback || "-";
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
        const purposeMatch = line.match(/^\*\*Purpose:\*\*\s*(.+)$/);
        if (purposeMatch) {
          entry.purpose = purposeMatch[1].trim();
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

  function deriveRisk(entry) {
    if (entry.deprecated) return "Deprecated: Feld nur fuer Rueckwaertskompatibilitaet verwenden.";
    if (entry.range) return "Ausserhalb des erlaubten Bereichs drohen Validierungsfehler oder instabiles Verhalten.";
    if (entry.scenarioHint && entry.scenarioHint !== "-") return `Im Szenario '${entry.scenarioHint}' sorgfaeltig abstimmen.`;
    return "Kein expliziter Risiko-Hinweis in den Quellen.";
  }

  function buildExplainEntry(path, schemaEntry, katalogEntry, refDeEntry, refEnEntry, editorEntry) {
    const firstKey = String(path || "").split(".")[0];
    const range = computeRange({ ...(schemaEntry || {}), ...(editorEntry || {}) });
    const description = editorEntry?.description || katalogEntry?.shortExplanation || schemaEntry?.description || refDeEntry?.purpose || refEnEntry?.purpose || "";
    const category = editorEntry?.category || firstKey;
    return {
      path,
      label: labelForPath(path),
      category,
      phase: PHASE_MAP[firstKey] || String(firstKey || "").toUpperCase(),
      defaultValue: editorEntry?.yaml_default ?? katalogEntry?.katalogDefault ?? "",
      range,
      description,
      shortExplanation: katalogEntry?.shortExplanation || description,
      risk: deriveRisk({ deprecated: Boolean(schemaEntry?.deprecated || editorEntry?.deprecated), range, scenarioHint: katalogEntry?.scenarioHint || "" }),
      scenarioHint: katalogEntry?.scenarioHint || "-",
      guiTarget: katalogEntry?.guiTarget || category,
      referenceDe: refDeEntry?.purpose || "-",
      referenceEn: refEnEntry?.purpose || "-",
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
    const [katalogText, schemaJson, refDeText, refEnText] = await Promise.all([
      fetchText("../doc/gui2/parameter_katalog.md").catch(() => ""),
      fetchJson("../tile_compile_cpp/tile_compile.schema.json").catch(() => ({})),
      fetchText("../doc/v3/configuration_reference.md").catch(() => ""),
      fetchText("../doc/v3/configuration_reference_en.md").catch(() => ""),
    ]);
    const katalogMap = parseParameterKatalog(katalogText);
    const schemaMap = flattenSchema(schemaJson);
    const refDeMap = parseReferenceMarkdown(refDeText);
    const refEnMap = parseReferenceMarkdown(refEnText);

    paramEditorIndex.forEach((editorEntry) => {
      const path = String(editorEntry.path || "");
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
      risk: "-",
      scenarioHint: "-",
      guiTarget: "-",
      referenceDe: "-",
      referenceEn: "-",
    };
    activeExplainPath = normalizedPath;
    setExplainField("parameter-explain-label", entry.label || normalizedPath);
    setExplainField("parameter-explain-path", normalizedPath);
    setExplainField("parameter-explain-category", entry.category || "-");
    setExplainField("parameter-explain-short", shortHelpForPath(normalizedPath, entry.shortExplanation || entry.description || "-"));
    setExplainField("parameter-explain-default", entry.defaultValue);
    setExplainField("parameter-explain-range", entry.range || "-");
    setExplainField("parameter-explain-risk", entry.risk || "-");
    setExplainField("parameter-explain-phase", entry.phase || "-");
    setExplainField("parameter-explain-scenario", entry.scenarioHint || "-");
    setExplainField("parameter-explain-target", entry.guiTarget || "-");
    setExplainField("parameter-explain-reference-de", entry.referenceDe || "-");
    setExplainField("parameter-explain-reference-en", entry.referenceEn || "-");
  }

  function resolvePathFromElement(el) {
    const dynamicRow = el.closest(".ps-dyn-row");
    if (dynamicRow?.dataset.path) return dynamicRow.dataset.path;
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

  function inputControlHtml(entry, value, fieldId) {
    const safeTitle = escapeHtml((entry.description || "").toString());
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

  function renderDynamicEditor(category) {
    if (!editorMetaEl || !editorFieldsEl) return;
    const entries = paramEditorIndex
      .filter((entry) => category === "all" || entry.category === category)
      .sort((a, b) => String(a.path).localeCompare(String(b.path)));
    editorMetaEl.innerHTML = `<b>${escapeHtml(category)}</b> - ${entries.length} editierbare Parameter`;
    if (entries.length === 0) {
      editorFieldsEl.innerHTML = '<div class="ps-note">Keine Parameter in dieser Kategorie.</div>';
      return;
    }
    editorFieldsEl.innerHTML = entries.map((entry) => {
      const fieldId = `param-edit-${entry.path.replace(/[^a-zA-Z0-9_]+/g, "_")}`;
      const value = hasOwn(entry, "yaml_default") ? formatValue(entry.yaml_default) : "";
      const hints = [entry.type || "any", entry.source === "yaml_only" ? "yaml-only" : "schema"];
      const range = computeRange(entry);
      if (range) hints.push(range);
      if (entry.deprecated) hints.push("deprecated");
      return `<div class="ps-row ps-dyn-row" data-path="${escapeHtml(entry.path)}"><label for="${fieldId}">${escapeHtml(entry.path)}</label>${inputControlHtml(entry, value, fieldId)}<span class="ps-hint">${escapeHtml(hints.join(" | "))}</span></div>`;
    }).join("");
    bindExplainInteractions(editorFieldsEl);
  }

  function setCategory(category) {
    activeCategory = category;
    categoryButtons.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.category === category);
    });
    parameterGroups.forEach((group) => {
      group.style.display = "none";
    });
    if (category === "all") {
      parameterGroups.forEach((group) => {
        group.style.display = "";
      });
      if (editorGroup) editorGroup.style.display = "";
      renderDynamicEditor("all");
      bindExplainInteractions(document);
      return;
    }
    if (editorGroup) editorGroup.style.display = "";
    document.querySelectorAll(`.ps-parameter-group[data-category="${category}"]`).forEach((group) => {
      group.style.display = "";
    });
    renderDynamicEditor(category);
    bindExplainInteractions(document);
  }

  function clearSearchHits() {
    document.querySelectorAll(".ps-dyn-row.ps-search-hit").forEach((row) => row.classList.remove("ps-search-hit"));
  }

  function jumpToPath(path) {
    const entry = paramEditorIndex.find((item) => item.path === path);
    if (!entry) {
      updateExplainPanel(path);
      return;
    }
    setCategory(entry.category || "all");
    clearSearchHits();
    const row = editorFieldsEl.querySelector(`.ps-dyn-row[data-path="${CSS.escape(path)}"]`);
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
      searchSummaryEl.textContent = "Keine Suche aktiv.";
      searchResultsEl.innerHTML = "";
      clearSearchHits();
      return;
    }
    const matches = paramEditorIndex.filter((entry) => String(entry.path || "").toLowerCase().includes(query));
    searchSummaryEl.innerHTML = `<b>${matches.length}</b> Treffer fuer <code>${escapeHtml(queryRaw.trim())}</code>.`;
    const lines = matches.slice(0, 40).map((entry) => {
      const source = entry.source === "yaml_only" ? "yaml-only" : "schema";
      return `<button class="ps-search-item is-form" type="button" data-path="${escapeHtml(entry.path)}"><code>${escapeHtml(entry.path)}</code><span>${escapeHtml(entry.category + " | " + source)}</span></button>`;
    });
    if (matches.length > 40) lines.push('<div class="ps-note">Weitere Treffer ausgeblendet ...</div>');
    if (lines.length === 0) lines.push('<div class="ps-note">Keine Treffer.</div>');
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
    const activeScenarios = Array.from(document.querySelectorAll(".ps-chip-btn.active"))
      .map((el) => el.dataset.scenario)
      .filter(Boolean);
    if (!summaryEl || !deltasEl) return;
    if (activeScenarios.length === 0) {
      summaryEl.textContent = "Keine Situation aktiv.";
      deltasEl.textContent = "Keine empfohlenen Deltas.";
      return;
    }
    summaryEl.innerHTML = `Aktive Situationen: <b>${activeScenarios.map((key) => scenarioNames[key] || key).join(", ")}</b>`;
    const merged = new Map();
    activeScenarios.forEach((scenarioKey) => {
      (scenarioDeltas[scenarioKey] || []).forEach(([path, value, reason]) => {
        if (!merged.has(path)) merged.set(path, { values: new Set(), reasons: [] });
        const info = merged.get(path);
        info.values.add(value);
        info.reasons.push(reason);
      });
    });
    deltasEl.innerHTML = Array.from(merged.entries()).map(([path, info]) => {
      const values = Array.from(info.values);
      const valueText = values.length > 1 ? `${values.join(" | ")} (Konflikt)` : values[0];
      return `<div><code>${escapeHtml(path)}=${escapeHtml(valueText)}</code> - ${escapeHtml(info.reasons[0] || "")}</div>`;
    }).join("");
  }

  async function refreshLocaleSensitiveUi() {
    await loadLocaleMessages();
    if (activeExplainPath) updateExplainPanel(activeExplainPath);
    renderSearchResults();
  }

  async function init() {
    await loadLocaleMessages();
    await buildExplainIndex();
    categoryButtons.forEach((btn) => {
      btn.addEventListener("click", () => setCategory(btn.dataset.category || "all"));
    });
    document.querySelectorAll(".ps-chip-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        btn.classList.toggle("active");
        renderSituationDeltas();
      });
    });
    if (searchInput) {
      searchInput.addEventListener("input", renderSearchResults);
      searchInput.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        const first = paramEditorIndex.find((entry) => String(entry.path || "").toLowerCase().includes(String(searchInput.value || "").trim().toLowerCase()));
        if (first) jumpToPath(first.path);
      });
    }
    document.getElementById("locale-de")?.addEventListener("click", () => {
      window.setTimeout(() => void refreshLocaleSensitiveUi(), 0);
    });
    document.getElementById("locale-en")?.addEventListener("click", () => {
      window.setTimeout(() => void refreshLocaleSensitiveUi(), 0);
    });
    setCategory(activeCategory);
    bindExplainInteractions(document);
    updateExplainPanel(activeExplainPath);
    renderSearchResults();
    renderSituationDeltas();
    staticRows.forEach((row) => {
      const path = resolvePathFromElement(row) || String(row.querySelector("label")?.textContent || "").trim();
      if (path && !explainIndex.has(path)) {
        explainIndex.set(path, buildExplainEntry(path, {}, {}, {}, {}, { path, category: path.split(".")[0], type: "string", source: "manual" }));
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      void init();
    }, { once: true });
  } else {
    void init();
  }
})();
