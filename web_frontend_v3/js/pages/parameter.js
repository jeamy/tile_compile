// js/pages/parameter.js – Sub-Tab: Parameter + Assumptions + AI-Tab

import { el, clear } from "../utils/dom.js";
import { getUiState, setUiState } from "../state/ui-state.js";
import { goToSubTab } from "../utils/navigation.js";
import { getConfigState, loadSchema, loadConfig, validateConfig, saveConfig, humanizeCategory, setConfigState, markDirty, deepClone } from "../state/config-state.js";
import { t } from "../i18n/i18n.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { createAiEmpfehlungPage } from "./ai-empfehlung.js";
import { createExplainPanel, updateExplainPanel } from "../components/explain-panel.js";
import { createSituationAssistant, getScenarioDeltas } from "../components/situation-assistant.js";
import { createYamlDiff } from "../components/yaml-diff.js";
import { stringifyYaml } from "../utils/yaml-parse.js";
import { parseYaml } from "../utils/yaml-parse.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { refreshGuardrails } from "../services/guardrail-service.js";
import { getStore } from "../state/store.js";

export function createParameterPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });
  const paramView = getUiState().paramView || "parameter";

  const paramTab = el("button", {
    class: `tc-tab${paramView === "parameter" ? " active" : ""}`,
    id: "tab-param",
    role: "tab",
    "aria-selected": paramView === "parameter" ? "true" : "false",
  }, t("ui.tab.parameter", "Parameter"));
  const aiTab = el("button", {
    class: `tc-tab${paramView === "ai" ? " active" : ""}`,
    id: "tab-ai",
    role: "tab",
    "aria-selected": paramView === "ai" ? "true" : "false",
  }, t("ui.tab.ai", "AI Empfehlung"));
  paramTab.onclick = () => switchView("parameter", page, paramTab, aiTab);
  aiTab.onclick = () => switchView("ai", page, paramTab, aiTab);

  const topBar = el("div", { class: "tc-card", id: "param-switchbar" },
    el("div", { class: "tc-card-title" }, t("ui.title.view", "Ansicht")),
    el("div", { class: "tc-tabs", role: "tablist" }, paramTab, aiTab),
  );

  // 3-column grid
  const grid = el("div", { class: "tc-param-grid" });

  // Category sidebar with search
  const sidebar = el("div", { class: "tc-card tc-scroll", style: { maxHeight: "70vh", overflowY: "auto" } },
    el("div", { class: "tc-card-title" }, t("ui.title.categories", "Kategorien")),
    el("input", {
      type: "text",
      class: "tc-input tc-mb-2",
      placeholder: t("ui.placeholder.search", "Parameter suchen..."),
      id: "param-search",
      oninput: () => {
        renderCategories();
        renderEditorForCategory(getUiState().selectedCategory || "all");
      },
    }),
    el("div", { class: "tc-flex-col", id: "param-categories" },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.loading", "L\u00e4dt...")),
    ),
  );

  // Editor area
  const editor = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, "Editor"),
    el("div", { class: "tc-flex-col tc-gap-3", id: "param-editor-body" },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.loading", "Lädt...")),
    ),
    el("div", { class: "tc-mt-4 tc-flex tc-items-center tc-gap-2" },
      el("span", { class: "tc-text-sm tc-text-muted" }, "Preset:"),
      el("select", { class: "tc-select", style: { width: "180px" }, id: "param-preset-select" },
        el("option", {}, "—"),
      ),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => loadPresets() }, t("ui.button.reload", "Reload")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => applyPreset() }, t("ui.button.apply", "Apply")),
    ),
    el("div", { class: "tc-mt-4 tc-flex tc-gap-3 tc-flex-wrap tc-items-center" },
      el("button", { class: "tc-btn tc-btn-primary", onclick: () => doValidate() }, t("ui.button.validate", "Validate")),
      el("button", { class: "tc-btn", onclick: () => doReset() }, t("ui.button.reset_default", "Reset")),
      el("div", { class: "tc-flex tc-gap-2", style: { flexWrap: "nowrap" } },
        el("button", { class: "tc-btn", onclick: () => doSave() }, t("ui.button.save", "Save")),
        el("button", { class: "tc-btn", onclick: () => doSaveAs() }, t("ui.button.save_as", "Save as")),
      ),
    ),
    el("div", { class: "tc-mt-2", id: "param-validation-result" }),
    el("div", { class: "tc-mt-4 tc-card", style: { background: "var(--bg)" }, id: "param-yaml-diff" },
      el("div", { class: "tc-card-title" }, "YAML Diff"),
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_changes", "Keine Änderungen")),
    ),
  );

  // Explain panel (component-based)
  const explain = createExplainPanel();

  grid.append(sidebar, editor, explain);
  grid.id = "param-grid";

  // Situation assistant as separate panel below the grid
  const savedSituations = getUiState().selectedSituations || [];
  const situation = createSituationAssistant({
    selected: savedSituations,
    onApply: (scenarios) => applySituationDeltas(scenarios),
    onChange: (scenarios) => {
      setUiState({ selectedSituations: scenarios });
    },
  });
  const situationPanel = el("div", { class: "tc-card tc-mt-4", id: "param-situation-panel" },
    el("div", { class: "tc-card-title" }, t("page.parameter_studio.situation_assistant", "Situation-Assistent")),
    situation,
  );

  // Next button OUTSIDE the grid, below
  const nextBar = el("div", { class: "tc-flex tc-justify-end tc-mt-4", id: "param-nextbar" },
    el("button", {
      class: "tc-btn",
      onclick: () => goToSubTab("processing", "run-monitor"),
    }, "\u25b6 " + t("ui.button.next", "Next")),
  );

  page.append(topBar, grid, situationPanel, nextBar);

  // Load schema + config from API, then render categories
  initParameterData(paramView === "ai" ? "ai" : null, page, paramTab, aiTab);

  return page;
}

function switchView(view, page, paramTab, aiTab) {
  setUiState({ paramView: view });
  paramTab.classList.toggle("active", view === "parameter");
  aiTab.classList.toggle("active", view === "ai");
  paramTab.setAttribute("aria-selected", view === "parameter" ? "true" : "false");
  aiTab.setAttribute("aria-selected", view === "ai" ? "true" : "false");

  const grid = document.getElementById("param-grid");
  const nextBar = document.getElementById("param-nextbar");
  const sitPanel = document.getElementById("param-situation-panel");
  const aiPage = document.getElementById("param-ai-page");

  if (view === "parameter") {
    if (grid) grid.style.display = "";
    if (nextBar) nextBar.style.display = "";
    if (sitPanel) sitPanel.style.display = "";
    if (aiPage) aiPage.style.display = "none";
  } else {
    if (grid) grid.style.display = "none";
    if (nextBar) nextBar.style.display = "none";
    if (sitPanel) sitPanel.style.display = "none";
    if (!aiPage) {
      const ai = createAiEmpfehlungPage();
      ai.id = "param-ai-page";
      page.appendChild(ai);
    } else {
      aiPage.style.display = "";
    }
  }
}

async function initParameterData(restoreView = null, page = null, paramTab = null, aiTab = null) {
  await loadSchema();
  await loadConfig();
  // Sync calibration values from Input & Scan tab into config draft
  syncCalibrationToDraft();
  renderCategories();
  const savedCat = getUiState().selectedCategory || "all";
  renderEditorForCategory(savedCat);
  loadPresets();
  if (restoreView === "ai" && page && paramTab && aiTab) {
    switchView("ai", page, paramTab, aiTab);
  }
}

function normalizeParameterSearchText(value) {
  return String(value || "")
    .toLocaleLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function parameterSearchTerm() {
  return normalizeParameterSearchText(
    document.getElementById("param-search")?.value || "",
  ).trim();
}

function parameterSearchEntries(schemaPaths, schema) {
  return (schemaPaths || []).map((path) => {
    const fieldSchema = getSchemaForPath(schema, path);
    const category = path.split(".")[0] || "";
    const label = t(`param.${path}.label`, path);
    const shortHelp = shortHelpForPath(path, fieldSchema);
    const explanation = explainHelpForPath(path, fieldSchema);
    const searchable = normalizeParameterSearchText([
      path,
      label,
      category,
      humanizeCategory(category),
      shortHelp,
      explanation,
      ...(fieldSchema?.enum || []),
    ].join(" "));
    return { path, category, searchable };
  });
}

function parameterMatchesSearch(entry, filter) {
  return !filter || entry.searchable.includes(filter);
}

function matchingParameterPaths(schemaPaths, schema, filter) {
  return parameterSearchEntries(schemaPaths, schema)
    .filter((entry) => parameterMatchesSearch(entry, filter))
    .map((entry) => entry.path);
}

function renderCategories() {
  const container = document.getElementById("param-categories");
  if (!container) return;
  clear(container);

  const { categories, schema, schemaPaths } = getConfigState();
  if (!categories || !schemaPaths) {
    container.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_data", "Keine Daten")));
    return;
  }

  const filter = parameterSearchTerm();
  const entries = parameterSearchEntries(schemaPaths, schema);
  const matchingEntries = entries.filter((entry) => parameterMatchesSearch(entry, filter));
  const matchingCategories = new Set(matchingEntries.map((entry) => entry.category));
  const visibleCategories = filter
    ? categories.filter((cat) => cat === "all" || matchingCategories.has(cat))
    : categories;
  const savedCat = getUiState().selectedCategory || "all";

  if (filter && matchingEntries.length === 0) {
    container.appendChild(el("div", { class: "tc-text-muted tc-text-sm" },
      t("ui.state.no_parameter_results", "Keine passenden Parameter gefunden")));
    return;
  }

  for (const cat of visibleCategories) {
    const label = humanizeCategory(cat);
    const count = filter && cat !== "all"
      ? matchingEntries.filter((entry) => entry.category === cat).length
      : null;
    const item = categoryItem(count === null ? label : `${label} (${count})`, cat === savedCat);
    item.dataset.category = cat;
    item.onclick = (e) => {
      e.target.parentElement.querySelectorAll(".tc-param-category")
        .forEach((c) => c.classList.remove("active"));
      e.target.classList.add("active");
      setUiState({ selectedCategory: cat });
      renderEditorForCategory(cat);
    };
    container.appendChild(item);
  }

  if (filter) {
    container.appendChild(el("div", { class: "tc-text-muted tc-text-sm tc-mt-2" },
      t("ui.state.parameter_results", "{count} Parameter gefunden").replace("{count}", String(matchingEntries.length))));
  }
}

const BGE_CLASSIC_ONLY_PREFIXES = [
  "bge.fit.", "bge.grid.", "bge.mask.", "bge.autotune.",
  "bge.sample_quantile", "bge.sample_estimator",
  "bge.min_sample_bg_value", "bge.min_tiles_per_cell",
  "bge.min_valid_sample_fraction_for_apply", "bge.min_valid_samples_for_apply",
  "bge.tile_weight_lambda_structure",
];
const BGE_AUTOBGE_ONLY_PREFIXES = ["bge.autobge."];

function isBgeParamVisible(path, bgeMethod) {
  const isClassicOnly = BGE_CLASSIC_ONLY_PREFIXES.some(p => path === p || path.startsWith(p));
  const isAutobgeOnly = BGE_AUTOBGE_ONLY_PREFIXES.some(p => path === p || path.startsWith(p));
  if (!isClassicOnly && !isAutobgeOnly) return true;
  if (bgeMethod === "autobge") return isAutobgeOnly;
  return isClassicOnly;
}

export function renderEditorForCategory(category) {
  const { schema, schemaPaths, config, draft } = getConfigState();
  const editorBody = document.getElementById("param-editor-body");
  if (!editorBody || !schemaPaths) return;
  clear(editorBody);

  const bgeMethod = draft?.bge?.method ?? "none";

  const filter = parameterSearchTerm();
  const categoryPaths = category === "all"
    ? [...schemaPaths]
    : [...schemaPaths].filter(p => p.startsWith(category + ".") || p === category);
  const paths = filter
    ? matchingParameterPaths(schemaPaths, schema, filter)
    : categoryPaths;

  let renderedCount = 0;
  for (const path of paths) {
    if (!isBgeParamVisible(path, bgeMethod)) continue;
    const fieldSchema = getSchemaForPath(schema, path);
    const value = draft ? (getConfigValue(draft, path) ?? fieldSchema?.default) : "";
    editorBody.appendChild(editableParamRow(path, value, fieldSchema));
    renderedCount += 1;
  }

  if (filter && renderedCount === 0) {
    editorBody.appendChild(el("div", { class: "tc-text-muted tc-text-sm" },
      t("ui.state.no_parameter_results", "Keine passenden Parameter gefunden")));
  }

  updateDiff();
}

function getSchemaForPath(schema, path) {
  if (!schema || !schema.properties) return null;
  const parts = path.split(".");
  let node = schema;
  for (const part of parts) {
    if (!node || !node.properties || !node.properties[part]) return null;
    node = node.properties[part];
  }
  return node;
}

function shortHelpForPath(path, fieldSchema) {
  const i18nKey = `param.${path}.short_help`;
  const localized = t(i18nKey, "");
  if (localized) return localized;
  if (fieldSchema && fieldSchema.description) return fieldSchema.description;
  return "";
}

function explainHelpForPath(path, fieldSchema) {
  const localized = t(`param.${path}.explain`, "");
  if (localized) return localized;
  return shortHelpForPath(path, fieldSchema);
}

function editableParamRow(path, value, fieldSchema) {
  const onChange = (rawVal) => {
    setConfigValue(getConfigState().draft, path, rawVal);
    markDirty();
    setConfigState({ draftYaml: stringifyYaml(getConfigState().draft) });
    updateDiff();
    if (path === "bge.method") {
      const draft = getConfigState().draft;
      if (draft?.bge) {
        draft.bge.enabled = (rawVal !== "none");
      }
      const savedCat = getUiState().selectedCategory || "all";
      renderEditorForCategory(savedCat);
    }
  };

  const shortHelp = shortHelpForPath(path, fieldSchema);
  const tooltipText = shortHelp || path;

  let control;

  if (fieldSchema && fieldSchema.enum) {
    control = el("select", {
      class: "tc-select",
      title: tooltipText,
      onchange: (e) => onChange(parseValue(e.target.value)),
    },
      el("option", { value: "" }, "—"),
      ...fieldSchema.enum.map(opt =>
        el("option", {
          value: String(opt),
          ...(String(value) === String(opt) ? { selected: true } : {}),
        }, String(opt)),
      ),
    );
  } else if (fieldSchema && fieldSchema.type === "boolean") {
    control = el("select", {
      class: "tc-select",
      title: tooltipText,
      onchange: (e) => {
        const v = e.target.value;
        onChange(v === "" ? null : v === "true");
      },
    },
      el("option", { value: "" }, "—"),
      el("option", { value: "true", ...(value === true ? { selected: true } : {}) }, "true"),
      el("option", { value: "false", ...(value === false ? { selected: true } : {}) }, "false"),
    );
  } else if (fieldSchema && (fieldSchema.type === "integer" || fieldSchema.type === "number")) {
    control = el("input", {
      type: "number",
      class: "tc-input",
      title: tooltipText,
      value: formatValue(value),
      ...(fieldSchema.minimum !== undefined ? { min: fieldSchema.minimum } : {}),
      ...(fieldSchema.maximum !== undefined ? { max: fieldSchema.maximum } : {}),
      ...(fieldSchema.type === "integer" ? { step: "1" } : {}),
      oninput: (e) => {
        const v = e.target.value.trim();
        if (v === "") { onChange(null); return; }
        const num = fieldSchema.type === "integer" ? parseInt(v, 10) : parseFloat(v);
        onChange(isNaN(num) ? v : num);
      },
    });
  } else {
    control = el("input", {
      type: "text",
      class: "tc-input",
      title: tooltipText,
      value: formatValue(value),
      oninput: (e) => {
        setConfigValue(getConfigState().draft, path, parseValue(e.target.value));
        markDirty();
        setConfigState({ draftYaml: stringifyYaml(getConfigState().draft) });
        updateDiff();
      },
    });
  }

  const label = el("label", {
    class: "tc-label tc-cursor-pointer",
    title: tooltipText,
    onclick: () => showExplainForPath(path, fieldSchema),
  }, path);

  return el("div", { class: "tc-grid-2" },
    label,
    control,
  );
}

function showExplainForPath(path, fieldSchema) {
  const entry = {
    path,
    label: t(`param.${path}.label`, path),
    category: path.split(".")[0] || "",
    type: fieldSchema?.type || "",
    default: fieldSchema?.default ?? "",
    minimum: fieldSchema?.minimum,
    maximum: fieldSchema?.maximum,
    enum: fieldSchema?.enum || [],
    description: explainHelpForPath(path, fieldSchema),
    deprecated: Boolean(fieldSchema?.deprecated),
  };
  updateExplainPanel(entry);
}

function syncCalibrationToDraft() {
  const inputStore = getStore("input-scan", { calValues: {} });
  const cal = inputStore.getState().calValues;
  if (!cal || Object.keys(cal).length === 0) return;
  const { draft } = getConfigState();
  if (!draft) return;

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
    const master = (cal[`${type}_master`] || "").trim();
    const dir = cal[`${type}_dir`] || "";
    return {
      useMaster: source === "master",
      dir: source === "master" ? "" : dir,
      master: source === "master" ? master : "",
    };
  };

  const bias = effective("bias");
  const dark = effective("dark");
  const flat = effective("flat");

  const useBias = cal.bias_enabled ?? false;
  const useDark = cal.dark_enabled ?? false;
  const useFlat = cal.flat_enabled ?? false;

  const entries = [
    ["calibration.use_bias", useBias],
    ["calibration.bias_use_master", useBias && bias.useMaster],
    ["calibration.bias_dir", useBias ? bias.dir : ""],
    ["calibration.bias_master", useBias ? bias.master : ""],
    ["calibration.use_dark", useDark],
    ["calibration.dark_use_master", useDark && dark.useMaster],
    ["calibration.darks_dir", useDark ? dark.dir : ""],
    ["calibration.dark_master", useDark ? dark.master : ""],
    ["calibration.use_flat", useFlat],
    ["calibration.flat_use_master", useFlat && flat.useMaster],
    ["calibration.flats_dir", useFlat ? flat.dir : ""],
    ["calibration.flat_master", useFlat ? flat.master : ""],
  ];

  let changed = false;
  for (const [path, val] of entries) {
    if (val === undefined || val === null) continue;
    // Allow empty string to clear existing values, but skip booleans that are false
    // (false is the default and would cause unnecessary dirty flags)
    if (val === "" && !getNestedValue(draft, path)) continue;
    setConfigValue(draft, path, val);
    changed = true;
  }
  if (changed) {
    markDirty();
    setConfigState({ draft, draftYaml: stringifyYaml(draft) });
  }
}

export function setConfigValue(obj, path, value) {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!cur[parts[i]] || typeof cur[parts[i]] !== "object") cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
}

function getNestedValue(obj, path) {
  const parts = path.split(".");
  let cur = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = cur[p];
  }
  return cur;
}

function formatValue(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "boolean") return String(v);
  if (typeof v === "number") return String(v);
  if (Array.isArray(v)) return JSON.stringify(v);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function parseValue(s) {
  const str = s.trim();
  if (str.startsWith("[") || str.startsWith("{")) {
    return parseYaml(`value: ${str}`).value;
  }
  if (str === "true") return true;
  if (str === "false") return false;
  if (str === "null" || str === "") return null;
  const num = Number(str);
  if (str !== "" && !isNaN(num)) return num;
  return str;
}

export function updateDiff() {
  const { config, draft } = getConfigState();
  const diffContainer = document.getElementById("param-yaml-diff");
  if (!diffContainer) return;
  clear(diffContainer);

  const diff = createYamlDiff(config, draft);
  const diffBody = diff.querySelector("#yaml-diff-body");
  const hasChanges = diffBody && diffBody.children.length > 0 && !diffBody.querySelector(".tc-text-muted");

  diffContainer.appendChild(el("div", { class: "tc-card-title" }, t("ui.title.yaml_diff", "YAML Diff")));
  if (!hasChanges) {
    diffContainer.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_changes", "Keine Änderungen")));
  } else {
    const changeCount = diffBody.querySelectorAll(".tc-diff-added, .tc-diff-removed").length;
    diffContainer.appendChild(el("div", { class: "tc-text-sm tc-mb-2" }, `${changeCount} ${t("ui.label.changed_values", "geänderte Werte")}`));
    diffContainer.appendChild(diffBody);
  }

}

function getConfigValue(obj, path) {
  const parts = path.split(".");
  let val = obj;
  for (const p of parts) {
    val = val?.[p];
    if (val === undefined) break;
  }
  return val ?? "";
}

async function doValidate() {
  toast(t("ui.toast.validating", "Validiere..."), "", "info");
  const result = await validateConfig();
  const valPanel = document.getElementById("param-validation-result");
  if (valPanel) clear(valPanel);

  if (result?.errors?.length) {
    toastError(t("ui.toast.validation_failed", "Validation failed"), `${result.errors.length} Fehler`);
    if (valPanel) {
      valPanel.appendChild(el("div", { class: "tc-card", style: { background: "var(--error-bg)", borderColor: "var(--error)" } },
        el("div", { class: "tc-card-title", style: { color: "var(--error)" } }, t("ui.toast.validation_failed", "Validation failed")),
        ...result.errors.map(e => {
          if (typeof e === "string") return el("div", { class: "tc-text-sm tc-mono", style: { color: "var(--error)" } }, e);
          const path = e.path || e.field || "";
          const msg = e.message || e.msg || String(e);
          return el("div", { class: "tc-text-sm tc-mono", style: { color: "var(--error)" } },
            path ? `${path}: ${msg}` : msg,
          );
        }),
      ));
    }
  } else if (result?.warnings?.length) {
    if (valPanel) {
      valPanel.appendChild(el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
        el("div", { class: "tc-card-title" }, t("ui.toast.validation_ok", "Config valid")),
        ...result.warnings.map(w => el("div", { class: "tc-text-sm tc-text-muted" },
          `${w.path || w.field || "?"}: ${w.message || w.msg || String(w)}`,
        )),
      ));
    }
    toastSuccess(t("ui.toast.validation_ok", "Config valid"));
    resetSituationCheckboxes();
    refreshGuardrails();
  } else {
    if (valPanel) {
      valPanel.appendChild(el("div", { class: "tc-text-sm", style: { color: "var(--success)" } },
        t("ui.toast.validation_ok", "Config valid"),
      ));
    }
    toastSuccess(t("ui.toast.validation_ok", "Config valid"));
    resetSituationCheckboxes();
    refreshGuardrails();
  }
}

async function doSave() {
  toast(t("ui.toast.saving", "Speichere..."), "", "info");
  const result = await saveConfig();
  if (result) {
    toastSuccess(t("ui.toast.saved", "Config saved"));
    refreshGuardrails();
  } else {
    toastError(t("ui.toast.save_failed", "Save failed"));
  }
}

async function doSaveAs() {
  const { draft, draftYaml } = getConfigState();
  if (!draft && !draftYaml) return;

  // Build modal dialog
  const overlay = el("div", {
    class: "tc-modal-overlay",
    style: { position: "fixed", top: "0", left: "0", width: "100%", height: "100%", background: "rgba(0,0,0,0.5)", zIndex: "9999", display: "flex", alignItems: "center", justifyContent: "center" },
  });

  const modal = el("div", {
    class: "tc-modal",
    style: { minWidth: "520px", maxWidth: "640px" },
  });

  let selectedDir = "";
  let selectedFile = "tile_compile.yaml";

  // Directory input with browse button
  const dirInput = el("input", {
    class: "tc-input",
    type: "text",
    placeholder: t("ui.dialog.select_dir", "Verzeichnis wählen oder eingeben"),
    value: "",
    style: { flex: "1 1 auto" },
    id: "saveas-dir-input",
  });

  // File list area
  const fileList = el("div", {
    class: "tc-flex-col tc-gap-1",
    style: { maxHeight: "200px", overflow: "auto", border: "1px solid var(--border)", borderRadius: "4px", padding: "8px", marginTop: "8px" },
    id: "saveas-file-list",
  });

  // Filename input
  const fileInput = el("input", {
    class: "tc-input",
    type: "text",
    value: selectedFile,
    placeholder: "tile_compile.yaml",
    style: { flex: "1 1 auto" },
    id: "saveas-file-input",
  });

  // Current browsing directory
  let currentDir = "";

  // Load presets/directory listing
  async function loadDirListing(dir) {
    try {
      const data = await api.get(API_ENDPOINTS.config.presets(dir || undefined));
      currentDir = data?.dir || dir || "";
      dirInput.value = currentDir;
      clear(fileList);

      // Up button
      if (currentDir) {
        const upRow = el("div", {
          class: "tc-text-sm tc-flex tc-items-center tc-gap-2",
          style: { padding: "4px 6px", borderRadius: "4px", cursor: "pointer" },
          onclick: () => {
            const parent = currentDir.replace(/\/[^\/]+$/, "") || "";
            loadDirListing(parent);
          },
          onmouseenter: (e) => { e.target.style.background = "var(--surface-2)"; },
          onmouseleave: (e) => { e.target.style.background = ""; },
        },
          el("span", { style: { flexShrink: "0" } }, "⬆"),
          el("span", {}, ".."),
        );
        fileList.appendChild(upRow);
      }

      const items = data?.items || [];
      if (items.length === 0 && !currentDir) {
        fileList.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_files", "Keine Dateien gefunden")));
      }
      for (const item of items) {
        const name = item.name || item.label || item.path || "";
        const path = item.path || name;
        const isDir = item.is_dir || false;
        const row = el("div", {
          class: "tc-text-sm tc-flex tc-items-center tc-gap-2",
          style: { padding: "4px 6px", borderRadius: "4px", cursor: "pointer" },
          onclick: () => {
            if (isDir) {
              dirInput.value = path;
              selectedDir = path;
              loadDirListing(path);
            } else {
              fileInput.value = name;
              selectedFile = name;
            }
          },
          onmouseenter: (e) => { e.target.style.background = "var(--surface-2)"; },
          onmouseleave: (e) => { e.target.style.background = ""; },
        },
          el("span", { style: { flexShrink: "0" } }, isDir ? "📁" : "📄"),
          el("span", {}, name),
        );
        fileList.appendChild(row);
      }
    } catch (e) {
      clear(fileList);
      fileList.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_files", "Keine Dateien gefunden")));
    }
  }

  // Browse button
  const browseBtn = el("button", {
    class: "tc-btn tc-btn-sm",
    onclick: () => {
      loadDirListing(dirInput.value.trim());
    },
  }, t("ui.button.browse", "Browse"));

  dirInput.addEventListener("input", () => { selectedDir = dirInput.value.trim(); });
  fileInput.addEventListener("input", () => { selectedFile = fileInput.value.trim(); });

  // Buttons
  const cancelBtn = el("button", {
    class: "tc-btn",
    onclick: () => overlay.remove(),
  }, t("ui.button.cancel", "Abbrechen"));

  const confirmBtn = el("button", {
    class: "tc-btn tc-btn-primary",
    onclick: async () => {
      const dir = dirInput.value.trim();
      const file = fileInput.value.trim();
      if (!file) {
        toastError(t("ui.toast.save_failed", "Save failed"), t("ui.error.no_filename", "Kein Dateiname angegeben"));
        return;
      }
      let fullPath = file;
      if (dir) {
        fullPath = dir.endsWith("/") ? dir + file : dir + "/" + file;
      }
      overlay.remove();
      toast(t("ui.toast.saving", "Speichere..."), "", "info");
      try {
        const yamlText = draftYaml || stringifyYaml(draft);
        const result = await api.post(API_ENDPOINTS.config.save, { yaml: yamlText, path: fullPath });
        toastSuccess(t("ui.toast.saved", "Config saved"), result?.path || fullPath);
        loadPresets();
      } catch (e) {
        toastError(t("ui.toast.save_failed", "Save failed"), e.message || String(e));
      }
    },
  }, t("ui.button.save", "Speichern"));

  const modalBody = el("div", {
    class: "tc-modal-body",
    style: { padding: "16px 20px", overflow: "auto" },
  });

  modalBody.append(
    el("div", { class: "tc-flex tc-gap-2 tc-items-center" },
      el("span", { class: "tc-text-sm", style: { flexShrink: "0" } }, t("ui.label.directory", "Verzeichnis")),
      dirInput,
      browseBtn,
    ),
    fileList,
    el("div", { class: "tc-flex tc-gap-2 tc-items-center tc-mt-2" },
      el("span", { class: "tc-text-sm", style: { flexShrink: "0" } }, t("ui.label.filename", "Dateiname")),
      fileInput,
    ),
  );

  const modalFooter = el("div", {
    class: "tc-modal-footer",
    style: { padding: "12px 20px", borderTop: "1px solid var(--border)", display: "flex", gap: "8px", justifyContent: "flex-end" },
  });
  modalFooter.append(cancelBtn, confirmBtn);

  modal.append(
    el("div", { class: "tc-modal-header" },
      el("div", { class: "tc-modal-title" }, t("ui.dialog.save_file", "Datei speichern unter")),
    ),
    modalBody,
    modalFooter,
  );

  overlay.append(modal);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);

  // Load initial directory listing
  loadDirListing("");
}

function categoryItem(label, active = false) {
  return el("div", {
    class: `tc-param-category${active ? " active" : ""}`,
    onclick: (e) => {
      e.target.parentElement.querySelectorAll(".tc-param-category")
        .forEach(c => c.classList.remove("active"));
      e.target.classList.add("active");
    },
  }, label);
}

async function doReset() {
  toast(t("ui.toast.resetting", "Reset..."), "", "info");
  try {
    const defaults = await api.get(API_ENDPOINTS.config.defaults);
    if (!defaults) {
      toastError(t("ui.toast.reset_failed", "Reset fehlgeschlagen"), "Keine Defaults erhalten");
      return;
    }
    const yamlText = stringifyYaml(defaults);
    setConfigState({
      config: deepClone(defaults),
      configYaml: yamlText,
      draft: deepClone(defaults),
      draftYaml: yamlText,
      dirty: false,
    });
    renderCategories();
    resetSituationCheckboxes();
    const savedCat = getUiState().selectedCategory || "all";
    renderEditorForCategory(savedCat);
    updateDiff();
    toastSuccess(t("ui.toast.reset_ok", "Auf Defaults zurückgesetzt"));
  } catch (e) {
    toastError(t("ui.toast.reset_failed", "Reset fehlgeschlagen"), e.message || String(e));
  }
}

function resetSituationCheckboxes() {
  setUiState({ selectedSituations: [] });
  document.querySelectorAll("#scenario-list input[type=checkbox]").forEach(cb => {
    cb.checked = false;
  });
  const previewEl = document.getElementById("situation-preview");
  if (previewEl) {
    clear(previewEl);
    previewEl.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.select_situation", "Situation auswählen um Änderungen zu sehen")));
  }
}

async function loadPresets() {
  try {
    const data = await api.get(API_ENDPOINTS.config.presets());
    const select = document.getElementById("param-preset-select");
    if (!select) return;
    const savedPreset = getUiState().selectedPreset || "";
    clear(select);
    select.appendChild(el("option", { value: "" }, "—"));
    for (const item of (data?.items || [])) {
      const name = item.name || item.label || item.path || "";
      const val = item.path || name;
      const opt = el("option", { value: val }, name);
      if (val === savedPreset) opt.selected = true;
      select.appendChild(opt);
    }
    select.value = savedPreset;
    select.onchange = () => setUiState({ selectedPreset: select.value });
  } catch (e) {
    toastError("Presets load failed", e.message);
  }
}

async function applyPreset() {
  const select = document.getElementById("param-preset-select");
  if (!select || !select.value || select.value === "—") return;
  setUiState({ selectedPreset: select.value });
  try {
    const result = await api.post(API_ENDPOINTS.config.applyPreset, { path: select.value });
    const yamlText = result?.config || "";
    const parsed = parseYaml(yamlText);
    setConfigState({
      config: deepClone(parsed),
      configYaml: yamlText,
      draft: deepClone(parsed),
      draftYaml: yamlText,
      dirty: false,
    });
    resetSituationCheckboxes();
    const savedCat = getUiState().selectedCategory || "all";
    renderEditorForCategory(savedCat);
    updateDiff();
    toastSuccess(t("ui.toast.preset_applied", "Preset angewendet"));
  } catch (e) {
    toastError("Preset apply failed", e.message);
  }
}

function applySituationDeltas(scenarios) {
  const { config } = getConfigState();
  if (!config) {
    toastError("Config nicht geladen");
    return;
  }
  const freshDraft = deepClone(config);
  let applied = 0;
  if (scenarios && scenarios.length > 0) {
    const deltas = getScenarioDeltas(scenarios);
    for (const [path, info] of deltas) {
      const value = info.values[info.values.length - 1];
      setConfigValue(freshDraft, path, value);
      applied++;
    }
  }
  markDirty();
  setConfigState({ draft: freshDraft, draftYaml: stringifyYaml(freshDraft) });
  const savedCat = getUiState().selectedCategory || "all";
  renderEditorForCategory(savedCat);
  updateDiff();
  if (applied > 0) {
    toastSuccess(t("ui.toast.situation_applied", "Situation angewendet"), `${applied} Parameter aktualisiert`);
  } else {
    toast(t("ui.toast.situation_cleared", "Situation zurückgesetzt"), "", "info");
  }
}
