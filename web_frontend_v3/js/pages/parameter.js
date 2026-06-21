// js/pages/parameter.js – Sub-Tab: Parameter + Assumptions + AI-Tab

import { el, clear } from "../utils/dom.js";
import { getUiState, setUiState } from "../state/ui-state.js";
import { getConfigState, loadSchema, loadConfig, validateConfig, saveConfig, humanizeCategory, setConfigState, markDirty } from "../state/config-state.js";
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

export function createParameterPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });
  const paramView = getUiState().paramView || "parameter";

  // Parameter / AI switch – card style like Run Control
  const paramTab = el("button", {
    class: `tc-btn${paramView === "parameter" ? " tc-btn-primary" : ""}`,
    id: "tab-param",
  }, t("ui.tab.parameter", "Parameter"));
  const aiTab = el("button", {
    class: `tc-btn${paramView === "ai" ? " tc-btn-primary" : ""}`,
    id: "tab-ai",
  }, t("ui.tab.ai", "AI Empfehlung"));
  paramTab.onclick = () => switchView("parameter", page, paramTab, aiTab);
  aiTab.onclick = () => switchView("ai", page, paramTab, aiTab);

  const topBar = el("div", { class: "tc-card", id: "param-switchbar" },
    el("div", { class: "tc-card-title" }, t("ui.title.view", "Ansicht")),
    el("div", { class: "tc-flex tc-gap-3" }, paramTab, aiTab),
  );

  // 3-column grid
  const grid = el("div", { class: "tc-param-grid" });

  // Category sidebar with search
  const sidebar = el("div", { class: "tc-card tc-scroll", style: { maxHeight: "70vh", overflowY: "auto" } },
    el("div", { class: "tc-card-title" }, t("ui.title.categories", "Kategorien")),
    el("input", { type: "text", class: "tc-input tc-mb-2", placeholder: t("ui.placeholder.search", "Suche..."), id: "param-search" }),
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
    el("div", { class: "tc-mt-4 tc-flex tc-gap-3" },
      el("button", { class: "tc-btn tc-btn-primary", onclick: () => doValidate() }, t("ui.button.validate", "Validate")),
      el("button", { class: "tc-btn", onclick: () => doReset() }, t("ui.button.reset_default", "Reset")),
      el("button", { class: "tc-btn", onclick: () => doSave() }, t("ui.button.save", "Save")),
    ),
    el("div", { class: "tc-mt-2", id: "param-validation-result" }),
    el("div", { class: "tc-mt-4 tc-card", style: { background: "var(--bg)" }, id: "param-yaml-diff" },
      el("div", { class: "tc-card-title" }, "YAML Diff"),
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_changes", "Keine Änderungen")),
    ),
  );

  // Explain panel (component-based)
  const explain = createExplainPanel();

  // Situation assistant inside explain
  const savedSituations = getUiState().selectedSituations || [];
  const situation = createSituationAssistant({
    selected: savedSituations,
    onApply: (scenarios) => applySituationDeltas(scenarios),
    onChange: (scenarios) => setUiState({ selectedSituations: scenarios }),
  });
  explain.querySelector("#explain-body").appendChild(situation);

  // Changed parameters summary inside explain panel
  const changedPanel = el("div", { class: "tc-mt-3", id: "param-changed-panel" },
    el("div", { class: "tc-card-title" }, t("ui.title.changed_params", "Geänderte Parameter")),
    el("div", { class: "tc-flex-col tc-gap-1", id: "param-changed-list" },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_changes", "Keine Änderungen")),
    ),
  );
  explain.querySelector("#explain-body").appendChild(changedPanel);

  grid.append(sidebar, editor, explain);
  grid.id = "param-grid";

  // Next button OUTSIDE the grid, below
  const nextBar = el("div", { class: "tc-flex tc-justify-end tc-mt-4", id: "param-nextbar" },
    el("button", {
      class: "tc-btn",
      onclick: () => goToSubTab("run-monitor"),
    }, "\u25b6 " + t("ui.button.next", "Next")),
  );

  page.append(topBar, grid, nextBar);

  // Load schema + config from API, then render categories
  initParameterData(paramView === "ai" ? "ai" : null, page, paramTab, aiTab);

  return page;
}

function switchView(view, page, paramTab, aiTab) {
  setUiState({ paramView: view });
  paramTab.classList.toggle("tc-btn-primary", view === "parameter");
  aiTab.classList.toggle("tc-btn-primary", view === "ai");

  const grid = document.getElementById("param-grid");
  const nextBar = document.getElementById("param-nextbar");
  const aiPage = document.getElementById("param-ai-page");

  if (view === "parameter") {
    if (grid) grid.style.display = "";
    if (nextBar) nextBar.style.display = "";
    if (aiPage) aiPage.style.display = "none";
  } else {
    if (grid) grid.style.display = "none";
    if (nextBar) nextBar.style.display = "none";
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
  const { schemaPaths } = getConfigState();
  if (!schemaPaths || !Array.isArray(schemaPaths)) {
    await loadSchema();
  }
  await loadConfig();
  renderCategories();
  const savedCat = getUiState().selectedCategory || "all";
  renderEditorForCategory(savedCat);
  loadPresets();
  if (restoreView === "ai" && page && paramTab && aiTab) {
    switchView("ai", page, paramTab, aiTab);
  }
}

function renderCategories() {
  const container = document.getElementById("param-categories");
  if (!container) return;
  clear(container);

  const { categories, schemaPaths, config } = getConfigState();
  if (!categories) {
    container.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_data", "Keine Daten")));
    return;
  }

  const searchInput = document.getElementById("param-search");
  const filter = (searchInput?.value || "").toLowerCase().trim();

  for (const cat of categories) {
    const label = humanizeCategory(cat);
    if (filter && !label.toLowerCase().includes(filter) && !cat.includes(filter)) continue;
    const savedCat = getUiState().selectedCategory || "all";
    const item = categoryItem(label, cat === savedCat);
    item.dataset.category = cat;
    item.onclick = (e) => {
      e.target.parentElement.querySelectorAll(".tc-param-category")
        .forEach(c => c.classList.remove("active"));
      e.target.classList.add("active");
      setUiState({ selectedCategory: cat });
      renderEditorForCategory(cat);
    };
    container.appendChild(item);
  }
}

function renderEditorForCategory(category) {
  const { schema, schemaPaths, config, draft } = getConfigState();
  const editorBody = document.getElementById("param-editor-body");
  if (!editorBody || !schemaPaths) return;
  clear(editorBody);

  const paths = category === "all"
    ? [...schemaPaths]
    : [...schemaPaths].filter(p => p.startsWith(category + ".") || p === category);

  for (const path of paths) {
    const value = draft ? getConfigValue(draft, path) : "";
    const fieldSchema = getSchemaForPath(schema, path);
    editorBody.appendChild(editableParamRow(path, value, fieldSchema));
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

function editableParamRow(path, value, fieldSchema) {
  const onChange = (rawVal) => {
    setConfigValue(getConfigState().draft, path, rawVal);
    markDirty();
    setConfigState({ draftYaml: stringifyYaml(getConfigState().draft) });
    updateDiff();
  };

  let control;

  if (fieldSchema && fieldSchema.enum) {
    control = el("select", {
      class: "tc-select",
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
      value: formatValue(value),
      oninput: (e) => {
        setConfigValue(getConfigState().draft, path, parseValue(e.target.value));
        markDirty();
        setConfigState({ draftYaml: stringifyYaml(getConfigState().draft) });
        updateDiff();
      },
    });
  }

  return el("div", { class: "tc-grid-2" },
    el("label", { class: "tc-label", title: path }, path),
    control,
  );
}

function setConfigValue(obj, path, value) {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!cur[parts[i]] || typeof cur[parts[i]] !== "object") cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
}

function formatValue(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "boolean") return String(v);
  if (typeof v === "number") return String(v);
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function parseValue(s) {
  const str = s.trim();
  if (str === "true") return true;
  if (str === "false") return false;
  if (str === "null" || str === "") return null;
  const num = Number(str);
  if (str !== "" && !isNaN(num)) return num;
  return str;
}

function updateDiff() {
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

  updateChangedParamsList(config, draft);
}

function updateChangedParamsList(config, draft) {
  const listEl = document.getElementById("param-changed-list");
  if (!listEl) return;
  clear(listEl);

  const changes = deepDiffPaths(config, draft);
  if (changes.length === 0) {
    listEl.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_changes", "Keine Änderungen")));
    return;
  }

  for (const c of changes) {
    const row = el("div", { class: "tc-text-sm tc-mono" },
      el("span", { class: "tc-diff-removed" }, `- ${c.path}: ${formatVal(c.oldValue)}`),
    );
    listEl.appendChild(row);
    listEl.appendChild(el("div", { class: "tc-text-sm tc-mono" },
      el("span", { class: "tc-diff-added" }, `+ ${c.path}: ${formatVal(c.newValue)}`),
    ));
  }
}

function deepDiffPaths(before, after, prefix = "", out = []) {
  const allKeys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
  for (const key of allKeys) {
    const path = prefix ? `${prefix}.${key}` : key;
    const bv = before?.[key];
    const av = after?.[key];
    if (bv === av) continue;
    if (bv !== null && av !== null && typeof bv === "object" && !Array.isArray(bv) && typeof av === "object" && !Array.isArray(av)) {
      deepDiffPaths(bv, av, path, out);
    } else if (Array.isArray(bv) && Array.isArray(av)) {
      if (JSON.stringify(bv) !== JSON.stringify(av)) out.push({ path, oldValue: bv, newValue: av });
    } else {
      if (bv === undefined) out.push({ path, oldValue: null, newValue: av });
      else if (av === undefined) out.push({ path, oldValue: bv, newValue: null });
      else out.push({ path, oldValue: bv, newValue: av });
    }
  }
  return out;
}

function formatVal(v) {
  if (v === null) return "null";
  if (v === true) return "true";
  if (v === false) return "false";
  if (Array.isArray(v)) return `[${v.map(formatVal).join(", ")}]`;
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
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
    refreshGuardrails();
  } else {
    if (valPanel) {
      valPanel.appendChild(el("div", { class: "tc-text-sm", style: { color: "var(--success)" } },
        t("ui.toast.validation_ok", "Config valid"),
      ));
    }
    toastSuccess(t("ui.toast.validation_ok", "Config valid"));
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
  const defaults = await api.get(API_ENDPOINTS.config.defaults);
  if (defaults) {
    const yamlText = stringifyYaml(defaults);
    const parsed = defaults;
    setConfigState({
      config: JSON.parse(JSON.stringify(parsed)),
      configYaml: yamlText,
      draft: JSON.parse(JSON.stringify(parsed)),
      draftYaml: yamlText,
      dirty: false,
    });
    renderCategories();
    renderEditorForCategory("all");
    toastSuccess(t("ui.toast.reset_ok", "Auf Defaults zurückgesetzt"));
  }
}

async function loadPresets() {
  try {
    const data = await api.get(API_ENDPOINTS.config.presets());
    const select = document.getElementById("param-preset-select");
    if (!select) return;
    clear(select);
    select.appendChild(el("option", {}, "—"));
    for (const item of (data?.items || [])) {
      const name = item.name || item.label || item.path || "";
      select.appendChild(el("option", { value: item.path || name }, name));
    }
  } catch (e) {
    toastError("Presets load failed", e.message);
  }
}

async function applyPreset() {
  const select = document.getElementById("param-preset-select");
  if (!select || !select.value || select.value === "—") return;
  try {
    const result = await api.post(API_ENDPOINTS.config.applyPreset, { path: select.value });
    const yamlText = result?.config || "";
    const parsed = parseYaml(yamlText);
    setConfigState({
      draft: parsed,
      draftYaml: yamlText,
      dirty: true,
    });
    renderEditorForCategory("all");
    toastSuccess(t("ui.toast.preset_applied", "Preset angewendet"));
  } catch (e) {
    toastError("Preset apply failed", e.message);
  }
}

function goToSubTab(subId) {
  const ui = getUiState();
  setUiState({ activeSubTab: { ...ui.activeSubTab, processing: subId } });
  window.location.hash = "#processing";
  window.dispatchEvent(new Event("tc-subtab-change"));
}

function applySituationDeltas(scenarios) {
  if (!scenarios || scenarios.length === 0) {
    toast(t("ui.toast.situation_none", "Keine Situation ausgewählt"), "", "info");
    return;
  }
  const deltas = getScenarioDeltas(scenarios);
  if (deltas.size === 0) {
    toast(t("ui.toast.situation_none", "Keine Situation ausgewählt"), "", "info");
    return;
  }
  const { draft } = getConfigState();
  let applied = 0;
  for (const [path, info] of deltas) {
    const value = info.values[info.values.length - 1];
    setConfigValue(draft, path, value);
    applied++;
  }
  markDirty();
  setConfigState({ draftYaml: stringifyYaml(draft) });
  const savedCat = getUiState().selectedCategory || "all";
  renderEditorForCategory(savedCat);
  toastSuccess(t("ui.toast.situation_applied", "Situation angewendet"), `${applied} Parameter aktualisiert`);
}
