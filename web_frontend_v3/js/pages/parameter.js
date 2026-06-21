// js/pages/parameter.js – Sub-Tab: Parameter + Assumptions + AI-Tab

import { el, clear } from "../utils/dom.js";
import { getUiState, setUiState } from "../state/ui-state.js";
import { getConfigState, loadSchema, loadConfig, validateConfig, saveConfig, humanizeCategory, setConfigState, markDirty } from "../state/config-state.js";
import { t } from "../i18n/i18n.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { createAiEmpfehlungPage } from "./ai-empfehlung.js";
import { createExplainPanel, updateExplainPanel } from "../components/explain-panel.js";
import { createSituationAssistant } from "../components/situation-assistant.js";
import { createYamlDiff, updateYamlDiff } from "../components/yaml-diff.js";
import { stringifyYaml } from "../utils/yaml-parse.js";
import { parseYaml } from "../utils/yaml-parse.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";

let paramView = "parameter";

export function createParameterPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  // Parameter / AI switch – card style like Run Control
  const paramTab = el("button", {
    class: "tc-btn tc-btn-primary",
    id: "tab-param",
  }, t("ui.tab.parameter", "Parameter"));
  const aiTab = el("button", {
    class: "tc-btn",
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
    el("div", { class: "tc-mt-4 tc-card", style: { background: "var(--bg)" }, id: "param-yaml-diff" },
      el("div", { class: "tc-card-title" }, "YAML Diff"),
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_changes", "Keine Änderungen")),
    ),
  );

  // Explain panel (component-based)
  const explain = createExplainPanel();

  // Situation assistant inside explain
  const situation = createSituationAssistant({
    selected: [],
    onApply: (scenarios) => {
      toast(t("ui.toast.situation_applied", "Situation angewendet"), `${scenarios.length} Szenarien`, "info");
    },
  });
  explain.querySelector("#explain-body").appendChild(situation);

  // YAML diff inside explain
  const yamlDiff = createYamlDiff("", "");
  explain.querySelector("#explain-body").appendChild(yamlDiff);

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
  initParameterData();

  return page;
}

function switchView(view, page, paramTab, aiTab) {
  paramView = view;
  paramTab.classList.toggle("tc-btn-primary", view === "parameter");
  aiTab.classList.toggle("tc-btn-primary", view === "ai");

  const grid = document.getElementById("param-grid");
  const nextBar = document.getElementById("param-nextbar");
  const aiPage = document.getElementById("param-ai-page");

  if (view === "parameter") {
    if (grid) grid.style.display = "";
    if (nextBar) nextBar.style.display = "";
    if (aiPage) aiPage.remove();
  } else {
    if (grid) grid.style.display = "none";
    if (nextBar) nextBar.style.display = "none";
    if (!aiPage) {
      const ai = createAiEmpfehlungPage();
      ai.id = "param-ai-page";
      page.appendChild(ai);
    }
  }
}

async function initParameterData() {
  const { categories } = getConfigState();
  if (!categories) {
    await loadSchema();
  }
  await loadConfig();
  renderCategories();
  renderEditorForCategory("all");
  loadPresets();
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
    const item = categoryItem(label, cat === "all");
    item.dataset.category = cat;
    item.onclick = (e) => {
      e.target.parentElement.querySelectorAll(".tc-param-category")
        .forEach(c => c.classList.remove("active"));
      e.target.classList.add("active");
      renderEditorForCategory(cat);
    };
    container.appendChild(item);
  }
}

function renderEditorForCategory(category) {
  const { schemaPaths, config, draft } = getConfigState();
  const editorBody = document.getElementById("param-editor-body");
  if (!editorBody || !schemaPaths) return;
  clear(editorBody);

  const paths = category === "all"
    ? [...schemaPaths]
    : [...schemaPaths].filter(p => p.startsWith(category + ".") || p === category);

  for (const path of paths) {
    const value = draft ? getConfigValue(draft, path) : "";
    editorBody.appendChild(editableParamRow(path, value));
  }

  updateDiff();
}

function editableParamRow(path, value) {
  const input = el("input", {
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
  return el("div", { class: "tc-grid-2" },
    el("label", { class: "tc-label", title: path }, path),
    input,
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
  const { configYaml, draftYaml } = getConfigState();
  const diffContainer = document.getElementById("param-yaml-diff");
  if (!diffContainer) return;
  clear(diffContainer);
  if (configYaml === draftYaml) {
    diffContainer.appendChild(el("div", { class: "tc-card-title" }, "YAML Diff"));
    diffContainer.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_changes", "Keine Änderungen")));
  } else {
    const diff = createYamlDiff(configYaml, draftYaml);
    diffContainer.appendChild(diff);
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
  if (result?.errors?.length) {
    toastError(t("ui.toast.validation_failed", "Validation failed"), result.errors.map(e => e.message).join(", "));
  } else {
    toastSuccess(t("ui.toast.validation_ok", "Config valid"));
  }
}

async function doSave() {
  toast(t("ui.toast.saving", "Speichere..."), "", "info");
  const result = await saveConfig();
  if (result) {
    toastSuccess(t("ui.toast.saved", "Config saved"));
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
      config: parsed,
      configYaml: yamlText,
      draft: parsed,
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
