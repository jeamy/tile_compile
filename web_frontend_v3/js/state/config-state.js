// js/state/config-state.js – Config-Draft, Validation, Dirty-State

import { getStore } from "./store.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { parseYaml, stringifyYaml } from "../utils/yaml-parse.js";

const store = getStore("config-state", {
  config: null,       // parsed JS object
  configYaml: "",     // raw YAML string
  draft: null,        // parsed JS object (editable)
  draftYaml: "",      // raw YAML string (editable)
  schema: null,
  schemaPaths: null,
  categories: null,
  validation: null,
  dirty: false,
  loading: false,
  error: null,
});

export function getConfigState() { return store.getState(); }
export function setConfigState(patch) { store.setState(patch); }
export function onConfigChange(fn) { return store.subscribe(fn); }

export function markDirty() { store.setState({ dirty: true }); }
export function markClean() { store.setState({ dirty: false }); }

function flattenSchemaPaths(node, prefix = [], out = new Set()) {
  if (!node || typeof node !== "object" || !node.properties || typeof node.properties !== "object") return out;
  for (const [key, value] of Object.entries(node.properties)) {
    const path = [...prefix, key];
    if (value && typeof value === "object" && value.type === "object" && value.properties) {
      flattenSchemaPaths(value, path, out);
      continue;
    }
    out.add(path.join("."));
  }
  return out;
}

function extractCategories(paths) {
  const cats = new Set();
  for (const p of paths) {
    const top = String(p).split(".")[0];
    if (top) cats.add(top);
  }
  return ["all", ...cats];
}

export async function loadSchema() {
  try {
    store.setState({ loading: true, error: null });
    const schema = await api.get(API_ENDPOINTS.config.schema);
    const paths = flattenSchemaPaths(schema);
    const categories = extractCategories(paths);
    store.setState({ schema, schemaPaths: paths, categories, loading: false });
    return { schema, paths, categories };
  } catch (e) {
    store.setState({ loading: false, error: e.message });
    return null;
  }
}

export async function loadConfig() {
  try {
    store.setState({ loading: true, error: null });
    const resp = await api.get(API_ENDPOINTS.config.current);
    const yamlText = resp?.config || resp?.yaml || (typeof resp === "string" ? resp : "");
    const parsed = parseYaml(yamlText);
    store.setState({
      config: parsed,
      configYaml: yamlText,
      draft: parsed,
      draftYaml: yamlText,
      loading: false,
      dirty: false,
    });
    return parsed;
  } catch (e) {
    store.setState({ loading: false, error: e.message });
    return null;
  }
}

export async function validateConfig() {
  const { draft, draftYaml } = store.getState();
  if (!draft && !draftYaml) return null;
  try {
    const yamlText = draftYaml || stringifyYaml(draft);
    const result = await api.post(API_ENDPOINTS.config.validate, { yaml: yamlText });
    store.setState({ validation: result });
    return result;
  } catch (e) {
    store.setState({ validation: { errors: [{ message: e.message }] } });
    return null;
  }
}

export async function saveConfig() {
  const { draft, draftYaml } = store.getState();
  if (!draft && !draftYaml) return null;
  try {
    const yamlText = draftYaml || stringifyYaml(draft);
    const result = await api.post(API_ENDPOINTS.config.save, { yaml: yamlText });
    store.setState({ config: draft, configYaml: yamlText, dirty: false });
    return result;
  } catch (e) {
    store.setState({ error: e.message });
    return null;
  }
}

export function humanizeCategory(category) {
  const normalized = String(category || "").trim();
  if (!normalized) return "";
  if (normalized === "all") return "Alle";
  return normalized
    .split("_")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
