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
export function markDirty() { store.setState({ dirty: true }); }

// `config-state` (including `draft`/`draftYaml`) is persisted to
// localStorage (see state/store.js PERSIST_LOCAL), so a draft saved to the
// browser before bge.enabled was removed from the schema can sit there
// indefinitely -- and loadConfig() now deliberately skips re-fetching from
// disk while `dirty` is true (see pages/parameter.js), so a stale draft
// with bge.enabled never self-heals just by revisiting the Parameter tab.
// Migrate it defensively wherever the draft is about to leave the browser
// (validate/save), so a leftover legacy field never round-trips as an
// unfixable "Validation error: bge.enabled is no longer supported" loop.
function migrateLegacyBgeEnabled(draft) {
  if (!draft || typeof draft.bge !== "object" || draft.bge === null || !("enabled" in draft.bge)) {
    return false;
  }
  const wasEnabled = draft.bge.enabled;
  draft.bge.method = wasEnabled
    ? (draft.bge.method && draft.bge.method !== "none" ? draft.bge.method : "classic")
    : "none";
  delete draft.bge.enabled;
  return true;
}

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
    const paths = [...flattenSchemaPaths(schema)];
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
      config: deepClone(parsed),
      configYaml: yamlText,
      draft: deepClone(parsed),
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

// Returns {draft, yaml} for the current draft, migrating any legacy
// bge.enabled found in it first and writing the migration back to the
// store (which also re-persists the sanitized draft to localStorage) so
// every subsequent reader -- Start, Resume, Save, Save As, PI action-plan
// preview, etc. -- sees the fixed value instead of re-discovering the same
// stale field on every call. Use this (not `store.getState().draftYaml`
// directly) anywhere a draft is about to be sent to the backend.
export function getOutgoingConfig() {
  let { draft, draftYaml } = store.getState();
  if (draft && migrateLegacyBgeEnabled(draft)) {
    draftYaml = stringifyYaml(draft);
    store.setState({ draft, draftYaml });
  } else if (draftYaml) {
    // Normally both representations are updated together. Still inspect the
    // serialized form as well so a partially persisted/older localStorage
    // entry cannot bypass the migration merely because its parsed draft is
    // missing or already differs from draftYaml.
    try {
      const parsed = parseYaml(draftYaml);
      if (migrateLegacyBgeEnabled(parsed)) {
        draft = parsed;
        draftYaml = stringifyYaml(parsed);
        store.setState({ draft, draftYaml });
      }
    } catch {
      // Keep malformed YAML unchanged; backend validation must report the
      // actual syntax error instead of this compatibility migration hiding it.
    }
  }
  return { draft, yaml: draftYaml || (draft ? stringifyYaml(draft) : "") };
}

export async function validateConfig() {
  const { draft, yaml: yamlText } = getOutgoingConfig();
  if (!draft && !yamlText) return null;
  try {
    const result = await api.post(API_ENDPOINTS.config.validate, { yaml: yamlText });
    store.setState({ validation: result });
    return result;
  } catch (e) {
    store.setState({ validation: { errors: [{ message: e.message }] } });
    return null;
  }
}

export async function saveConfig() {
  const { draft, yaml: yamlText } = getOutgoingConfig();
  if (!draft && !yamlText) return null;
  try {
    const result = await api.post(API_ENDPOINTS.config.save, { yaml: yamlText });
    store.setState({ config: deepClone(draft), configYaml: yamlText, dirty: false });
    return result;
  } catch (e) {
    store.setState({ error: e.message });
    return null;
  }
}

export function deepClone(obj) {
  if (obj === null || typeof obj !== "object") return obj;
  if (Array.isArray(obj)) return obj.map(deepClone);
  const clone = {};
  for (const [k, v] of Object.entries(obj)) clone[k] = deepClone(v);
  return clone;
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
