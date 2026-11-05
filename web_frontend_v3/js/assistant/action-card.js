import { el, clear } from "../utils/dom.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { getRunState } from "../state/run-state.js";
import { createYamlDiff } from "../components/yaml-diff.js";
import { t } from "../i18n/i18n.js";
const text = key => t(`ui.dock.actions.${key}`);

// Only explicit config actions enter previews. No parsing prose into parameter values.
export function assistantActionPlans(result) {
  const plans = [];
  if (result?.action_plan?.schema_version === "pi.action-plan.v1" && Array.isArray(result.action_plan.actions)) plans.push(result.action_plan);
  for (const suggestion of Array.isArray(result?.advice?.suggestions) ? result.advice.suggestions : []) {
    if (!suggestion || !Array.isArray(suggestion.updates) || !suggestion.updates.length) continue;
    plans.push({ schema_version: "pi.action-plan.v1", source: "jev.post-run", mutation_free: true,
      candidate_id: suggestion.candidate_id, resume_mode: suggestion.resume_mode,
      actions: [{ id: suggestion.candidate_id, type: "config.patch", updates: suggestion.updates }] });
  }
  return plans;
}

async function verifyScope(scope) {
  const identity = await api.get(API_ENDPOINTS.pi.assistantRunContext(scope.run_key));
  if (identity.run_uid !== scope.run_uid) throw new Error(text("context_changed"));
}
async function resumeTarget(scope) {
  const editor = document.getElementById("resume-config-yaml");
  const state = getRunState();
  const ref = state.currentRunDir || state.currentRunId;
  if (!editor || !ref) return null;
  const identity = await api.get(API_ENDPOINTS.pi.assistantRunContext(ref));
  const now = getRunState();
  if (document.getElementById("resume-config-yaml") !== editor || (now.currentRunDir || now.currentRunId) !== ref || identity.run_uid !== scope.run_uid) return null;
  return { editor, value: editor.value, ref };
}

export function createAssistantActionCard(plan, scope, isCurrent) {
  const updates = [];
  for (const [i, action] of (plan.actions || []).entries()) {
    if (!action || typeof action !== "object") continue;
    if (action.type === "config.set" && typeof action.path === "string" && Object.hasOwn(action, "value"))
      updates.push({ ...action, key: `${i}`, actionIndex: i });
    if (action.type === "config.patch") for (const [j, update] of (Array.isArray(action.updates) ? action.updates : []).entries())
      if (update && typeof update.path === "string" && Object.hasOwn(update, "value")) updates.push({ ...update, key: `${i}:${j}`, actionIndex: i, updateIndex: j });
  }
  if (!updates.length) return null;
  const selected = new Set(); // Explicit opt-in, including experimental suggestions.
  let revision = 0, working = false, snapshot = null;
  const output = el("div", { class: "tc-flex-col tc-gap-2" });
  const status = el("p", { class: "tc-text-muted tc-text-sm", "aria-live": "polite" });
  const apply = el("button", { class: "tc-btn tc-btn-sm", disabled: true, onclick: () => handoff() }, text("handoff"));
  const preview = el("button", { class: "tc-btn tc-btn-sm", onclick: () => build() }, text("preview"));
  const root = el("section", { class: "tc-flex-col tc-gap-2" },
    el("strong", {}, text("title")),
    el("p", { class: "tc-mono tc-text-sm" }, scope.run_key),
    plan.candidate_id ? el("p", { class: "tc-text-sm" }, String(plan.candidate_id)) : null,
    plan.resume_mode === "full_run" ? el("p", { class: "tc-text-warning tc-text-sm" }, text("full_run")) : null,
    el("p", { class: "tc-text-muted tc-text-sm" }, text("draft_only")),
    ...updates.map(update => el("label", { class: "tc-checkbox" },
      el("input", { type: "checkbox", "aria-label": `${update.path}: ${JSON.stringify(update.value)}`, onchange: event => {
        if (event.target.checked) selected.add(update.key); else selected.delete(update.key);
        revision++; snapshot = null; clear(output); status.textContent = ""; buttons();
      } }), el("span", { class: "tc-mono tc-text-sm" }, `${update.path}: ${JSON.stringify(update.value)}`))),
    el("div", { class: "tc-archive-toolbar" }, preview, apply), status, output);
  const active = () => root.isConnected && isCurrent() && !scope.readOnly;
  function buttons() { preview.disabled = working || !selected.size || scope.readOnly; apply.disabled = working || !snapshot?.preview?.config_valid; }
  function selectedPlan() {
    return { ...plan, actions: (plan.actions || []).flatMap((action, i) => {
      if (!action || typeof action !== "object") return [];
      if (action.type === "config.set") return selected.has(`${i}`) ? [action] : [];
      if (action.type === "config.patch") {
        const chosen = (Array.isArray(action.updates) ? action.updates : []).filter((_, j) => selected.has(`${i}:${j}`));
        return chosen.length ? [{ ...action, updates: chosen }] : [];
      }
      return [];
    }) };
  }
  async function build() {
    if (!active() || working || !selected.size) return;
    const version = revision;
    working = true; snapshot = null; buttons(); clear(output); status.textContent = text("working");
    try {
      await verifyScope(scope);
      const target = await resumeTarget(scope);
      const loaded = await api.get(API_ENDPOINTS.runs.config(scope.run_key));
      const sourceFileYaml = loaded?.config_yaml || loaded?.config;
      const yaml = target?.value?.trim() ? target.value : sourceFileYaml;
      if (typeof yaml !== "string" || !yaml.trim()) throw new Error(text("no_config"));
      await verifyScope(scope);
      const result = await api.post(API_ENDPOINTS.pi.actionPlanPreview, { plan: selectedPlan(), yaml });
      if (!active() || version !== revision) return;
      const value = result.preview;
      if (!value?.preview_id || !value.base_yaml || !value.patched_yaml) throw new Error(text("invalid_preview"));
      snapshot = { preview: value, sourceYaml: yaml, sourceFileYaml, target, version };
      status.textContent = value.config_valid === true ? text("valid") : text("invalid");
      output.append(createYamlDiff(value.base_yaml, value.patched_yaml));
      if (value.validation?.errors?.length) output.append(el("pre", { class: "tc-archive-json" }, JSON.stringify(value.validation.errors, null, 2)));
    } catch (error) { if (active() && version === revision) status.textContent = error.message; }
    finally { working = false; buttons(); }
  }
  async function handoff() {
    const saved = snapshot;
    if (!active() || working || !saved?.preview?.config_valid) return;
    working = true; buttons();
    try {
      await verifyScope(scope);
      const object = await api.get(API_ENDPOINTS.pi.actionPlanPreviewById(saved.preview.preview_id));
      if (object.state !== "pending" || object.action_plan_id !== saved.preview.action_plan_id || object.config_sha256 !== saved.preview.config_sha256)
        throw new Error(text("stale"));
      const target = await resumeTarget(scope);
      if (!target) throw new Error(text("open_monitor"));
      // Never overwrite edits made since the preview, or a newly opened different draft.
      if (saved.target ? target.editor !== saved.target.editor || target.value !== saved.target.value
        : target.value.trim() && target.value !== saved.sourceYaml) throw new Error(text("stale"));
      const loaded = await api.get(API_ENDPOINTS.runs.config(scope.run_key));
      if ((loaded.config_yaml || loaded.config) !== saved.sourceFileYaml) throw new Error(text("stale"));
      if (!active() || saved.version !== revision || target.editor.value !== target.value) throw new Error(text("context_changed"));
      await verifyScope(scope);
      if (!active() || target.editor.value !== target.value || document.getElementById("resume-config-yaml") !== target.editor) throw new Error(text("context_changed"));
      if (!window.confirm(`${text("confirm")}\n${scope.run_key}`)) return;
      const detail = { editor: target.editor, ref: target.ref, baseline: target.value, yaml: saved.preview.patched_yaml, accepted: false };
      window.dispatchEvent(new CustomEvent("tc-assistant-resume-draft", { detail }));
      if (!detail.accepted) throw new Error(text("context_changed"));
      status.textContent = text("accepted"); snapshot = null;
    } catch (error) { if (active()) { status.textContent = error.message; snapshot = null; } }
    finally { working = false; buttons(); }
  }
  buttons(); return root;
}
