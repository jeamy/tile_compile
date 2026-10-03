import { el, clear } from "../utils/dom.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { getStore } from "../state/store.js";
import { getRunState } from "../state/run-state.js";
import { getFeatureFlags, onFeatureFlagsChange } from "../state/feature-flags.js";
import { t } from "../i18n/i18n.js";
import { assistantActionPlans, createAssistantActionCard } from "./action-card.js";

const label = key => t(`ui.dock.${key}`);
const preferences = getStore("assistant-dock", { open: true, width: 384, pinnedUid: null });
const jsonView = value => el("pre", { class: "tc-archive-json" }, JSON.stringify(value, null, 2).slice(0, 12000));

export function createAssistantDock() {
  let context = null;
  let generation = 0;
  let historyGeneration = 0;
  let capable = false;
  let threadCapable = false;
  let durableJev = false;
  let lastRun = "";
  let selectionTail = Promise.resolve();
  const busy = new Set();
  const drafts = new Map(); // Draft text stays in memory, never in localStorage.
  const title = el("div", { class: "tc-mono tc-dock-context" });
  const status = el("p", { class: "tc-text-muted tc-text-sm", "aria-live": "polite" });
  const thread = el("div", { class: "tc-flex-col tc-gap-2 tc-dock-thread", "aria-live": "polite" });
  const input = el("textarea", { class: "tc-input", rows: 3, "aria-label": label("message"), placeholder: label("message"),
    oninput: () => { if (context) drafts.set(context.context_id, input.value); } });
  const send = el("button", { class: "tc-btn tc-btn-sm", onclick: () => submit("pi") }, label("send"));
  const jev = el("button", { class: "tc-btn tc-btn-sm", onclick: () => submit("jev") }, label("jev"));
  const controls = el("div", { class: "tc-archive-toolbar" }, send, jev);
  const panel = el("aside", { class: "tc-assistant-dock", id: "assistant-dock", "aria-label": label("title") },
    el("div", { class: "tc-dock-header" }, el("strong", {}, label("title")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => toggle(false) }, label("close"))),
    el("div", { class: "tc-dock-body" }, title,
      el("div", { class: "tc-archive-toolbar" },
        el("button", { class: "tc-btn tc-btn-sm", onclick: () => { preferences.setState({ pinnedUid: null }); followRun(true); } }, label("follow")),
        el("button", { class: "tc-btn tc-btn-sm", onclick: () => refresh() }, t("ui.button.refresh", "Refresh"))),
      status, el("p", { class: "tc-text-muted tc-text-sm" }, label("stage")), thread, input, controls));
  const launcher = el("button", { class: "tc-btn tc-dock-launcher", "aria-controls": "assistant-dock", onclick: () => toggle(true) }, label("title"));
  const width = el("input", { class: "tc-dock-width", type: "range", min: 320, max: 560,
    value: preferences.getState().width, "aria-label": label("width"), oninput: event => {
      preferences.setState({ width: Number(event.target.value) }); layout();
    } });
  panel.append(width);
  const root = el("div", {}, launcher, panel);

  function layout() {
    const open = !!preferences.getState().open;
    const size = Math.max(320, Math.min(560, Number(preferences.getState().width) || 384));
    document.documentElement.style.setProperty("--assistant-width", `${size}px`);
    document.documentElement.classList.toggle("tc-dock-open", open);
    panel.classList.toggle("tc-hidden", !open);
    launcher.setAttribute("aria-expanded", String(open));
  }
  function toggle(open) {
    preferences.setState({ open }); layout();
    if (open) { refresh(); input.focus(); } else launcher.focus();
  }
  function buttons() {
    const flags = getFeatureFlags();
    input.classList.toggle("tc-hidden", !flags.aiEnabled);
    send.classList.toggle("tc-hidden", !flags.aiEnabled);
    jev.classList.toggle("tc-hidden", !flags.jevEnabled);
    const blocked = !capable || !context?.run_key || context.readOnly || busy.has(context?.context_id);
    send.disabled = blocked; jev.disabled = blocked || !durableJev; input.disabled = blocked;
    jev.title = durableJev ? label("jev") : label("jev_backend_required");
  }
  async function select(uid, pin = false) {
    const version = ++generation;
    if (pin) preferences.setState({ pinnedUid: uid, open: true });
    context = null; input.value = ""; title.textContent = label("loading");
    layout(); clear(thread); status.textContent = label("loading"); buttons();
    try {
      const next = await (selectionTail = selectionTail.catch(() => {}).then(() => {
        if (version !== generation) throw new Error("Superseded context selection");
        return api.post(API_ENDPOINTS.pi.assistantSelectContext, { run_uid: uid });
      }));
      const identity = await api.get(API_ENDPOINTS.pi.runIdentity(uid));
      if (version !== generation) return;
      // The selected backend path is authoritative. Validate it again when sending.
      context = { ...next, readOnly: !next.artifacts_reachable, aliases: identity.run_keys };
      title.textContent = `${context.context_id}\n${context.run_key || ""}`;
      input.value = drafts.get(context.context_id) || "";
      status.textContent = context.readOnly ? label("readonly") : label("ready");
      buttons(); await history(version);
    } catch (error) {
      if (version !== generation) return;
      context = null; clear(thread); status.textContent = `${label("unavailable")} ${error.message}`; buttons();
    }
  }
  async function followRun(force = false) {
    if (!capable || (preferences.getState().pinnedUid && !force)) return;
    const state = getRunState();
    const ref = state.currentRunDir || state.currentRunId || "";
    if (!force && ref === lastRun) return;
    lastRun = ref;
    const version = ++generation;
    context = null; clear(thread); buttons();
    if (!ref) {
      const active = await api.get(API_ENDPOINTS.pi.activeContext).catch(() => null);
      if (version !== generation) return;
      const saved = active?.context || active;
      if (saved?.kind === "run" && saved.run_uid) { await select(saved.run_uid); return; }
      title.textContent = label("no_context"); status.textContent = label("no_context"); return;
    }
    try {
      const resolved = await api.get(API_ENDPOINTS.pi.assistantRunContext(ref));
      if (version !== generation) return;
      await select(resolved.run_uid);
    } catch (error) { if (version === generation) { status.textContent = `${label("unavailable")} ${error.message}`; buttons(); } }
  }
  function card(result, provider, scope, message = "", method = "") {
    const why = el("details", { class: "tc-archive-section" }, el("summary", {}, label("why")));
    let loaded = false;
    why.addEventListener("toggle", async () => {
      if (!why.open || loaded) return;
      loaded = true;
      why.append(el("p", { class: "tc-text-muted tc-text-sm" }, label("explanation")), jsonView(result));
      try {
        const records = await api.get(API_ENDPOINTS.pi.decisionRecords(scope.context_id));
        why.append(el("p", { class: "tc-text-muted tc-text-sm" }, label("context_records")),
          records.items?.length ? jsonView(records.items) : el("p", {}, label("no_records")));
      } catch { why.append(el("p", {}, label("records_unavailable"))); }
    });
    return el("article", { class: "tc-card tc-dock-card" },
      el("div", { class: "tc-card-title" }, provider === "jev" ? "Jev" : "PI"),
      method === "backend_rules" ? el("p", { class: "tc-text-muted tc-text-sm" }, label("backend_rules")) : null,
      message ? el("p", { class: "tc-text-sm" }, message) : null,
      el("p", { class: "tc-text-sm" }, result?.summary || result?.message || label("structured")),
      why, ...assistantActionPlans(result).map(plan => createAssistantActionCard(plan, scope,
        () => context?.context_id === scope.context_id && !context.readOnly)),
      el("p", { class: "tc-text-muted tc-text-sm" }, label("read_only_actions")));
  }
  async function history(version = generation) {
    if (!context) return;
    const request = ++historyGeneration;
    const scope = { ...context };
    try {
      const flags = getFeatureFlags();
      const result = threadCapable
        ? await api.get(API_ENDPOINTS.pi.assistantThread(scope.run_uid, flags.aiEnabled, flags.jevEnabled))
        : flags.aiEnabled ? await api.get(API_ENDPOINTS.pi.runChatHistoryUid(scope.run_uid)) : { turns: [] };
      if (version !== generation || request !== historyGeneration) return;
      clear(thread);
      if (threadCapable) {
        if (result.context_id !== scope.context_id) throw new Error(label("context_mismatch"));
        for (const item of result.items || []) thread.append(card(item.result || {}, item.provider, scope, item.message, item.evaluation_method));
        if (thread.childNodes.length) thread.append(el("p", { class: "tc-text-muted tc-text-sm" }, label("history_window")));
      } else for (const turn of result.turns || []) thread.append(card(turn.result || { summary: turn.error || "" }, "pi", scope, turn.message));
      if (!thread.childNodes.length) thread.append(el("p", { class: "tc-text-muted tc-text-sm" }, label("empty")));
    } catch (error) { if (version === generation && request === historyGeneration) status.textContent = `${label("unavailable")} ${error.message}`; }
  }
  async function submit(provider) {
    if (!context || context.readOnly || busy.has(context.context_id) || !capable || (provider === "jev" && !durableJev)) return;
    const scope = { ...context };
    const message = input.value.trim();
    if (provider === "pi" && !message) return;
    busy.add(scope.context_id); buttons(); status.textContent = label("working");
    try {
      const result = provider === "pi" ? await api.post(API_ENDPOINTS.pi.runChat, { run_id: scope.run_key, run_uid: scope.run_uid, message })
        : await api.post(API_ENDPOINTS.decisions.postRunAdvice, { run_id: scope.run_key, run_uid: scope.run_uid,
          request_id: Array.from(crypto.getRandomValues(new Uint8Array(16)), n => n.toString(16).padStart(2, "0")).join(""), allow_experimental: false });
      if (provider === "pi") drafts.delete(scope.context_id);
      if (context?.context_id === scope.context_id) {
        if (provider === "pi") input.value = "";
        status.textContent = label("ready"); await history();
      }
    } catch (error) { if (context?.context_id === scope.context_id) status.textContent = `${label("unavailable")} ${error.message}`; }
    finally { busy.delete(scope.context_id); buttons(); }
  }
  async function refresh() {
    try {
      const capabilities = await api.get(API_ENDPOINTS.pi.assistantCapabilities);
      capable = capabilities.schema_version === "pi.assistant-capabilities.v1" && capabilities.run_uid_threads === true;
      threadCapable = capable && capabilities.thread_history === true;
      durableJev = threadCapable && capabilities.durable_jev_post_run === true;
    } catch { capable = false; threadCapable = false; durableJev = false; }
    if (!capable) { status.textContent = label("backend_required"); buttons(); return; }
    if (preferences.getState().pinnedUid) await select(preferences.getState().pinnedUid);
    else await followRun(true);
  }
  getStore("run-state").subscribe(() => followRun());
  onFeatureFlagsChange(() => { buttons(); history(); });
  window.addEventListener("tc-assistant-context", event => { if (event.detail?.run_uid) select(event.detail.run_uid, true); });
  panel.addEventListener("keydown", event => { if (event.key === "Escape") { event.preventDefault(); toggle(false); } });
  layout();
  setTimeout(() => refresh(), 0);
  return root;
}
