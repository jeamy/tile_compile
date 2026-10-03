import { el, clear, statItem } from "../utils/dom.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { t } from "../i18n/i18n.js";
import { toastError, toastSuccess } from "./toast.js";

const text = (key, params) => t(`ui.archive.${key}`, undefined, params);
const date = (epoch) => epoch ? new Date(epoch * 1000).toLocaleString(document.documentElement.lang || undefined) : "\u2014";
const stateLabel = (state) => text(`state.${["available", "deletion_pending", "deleted", "delete_failed", "missing"].includes(state) ? state : "unknown"}`);
const reasons = ["test_run", "invalid_data", "unreliable_measurements", "user_choice"];

// Bound rendering independently of the full, immutable archive payload.
function compact(value, depth = 0) {
  if (typeof value === "string") return value.length > 4000 ? value.slice(0, 4000) + "\u2026" : value;
  if (!value || typeof value !== "object") return value;
  if (depth >= 6) return "\u2026";
  if (Array.isArray(value)) return value.slice(0, 30).map(v => compact(v, depth + 1));
  return Object.fromEntries(Object.entries(value).slice(0, 50).map(([k, v]) => [k, compact(v, depth + 1)]));
}
function section(label, data) {
  const detail = el("details", { class: "tc-archive-section" }, el("summary", {}, label));
  detail.addEventListener("toggle", () => {
    if (!detail.open || detail.childNodes.length > 1) return;
    detail.append(el("p", { class: "tc-text-muted tc-text-sm" }, text("compact_notice")),
      el("pre", { class: "tc-archive-json" }, JSON.stringify(compact(data), null, 2)));
  });
  return detail;
}
async function download(snapshot) {
  const response = await fetch(api.httpUrl(API_ENDPOINTS.pi.runLearning.snapshot(snapshot.run_uid, snapshot.snapshot_id)));
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  // Keep server JSON bytes: JS number parsing would round 64-bit file-clock ticks.
  const raw = await response.text();
  const url = URL.createObjectURL(new Blob([raw], { type: "application/json" }));
  const link = el("a", { href: url, download: `${snapshot.snapshot_id}.json` });
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function createRunLearningArchive({ selectedUid = null, onSelect = () => {} } = {}) {
  let uid = selectedUid;
  let request = 0;
  let listRequest = 0;
  let limit = 100;
  const list = el("div", { class: "tc-flex-col tc-gap-1", "data-archive-list": "" });
  const body = el("div", { class: "tc-flex-col tc-gap-2", "data-archive-detail": "", "aria-live": "polite" });
  const more = el("button", { class: "tc-btn tc-btn-sm tc-hidden", hidden: true, onclick: () => { limit = Math.min(1000, limit + 100); refresh(); } }, text("more"));
  const root = el("section", { class: "tc-card tc-run-archive" },
    el("div", { class: "tc-card-title tc-archive-toolbar" }, el("span", {}, text("title")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => refresh() }, t("ui.button.refresh", "Refresh"))),
    el("p", { class: "tc-text-sm tc-text-muted" }, text("retention")), list, more, body);

  async function refresh() {
    if (!root.isConnected) return;
    const version = ++listRequest;
    try {
      const capabilities = await api.get(API_ENDPOINTS.pi.runLearning.capabilities).catch(() => null);
      if (capabilities?.schema_version !== "pi.run-learning-capabilities.v1" || capabilities.history_summaries !== true || capabilities.snapshot_lookup !== true) {
        if (!root.isConnected || version !== listRequest) return;
        clear(list); list.append(el("p", { class: "tc-text-muted" }, text("backend_required")));
        ++request; clear(body); more.hidden = true; more.classList.add("tc-hidden");
        return;
      }
      const result = await api.get(API_ENDPOINTS.pi.runLearning.list(limit));
      if (!root.isConnected || version !== listRequest) return;
      clear(list);
      const items = result?.items || [];
      for (const item of items) {
        list.append(el("button", { class: `tc-run-item tc-archive-row${uid === item.run_uid ? " active" : ""}`,
          type: "button", "aria-pressed": String(uid === item.run_uid), "data-archive-uid": item.run_uid,
          onclick: () => select(item.run_uid) },
          el("span", { class: "tc-mono" }, item.run_id || item.run_uid),
          el("span", { class: "tc-badge" }, stateLabel(item.artifacts_state)),
          el("span", { class: "tc-badge" }, text(item.excluded_from_learning ? "excluded" : "not_excluded")),
          el("span", { class: "tc-text-muted" }, date(item.captured_at_epoch))));
      }
      if (!items.length) list.append(el("p", { class: "tc-text-muted" }, text("empty")));
      more.hidden = items.length < limit || limit >= 1000;
      more.classList.toggle("tc-hidden", more.hidden);
      if (items.length >= 1000) list.append(el("p", { class: "tc-text-muted" }, text("limit")));
      if (uid) await select(uid, false);
    } catch (error) {
      if (version !== listRequest || !root.isConnected) return;
      clear(list); list.append(el("p", { class: "tc-text-muted" }, text("load_failed")));
      toastError(text("load_failed"), error.message);
    }
  }
  async function select(next, notify = true) {
    if (!root.isConnected) return;
    uid = next;
    if (notify) onSelect(uid);
    for (const row of list.querySelectorAll("[data-archive-uid]")) {
      const active = row.dataset.archiveUid === uid;
      row.classList.toggle("active", active); row.setAttribute("aria-pressed", String(active));
    }
    const version = ++request;
    clear(body); body.append(el("p", {}, t("ui.state.loading", "Loading...")));
    try {
      const current = await api.get(API_ENDPOINTS.pi.runLearning.byUid(uid));
      if (version !== request || !root.isConnected) return;
      clear(body);
      body.append(el("h3", { class: "tc-text-sm tc-mono tc-archive-path" }, current.run_uid),
        el("div", { class: "tc-grid-2" },
          statItem(text("files"), stateLabel(current.artifacts_state)),
          statItem(text("reachable"), text(current.artifacts_reachable ? "yes" : "missing")),
          statItem(text("learning"), text(current.excluded_from_learning ? "excluded" : "not_excluded")),
          statItem(text("reason"), current.exclusion_code ? text(`reason.${current.exclusion_code}`) : "\u2014")),
        el("p", { class: "tc-text-muted tc-text-sm" }, text("unreviewed")),
        el("p", { class: "tc-text-muted tc-text-sm" }, text("exclusion_scope")));
      const controls = el("div", { class: "tc-archive-toolbar" });
      const reason = el("select", { class: "tc-select", "aria-label": text("reason") },
        ...reasons.map(code => el("option", { value: code, selected: code === (current.exclusion_code || "user_choice") }, text(`reason.${code}`))));
      const toggle = el("button", { class: "tc-btn tc-btn-sm", onclick: async () => {
        const excluded = !current.excluded_from_learning;
        if (!window.confirm(text(excluded ? "confirm_exclude" : "confirm_include", { run: current.run_id || uid }))) return;
        toggle.disabled = true;
        try {
          await api.post(API_ENDPOINTS.pi.runLearning.exclusion(current.run_uid), { confirmed: true, excluded, ...(excluded ? { reason_code: reason.value } : {}) });
          toastSuccess(text("updated")); await refresh();
        } catch (error) { toastError(text("update_failed"), error.message); toggle.disabled = false; }
      } }, text(current.excluded_from_learning ? "include" : "exclude"));
      if (!current.excluded_from_learning) controls.append(reason);
      controls.append(toggle); body.append(controls);

      const targetUid = current.run_uid;
      const path = el("input", { class: "tc-input tc-archive-path-input", type: "text", placeholder: text("new_path"), "aria-label": text("new_path") });
      const relink = el("button", { class: "tc-btn tc-btn-sm", onclick: async () => {
        const runDir = path.value.trim();
        if (!runDir || !window.confirm(text("confirm_relink", { uid: targetUid, path: runDir }))) return;
        relink.disabled = true;
        try {
          await api.post(API_ENDPOINTS.pi.runRelink(targetUid), { run_dir: runDir, confirmed: true });
          toastSuccess(text("relinked")); await refresh();
        } catch (error) { toastError(text("relink_failed"), error.message); relink.disabled = false; }
      } }, text("relink"));
      body.append(el("details", { class: "tc-archive-section" }, el("summary", {}, text("relink")),
        el("p", { class: "tc-text-sm tc-text-muted" }, text("relink_help")), el("div", { class: "tc-archive-toolbar" }, path, relink)));

      const versions = el("select", { class: "tc-select", "aria-label": text("versions") }, el("option", { value: current.snapshot_id }, text("latest")));
      const payload = el("div", { class: "tc-flex-col tc-gap-2" });
      let snapshotRequest = 0;
      function render(snapshot) {
        clear(payload);
        payload.append(el("div", { class: "tc-grid-2" }, statItem(text("snapshot"), snapshot.snapshot_id),
          statItem(text("captured"), date(snapshot.captured_at_epoch)), statItem(text("stage"), snapshot.stage),
          statItem(text("status"), snapshot.status)),
          el("p", { class: "tc-archive-path tc-text-sm tc-mono" }, snapshot.run_key),
          el("button", { class: "tc-btn tc-btn-sm", onclick: async (event) => {
            const button = event.currentTarget; button.disabled = true;
            try { await download(snapshot); } catch (error) { toastError(text("download_failed"), error.message); }
            finally { button.disabled = false; }
          } }, text("download")),
          el("p", { class: "tc-text-muted tc-text-sm" }, text("local_paths")),
          section(text("config"), snapshot.config), section(text("source"), snapshot.source),
          section(text("metrics"), snapshot.artifacts), section(text("phases"), snapshot.phase_events),
          section(text("gaps"), { capture_issues: snapshot.capture_issues, missing_optional_files: snapshot.missing_optional_files,
            light_manifest_available: snapshot.light_manifest_available, parser_defaults_included: snapshot.config?.parser_defaults_included }));
        if (current.preview_snapshot_id === snapshot.snapshot_id) {
          const img = el("img", { class: "tc-archive-preview", alt: text("preview"),
            src: api.httpUrl(API_ENDPOINTS.pi.runLearning.preview(targetUid)) });
          img.addEventListener("error", () => img.replaceWith(el("p", { class: "tc-text-muted" }, text("preview_missing"))), { once: true });
          payload.append(img, statItem(text("preview_source"), current.preview_source_artifact || "\u2014"),
            el("p", { class: "tc-text-muted tc-text-sm" }, text("preview_help")));
        } else payload.append(el("p", { class: "tc-text-muted tc-text-sm" }, text("preview_missing")));
      }
      versions.addEventListener("change", async () => {
        const snapshotVersion = ++snapshotRequest;
        clear(payload); payload.append(el("p", {}, t("ui.state.loading", "Loading...")));
        try {
          const snapshot = versions.value === current.snapshot_id ? current : await api.get(API_ENDPOINTS.pi.runLearning.snapshot(targetUid, versions.value));
          if (version === request && snapshotVersion === snapshotRequest && root.isConnected) render(snapshot);
        } catch (error) {
          if (version !== request || snapshotVersion !== snapshotRequest || !root.isConnected) return;
          clear(payload); payload.append(el("p", {}, text("load_failed"))); toastError(text("load_failed"), error.message);
        }
      });
      body.append(el("label", { class: "tc-label" }, text("versions"), versions), payload);
      render(current);
      try {
        const [history, identity] = await Promise.all([
          api.get(API_ENDPOINTS.pi.runLearning.history(targetUid)),
          api.get(API_ENDPOINTS.pi.runIdentity(targetUid)).catch(() => null),
        ]);
        if (version !== request || !root.isConnected) return;
        if (identity?.run_keys) body.append(section(text("known_paths"), identity.run_keys));
        for (const item of history.items || []) {
          if (item.snapshot_id !== current.snapshot_id) versions.append(el("option", { value: item.snapshot_id }, `${date(item.captured_at_epoch)} / ${item.stage}`));
        }
        if (history.items?.length === 50) body.append(el("p", { class: "tc-text-muted tc-text-sm" }, text("history_limit")));
      } catch { if (version === request && root.isConnected) body.append(el("p", {}, text("history_failed"))); }
    } catch (error) {
      if (version !== request || !root.isConnected) return;
      clear(body); body.append(el("p", {}, text("load_failed"))); toastError(text("load_failed"), error.message);
    }
  }
  return { element: root, refresh, select };
}
