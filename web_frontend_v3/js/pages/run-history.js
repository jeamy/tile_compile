// js/pages/run-history.js – Sub-Tab: Run-Historie mit Detail-View

import { el, clear } from "../utils/dom.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastError, toastSuccess } from "../components/toast.js";
import { t } from "../i18n/i18n.js";

let selectedRunId = null;

export function createRunHistoryPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  const header = el("div", { class: "tc-flex tc-items-center tc-justify-between" },
    el("span", { class: "tc-text-sm" }, t("ui.label.source", "Quelle") + ": /data/runs"),
    el("div", { class: "tc-flex tc-gap-2" },
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => loadRuns() }, t("ui.button.refresh", "Refresh")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => compareRuns() }, t("ui.button.compare", "Vergleichen")),
    ),
  );

  const listCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.run_list", "Run-Liste")),
    el("div", { class: "tc-run-list", id: "run-list" },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.loading", "L\u00e4dt...")),
    ),
  );

  const detailCard = el("div", { class: "tc-card", id: "run-detail-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.selected_run", "Ausgew\u00e4hlter Run")),
    el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_run_selected", "Kein Run ausgew\u00e4hlt")),
  );

  page.append(header, listCard, detailCard);

  setTimeout(() => loadRuns(), 100);
  return page;
}

async function loadRuns() {
  try {
    const runs = await api.get(API_ENDPOINTS.runs.list);
    const list = document.getElementById("run-list");
    if (!list) return;
    clear(list);
    if (!runs || (Array.isArray(runs) && runs.length === 0)) {
      list.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_runs", "Keine Runs gefunden")));
      return;
    }
    const runArr = Array.isArray(runs) ? runs : (runs.runs || runs.items || []);
    for (const run of runArr) {
      list.appendChild(runItem(run));
    }
  } catch (e) {
    toastError(t("ui.toast.load_runs_failed", "Runs laden fehlgeschlagen"), e.message);
  }
}

function runItem(run) {
  const status = run.status || "UNKNOWN";
  const badgeClass = status === "OK" || status === "done" ? "tc-badge-success" :
    status === "ERROR" || status === "failed" ? "tc-badge-error" :
    status === "RUNNING" || status === "running" ? "tc-badge-info" : "";
  const runId = run.run_id || run.id || "";

  const item = el("div", {
    class: "tc-run-item",
    onclick: () => selectRun(runId),
  },
    el("span", { class: "tc-badge" }, run.pipeline || "AQMH"),
    el("span", { class: `tc-badge ${badgeClass}` }, status),
    el("span", { class: "tc-mono tc-text-sm" }, runId),
    el("span", { class: "tc-text-sm" }, run.run_name || run.name || ""),
  );

  if (runId === selectedRunId) item.classList.add("active");
  return item;
}

async function selectRun(runId) {
  selectedRunId = runId;
  const card = document.getElementById("run-detail-card");
  if (!card) return;
  clear(card);

  card.appendChild(el("div", { class: "tc-card-title" }, t("ui.title.selected_run", "Ausgew\u00e4hlter Run")));
  card.appendChild(el("div", { class: "tc-text-muted tc-text-sm tc-mb-2" }, runId));

  try {
    const [status, stats, artifacts] = await Promise.all([
      api.get(API_ENDPOINTS.runs.status(runId)).catch(() => null),
      api.get(API_ENDPOINTS.runs.stats(runId)).catch(() => null),
      api.get(API_ENDPOINTS.runs.artifacts(runId)).catch(() => null),
    ]);

    if (status) {
      card.appendChild(el("div", { class: "tc-grid-2 tc-mt-2" },
        statItem(t("ui.label.status", "Status"), status.status || "\u2014"),
        statItem(t("ui.label.phase", "Phase"), status.phase || "\u2014"),
        statItem(t("ui.label.created", "Erstellt"), status.created_at || status.started_at || "\u2014"),
        statItem(t("ui.label.elapsed", "Elapsed"), status.elapsed || "\u2014"),
      ));
    }

    if (stats) {
      card.appendChild(el("div", { class: "tc-mt-3 tc-label" }, t("ui.title.stats", "Stats")));
      card.appendChild(el("div", { class: "tc-grid-2" },
        statItem(t("ui.label.frames", "Frames"), stats.frames ?? "\u2014"),
        statItem(t("ui.label.registered", "Registered"), stats.registered ?? "\u2014"),
        statItem(t("ui.label.sqm", "SQM"), stats.sqm ?? "\u2014"),
        statItem(t("ui.label.snr", "SNR"), stats.snr ?? "\u2014"),
      ));
    }

    if (artifacts && (artifacts.items || artifacts).length > 0) {
      const items = artifacts.items || artifacts;
      card.appendChild(el("div", { class: "tc-mt-3 tc-label" }, t("ui.title.artifacts", "Artifacts")));
      const artList = el("div", { class: "tc-flex-col tc-gap-1" });
      for (const art of items) {
        const name = art.name || art.path || art.filename || String(art);
        artList.appendChild(el("div", { class: "tc-text-sm tc-mono" },
          el("a", {
            href: "#",
            onclick: (e) => { e.preventDefault(); viewArtifact(runId, art.path || name); },
          }, name),
        ));
      }
      card.appendChild(artList);
    }

    card.appendChild(el("div", { class: "tc-mt-3 tc-flex tc-gap-2" },
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => setRunCurrent(runId) }, t("ui.button.set_current", "Als aktuell setzen")),
      el("button", { class: "tc-btn tc-btn-sm tc-btn-danger", onclick: () => deleteRun(runId) }, t("ui.button.delete", "L\u00f6schen")),
    ));
  } catch (e) {
    card.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, e.message));
  }
}

function statItem(label, value) {
  return el("div", {},
    el("div", { class: "tc-label" }, label),
    el("div", { class: "tc-text-sm tc-mono" }, String(value)),
  );
}

async function viewArtifact(runId, path) {
  try {
    const url = api._toHttpUrl(API_ENDPOINTS.runs.artifactView(runId, path));
    window.open(url, "_blank");
  } catch (e) {
    toastError(t("ui.toast.view_failed", "Anzeigen fehlgeschlagen"), e.message);
  }
}

async function setRunCurrent(runId) {
  try {
    await api.post(API_ENDPOINTS.runs.setCurrent(runId), {});
    toastSuccess(t("ui.toast.set_current", "Als aktuell gesetzt"));
  } catch (e) {
    toastError(t("ui.toast.set_current_failed", "Fehlgeschlagen"), e.message);
  }
}

async function deleteRun(runId) {
  if (!confirm(t("ui.confirm.delete_run", "Run wirklich l\u00f6schen?"))) return;
  try {
    await api.delete(API_ENDPOINTS.runs.delete(runId));
    toastSuccess(t("ui.toast.deleted", "Gel\u00f6scht"));
    selectedRunId = null;
    loadRuns();
  } catch (e) {
    toastError(t("ui.toast.delete_failed", "L\u00f6schen fehlgeschlagen"), e.message);
  }
}

function compareRuns() {
  toast(t("ui.toast.compare_todo", "Vergleich noch nicht implementiert"), "", "info");
}
