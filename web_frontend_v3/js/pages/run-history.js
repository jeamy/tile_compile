// js/pages/run-history.js – Sub-Tab: Run-Historie mit Detail-View

import { el, clear, statItem } from "../utils/dom.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastError, toastSuccess } from "../components/toast.js";
import { t } from "../i18n/i18n.js";
import { getStore } from "../state/store.js";
import { getUiState, setUiState } from "../state/ui-state.js";
import { pollJob } from "../utils/poll.js";
import { openStatsFolder, openStatsReport } from "../utils/stats-utils.js";

const store = getStore("run-history", {
  selectedRunId: null,
  compareRunId: null,
  runsCache: [],
});

function getSelectedRunId() { return store.getState().selectedRunId; }
function setSelectedRunId(id) { store.setState({ selectedRunId: id }); }
function getCompareRunId() { return store.getState().compareRunId; }
function setCompareRunId(id) { store.setState({ compareRunId: id }); }
function getRunsCache() { return store.getState().runsCache || []; }
function setRunsCache(runs) { store.setState({ runsCache: runs }); }

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
    el("div", { class: "tc-card-title tc-flex tc-items-center tc-justify-between", style: "cursor:pointer", onclick: () => {
      const body = document.getElementById("run-detail-body");
      const arrow = document.getElementById("run-detail-toggle");
      if (body) body.classList.toggle("tc-hidden");
      if (arrow) arrow.textContent = body?.classList.contains("tc-hidden") ? "\u25b6" : "\u25bc";
    } },
      el("span", {}, t("ui.title.selected_run", "Ausgew\u00e4hlter Run")),
      el("span", { class: "tc-text-muted", id: "run-detail-toggle" }, "\u25b6"),
    ),
    el("div", { id: "run-detail-body", class: "tc-hidden" },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_run_selected", "Kein Run ausgew\u00e4hlt")),
    ),
  );

  const compareCard = el("div", { class: "tc-card", id: "compare-card" },
    el("div", { class: "tc-card-title tc-flex tc-items-center tc-justify-between" },
      el("span", {}, t("ui.title.compare", "Vergleich")),
      el("div", { class: "tc-flex tc-items-center tc-gap-2" },
        el("select", { class: "tc-select", id: "compare-run-select", onchange: (e) => { setCompareRunId(e.target.value || null); renderCompare(); } },
          el("option", { value: "" }, "-"),
        ),
        el("button", { class: "tc-btn tc-btn-sm", onclick: () => { setCompareRunId(null); renderCompare(); } }, t("ui.button.clear", "Zur\u00fccksetzen")),
      ),
    ),
    el("div", { id: "compare-body" },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.select_compare_run", "Vergleichs-Run w\u00e4hlen")),
    ),
  );

  const actionsBar = el("div", { class: "tc-flex tc-gap-2", id: "run-actions" });

  page.append(header, listCard, detailCard, compareCard, actionsBar);

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
    setRunsCache(runArr);
    for (const run of runArr) {
      list.appendChild(runItem(run));
    }
    updateCompareDropdown();
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
    "data-run-id": runId,
    onclick: () => selectRun(runId),
  },
    el("span", { class: "tc-badge" }, run.pipeline || "AQMH"),
    el("span", { class: `tc-badge ${badgeClass}` }, status),
    el("span", { class: "tc-mono tc-text-sm" }, runId),
    el("span", { class: "tc-text-sm" }, run.run_name || run.name || ""),
  );

  if (runId === getSelectedRunId()) item.classList.add("selected");
  return item;
}

async function selectRun(runId) {
  setSelectedRunId(runId);
  document.querySelectorAll(".tc-run-item").forEach(item => {
    item.classList.toggle("selected", item.getAttribute("data-run-id") === runId);
  });
  updateCompareDropdown();
  const body = document.getElementById("run-detail-body");
  if (!body) return;
  clear(body);

  body.appendChild(el("div", { class: "tc-text-muted tc-text-sm tc-mb-2" }, runId));

  try {
    const [status, stats, artifacts] = await Promise.all([
      api.get(API_ENDPOINTS.runs.status(runId)).catch(() => null),
      api.get(API_ENDPOINTS.runs.stats(runId)).catch(() => null),
      api.get(API_ENDPOINTS.runs.artifacts(runId)).catch(() => null),
    ]);

    if (status) {
      body.appendChild(el("div", { class: "tc-grid-2 tc-mt-2" },
        statItem(t("ui.label.status", "Status"), status.status || "\u2014"),
        statItem(t("ui.label.phase", "Phase"), status.phase || "\u2014"),
        statItem(t("ui.label.created", "Erstellt"), status.created_at || status.started_at || "\u2014"),
        statItem(t("ui.label.elapsed", "Elapsed"), status.elapsed || "\u2014"),
      ));
    }

    if (stats) {
      body.appendChild(el("div", { class: "tc-mt-3 tc-label" }, t("ui.title.stats", "Stats")));
      body.appendChild(el("div", { class: "tc-grid-2" },
        statItem(t("ui.label.frames", "Frames"), stats.frames ?? "\u2014"),
        statItem(t("ui.label.registered", "Registered"), stats.registered ?? "\u2014"),
        statItem(t("ui.label.sqm", "SQM"), stats.sqm ?? "\u2014"),
        statItem(t("ui.label.snr", "SNR"), stats.snr ?? "\u2014"),
      ));
    }

    if (artifacts && (artifacts.items || artifacts).length > 0) {
      const items = artifacts.items || artifacts;
      body.appendChild(el("div", { class: "tc-mt-3 tc-label" }, t("ui.title.artifacts", "Artifacts")));
      const artList = el("div", { class: "tc-flex-col tc-gap-1" });
      for (const art of items) {
        const name = art.name || art.path || art.filename || String(art);
        artList.appendChild(el("div", { class: "tc-text-sm tc-mono" },
          el("a", {
            href: "#",
            onclick: (e) => { e.preventDefault(); viewArtifact(runId, art.path || name, status?.run_dir); },
          }, name),
        ));
      }
      body.appendChild(artList);
    }

    const actions = document.getElementById("run-actions");
    if (actions) {
      clear(actions);
      const runStatus = (status?.status || "").toLowerCase();
      const isTerminal = ["ok", "done", "completed", "error", "failed", "stopped", "aborted"].includes(runStatus);

      let hasReport = false;
      try {
        const statsStatus = await api.get(API_ENDPOINTS.runs.statsStatus(runId, status?.run_dir || ""));
        hasReport = !!statsStatus?.report_path;
      } catch {}

      actions.appendChild(el("button", { class: "tc-btn tc-btn-sm", onclick: () => setRunCurrent(runId) }, t("ui.button.set_current", "Als aktuell setzen")));

      const genBtn = el("button", { class: "tc-btn tc-btn-sm", disabled: !isTerminal, onclick: () => generateStatsForRun(runId) }, t("ui.button.generate_stats", "Generate Stats"));
      const openBtn = el("button", { class: "tc-btn tc-btn-sm", disabled: !hasReport, onclick: () => openStatsFolder(runId, status?.run_dir) }, t("ui.button.open_stats_folder", "Open Stats Folder"));
      const reportBtn = el("button", { class: "tc-btn tc-btn-sm", disabled: !hasReport, onclick: () => openStatsReport(runId, status?.run_dir) }, t("ui.button.open_stats_report", "Open Report"));
      actions.appendChild(genBtn);
      actions.appendChild(openBtn);
      actions.appendChild(reportBtn);

      actions.appendChild(el("button", { class: "tc-btn tc-btn-sm tc-btn-danger", onclick: () => deleteRun(runId) }, t("ui.button.delete", "L\u00f6schen")));
    }
  } catch (e) {
    body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, e.message));
  }
}

async function viewArtifact(runId, path, runDir = "") {
  try {
    const url = api._toHttpUrl(API_ENDPOINTS.runs.artifactView(runId, path, runDir));
    window.open(url, "_blank");
  } catch (e) {
    toastError(t("ui.toast.view_failed", "Anzeigen fehlgeschlagen"), e.message);
  }
}

async function setRunCurrent(runId) {
  try {
    const cached = getRunsCache().find(r => (r.run_id || r.id) === runId);
    const runDir = cached?.path || cached?.run_dir || "";
    await api.post(API_ENDPOINTS.runs.setCurrent(runId), runDir ? { run_dir: runDir } : {});
    toastSuccess(t("ui.toast.set_current", "Als aktuell gesetzt"));
    const ui = getUiState();
    setUiState({ activeTab: "processing", activeSubTab: { ...ui.activeSubTab, processing: "run-monitor" } });
    window.location.hash = "#processing";
    window.dispatchEvent(new Event("tc-subtab-change"));
  } catch (e) {
    toastError(t("ui.toast.set_current_failed", "Fehlgeschlagen"), e.message);
  }
}

async function deleteRun(runId) {
  if (!confirm(t("ui.confirm.delete_run", "Run wirklich l\u00f6schen?"))) return;
  try {
    await api.post(API_ENDPOINTS.runs.delete(runId), {});
    toastSuccess(t("ui.toast.deleted", "Gel\u00f6scht"));
    setSelectedRunId(null);
    loadRuns();
  } catch (e) {
    toastError(t("ui.toast.delete_failed", "L\u00f6schen fehlgeschlagen"), e.message);
  }
}

function compareRuns() {
  const compareId = getCompareRunId();
  if (!compareId) {
    toast(t("ui.toast.select_compare_first", "Bitte zuerst einen Vergleichs-Run auswählen"), "", "info");
    return;
  }
  renderCompare();
}

function updateCompareDropdown() {
  const select = document.getElementById("compare-run-select");
  if (!select) return;
  const selectedId = getSelectedRunId();
  const compareId = getCompareRunId();
  const candidates = getRunsCache().filter(r => (r.run_id || r.id) !== selectedId);
  const current = compareId && candidates.some(r => (r.run_id || r.id) === compareId) ? compareId : "";
  select.innerHTML = "";
  select.appendChild(el("option", { value: "" }, "-"));
  for (const run of candidates) {
    const rid = run.run_id || run.id || "";
    const label = `${rid} (${run.status || "?"})`;
    const opt = el("option", { value: rid, ...(rid === current ? { selected: true } : {}) }, label);
    select.appendChild(opt);
  }
  if (current !== compareId) setCompareRunId(current || null);
}

async function loadRunSnapshot(runId) {
  if (!runId) return null;
  try {
    const status = await api.get(API_ENDPOINTS.runs.status(runId));
    if (!status) return null;
    const runDir = status.run_dir || "";

    const [statsStatus, artifactsResult, logs] = await Promise.all([
      api.get(API_ENDPOINTS.runs.statsStatus(runId, runDir)).catch(() => null),
      api.get(API_ENDPOINTS.runs.artifacts(runId)).catch(() => null),
      api.get(API_ENDPOINTS.runs.logs(runId, 99999)).catch(() => null),
    ]);
    const artifacts = artifactsResult?.items || artifactsResult || [];

    let frames = "-";
    let elapsed = "-";
    let firstTs = null;
    let lastTs = null;
    let method = status.method || "-";

    const events = status.events || [];
    for (const ev of events) {
      if (ev.loaded_frames != null && frames === "-") frames = ev.loaded_frames;
      if (ev.ts) {
        if (!firstTs || ev.ts < firstTs) firstTs = ev.ts;
        if (!lastTs || ev.ts > lastTs) lastTs = ev.ts;
      }
    }

    if (logs?.lines) {
      for (const line of logs.lines) {
        try {
          const ev = typeof line === "string" ? JSON.parse(line) : line;
          if (ev.frames_discovered != null) frames = ev.frames_discovered;
          if (ev.type === "run_start" && ev.ts) firstTs = ev.ts;
          if (ev.type === "run_end" && ev.ts) lastTs = ev.ts;
          if (ev.method) method = ev.method;
        } catch {}
      }
    }

    if (firstTs && lastTs) {
      const ms = new Date(lastTs) - new Date(firstTs);
      if (ms > 0) {
        const mins = Math.floor(ms / 60000);
        const secs = Math.floor((ms % 60000) / 1000);
        elapsed = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
      }
    }

    let phase = "-";
    const phases = status.phases || [];
    if (status.current_phase) {
      phase = status.current_phase;
    } else if (phases.length > 0) {
      const lastPhase = phases[phases.length - 1];
      phase = lastPhase.phase || lastPhase.phase_name || "DONE";
    }

    const hasReport = !!statsStatus?.report_path;
    return {
      runId,
      status: status.status || "-",
      phase,
      progress: status.progress != null ? `${Math.round(status.progress * 100)}%` : "-",
      elapsed,
      frames,
      method,
      colorMode: status.color_mode || "-",
      runDir,
      artifactCount: Array.isArray(artifacts) ? artifacts.length : 0,
      hasReport,
      reportPath: statsStatus?.report_path || "",
    };
  } catch {
    return null;
  }
}

async function renderCompare() {
  const body = document.getElementById("compare-body");
  if (!body) return;
  clear(body);

  const selectedId = getSelectedRunId();
  const compareId = getCompareRunId();

  if (!selectedId || !compareId) {
    body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.select_compare_run", "Vergleichs-Run wählen")));
    return;
  }

  body.appendChild(el("div", { class: "tc-text-muted tc-text-sm tc-mb-2" }, `${selectedId} vs ${compareId}`));

  const [snapA, snapB] = await Promise.all([
    loadRunSnapshot(selectedId),
    loadRunSnapshot(compareId),
  ]);

  if (!snapA || !snapB) {
    body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.compare_failed", "Run-Daten konnten nicht geladen werden")));
    return;
  }

  const rows = [
    { label: t("ui.label.run_id", "Run ID"), a: snapA.runId, b: snapB.runId },
    { label: t("ui.label.status", "Status"), a: snapA.status, b: snapB.status },
    { label: t("ui.label.phase", "Phase"), a: snapA.phase, b: snapB.phase },
    { label: t("ui.label.progress", "Progress"), a: snapA.progress, b: snapB.progress },
    { label: t("ui.label.elapsed", "Elapsed"), a: snapA.elapsed, b: snapB.elapsed },
    { label: t("ui.label.frames", "Frames"), a: snapA.frames, b: snapB.frames },
    { label: t("ui.label.method", "Method"), a: snapA.method, b: snapB.method },
    { label: t("ui.label.color_mode", "OSC/Mono"), a: snapA.colorMode, b: snapB.colorMode },
    { label: t("ui.label.report", "Report"), a: snapA.hasReport ? "✓" : "✗", b: snapB.hasReport ? "✓" : "✗" },
  ];

  const table = el("table", { class: "tc-compare-table", style: { width: "100%", borderCollapse: "collapse" } });
  table.appendChild(el("thead", {},
    el("tr", {},
      el("th", { class: "tc-label", style: { textAlign: "left", padding: "4px 8px" } }, ""),
      el("th", { class: "tc-label", style: { textAlign: "left", padding: "4px 8px" } }, t("ui.label.selected", "Ausgewählt")),
      el("th", { class: "tc-label", style: { textAlign: "left", padding: "4px 8px" } }, t("ui.label.compare", "Vergleich")),
    ),
  ));
  const tbody = el("tbody");
  for (const row of rows) {
    const diff = String(row.a) !== String(row.b);
    tbody.appendChild(el("tr", { style: { borderBottom: "1px solid var(--border)" } },
      el("td", { class: "tc-text-sm tc-label", style: { padding: "4px 8px" } }, row.label),
      el("td", { class: `tc-text-sm tc-mono ${diff ? "tc-text-warning" : ""}`, style: { padding: "4px 8px" } }, String(row.a)),
      el("td", { class: `tc-text-sm tc-mono ${diff ? "tc-text-warning" : ""}`, style: { padding: "4px 8px" } }, String(row.b)),
    ));
  }
  table.appendChild(tbody);
  body.appendChild(table);

  const btnRow = el("div", { class: "tc-flex tc-gap-2 tc-mt-3" });
  if (snapA.hasReport) {
    btnRow.appendChild(el("button", { class: "tc-btn tc-btn-sm", onclick: () => openStatsReport(selectedId, snapA.runDir) }, t("ui.button.open_report_a", "Report A")));
  }
  if (snapB.hasReport) {
    btnRow.appendChild(el("button", { class: "tc-btn tc-btn-sm", onclick: () => openStatsReport(compareId, snapB.runDir) }, t("ui.button.open_report_b", "Report B")));
  }
  body.appendChild(btnRow);
}

async function generateStatsForRun(runId) {
  try {
    toast(t("ui.toast.stats_generating", "Stats werden generiert..."), "", "info");
    const result = await api.post(API_ENDPOINTS.runs.stats(runId), {});
    const jobId = result?.job_id;
    if (jobId) {
      await pollJob(jobId, { intervalMs: 1000, timeoutMs: 120000 });
      toastSuccess(t("ui.toast.stats_done", "Stats generiert"));
      selectRun(runId);
    }
  } catch (e) {
    toastError(t("ui.toast.stats_failed", "Stats-Generierung fehlgeschlagen"), e.message);
  }
}

