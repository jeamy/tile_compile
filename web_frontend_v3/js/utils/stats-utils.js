// js/utils/stats-utils.js – Shared stats folder/report helpers

import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toastError } from "../components/toast.js";
import { t } from "../i18n/i18n.js";

export async function openStatsFolder(runId, runDir = "") {
  try {
    const status = await api.get(API_ENDPOINTS.runs.statsStatus(runId, runDir));
    const dir = status?.output_dir;
    if (!dir) {
      toastError(t("ui.toast.stats_folder_unavailable", "Stats-Ordner nicht verfügbar"));
      return;
    }
    await api.post(API_ENDPOINTS.fs.openPath, { path: dir });
  } catch (e) {
    toastError(t("ui.toast.open_failed", "Öffnen fehlgeschlagen"), e.message);
  }
}

export async function openStatsReport(runId, runDir = "") {
  try {
    const status = await api.get(API_ENDPOINTS.runs.statsStatus(runId, runDir));
    const reportPath = status?.report_path;
    if (!reportPath) {
      toastError(t("ui.toast.stats_folder_unavailable", "Report nicht verfügbar"));
      return;
    }
    const artifactRelPath = reportPath.includes("/artifacts/")
      ? "artifacts/" + reportPath.substring(reportPath.indexOf("/artifacts/") + "/artifacts/".length)
      : "artifacts/report.html";
    const url = api._toHttpUrl(API_ENDPOINTS.runs.artifactRaw(runId, artifactRelPath, runDir));
    window.open(url, "_blank");
  } catch (e) {
    toastError(t("ui.toast.open_failed", "Öffnen fehlgeschlagen"), e.message);
  }
}
