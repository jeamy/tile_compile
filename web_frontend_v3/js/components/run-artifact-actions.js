import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { t } from "../i18n/i18n.js";

export function deletionErrorMessage(error) {
  const code = error?.payload?.error?.code || error?.payload?.code;
  const keys = {
    RAW_SOURCE_INSIDE_RUN: "raw_protected",
    LEARNING_SNAPSHOT_FAILED: "archive_failed",
    RUN_IDENTITY_CONFLICT: "identity_conflict",
    LEARNING_SNAPSHOT_INCOMPLETE: "incomplete_blocked",
    RUN_ACTIVE: "run_active",
    LEARNING_ARCHIVE_UNAVAILABLE: "backend_required",
  };
  return keys[code] ? t(`ui.archive.${keys[code]}`) : error.message;
}

export async function deleteRunFiles(runId, runDir = "", { confirm = message => window.confirm(message), client = api } = {}) {
  const target = runDir || runId;
  if (!confirm(t("ui.archive.confirm_delete_files", undefined, { run: runId, path: target }))) return null;
  try {
    const capabilities = await client.get(API_ENDPOINTS.pi.runLearning.capabilities);
    if (capabilities?.schema_version !== "pi.run-learning-capabilities.v1" || capabilities.delete_preserves_learning !== true)
      throw new Error("Archive-preserving deletion unavailable");
  } catch {
    const error = new Error(t("ui.archive.backend_required"));
    error.payload = { code: "LEARNING_ARCHIVE_UNAVAILABLE" };
    throw error;
  }
  const body = runDir ? { run_dir: runDir } : {};
  try {
    return await client.post(API_ENDPOINTS.runs.delete(runId), body);
  } catch (error) {
    const code = error?.payload?.error?.code || error?.payload?.code;
    if (code !== "LEARNING_SNAPSHOT_INCOMPLETE" || error.status !== 409) throw error;
    if (!confirm(t("ui.archive.confirm_incomplete", undefined, { run: runId, path: target }))) return null;
    return client.post(API_ENDPOINTS.runs.delete(runId), { ...body, allow_incomplete_snapshot: true });
  }
}
