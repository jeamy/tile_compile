// js/utils/poll.js – Generic job polling utility

import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";

const DONE_STATES = new Set(["ok", "done", "completed"]);
const ERROR_STATES = new Set(["error", "failed", "cancelled"]);

/**
 * Poll a job endpoint until it reaches a terminal state.
 *
 * @param {string} jobId - The job ID to poll.
 * @param {object} opts
 * @param {string} opts.endpoint - API endpoint function (e.g. API_ENDPOINTS.jobs.byId or API_ENDPOINTS.scan.jobStatus).
 * @param {number} opts.intervalMs - Polling interval in ms (default 2000).
 * @param {number} opts.timeoutMs - Max total wait time in ms (default 300000).
 * @param {function|null} opts.onProgress - Optional callback invoked with each job response.
 * @param {function|null} opts.onDone - Optional transform applied to the job on success (e.g. fetch latest scan).
 * @param {string} opts.errorLabel - Label used in timeout error message (default "Job").
 * @returns {Promise<any>} The job result (or transformed result if onDone provided).
 */
export async function pollJob(jobId, {
  endpoint = API_ENDPOINTS.jobs.byId,
  intervalMs = 2000,
  timeoutMs = 300000,
  onProgress = null,
  onDone = null,
  errorLabel = "Job",
} = {}) {
  const maxAttempts = Math.ceil(timeoutMs / intervalMs);
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, intervalMs));
    const job = await api.get(endpoint(jobId));
    const state = job?.state;
    if (onProgress) onProgress(job);
    if (DONE_STATES.has(state)) {
      return onDone ? await onDone(job) : (job?.data || job);
    }
    if (ERROR_STATES.has(state)) {
      const stderr = job?.data?.stderr || "";
      const stdout = job?.data?.stdout || "";
      const detail = stderr || stdout || job?.error || `${errorLabel} failed`;
      throw new Error(String(detail).substring(0, 500));
    }
  }
  throw new Error(`${errorLabel} timeout`);
}
