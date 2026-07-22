// js/api/endpoints.js – API_ENDPOINTS (aus GUI2 constants.js migriert)

import { encodeRunIdPathSegment } from "../utils/path.js";

export const API_ENDPOINTS = {
  fs: {
    grantRoot: "/api/fs/grant-root",
    openPath: "/api/fs/open",
  },
  jobs: {
    list: "/api/jobs",
    byId: (jobId) => `/api/jobs/${encodeURIComponent(String(jobId || ""))}`,
  },
  guardrails: {
    root: "/api/guardrails",
  },
  app: {
    state: "/api/app/state",
    constants: "/api/app/constants",
    uiState: "/api/app/ui-state",
  },
  scan: {
    root: "/api/scan",
    latest: "/api/scan/latest",
    jobStatus: (jobId) => `/api/jobs/${encodeURIComponent(String(jobId || ""))}`,
    quality: "/api/scan/quality",
    metrics: "/api/scan/metrics",
    metricsLatest: "/api/scan/metrics/latest",
    analysis: "/api/scan/analysis",
    analysisLatest: "/api/scan/analysis/latest",
    analysisStore: "/api/scan/analysis/store",
    analysisHistory: "/api/scan/analysis/history",
    analysisHistoryItem: (filename) => `/api/scan/analysis/history/${encodeURIComponent(String(filename || ""))}`,
    analysisApply: "/api/scan/analysis/apply",
  },
  ai: {
    config: "/api/ai/config",
    models: "/api/ai/models",
    account: (provider = "") => `/api/ai/account?provider=${encodeURIComponent(String(provider || ""))}`,
    traffic: (limit = 500) => `/api/ai/traffic?limit=${encodeURIComponent(String(limit || 500))}`,
    auth: "/api/ai/auth",
    authProvider: (provider = "") => `/api/ai/auth/${encodeURIComponent(String(provider || ""))}`,
    test: "/api/ai/test",
  },
  pi: {
    tools: "/api/pi/tools",
    toolsCall: "/api/pi/tools/call",
    context: "/api/pi/context",
    ask: "/api/pi/assistant/ask",
    runChat: "/api/pi/run-chat",
    runChatHistory: (runId = "") => `/api/pi/run-chat/history?run_id=${encodeURIComponent(String(runId || ""))}`,
    actionPlanValidate: "/api/pi/action-plans/validate",
    actionPlanPreview: "/api/pi/action-plans/preview",
    actionPlanApply: "/api/pi/action-plans/apply",
    storage: "/api/pi/storage",
    memories: "/api/pi/memories",
    memoriesExport: "/api/pi/memories/export",
    memoriesImport: "/api/pi/memories/import",
    memoriesDedupe: "/api/pi/memories/dedupe",
    memoryReview: (memoryId = "") => `/api/pi/memories/${encodeURIComponent(String(memoryId || ""))}/review`,
    memoryRetrieve: "/api/pi/memories/retrieve",
    audit: "/api/pi/audit",
    liveImageChat: {
      create: "/api/pi/live-image-chat/create",
      chat: "/api/pi/live-image-chat",
      adjust: "/api/pi/live-image-chat/adjust",
      undo: "/api/pi/live-image-chat/undo",
      redo: "/api/pi/live-image-chat/redo",
      reset: "/api/pi/live-image-chat/reset",
      export: "/api/pi/live-image-chat/export",
      history: (runId) => `/api/pi/live-image-chat/history?run_id=${encodeURIComponent(String(runId || ""))}`,
      close: "/api/pi/live-image-chat/close",
    },
  },
  config: {
    schema: "/api/config/schema",
    defaults: "/api/config/defaults",
    current: "/api/config/current",
    patch: "/api/config/patch",
    presets: (dir = "") => {
      const query = String(dir || "").trim()
        ? `?dir=${encodeURIComponent(String(dir || "").trim())}`
        : "";
      return `/api/config/presets${query}`;
    },
    applyPreset: "/api/config/presets/apply",
    validate: "/api/config/validate",
    save: "/api/config/save",
    revisions: "/api/config/revisions",
  },
  runs: {
    list: "/api/runs",
    start: "/api/runs/start",
    status: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/status`,
    config: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/config`,
    configRevisions: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/config-revisions`,
    configRevision: (runId, revisionId) => `/api/runs/${encodeRunIdPathSegment(runId)}/config-revisions/${encodeURIComponent(String(revisionId || ""))}`,
    artifacts: (runId, runDir = "") => {
      const query = String(runDir || "").trim() ? `?run_dir=${encodeURIComponent(String(runDir || "").trim())}` : "";
      return `/api/runs/${encodeRunIdPathSegment(runId)}/artifacts${query}`;
    },
    artifactView: (runId, path = "", runDir = "") => {
      const params = new URLSearchParams();
      params.set("path", String(path || ""));
      if (String(runDir || "").trim()) params.set("run_dir", String(runDir || "").trim());
      return `/api/runs/${encodeRunIdPathSegment(runId)}/artifacts/view?${params.toString()}`;
    },
    artifactRaw: (runId, path = "", runDir = "") => {
      const base = `/api/runs/${encodeRunIdPathSegment(runId)}/artifacts/raw/${String(path || "").split("/").map((part) => encodeURIComponent(part)).join("/")}`;
      const query = String(runDir || "").trim() ? `?run_dir=${encodeURIComponent(String(runDir || "").trim())}` : "";
      return `${base}${query}`;
    },
    imagePreview: (runId, path = "", runDir = "") => {
      const params = new URLSearchParams();
      params.set("path", String(path || ""));
      if (String(runDir || "").trim()) params.set("run_dir", String(runDir || "").trim());
      return `/api/runs/${encodeRunIdPathSegment(runId)}/image-preview?${params.toString()}`;
    },
    delete: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/delete`,
    stop: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/stop`,
    resume: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/resume`,
    hmePreview: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/hme-preview`,
    bgePreview: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/bge-preview`,
    stats: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/stats`,
    statsStatus: (runId, runDir = "") => {
      const query = String(runDir || "").trim()
        ? `?run_dir=${encodeURIComponent(String(runDir || "").trim())}`
        : "";
      return `/api/runs/${encodeRunIdPathSegment(runId)}/stats/status${query}`;
    },
    logs: (runId, tail = 250, runDir = "") => {
      const params = new URLSearchParams();
      params.set("tail", String(tail));
      if (String(runDir || "").trim()) params.set("run_dir", String(runDir || "").trim());
      return `/api/runs/${encodeRunIdPathSegment(runId)}/logs?${params.toString()}`;
    },
    setCurrent: (runId) => `/api/runs/${encodeRunIdPathSegment(runId)}/set-current`,
    restoreRevision: (runId, revisionId) => `/api/runs/${encodeRunIdPathSegment(runId)}/config-revisions/${encodeURIComponent(String(revisionId || ""))}/restore`,
  },
  ws: {
    run: (runId, runDir = "") => {
      const query = String(runDir || "").trim() ? `?run_dir=${encodeURIComponent(String(runDir || "").trim())}` : "";
      return `/api/ws/runs/${encodeRunIdPathSegment(runId)}${query}`;
    },
  },
  astrometry: {
    detect: "/api/tools/astrometry/detect",
    installCli: "/api/tools/astrometry/install-cli",
    downloadCatalog: "/api/tools/astrometry/catalog/download",
    cancelDownload: "/api/tools/astrometry/catalog/cancel",
    solve: "/api/tools/astrometry/solve",
    saveSolved: "/api/tools/astrometry/save-solved",
  },
  pcc: {
    sirilStatus: (catalogDir = "") => `/api/tools/pcc/siril/status?catalog_dir=${encodeURIComponent(String(catalogDir || ""))}`,
    downloadMissing: "/api/tools/pcc/siril/download-missing",
    cancelDownload: "/api/tools/pcc/siril/cancel",
    checkOnline: "/api/tools/pcc/check-online",
    run: "/api/tools/pcc/run",
    saveCorrected: "/api/tools/pcc/save-corrected",
  },
  preprocessing: {
    defaults: "/api/tools/preprocessing/defaults",
    parameters: "/api/tools/preprocessing/parameters",
    scan: "/api/tools/preprocessing/scan",
    run: "/api/tools/preprocessing/run",
    cancel: "/api/tools/preprocessing/cancel",
    status: (jobId = "") => `/api/tools/preprocessing/status?job_id=${encodeURIComponent(String(jobId || ""))}`,
    report: (jobId = "") => `/api/tools/preprocessing/report?job_id=${encodeURIComponent(String(jobId || ""))}`,
  },
};
