export const API_ENDPOINTS = {
  fs: {
    grantRoot: "/api/fs/grant-root",
    openPath: "/api/fs/open",
  },
  jobs: {
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
    quality: "/api/scan/quality",
  },
  config: {
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
    status: (runId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/status`,
    config: (runId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/config`,
    configRevisions: (runId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/config-revisions`,
    configRevision: (runId, revisionId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/config-revisions/${encodeURIComponent(String(revisionId || ""))}`,
    artifacts: (runId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/artifacts`,
    artifactView: (runId, path = "") => `/api/runs/${encodeURIComponent(String(runId || ""))}/artifacts/view?path=${encodeURIComponent(String(path || ""))}`,
    artifactRaw: (runId, path = "") => `/api/runs/${encodeURIComponent(String(runId || ""))}/artifacts/raw/${String(path || "").split("/").map((part) => encodeURIComponent(part)).join("/")}`,
    delete: (runId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/delete`,
    stop: (runId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/stop`,
    resume: (runId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/resume`,
    stats: (runId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/stats`,
    statsStatus: (runId, runDir = "") => {
      const query = String(runDir || "").trim()
        ? `?run_dir=${encodeURIComponent(String(runDir || ""))}`
        : "";
      return `/api/runs/${encodeURIComponent(String(runId || ""))}/stats/status${query}`;
    },
    logs: (runId, tail = 250) => `/api/runs/${encodeURIComponent(String(runId || ""))}/logs?tail=${encodeURIComponent(String(tail))}`,
    setCurrent: (runId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/set-current`,
    restoreRevision: (runId, revisionId) => `/api/runs/${encodeURIComponent(String(runId || ""))}/config-revisions/${encodeURIComponent(String(revisionId || ""))}/restore`,
  },
  ws: {
    run: (runId) => `/api/ws/runs/${encodeURIComponent(String(runId || ""))}`,
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
};
