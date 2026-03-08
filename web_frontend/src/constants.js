export const API_ENDPOINTS = {
  pcc: {
    sirilStatus: (catalogDir = "") => `/api/tools/pcc/siril/status?catalog_dir=${encodeURIComponent(String(catalogDir || ""))}`,
    downloadMissing: "/api/tools/pcc/siril/download-missing",
    cancelDownload: "/api/tools/pcc/siril/cancel",
    checkOnline: "/api/tools/pcc/check-online",
    run: "/api/tools/pcc/run",
    saveCorrected: "/api/tools/pcc/save-corrected",
  },
};
