// js/pages/pcc.js – Sub-Tab: PCC (Photometric Color Calibration)

import { el, setBadge } from "../utils/dom.js";
import { createPathInput } from "../components/path-input.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { t } from "../i18n/i18n.js";
import { getStore } from "../state/store.js";
import { pollJob } from "../utils/poll.js";

const store = getStore("pcc", {
  pccData: { rgb_fits: "", wcs_file: "", output_rgb: "", catalog_source: "siril", catalog_dir: "", mag_limit: 14.0, mag_bright_limit: 6.0, min_stars: 10, sigma_clip: 2.5 },
  downloadJobId: null,
});

function getDownloadJobId() { return store.getState().downloadJobId; }
function setDownloadJobId(id) { store.setState({ downloadJobId: id }); }

function getPccData() { return store.getState().pccData; }
function setPccData(patch) { store.setState({ pccData: { ...getPccData(), ...patch } }); }

export function createPccPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  const inputCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.input", "Input")),
    createPathInput({ label: t("ui.field.rgb_fits", "RGB FITS"), mode: "file", filter: "*.fits;*.fit;*.fts", placeholder: "/data/runs/M31/outputs/stack_M31.fits", value: getPccData().rgb_fits, onInput: (v) => {
      const defaultWcs = v.replace(/\.(fits|fit|fts|fits\.fz|fit\.fz|fts\.fz)$/i, ".wcs");
      const defaultOut = v.replace(/\.(fits|fit|fts|fits\.fz|fit\.fz|fts\.fz)$/i, "_pcc.fits");
      setPccData({ rgb_fits: v, wcs_file: defaultWcs, output_rgb: defaultOut });
    } }),
    el("div", { class: "tc-mt-2" },
      createPathInput({ label: t("ui.field.wcs_file", "WCS Datei"), mode: "file", filter: "*.wcs;*.fits;*.fit", placeholder: "/data/runs/M31/outputs/stack_M31.wcs", value: getPccData().wcs_file, onInput: (v) => setPccData({ wcs_file: v }) }),
    ),
    el("div", { class: "tc-mt-2" },
      createPathInput({ label: t("ui.field.output_rgb", "Output RGB"), mode: "file", filter: "*.fits;*.fit;*.fts", placeholder: "auto: *_pcc.fits", value: getPccData().output_rgb, onInput: (v) => setPccData({ output_rgb: v }) }),
    ),
  );

  const catalogCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.catalog_source", "Catalog Source")),
    el("div", {},
      el("label", { class: "tc-label" }, t("ui.field.source", "Source")),
      el("select", { class: "tc-select", onchange: (e) => setPccData({ catalog_source: e.target.value }) },
        el("option", { value: "siril", ...(getPccData().catalog_source === "siril" ? { selected: true } : {}) }, "Siril (lokale Gaia-DR3-XP Chunks)"),
        el("option", { value: "vizier_gaia", ...(getPccData().catalog_source === "vizier_gaia" ? { selected: true } : {}) }, "Online: VizieR Gaia"),
        el("option", { value: "vizier_apass", ...(getPccData().catalog_source === "vizier_apass" ? { selected: true } : {}) }, "Online: VizieR APASS"),
      ),
    ),
    el("div", { class: "tc-mt-2" },
      createPathInput({ label: t("ui.field.catalog_dir", "Catalog Dir"), mode: "dir", placeholder: "", value: getPccData().catalog_dir, onInput: (v) => { setPccData({ catalog_dir: v }); const inp = document.getElementById("pcc-catalog-dir"); if (inp) inp.value = v; } }),
      el("input", { type: "hidden", id: "pcc-catalog-dir", value: getPccData().catalog_dir }),
      el("span", { class: "tc-badge", id: "pcc-catalog-badge", style: { whiteSpace: "nowrap" } }, "…"),
    ),
    el("div", { class: "tc-flex tc-gap-3 tc-items-center tc-mt-2" },
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => checkSirilStatus() }, t("ui.button.check_catalog", "Check Catalog")),
      el("button", { class: "tc-btn tc-btn-sm", id: "pcc-download-btn", onclick: () => downloadMissing() }, t("ui.button.download_missing", "Download Missing")),
      el("span", { class: "tc-text-muted", id: "pcc-download-status", style: { fontSize: "0.85em" } }, ""),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => checkOnline() }, t("ui.button.check_online", "Check Online Source")),
    ),
  );

  const paramCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.pcc_params", "PCC Parameters")),
    el("div", { class: "tc-grid-2" },
      el("div", {}, el("label", { class: "tc-label" }, "mag_limit"), el("input", { type: "number", class: "tc-input", value: String(getPccData().mag_limit), oninput: (e) => setPccData({ mag_limit: parseFloat(e.target.value) || 14.0 }) })),
      el("div", {}, el("label", { class: "tc-label" }, "mag_bright_limit"), el("input", { type: "number", class: "tc-input", value: String(getPccData().mag_bright_limit), oninput: (e) => setPccData({ mag_bright_limit: parseFloat(e.target.value) || 6.0 }) })),
      el("div", {}, el("label", { class: "tc-label" }, "min_stars"), el("input", { type: "number", class: "tc-input", value: String(getPccData().min_stars), oninput: (e) => setPccData({ min_stars: parseInt(e.target.value) || 10 }) })),
      el("div", {}, el("label", { class: "tc-label" }, "sigma_clip"), el("input", { type: "number", class: "tc-input", value: String(getPccData().sigma_clip), oninput: (e) => setPccData({ sigma_clip: parseFloat(e.target.value) || 2.5 }) })),
    ),
  );

  const actions = el("div", { class: "tc-flex tc-gap-3" },
    el("button", { class: "tc-btn tc-btn-primary", onclick: () => runPcc() }, t("ui.button.run_pcc", "Run PCC")),
    el("button", { class: "tc-btn", onclick: () => saveCorrected() }, t("ui.button.save_corrected", "Save Corrected")),
  );

  const resultCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.result", "Result")),
    el("div", { class: "tc-wcs-results", id: "pcc-results" },
      el("div", {}, el("span", { class: "tc-text-muted" }, t("ui.label.stars_matched", "Stars matched") + ": "), el("span", {}, "\u2014")),
      el("div", {}, el("span", { class: "tc-text-muted" }, t("ui.label.stars_used", "Stars used") + ": "), el("span", {}, "\u2014")),
      el("div", {}, el("span", { class: "tc-text-muted" }, t("ui.label.residual_rms", "Residual RMS") + ": "), el("span", {}, "\u2014")),
    ),
  );

  page.append(inputCard, catalogCard, paramCard, actions, resultCard);

  checkSirilStatus();

  const activeJobId = getDownloadJobId();
  if (activeJobId) {
    setTimeout(() => {
      const btn = document.getElementById("pcc-download-btn");
      if (btn) btn.disabled = true;
      const statusEl = document.getElementById("pcc-download-status");
      if (statusEl) statusEl.textContent = "Läuft...";
      resumeDownloadPolling(activeJobId);
    }, 50);
  }

  return page;
}

async function checkSirilStatus() {
  const badge = document.getElementById("pcc-catalog-badge");
  try {
    const dir = getPccData().catalog_dir || "";
    const result = await api.get(API_ENDPOINTS.pcc.sirilStatus(dir));
    if (result?.catalog_dir && !getPccData().catalog_dir) {
      setPccData({ catalog_dir: result.catalog_dir });
    }
    const installed = result?.installed || 0;
    const total = result?.total || 0;
    const missing = result?.missing || [];
    const ok = installed > 0 && missing.length === 0;
    const text = ok ? `${installed}/${total}` : (installed > 0 ? `${installed}/${total} (${missing.length} fehlen)` : "✗");
    setBadge(badge, ok, text);
  } catch (e) {
    setBadge(badge, false, "✗");
  }
}

async function downloadMissing() {
  const statusEl = document.getElementById("pcc-download-status");
  const btn = document.getElementById("pcc-download-btn");
  try {
    const pd = getPccData();
    if (btn) btn.disabled = true;
    if (statusEl) statusEl.textContent = "Starte Download...";
    toast(t("ui.toast.downloading_missing", "Lade fehlende Kataloge..."), "", "info");
    const startResp = await api.post(API_ENDPOINTS.pcc.downloadMissing, { catalog_dir: pd.catalog_dir }, { timeoutMs: 30000 });
    const jobId = startResp?.job_id;
    if (!jobId) {
      toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), "No job_id returned");
      if (btn) btn.disabled = false;
      if (statusEl) statusEl.textContent = "";
      return;
    }
    setDownloadJobId(jobId);
    const result = await pollDownloadJob(jobId, 600000, (job) => {
      const completed = job?.data?.completed_chunks ?? 0;
      const total = job?.data?.total_chunks ?? 0;
      const stage = job?.data?.stage ?? "";
      if (stage === "decompress") {
        if (statusEl) statusEl.textContent = `Entpacke Chunk ${completed + 1}/${total}`;
      } else if (total > 0) {
        if (statusEl) statusEl.textContent = `Lade ${completed + 1}/${total}`;
      } else if (statusEl) {
        statusEl.textContent = "Läuft...";
      }
    });
    finishDownload(result, statusEl, btn);
  } catch (e) {
    toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), e.message);
    const statusEl2 = document.getElementById("pcc-download-status");
    if (statusEl2) statusEl2.textContent = "Fehler";
    setDownloadJobId(null);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function finishDownload(result, statusEl, btn) {
  setDownloadJobId(null);
  const missing = result?.missing_after || result?.missing || [];
  if (Array.isArray(missing) && missing.length > 0) {
    toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), `${missing.length} Chunks fehlen noch`);
    if (statusEl) statusEl.textContent = `${missing.length} Chunks fehlen`;
  } else {
    toastSuccess(t("ui.toast.download_complete", "Download abgeschlossen"));
    if (statusEl) statusEl.textContent = "Fertig";
    checkSirilStatus();
  }
}

async function resumeDownloadPolling(jobId) {
  const statusEl = document.getElementById("pcc-download-status");
  const btn = document.getElementById("pcc-download-btn");
  if (statusEl) statusEl.textContent = "Läuft...";
  if (btn) btn.disabled = true;
  try {
    const result = await pollDownloadJob(jobId, 600000, (job) => {
      const el2 = document.getElementById("pcc-download-status");
      const completed = job?.data?.completed_chunks ?? 0;
      const total = job?.data?.total_chunks ?? 0;
      const stage = job?.data?.stage ?? "";
      if (stage === "decompress") {
        if (el2) el2.textContent = `Entpacke Chunk ${completed + 1}/${total}`;
      } else if (total > 0) {
        if (el2) el2.textContent = `Lade ${completed + 1}/${total}`;
      } else if (el2) {
        el2.textContent = "Läuft...";
      }
    });
    finishDownload(result, document.getElementById("pcc-download-status"), btn);
  } catch (e) {
    toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), e.message);
    const el2 = document.getElementById("pcc-download-status");
    if (el2) el2.textContent = "Fehler";
    setDownloadJobId(null);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function pollDownloadJob(jobId, timeoutMs = 600000, onProgress = null) {
  return pollJob(jobId, { timeoutMs, onProgress, errorLabel: "Download" });
}

async function checkOnline() {
  try {
    const result = await api.post(API_ENDPOINTS.pcc.checkOnline, {});
    const sources = result?.sources || [];
    if (sources.length > 0) {
      const parts = sources.map(s => `${s.name}: ${s.ok ? "OK" : "FAIL"}${s.latency_ms ? ` (${s.latency_ms}ms)` : ""}`);
      if (result?.ok) {
        toastSuccess(t("ui.toast.online_checked", "Online-Quellen überprüft"), parts.join(", "));
      } else {
        toastError(t("ui.toast.check_failed", "Check fehlgeschlagen"), parts.join(", "));
      }
    } else {
      toastSuccess(t("ui.toast.online_checked", "Online-Quelle verfuegbar"), "");
    }
  } catch (e) {
    toastError(t("ui.toast.check_failed", "Check fehlgeschlagen"), e.message);
  }
}

async function runPcc() {
  try {
    const pd = getPccData();
    if (!pd.rgb_fits) {
      toastError(t("ui.toast.pcc_failed", "PCC fehlgeschlagen"), t("ui.error.no_file", "Bitte RGB FITS-Datei w\u00e4hlen"));
      return;
    }
    toast(t("ui.toast.pcc_running", "PCC l\u00e4uft..."), "", "info");
    const outputRgb = pd.output_rgb || pd.rgb_fits.replace(/\.(fits|fit|fts|fits\.fz|fit\.fz|fts\.fz)$/i, "_pcc.fits");
    const payload = {
      input_rgb: pd.rgb_fits,
      output_rgb: outputRgb,
      wcs_file: pd.wcs_file || "",
      source: pd.catalog_source || "siril",
      catalog_dir: pd.catalog_dir || "",
      mag_limit: pd.mag_limit,
      mag_bright_limit: pd.mag_bright_limit,
      min_stars: pd.min_stars,
      sigma_clip: pd.sigma_clip,
    };
    const startResp = await api.post(API_ENDPOINTS.pcc.run, payload, { timeoutMs: 30000 });
    const jobId = startResp?.job_id;
    if (!jobId) {
      toastError(t("ui.toast.pcc_failed", "PCC fehlgeschlagen"), "No job_id returned");
      return;
    }
    const result = await pollPccJob(jobId);
    const slot = document.getElementById("pcc-results");
    if (slot && result) {
      slot.innerHTML = "";
      const r = result.result || result;
      const fields = [
        [t("ui.label.stars_matched", "Stars matched"), r.stars_matched],
        [t("ui.label.stars_used", "Stars used"), r.stars_used],
        [t("ui.label.residual_rms", "Residual RMS"), r.residual_rms],
      ];
      for (const [label, val] of fields) {
        slot.appendChild(el("div", {}, el("span", { class: "tc-text-muted" }, `${label}: `), el("span", {}, val ?? "\u2014")));
      }
    }
    toastSuccess(t("ui.toast.pcc_done", "PCC abgeschlossen"));
  } catch (e) {
    toastError(t("ui.toast.pcc_failed", "PCC fehlgeschlagen"), e.message);
  }
}

async function pollPccJob(jobId, timeoutMs = 300000) {
  return pollJob(jobId, { timeoutMs, errorLabel: "PCC" });
}

async function saveCorrected() {
  try {
    const pd = getPccData();
    if (!pd.output_rgb) {
      toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), t("ui.error.no_file", "Bitte Output-Pfad w\u00e4hlen"));
      return;
    }
    await api.post(API_ENDPOINTS.pcc.saveCorrected, {
      source_output_rgb: pd.output_rgb,
      output_rgb: pd.output_rgb.replace(/\.(fits|fit|fts|fits\.fz|fit\.fz|fts\.fz)$/i, "_corrected.fits"),
      wcs_file: pd.wcs_file || "",
    });
    toastSuccess(t("ui.toast.saved", "Gespeichert"));
  } catch (e) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
  }
}
