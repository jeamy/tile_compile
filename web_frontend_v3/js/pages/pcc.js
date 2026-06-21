// js/pages/pcc.js – Sub-Tab: PCC (Photometric Color Calibration)

import { el } from "../utils/dom.js";
import { createPathInput } from "../components/path-input.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { t } from "../i18n/i18n.js";
import { getStore } from "../state/store.js";

const store = getStore("pcc", {
  pccData: { rgb_fits: "", wcs_file: "", output_rgb: "", catalog_source: "siril", catalog_dir: "", mag_limit: 14.0, mag_bright_limit: 6.0, min_stars: 10, sigma_clip: 2.5 },
});

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
      createPathInput({ label: t("ui.field.catalog_dir", "Catalog Dir"), value: getPccData().catalog_dir, onInput: (v) => setPccData({ catalog_dir: v }) }),
    ),
    el("div", { class: "tc-flex tc-gap-3 tc-mt-2" },
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => downloadMissing() }, t("ui.button.download_missing", "Download Missing")),
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
  return page;
}

async function downloadMissing() {
  try {
    const pd = getPccData();
    toast(t("ui.toast.downloading_missing", "Lade fehlende Kataloge..."), "", "info");
    const startResp = await api.post(API_ENDPOINTS.pcc.downloadMissing, { catalog_dir: pd.catalog_dir }, { timeoutMs: 30000 });
    const jobId = startResp?.job_id;
    if (!jobId) {
      toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), "No job_id returned");
      return;
    }
    const result = await pollDownloadJob(jobId);
    const missing = result?.missing_after || result?.missing || [];
    if (Array.isArray(missing) && missing.length > 0) {
      toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), `${missing.length} Chunks fehlen noch`);
    } else {
      toastSuccess(t("ui.toast.download_complete", "Download abgeschlossen"));
    }
  } catch (e) {
    toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), e.message);
  }
}

async function pollDownloadJob(jobId, timeoutMs = 600000) {
  const maxAttempts = Math.ceil(timeoutMs / 2000);
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 2000));
    const job = await api.get(API_ENDPOINTS.jobs.byId(jobId));
    const state = job?.state;
    if (state === "ok" || state === "done" || state === "completed") {
      return job?.data || job;
    }
    if (state === "error" || state === "failed") {
      const stderr = job?.data?.stderr || "";
      const detail = stderr || job?.error || "Download failed";
      throw new Error(detail.substring(0, 500));
    }
  }
  throw new Error("Download timeout");
}

async function checkOnline() {
  try {
    const result = await api.get(API_ENDPOINTS.pcc.checkOnline);
    toastSuccess(t("ui.toast.online_checked", "Online-Quelle verfuegbar"), result?.source || "");
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
  const maxAttempts = Math.ceil(timeoutMs / 2000);
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 2000));
    const job = await api.get(API_ENDPOINTS.jobs.byId(jobId));
    const state = job?.state;
    if (state === "ok" || state === "done" || state === "completed") {
      return job?.data || job;
    }
    if (state === "error" || state === "failed") {
      const stderr = job?.data?.stderr || "";
      const stdout = job?.data?.stdout || "";
      const detail = stderr || stdout || job?.error || "PCC failed";
      throw new Error(detail.substring(0, 500));
    }
  }
  throw new Error("PCC timeout");
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
