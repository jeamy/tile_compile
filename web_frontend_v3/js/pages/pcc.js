// js/pages/pcc.js – Sub-Tab: PCC (Photometric Color Calibration)

import { el } from "../utils/dom.js";
import { createPathInput } from "../components/path-input.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { t } from "../i18n/i18n.js";

let pccData = { rgb_fits: "", wcs_file: "", catalog_source: "siril", catalog_dir: "/media/data/Astro/siril_catalog", mag_limit: 14.0, mag_bright_limit: 6.0, min_stars: 10, sigma_clip: 2.5 };

export function createPccPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  const inputCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.input", "Input")),
    createPathInput({ label: t("ui.field.rgb_fits", "RGB FITS"), mode: "file", filter: "*.fits;*.fit;*.fts", placeholder: "/data/runs/M31/outputs/stack_M31.fits", onInput: (v) => pccData.rgb_fits = v }),
    el("div", { class: "tc-mt-2" },
      createPathInput({ label: t("ui.field.wcs_file", "WCS Datei"), mode: "file", filter: "*.wcs;*.fits;*.fit", placeholder: "/data/runs/M31/outputs/stack_M31.wcs", onInput: (v) => pccData.wcs_file = v }),
    ),
  );

  const catalogCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.catalog_source", "Catalog Source")),
    el("div", {},
      el("label", { class: "tc-label" }, t("ui.field.source", "Source")),
      el("select", { class: "tc-select", onchange: (e) => pccData.catalog_source = e.target.value },
        el("option", { value: "siril" }, "Siril (lokale Gaia-DR3-XP Chunks)"),
        el("option", { value: "vizier_gaia" }, "Online: VizieR Gaia"),
        el("option", { value: "vizier_apass" }, "Online: VizieR APASS"),
      ),
    ),
    el("div", { class: "tc-mt-2" },
      createPathInput({ label: t("ui.field.catalog_dir", "Catalog Dir"), value: pccData.catalog_dir, onInput: (v) => pccData.catalog_dir = v }),
    ),
    el("div", { class: "tc-flex tc-gap-3 tc-mt-2" },
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => downloadMissing() }, t("ui.button.download_missing", "Download Missing")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => checkOnline() }, t("ui.button.check_online", "Check Online Source")),
    ),
  );

  const paramCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.pcc_params", "PCC Parameters")),
    el("div", { class: "tc-grid-2" },
      el("div", {}, el("label", { class: "tc-label" }, "mag_limit"), el("input", { type: "number", class: "tc-input", value: "14.0", oninput: (e) => pccData.mag_limit = parseFloat(e.target.value) || 14.0 })),
      el("div", {}, el("label", { class: "tc-label" }, "mag_bright_limit"), el("input", { type: "number", class: "tc-input", value: "6.0", oninput: (e) => pccData.mag_bright_limit = parseFloat(e.target.value) || 6.0 })),
      el("div", {}, el("label", { class: "tc-label" }, "min_stars"), el("input", { type: "number", class: "tc-input", value: "10", oninput: (e) => pccData.min_stars = parseInt(e.target.value) || 10 })),
      el("div", {}, el("label", { class: "tc-label" }, "sigma_clip"), el("input", { type: "number", class: "tc-input", value: "2.5", oninput: (e) => pccData.sigma_clip = parseFloat(e.target.value) || 2.5 })),
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
    toast(t("ui.toast.downloading_missing", "Lade fehlende Kataloge..."), "", "info");
    await api.post(API_ENDPOINTS.pcc.downloadMissing, { catalog_dir: pccData.catalog_dir });
    toastSuccess(t("ui.toast.download_complete", "Download abgeschlossen"));
  } catch (e) {
    toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), e.message);
  }
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
    toast(t("ui.toast.pcc_running", "PCC l\u00e4uft..."), "", "info");
    const result = await api.post(API_ENDPOINTS.pcc.run, pccData);
    const slot = document.getElementById("pcc-results");
    if (slot && result) {
      slot.innerHTML = "";
      const fields = [
        [t("ui.label.stars_matched", "Stars matched"), result.stars_matched],
        [t("ui.label.stars_used", "Stars used"), result.stars_used],
        [t("ui.label.residual_rms", "Residual RMS"), result.residual_rms],
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

async function saveCorrected() {
  try {
    await api.post(API_ENDPOINTS.pcc.saveCorrected, pccData);
    toastSuccess(t("ui.toast.saved", "Gespeichert"));
  } catch (e) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
  }
}
