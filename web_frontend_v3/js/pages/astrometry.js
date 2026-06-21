// js/pages/astrometry.js – Sub-Tab: Astrometry (Plate Solving via ASTAP)

import { el } from "../utils/dom.js";
import { createPathInput } from "../components/path-input.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { t } from "../i18n/i18n.js";

let solveData = { fits_file: "", downsample: "0" };

export function createAstrometryPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  const setupCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.astap_setup", "ASTAP Setup")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.astap_cli", "ASTAP CLI")),
        el("input", { type: "text", class: "tc-input", value: "/usr/local/bin/astap", id: "astap-cli" }),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.star_db_dir", "Star Database Dir")),
        el("input", { type: "text", class: "tc-input", value: "/usr/local/share/astap", id: "astap-db" }),
      ),
    ),
    el("div", { class: "tc-flex tc-gap-3 tc-mt-2" },
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => detectAstap() }, t("ui.button.detect_astap", "Detect ASTAP")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => installCli() }, t("ui.button.install_cli", "Install CLI")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => downloadCatalog() }, t("ui.button.download_catalog", "Download Catalog")),
    ),
  );

  const solveCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.plate_solve", "Plate Solve")),
    createPathInput({ label: t("ui.field.fits_file", "FITS Datei"), mode: "file", filter: "*.fits;*.fit;*.fts;*.fits.fz", placeholder: "/data/runs/M31/outputs/stack_M31.fits", onInput: (v) => solveData.fits_file = v }),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.field.downsample", "Downsample")),
      el("select", { class: "tc-select", onchange: (e) => solveData.downsample = e.target.value },
        el("option", { value: "0" }, "Auto"),
        el("option", { value: "1" }, "1x"),
        el("option", { value: "2" }, "2x"),
        el("option", { value: "4" }, "4x"),
      ),
    ),
    el("div", { class: "tc-flex tc-gap-3 tc-mt-2" },
      el("button", { class: "tc-btn tc-btn-primary", onclick: () => solve() }, t("ui.button.solve", "Solve")),
      el("button", { class: "tc-btn", onclick: () => saveSolved() }, t("ui.button.save_solved", "Save Solved")),
    ),
  );

  const resultsCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.wcs_results", "WCS Results")),
    el("div", { class: "tc-wcs-results", id: "wcs-results" },
      el("div", {}, el("span", { class: "tc-text-muted" }, "RA: "), el("span", {}, "\u2014")),
      el("div", {}, el("span", { class: "tc-text-muted" }, "DEC: "), el("span", {}, "\u2014")),
      el("div", {}, el("span", { class: "tc-text-muted" }, "Scale: "), el("span", {}, "\u2014")),
      el("div", {}, el("span", { class: "tc-text-muted" }, "Rotation: "), el("span", {}, "\u2014")),
    ),
  );

  page.append(setupCard, solveCard, resultsCard);
  return page;
}

async function detectAstap() {
  try {
    const result = await api.get(API_ENDPOINTS.astrometry.detect);
    if (result?.cli_path) document.getElementById("astap-cli").value = result.cli_path;
    if (result?.db_dir) document.getElementById("astap-db").value = result.db_dir;
    toastSuccess(t("ui.toast.astap_detected", "ASTAP erkannt"), result?.version || "");
  } catch (e) {
    toastError(t("ui.toast.detect_failed", "Detect fehlgeschlagen"), e.message);
  }
}

async function installCli() {
  try {
    toast(t("ui.toast.installing_cli", "Installiere CLI..."), "", "info");
    await api.post(API_ENDPOINTS.astrometry.installCli, {});
    toastSuccess(t("ui.toast.cli_installed", "CLI installiert"));
  } catch (e) {
    toastError(t("ui.toast.install_failed", "Installation fehlgeschlagen"), e.message);
  }
}

async function downloadCatalog() {
  try {
    toast(t("ui.toast.downloading_catalog", "Katalog wird heruntergeladen..."), "", "info");
    await api.post(API_ENDPOINTS.astrometry.downloadCatalog, {});
    toastSuccess(t("ui.toast.catalog_downloaded", "Katalog heruntergeladen"));
  } catch (e) {
    toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), e.message);
  }
}

async function solve() {
  try {
    toast(t("ui.toast.solving", "Plate Solve l\u00e4uft..."), "", "info");
    const result = await api.post(API_ENDPOINTS.astrometry.solve, solveData);
    const slot = document.getElementById("wcs-results");
    if (slot && result) {
      slot.innerHTML = "";
      const fields = [["RA", result.ra], ["DEC", result.dec], ["Scale", result.scale], ["Rotation", result.rotation]];
      for (const [label, val] of fields) {
        slot.appendChild(el("div", {}, el("span", { class: "tc-text-muted" }, `${label}: `), el("span", {}, val ?? "\u2014")));
      }
    }
    toastSuccess(t("ui.toast.solved", "Plate Solve erfolgreich"));
  } catch (e) {
    toastError(t("ui.toast.solve_failed", "Solve fehlgeschlagen"), e.message);
  }
}

async function saveSolved() {
  try {
    await api.post(API_ENDPOINTS.astrometry.saveSolved, solveData);
    toastSuccess(t("ui.toast.saved", "Gespeichert"));
  } catch (e) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
  }
}
