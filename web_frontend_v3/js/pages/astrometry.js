// js/pages/astrometry.js – Sub-Tab: Astrometry (Plate Solving via ASTAP)

import { el, setBadge } from "../utils/dom.js";
import { createPathInput } from "../components/path-input.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { t } from "../i18n/i18n.js";
import { getStore } from "../state/store.js";
import { pollJob } from "../utils/poll.js";

const store = getStore("astrometry", {
  solveData: { solve_file: "", output_path: "", downsample: "0" },
  astapDataDir: "",
  astapCliPath: "",
});

function getSolveData() { return store.getState().solveData; }
function setSolveData(patch) { store.setState({ solveData: { ...getSolveData(), ...patch } }); }

let _astapCliInput = null;
let _astapDbInput = null;

export function createAstrometryPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  const astapCliGroup = createPathInput({ label: t("ui.field.astap_cli", "ASTAP CLI"), mode: "file", placeholder: "astap_cli", value: store.getState().astapCliPath || "", onInput: (v) => store.setState({ astapCliPath: v }) });
  _astapCliInput = astapCliGroup.querySelector("input[type=text]");

  const astapDbGroup = createPathInput({ label: t("ui.field.star_db_dir", "Star Database Dir"), mode: "dir", placeholder: "", value: store.getState().astapDataDir || "", onInput: (v) => store.setState({ astapDataDir: v }) });
  _astapDbInput = astapDbGroup.querySelector("input[type=text]");

  const setupCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.astap_setup", "ASTAP Setup")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        astapCliGroup,
        el("span", { class: "tc-badge", id: "astap-cli-badge", style: { whiteSpace: "nowrap" } }, "…"),
      ),
      el("div", {},
        astapDbGroup,
        el("span", { class: "tc-badge", id: "astap-db-badge", style: { whiteSpace: "nowrap" } }, "…"),
      ),
    ),
    el("div", { class: "tc-flex tc-gap-3 tc-mt-2" },
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => detectAstap(_astapCliInput, _astapDbInput) }, t("ui.button.detect_astap", "Detect ASTAP")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => installCli(_astapDbInput?.value || "") }, t("ui.button.install_cli", "Install CLI")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => downloadCatalog(_astapDbInput?.value || "") }, t("ui.button.download_catalog", "Download Catalog")),
    ),
  );

  const solveCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.plate_solve", "Plate Solve")),
    createPathInput({ label: t("ui.field.fits_file", "FITS Datei"), mode: "file", filter: "*.fits;*.fit;*.fts;*.fits.fz", placeholder: "/data/runs/M31/outputs/stack_M31.fits", value: getSolveData().solve_file, onInput: (v) => {
      const defaultOut = v.replace(/\.(fits|fit|fts|fits\.fz|fit\.fz|fts\.fz)$/i, "_solved.fits");
      setSolveData({ solve_file: v, output_path: defaultOut });
    } }),
    createPathInput({ label: t("ui.field.output_path", "Output Pfad"), mode: "file", filter: "*.fits;*.fit;*.fts", placeholder: "auto: *_solved.fits", value: getSolveData().output_path, onInput: (v) => setSolveData({ output_path: v }) }),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.field.downsample", "Downsample")),
      el("select", { class: "tc-select", onchange: (e) => setSolveData({ downsample: e.target.value }) },
        el("option", { value: "0", ...(getSolveData().downsample === "0" ? { selected: true } : {}) }, "Auto"),
        el("option", { value: "1", ...(getSolveData().downsample === "1" ? { selected: true } : {}) }, "1x"),
        el("option", { value: "2", ...(getSolveData().downsample === "2" ? { selected: true } : {}) }, "2x"),
        el("option", { value: "4", ...(getSolveData().downsample === "4" ? { selected: true } : {}) }, "4x"),
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

  detectAstap(_astapCliInput, _astapDbInput);

  return page;
}

async function detectAstap(cliInput, dbInput) {
  const cliBadge = document.getElementById("astap-cli-badge");
  const dbBadge = document.getElementById("astap-db-badge");
  try {
    const detectPayload = {};
    if (cliInput?.value) detectPayload.astap_cli = cliInput.value;
    if (dbInput?.value) detectPayload.astap_data_dir = dbInput.value;
    const result = await api.post(API_ENDPOINTS.astrometry.detect, detectPayload);
    if (result?.binary && cliInput) { cliInput.value = result.binary; store.setState({ astapCliPath: result.binary }); }
    if (result?.data_dir && dbInput) { dbInput.value = result.data_dir; store.setState({ astapDataDir: result.data_dir }); }

    const cliOk = !!result?.installed;
    setBadge(cliBadge, cliOk, cliOk ? "OK" : "✗");

    const catalogs = result?.catalogs || {};
    const catalogCount = Object.values(catalogs).filter(Boolean).length;
    const dbOk = catalogCount > 0;
    setBadge(dbBadge, dbOk, dbOk ? `${catalogCount} Kataloge` : "✗");

    if (cliOk) {
      toastSuccess(t("ui.toast.astap_detected", "ASTAP erkannt"), result.binary || "");
    } else {
      toast(t("ui.toast.astap_not_found", "ASTAP nicht gefunden"), t("ui.state.install_hint", "Bitte ASTAP installieren"), "info");
    }
  } catch (e) {
    setBadge(cliBadge, false, "✗");
    setBadge(dbBadge, false, "✗");
    toastError(t("ui.toast.detect_failed", "Detect fehlgeschlagen"), e.message);
  }
}

async function installCli(dataDir = "") {
  try {
    toast(t("ui.toast.installing_cli", "Installiere CLI..."), "", "info");
    await api.post(API_ENDPOINTS.astrometry.installCli, dataDir ? { astap_data_dir: dataDir } : {});
    toastSuccess(t("ui.toast.cli_installed", "CLI installiert"));
  } catch (e) {
    toastError(t("ui.toast.install_failed", "Installation fehlgeschlagen"), e.message);
  }
}

async function downloadCatalog(dataDir = "") {
  try {
    toast(t("ui.toast.downloading_catalog", "Katalog wird heruntergeladen..."), "", "info");
    await api.post(API_ENDPOINTS.astrometry.downloadCatalog, dataDir ? { astap_data_dir: dataDir } : {});
    toastSuccess(t("ui.toast.catalog_downloaded", "Katalog heruntergeladen"));
  } catch (e) {
    toastError(t("ui.toast.download_failed", "Download fehlgeschlagen"), e.message);
  }
}

async function solve() {
  try {
    const sd = getSolveData();
    if (!sd.solve_file) {
      toastError(t("ui.toast.solve_failed", "Solve fehlgeschlagen"), t("ui.error.no_file", "Bitte FITS-Datei w\u00e4hlen"));
      return;
    }
    toast(t("ui.toast.solving", "Plate Solve l\u00e4uft..."), "", "info");
    const astapCli = _astapCliInput?.value || "";
    const astapDb = _astapDbInput?.value || "";
    const payload = {
      solve_file: sd.solve_file,
      astap_cli: astapCli,
      astap_data_dir: astapDb,
    };
    const startResp = await api.post(API_ENDPOINTS.astrometry.solve, payload, { timeoutMs: 30000 });
    const jobId = startResp?.job_id;
    if (!jobId) {
      toastError(t("ui.toast.solve_failed", "Solve fehlgeschlagen"), "No job_id returned");
      return;
    }
    const result = await pollSolveJob(jobId);
    const slot = document.getElementById("wcs-results");
    if (slot && result) {
      slot.innerHTML = "";
      const r = result.result || result;
      const fields = [["RA", r.ra], ["DEC", r.dec], ["Scale", r.scale], ["Rotation", r.rotation]];
      for (const [label, val] of fields) {
        slot.appendChild(el("div", {}, el("span", { class: "tc-text-muted" }, `${label}: `), el("span", {}, val ?? "\u2014")));
      }
    }
    toastSuccess(t("ui.toast.solved", "Plate Solve erfolgreich"));
  } catch (e) {
    toastError(t("ui.toast.solve_failed", "Solve fehlgeschlagen"), e.message);
  }
}

async function pollSolveJob(jobId, timeoutMs = 300000) {
  return pollJob(jobId, { timeoutMs, errorLabel: "Solve" });
}

async function saveSolved() {
  try {
    const sd = getSolveData();
    if (!sd.solve_file) {
      toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), t("ui.error.no_file", "Bitte FITS-Datei w\u00e4hlen"));
      return;
    }
    const wcsPath = sd.solve_file.replace(/\.(fits|fit|fts|fits\.fz|fit\.fz|fts\.fz)$/i, ".wcs");
    const outputPath = sd.output_path || sd.solve_file.replace(/\.(fits|fit|fts|fits\.fz|fit\.fz|fts\.fz)$/i, "_solved.fits");
    await api.post(API_ENDPOINTS.astrometry.saveSolved, {
      input_path: sd.solve_file,
      output_path: outputPath,
      wcs_path: wcsPath,
    });
    toastSuccess(t("ui.toast.saved", "Gespeichert"));
  } catch (e) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
  }
}
