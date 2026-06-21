// js/pages/input-scan.js – Sub-Tab: Input & Scan

import { el, clear } from "../utils/dom.js";
import { createPathInput } from "../components/path-input.js";
import { createQueueEditor } from "../components/queue-editor.js";
import { createCalibrationPanel } from "../components/calibration-panel.js";
import { createScanResultCard } from "../components/scan-result-card.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastError, toastSuccess } from "../components/toast.js";
import { getUiState, setUiState } from "../state/ui-state.js";
import { getScanState, setScanState } from "../state/scan-state.js";
import { getStore } from "../state/store.js";
import { setRunState } from "../state/run-state.js";
import { refreshGuardrails } from "../services/guardrail-service.js";
import { t } from "../i18n/i18n.js";

const inputStore = getStore("input-scan", {
  scanData: { input_dir: "", pattern: "*.fits", runs_dir: "/media/data/programming/tile_compile/runs", run_name: "", color_mode: "OSC", frame_min: 30, max_frames: 0, sort: "numeric", with_checksums: false },
  queueItems: [],
  calValues: {},
});

function getScanData() { return inputStore.getState().scanData; }
function setScanData(patch) { inputStore.setState({ scanData: { ...getScanData(), ...patch } }); }
function getQueueItems() { return inputStore.getState().queueItems; }
function setQueueItems(items) { inputStore.setState({ queueItems: items }); }
function getCalValues() { return inputStore.getState().calValues; }
function setCalValues(v) { inputStore.setState({ calValues: v }); }

export function createInputScanPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  // Input section
  const inputCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.input", "Input")),
    createPathInput({
      label: t("ui.field.input_dirs", "Eingabeordner"),
      placeholder: "/data/M31/lights",
      value: getScanData().input_dir,
      onInput: (v) => setScanData({ input_dir: v }),
    }),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.field.pattern", "Dateimuster")),
      el("input", { type: "text", class: "tc-input", value: getScanData().pattern, placeholder: "*.fits",
        oninput: (e) => setScanData({ pattern: e.target.value }) }),
    ),
    el("div", { class: "tc-mt-2" },
      createPathInput({
        label: t("ui.field.runs_dir", "Ausgabeordner"),
        placeholder: "/data/runs",
        value: getScanData().runs_dir,
        onInput: (v) => setScanData({ runs_dir: v }),
      }),
    ),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.field.run_name", "Run Name")),
      el("input", { type: "text", class: "tc-input", value: getScanData().run_name, placeholder: "M31_altaz_test",
        oninput: (e) => setScanData({ run_name: e.target.value }) }),
    ),
  );

  // Parameters
  const paramCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.parameters", "Parameter")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.frame_min", "Frames Minimum")),
        el("input", { type: "number", class: "tc-input", value: String(getScanData().frame_min),
          oninput: (e) => setScanData({ frame_min: parseInt(e.target.value) || 0 }) }),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.max_frames", "Max. Frames (0=unbegrenzt)")),
        el("input", { type: "number", class: "tc-input", value: String(getScanData().max_frames),
          oninput: (e) => setScanData({ max_frames: parseInt(e.target.value) || 0 }) }),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.sort", "Sortierung")),
        el("select", { class: "tc-select", onchange: (e) => setScanData({ sort: e.target.value }) },
          el("option", { value: "numeric", ...(getScanData().sort === "numeric" ? { selected: true } : {}) }, "numeric"),
          el("option", { value: "alphabetic", ...(getScanData().sort === "alphabetic" ? { selected: true } : {}) }, "alphabetic"),
          el("option", { value: "timestamp", ...(getScanData().sort === "timestamp" ? { selected: true } : {}) }, "timestamp"),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.color_mode", "Farbmodus")),
        el("select", { class: "tc-select", onchange: (e) => setScanData({ color_mode: e.target.value }) },
          el("option", { value: "MONO", ...(getScanData().color_mode === "MONO" ? { selected: true } : {}) }, "MONO"),
          el("option", { value: "OSC", ...(getScanData().color_mode === "OSC" ? { selected: true } : {}) }, "OSC"),
        ),
      ),
    ),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-checkbox" },
        el("input", { type: "checkbox", checked: getScanData().with_checksums, onchange: (e) => setScanData({ with_checksums: e.target.checked }) }),
        t("ui.field.with_checksums", "Checksummen berechnen"),
      ),
    ),
  );

  // Queue editor (for MONO)
  const queueEditor = createQueueEditor({
    items: getQueueItems(),
    onChange: (items) => { setQueueItems(items); },
    currentInputDir: () => getScanData().input_dir,
  });

  // Calibration panel
  const calPanel = createCalibrationPanel({
    values: getCalValues(),
    onChange: (v) => { setCalValues(v); },
  });

  // Scan result card
  const resultCard = el("div", { id: "scan-result-slot" });
  if (getScanState().lastScan) {
    resultCard.appendChild(createScanResultCard(getScanState().lastScan));
  } else {
    resultCard.appendChild(createScanResultCard(null));
  }

  // Actions
  const actions = el("div", { class: "tc-flex tc-gap-3 tc-mt-2" },
    el("button", {
      class: "tc-btn tc-btn-primary",
      onclick: () => doScan(),
    }, t("ui.button.scan", "Scan starten")),
    el("button", {
      class: "tc-btn",
      onclick: () => goToSubTab("parameter"),
    }, "\u25b6 " + t("ui.button.next", "Next")),
  );

  page.append(inputCard, paramCard, queueEditor, calPanel, resultCard, actions);
  return page;
}

async function doScan() {
  try {
    const sd = getScanData();
    setRunState({ currentRunId: null, currentRunDir: null, status: null, phases: [], resumeActive: false, resumePending: false });
    toast(t("ui.toast.scan_starting", "Scan wird gestartet..."), "", "info");
    const payload = {
      input_dir: sd.input_dir,
      pattern: sd.pattern,
      runs_dir: sd.runs_dir,
      run_name: sd.run_name,
      color_mode: sd.color_mode,
      frame_min: sd.frame_min,
      max_frames: sd.max_frames,
      sort: sd.sort,
      with_checksums: sd.with_checksums,
      queue: getQueueItems(),
      calibration: getCalValues(),
    };
    const jobResult = await api.post(API_ENDPOINTS.scan.root, payload);
    const jobId = jobResult?.job_id;
    if (!jobId) {
      throw new Error("No job_id returned");
    }

    const slot = document.getElementById("scan-result-slot");
    if (slot) {
      clear(slot);
      slot.appendChild(el("div", { class: "tc-card" },
        el("div", { class: "tc-card-title" }, t("ui.title.scan_results", "Scan-Ergebnis")),
        el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.loading", "Lädt...")),
      ));
    }

    const finalResult = await pollScanJob(jobId);
    setScanState({ lastScan: finalResult });
    const frameCount = finalResult?.frames_detected || finalResult?.frame_count || 0;
    toastSuccess(t("ui.toast.scan_completed", "Scan abgeschlossen"), `${frameCount} ${t("ui.label.frames_detected", "Frames erkannt")}`);

    if (slot) {
      clear(slot);
      slot.appendChild(createScanResultCard(finalResult));
    }
    refreshGuardrails();
  } catch (e) {
    toastError(t("ui.toast.scan_failed", "Scan starten fehlgeschlagen"), e.message);
  }
}

async function pollScanJob(jobId) {
  const maxAttempts = 120;
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 1000));
    const job = await api.get(API_ENDPOINTS.scan.jobStatus(jobId));
    const state = job?.state;
    if (state === "done" || state === "completed" || state === "ok") {
      const latest = await api.get(API_ENDPOINTS.scan.latest);
      return latest;
    }
    if (state === "error" || state === "failed") {
      throw new Error(job?.error || "Scan failed");
    }
  }
  throw new Error("Scan timeout");
}

function goToSubTab(subId) {
  const ui = getUiState();
  setUiState({ activeSubTab: { ...ui.activeSubTab, processing: subId } });
  window.location.hash = "#processing";
  window.dispatchEvent(new Event("tc-subtab-change"));
}
