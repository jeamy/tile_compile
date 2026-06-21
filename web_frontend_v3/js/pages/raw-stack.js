// js/pages/raw-stack.js – Sub-Tab: Raw Stack (Preprocessing-Pipeline)

import { el } from "../utils/dom.js";
import { createPathInput } from "../components/path-input.js";
import { createCalibrationPanel } from "../components/calibration-panel.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { t } from "../i18n/i18n.js";
import { getStore } from "../state/store.js";

const store = getStore("raw-stack", {
  stackData: { input_dir: "", pattern: "*.fits", output_dir: "", method: "sigma_clip", sigma_low: 3.0, sigma_high: 3.0 },
  calValues: {},
  currentJobId: null,
});

function getStackData() { return store.getState().stackData; }
function setStackData(patch) { store.setState({ stackData: { ...getStackData(), ...patch } }); }
function getCalValues() { return store.getState().calValues; }
function setCalValues(v) { store.setState({ calValues: v }); }
function getJobId() { return store.getState().currentJobId; }
function setJobId(id) { store.setState({ currentJobId: id }); }

export function createRawStackPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  const inputCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.input", "Input")),
    createPathInput({ label: t("ui.field.input_dirs", "Eingabeordner"), placeholder: "/data/M31/lights", value: getStackData().input_dir, onInput: (v) => setStackData({ input_dir: v }) }),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.field.pattern", "Dateimuster")),
      el("input", { type: "text", class: "tc-input", value: getStackData().pattern, oninput: (e) => setStackData({ pattern: e.target.value }) }),
    ),
    el("div", { class: "tc-mt-2" },
      createPathInput({ label: t("ui.field.output_dir", "Ausgabeordner"), placeholder: "/data/runs", value: getStackData().output_dir, onInput: (v) => setStackData({ output_dir: v }) }),
    ),
  );

  const calPanel = createCalibrationPanel({ values: getCalValues(), onChange: (v) => { setCalValues(v); } });

  const stackCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.stack_params", "Stack Parameter")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.method", "Method")),
        el("select", { class: "tc-select", onchange: (e) => setStackData({ method: e.target.value }) },
          el("option", { value: "sigma_clip", ...(getStackData().method === "sigma_clip" ? { selected: true } : {}) }, "sigma_clip"),
          el("option", { value: "median", ...(getStackData().method === "median" ? { selected: true } : {}) }, "median"),
          el("option", { value: "average", ...(getStackData().method === "average" ? { selected: true } : {}) }, "average"),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.sigma_low", "Sigma Low")),
        el("input", { type: "number", class: "tc-input", value: String(getStackData().sigma_low), oninput: (e) => setStackData({ sigma_low: parseFloat(e.target.value) || 3.0 }) }),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.sigma_high", "Sigma High")),
        el("input", { type: "number", class: "tc-input", value: String(getStackData().sigma_high), oninput: (e) => setStackData({ sigma_high: parseFloat(e.target.value) || 3.0 }) }),
      ),
    ),
  );

  const actions = el("div", { class: "tc-flex tc-gap-3" },
    el("button", { class: "tc-btn tc-btn-primary", onclick: () => runStack() }, t("ui.button.run_raw_stack", "Run Raw Stack")),
    el("button", { class: "tc-btn", onclick: () => cancelStack() }, t("ui.button.cancel", "Cancel")),
  );

  const statusSlot = el("div", { id: "raw-stack-status" });

  page.append(inputCard, calPanel, stackCard, actions, statusSlot);

  restoreRunningJob();
  return page;
}

async function restoreRunningJob() {
  try {
    const status = await api.get(API_ENDPOINTS.preprocessing.status(""));
    if (status?.status === "running" || status?.status === "pending") {
      const jobId = status?.job_id || status?.job?.job_id || status?.id;
      if (jobId) {
        setJobId(jobId);
        pollStatus();
      }
    }
  } catch {}
}

async function runStack() {
  try {
    const sd = getStackData();
    toast(t("ui.toast.stack_starting", "Raw Stack wird gestartet..."), "", "info");
    const result = await api.post(API_ENDPOINTS.preprocessing.run, {
      lights_dir: sd.input_dir,
      pattern: sd.pattern,
      output_dir: sd.output_dir,
      method: sd.method,
      sigma_low: sd.sigma_low,
      sigma_high: sd.sigma_high,
      calibration: getCalValues(),
    });
    setJobId(result?.job_id || result?.id);
    toastSuccess(t("ui.toast.stack_started", "Raw Stack gestartet"), getJobId() || "");
    pollStatus();
  } catch (e) {
    toastError(t("ui.toast.stack_failed", "Start fehlgeschlagen"), e.message);
  }
}

async function cancelStack() {
  const jobId = getJobId();
  if (!jobId) return;
  try {
    await api.post(API_ENDPOINTS.preprocessing.cancel, { job_id: jobId });
    toastSuccess(t("ui.toast.stack_cancelled", "Abgebrochen"));
    setJobId(null);
  } catch (e) {
    toastError(t("ui.toast.cancel_failed", "Abbrechen fehlgeschlagen"), e.message);
  }
}

async function pollStatus() {
  const jobId = getJobId();
  if (!jobId) return;
  try {
    const status = await api.get(API_ENDPOINTS.preprocessing.status(jobId));
    const slot = document.getElementById("raw-stack-status");
    if (slot) {
      slot.innerHTML = "";
      slot.appendChild(el("div", { class: "tc-card" },
        el("div", { class: "tc-card-title" }, t("ui.title.status", "Status")),
        el("div", { class: "tc-text-sm" }, `${status?.status || "running"} - ${(Number(status?.progress) * 100 || 0).toFixed(2)}%`),
      ));
    }
    if (status?.status === "running" || status?.status === "pending") {
      setTimeout(pollStatus, 2000);
    } else {
      setJobId(null);
      if (status?.status === "done") toastSuccess(t("ui.toast.stack_done", "Raw Stack abgeschlossen"));
    }
  } catch {
    setTimeout(pollStatus, 5000);
  }
}
