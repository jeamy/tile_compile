import { IPC, EVENTS } from "./constants.js";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

window.addEventListener("error", (e) => {
  try {
    appendLog(`[error] ${String(e?.message || e)}`);
  } catch {
    // ignore
  }
});

window.addEventListener("unhandledrejection", (e) => {
  try {
    appendLog(`[error] unhandledrejection: ${String(e?.reason || e)}`);
  } catch {
    // ignore
  }
});

function appendLog(text) {
  const el = document.querySelector("#log");
  el.textContent += text + "\n";
  el.scrollTop = el.scrollHeight;
}

function statusClass(status) {
  const s = String(status || "").toUpperCase();
  if (s === "SUCCESS") return "success";
  if (s === "FAILED") return "failed";
  if (s === "ABORTED") return "aborted";
  if (s === "ABORTING") return "aborting";
  if (s === "RUNNING") return "running";
  return "pending";
}

function setCurrentRun(runId, runDir) {
  currentRunId = runId || null;
  currentRunDir = runDir || null;
  document.querySelector("#current-run-id").textContent = currentRunId ?? "-";
  document.querySelector("#current-run-dir").textContent = currentRunDir ?? "-";
}

function renderLogEvents(containerId, events, filterText) {
  const el = document.querySelector(containerId);
  el.innerHTML = "";
  const ft = (filterText || "").toLowerCase().trim();

  for (const ev of events || []) {
    const t = String(ev.type ?? "");
    const ts = String(ev.ts ?? "");
    const phase = ev.phase_name ? ` ${String(ev.phase_name)}` : "";
    const msg = t === "phase_end" ? ` status=${String(ev.status ?? "")}` : "";
    const lineText = `[${ts}] ${t}${phase}${msg}`;

    const hay = (lineText + " " + JSON.stringify(ev)).toLowerCase();
    if (ft && !hay.includes(ft)) continue;

    const div = document.createElement("div");
    div.className = "log-line";
    div.textContent = lineText;
    el.appendChild(div);
  }
}

function setCurrentRunStatus(text) {
  document.querySelector("#current-run-status").textContent = text;
}

function setRunsStatus(text) {
  document.querySelector("#runs-status").textContent = text;
}

function setStatus(text) {
  document.querySelector("#run-status").textContent = text;
}

function setScanStatus(text) {
  document.querySelector("#scan-status").textContent = text;
}

function setScanMessage(kind, text) {
  const el = document.querySelector("#scan-message");
  if (!text) {
    el.hidden = true;
    el.textContent = "";
    el.classList.remove("error");
    el.classList.remove("warn");
    return;
  }
  el.hidden = false;
  el.textContent = text;
  el.classList.remove("error");
  el.classList.remove("warn");
  if (kind === "error") el.classList.add("error");
  if (kind === "warn") el.classList.add("warn");
}

function setConfigStatus(text) {
  document.querySelector("#config-status").textContent = text;
}

function setButtons(running) {
  document.querySelector("#start-run").disabled = running;
  document.querySelector("#stop-run").disabled = !running;
}

function setConfigEditable(running) {
  const disabled = Boolean(running);
  document.querySelector("#config-path-editor").disabled = disabled;
  document.querySelector("#config-yaml").disabled = disabled;
  document.querySelector("#config-load").disabled = disabled;
  document.querySelector("#config-save").disabled = disabled;
  document.querySelector("#config-validate").disabled = disabled;
}

let lastScanResult = null;
let confirmedColorMode = null;
let configValidatedOk = false;
let currentRunId = null;
let currentRunDir = null;

function setStartAllowed(running) {
  const startBtn = document.querySelector("#start-run");
  if (running) {
    startBtn.disabled = true;
    startBtn.dataset.blockedReason = "";
    return;
  }

  // Keep the button clickable and explain missing prerequisites on click.
  startBtn.disabled = false;

  let reason = "";
  if (!lastScanResult) {
    reason = "please run Scan first";
  } else if (!lastScanResult.ok) {
    reason = "scan has errors";
  } else if (lastScanResult.requires_user_confirmation && !confirmedColorMode) {
    reason = "please confirm color mode";
  } else if (!configValidatedOk) {
    reason = "please validate config";
  }
  startBtn.dataset.blockedReason = reason;
}

function setConfigValidateStatus(ok, text) {
  const el = document.querySelector("#config-validate-status");
  el.textContent = text;
  el.classList.remove("badge-ok");
  el.classList.remove("badge-bad");
  if (ok === true) el.classList.add("badge-ok");
  if (ok === false) el.classList.add("badge-bad");
}

function renderScanResult(res) {
  document.querySelector("#scan-results").hidden = false;
  document.querySelector("#scan-frames-detected").textContent = String(res.frames_detected ?? "-");
  document.querySelector("#scan-image-size").textContent = `${res.image_width ?? "-"} x ${res.image_height ?? "-"}`;
  document.querySelector("#scan-manifest-id").textContent = String(res.frames_manifest_id ?? "-");
  document.querySelector("#scan-color-mode").textContent = String(res.color_mode ?? "-");

  const confirmBox = document.querySelector("#color-confirm");
  const select = document.querySelector("#color-mode-select");
  const hint = document.querySelector("#color-confirm-hint");

  if (res.requires_user_confirmation) {
    confirmBox.hidden = false;
    select.innerHTML = "";
    const candidates = Array.isArray(res.color_mode_candidates) ? res.color_mode_candidates : [];
    for (const c of candidates) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      select.appendChild(opt);
    }
    hint.textContent = "Scan could not determine color mode from FITS headers (missing BAYERPAT).";
  } else {
    confirmBox.hidden = true;
    hint.textContent = "";
  }
}

function getFormValues() {
  return {
    workingDir: document.querySelector("#working-dir").value,
    configPath: document.querySelector("#config-path").value,
    inputDir: document.querySelector("#input-dir").value,
    runsDir: document.querySelector("#runs-dir").value,
    pattern: document.querySelector("#pattern").value,
    dryRun: document.querySelector("#dry-run").checked,
  };
}

async function startRun() {
  const args = getFormValues();
  const blockedReason = document.querySelector("#start-run")?.dataset?.blockedReason || "";
  if (blockedReason) {
    setStatus(`blocked: ${blockedReason}`);
    appendLog(`[ui] ${blockedReason}`);
    return;
  }

  args.colorModeConfirmed = confirmedColorMode;
  setStatus("starting...");
  appendLog("[ui] start_run");
  try {
    const res = await invoke(IPC.START_RUN, args);
    setButtons(true);
    setStatus(`running (pid=${res.pid})`);
  } catch (e) {
    setButtons(false);
    setStatus("error");
    appendLog(`[error] ${String(e)}`);
  }
}

async function stopRun() {
  setStatus("aborting...");
  appendLog("[ui] abort_run");
  try {
    const res = await invoke(IPC.ABORT_RUN);
    setButtons(false);
    setStatus(res.requested ? (res.forced ? "aborted (forced)" : "aborting") : "idle");
  } catch (e) {
    setStatus("error");
    appendLog(`[error] ${String(e)}`);
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  document.querySelector("#working-dir").value ||= "";
  document.querySelector("#config-path").value ||= "tile_compile.yaml";
  document.querySelector("#config-path-editor").value ||= "tile_compile.yaml";
  document.querySelector("#runs-dir").value ||= "";
  document.querySelector("#pattern").value ||= "*.fit*";

  document.querySelector("#scan-input-dir").value ||= "";
  document.querySelector("#scan-frames-min").value ||= "1";

  try {
    const defaults = await invoke(IPC.GET_DEFAULT_PATHS);
    const pr = String(defaults.projectRoot || "");
    const rd = String(defaults.runsDir || "runs");
    document.querySelector("#working-dir").value ||= pr;
    document.querySelector("#runs-dir").value ||= rd;
  } catch (e) {
    // Fallback: old behavior
    document.querySelector("#working-dir").value ||= "..";
    document.querySelector("#runs-dir").value ||= "runs";
    appendLog(`[warning] get_default_paths failed: ${String(e)}`);
  }

  const persistGuiState = async () => {
    const workingDir = (document.querySelector("#working-dir")?.value || "").trim();
    if (!workingDir) return;
    const lastInputDir =
      (document.querySelector("#input-dir")?.value || "").trim() ||
      (document.querySelector("#scan-input-dir")?.value || "").trim();
    if (!lastInputDir) return;
    try {
      await invoke(IPC.SAVE_GUI_STATE, { workingDir, lastInputDir });
    } catch (e) {
      appendLog(`[warning] save_gui_state failed: ${String(e)}`);
    }
  };

  const loadGuiState = async () => {
    const workingDir = (document.querySelector("#working-dir")?.value || "").trim();
    if (!workingDir) return;
    try {
      const st = await invoke(IPC.LOAD_GUI_STATE, { workingDir });
      const last = st && typeof st.lastInputDir === "string" ? st.lastInputDir : "";
      if (last && last.trim()) {
        document.querySelector("#input-dir").value ||= last;
        document.querySelector("#scan-input-dir").value ||= last;
      }
    } catch (e) {
      appendLog(`[warning] load_gui_state failed: ${String(e)}`);
    }
  };

  await loadGuiState();

  const pickDir = async (defaultPath) => {
    const openDialog = window.__TAURI__?.dialog?.open;
    if (typeof openDialog !== "function") {
      appendLog("[error] dialog.open not available (missing tauri-plugin-dialog permissions or frontend plugin)");
      return null;
    }
    try {
      const wd = (document.querySelector("#working-dir")?.value || "").trim();
      const dp = String(defaultPath || "").trim() || wd || ".";
      const res = await openDialog({ directory: true, multiple: false, defaultPath: dp });
      if (typeof res === "string" && res.trim()) return res;
      return null;
    } catch (e) {
      appendLog(`[error] dialog.open failed: ${String(e)}`);
      return null;
    }
  };

  document.querySelector("#browse-scan-input-dir").addEventListener("click", async () => {
    const current = document.querySelector("#scan-input-dir").value;
    const dir = await pickDir(current);
    if (!dir) return;
    document.querySelector("#scan-input-dir").value = dir;
    await persistGuiState();
  });

  document.querySelector("#browse-input-dir").addEventListener("click", async () => {
    const current = document.querySelector("#input-dir").value;
    const dir = await pickDir(current);
    if (!dir) return;
    document.querySelector("#input-dir").value = dir;
    document.querySelector("#scan-input-dir").value ||= dir;
    await persistGuiState();
  });

  document.querySelector("#browse-working-dir").addEventListener("click", async () => {
    const current = document.querySelector("#working-dir").value;
    const dir = await pickDir(current);
    if (!dir) return;
    document.querySelector("#working-dir").value = dir;
    await loadGuiState();
  });

  document.querySelector("#browse-runs-dir").addEventListener("click", async () => {
    const current = document.querySelector("#runs-dir").value;
    const dir = await pickDir(current);
    if (!dir) return;
    document.querySelector("#runs-dir").value = dir;
  });

  document.querySelector("#input-dir").addEventListener("change", persistGuiState);
  document.querySelector("#scan-input-dir").addEventListener("change", persistGuiState);

  setButtons(false);
  setConfigEditable(false);
  setStatus("idle");
  setScanStatus("idle");
  setConfigStatus("idle");
  setConfigValidateStatus(null, "not validated");
  setStartAllowed(false);

  document.querySelector("#start-run").addEventListener("click", startRun);
  document.querySelector("#stop-run").addEventListener("click", stopRun);

  setCurrentRun(null, null);
  setCurrentRunStatus("idle");
  setRunsStatus("idle");

  document.querySelector("#refresh-runs").addEventListener("click", async () => {
    const workingDir = document.querySelector("#working-dir").value;
    const runsDir = document.querySelector("#runs-dir").value;
    setRunsStatus("loading...");
    appendLog("[ui] list_runs");
    try {
      const res = await invoke(IPC.LIST_RUNS, { workingDir, runsDir });
      const table = document.querySelector("#runs-table");
      const tbody = table.querySelector("tbody");
      tbody.innerHTML = "";

      const rows = Array.isArray(res) ? res : [];
      for (const r of rows) {
        const tr = document.createElement("tr");

        const createdAt = r.created_at ?? "";
        const runId = r.run_id ?? "";
        const status = r.status ?? "";
        const configHash = r.config_hash ?? "";
        const manifestId = r.frames_manifest_id ?? "";

        tr.dataset.runDir = r.run_dir ?? "";
        tr.dataset.runId = runId;
        const sClass = statusClass(status);
        tr.innerHTML = `
          <td>${createdAt}</td>
          <td class="monospace">${runId}</td>
          <td><span class="status-pill ${sClass}">${status}</span></td>
          <td class="monospace">${configHash}</td>
          <td class="monospace">${manifestId}</td>
        `;

        tr.addEventListener("click", async () => {
          for (const row of tbody.querySelectorAll("tr")) row.classList.remove("selected");
          tr.classList.add("selected");
          const rd = tr.dataset.runDir;
          setCurrentRun(tr.dataset.runId, rd);
          await refreshCurrentStatus();
          document.querySelector("#refresh-current-logs").click();
          document.querySelector("#refresh-current-artifacts").click();
        });

        tbody.appendChild(tr);
      }
      setRunsStatus("idle");
    } catch (e) {
      setRunsStatus("error");
      appendLog(`[error] list_runs failed: ${String(e)}`);
    }
  });

  async function refreshCurrentStatus() {
    const workingDir = document.querySelector("#working-dir").value;
    if (!currentRunDir) {
      appendLog("[ui] no current run_dir yet");
      return;
    }
    setCurrentRunStatus("loading status...");
    try {
      const res = await invoke(IPC.GET_RUN_STATUS, { workingDir, runDir: currentRunDir });
      const st = String(res.status ?? "");
      const ph = res.phase_name ? ` (${String(res.phase_name)})` : "";
      setCurrentRunStatus(`${st}${ph}`);
    } catch (e) {
      setCurrentRunStatus("error");
      appendLog(`[error] get_run_status failed: ${String(e)}`);
    }
  }

  document.querySelector("#refresh-current-status").addEventListener("click", refreshCurrentStatus);

  document.querySelector("#refresh-current-logs").addEventListener("click", async () => {
    const workingDir = document.querySelector("#working-dir").value;
    if (!currentRunDir) {
      appendLog("[ui] no current run_dir yet");
      return;
    }
    const tail = Number(document.querySelector("#current-tail").value || "200");
    const filterText = document.querySelector("#current-filter").value || "";
    setCurrentRunStatus("loading logs...");
    try {
      const res = await invoke(IPC.GET_RUN_LOGS, { workingDir, runDir: currentRunDir, tail });
      const events = res.events || [];
      renderLogEvents("#current-run-logs", events, filterText);
      setCurrentRunStatus("idle");
    } catch (e) {
      setCurrentRunStatus("error");
      appendLog(`[error] get_run_logs failed: ${String(e)}`);
    }
  });

  document.querySelector("#refresh-current-artifacts").addEventListener("click", async () => {
    const workingDir = document.querySelector("#working-dir").value;
    if (!currentRunDir) {
      appendLog("[ui] no current run_dir yet");
      return;
    }
    setCurrentRunStatus("loading artifacts...");
    try {
      const res = await invoke(IPC.LIST_ARTIFACTS, { workingDir, runDir: currentRunDir });
      document.querySelector("#current-run-artifacts").textContent = JSON.stringify(res, null, 2);
      setCurrentRunStatus("idle");
    } catch (e) {
      setCurrentRunStatus("error");
      appendLog(`[error] list_artifacts failed: ${String(e)}`);
    }
  });

  const getYamlText = () => document.querySelector("#config-yaml").value;
  const setYamlText = (t) => {
    document.querySelector("#config-yaml").value = t;
  };

  const getConfigPath = () => document.querySelector("#config-path-editor").value;

  document.querySelector("#config-yaml").addEventListener("input", () => {
    configValidatedOk = false;
    setConfigValidateStatus(null, "not validated");
    setStartAllowed(false);
  });

  document.querySelector("#config-path-editor").addEventListener("input", () => {
    configValidatedOk = false;
    setConfigValidateStatus(null, "not validated");
    setStartAllowed(false);
    document.querySelector("#config-path").value = getConfigPath();
  });

  document.querySelector("#config-load").addEventListener("click", async () => {
    const workingDir = document.querySelector("#working-dir").value;
    const path = getConfigPath();
    setConfigStatus("loading...");
    appendLog("[ui] load_config");
    try {
      const res = await invoke(IPC.LOAD_CONFIG, { workingDir, path });
      setYamlText(res.yaml ?? "");
      document.querySelector("#config-path").value = path;
      configValidatedOk = false;
      setConfigValidateStatus(null, "not validated");
      setConfigStatus("idle");
      setStartAllowed(false);
    } catch (e) {
      setConfigStatus("error");
      appendLog(`[error] load_config failed: ${String(e)}`);
    }
  });

  document.querySelector("#config-save").addEventListener("click", async () => {
    const workingDir = document.querySelector("#working-dir").value;
    const path = getConfigPath();
    const yaml_text = getYamlText();
    setConfigStatus("saving...");
    appendLog("[ui] save_config");
    try {
      await invoke(IPC.SAVE_CONFIG, { workingDir, path, yamlText: yaml_text });
      document.querySelector("#config-path").value = path;
      setConfigStatus("idle");
    } catch (e) {
      setConfigStatus("error");
      appendLog(`[error] save_config failed: ${String(e)}`);
    }
  });

  document.querySelector("#config-validate").addEventListener("click", async () => {
    const workingDir = document.querySelector("#working-dir").value;
    const yaml_text = getYamlText();
    setConfigStatus("validating...");
    appendLog("[ui] validate_config");
    try {
      const res = await invoke(IPC.VALIDATE_CONFIG, { workingDir, yamlText: yaml_text, schemaPath: null });
      const ok = Boolean(res.valid);
      configValidatedOk = ok;
      setConfigValidateStatus(ok, ok ? "valid" : "invalid");
      setConfigStatus(ok ? "ok" : "error");

      if (Array.isArray(res.warnings)) {
        for (const w of res.warnings) {
          appendLog(`[warning] ${w.code ?? "warning"}: ${w.message ?? ""}`);
        }
      }
      if (Array.isArray(res.errors)) {
        for (const err of res.errors) {
          appendLog(`[error] ${err.code ?? "error"}: ${err.message ?? ""}`);
        }
      }

      setStartAllowed(false);
    } catch (e) {
      configValidatedOk = false;
      setConfigValidateStatus(false, "error");
      setConfigStatus("error");
      appendLog(`[error] validate_config failed: ${String(e)}`);
      setStartAllowed(false);
    }
  });

  document.querySelector("#scan-run").addEventListener("click", async () => {
    const workingDir = document.querySelector("#working-dir").value;
    const input_path = document.querySelector("#scan-input-dir").value || document.querySelector("#input-dir").value;
    const frames_min = Number(document.querySelector("#scan-frames-min").value || "1");

    if (!String(input_path || "").trim()) {
      setScanStatus("error");
      setScanMessage("error", "Input dir is required.");
      appendLog("[error] scan_input: input dir missing");
      return;
    }

    confirmedColorMode = null;
    lastScanResult = null;
    document.querySelector("#scan-results").hidden = true;
    document.querySelector("#color-confirm").hidden = true;
    setScanMessage(null, null);
    setScanStatus("scanning...");
    appendLog("[ui] scan_input");

    try {
      const res = await invoke(IPC.SCAN_INPUT, { workingDir, inputPath: input_path, framesMin: frames_min });
      lastScanResult = res;
      renderScanResult(res);
      setScanStatus(res.ok ? "ok" : "error");

      if (Array.isArray(res.errors) && res.errors.length) {
        const first = res.errors[0];
        setScanMessage("error", `${first.code ?? "error"}: ${first.message ?? ""}`.trim());
      } else if (Array.isArray(res.warnings) && res.warnings.length) {
        const first = res.warnings[0];
        setScanMessage("warn", `${first.code ?? "warning"}: ${first.message ?? ""}`.trim());
      } else {
        setScanMessage(null, null);
      }

      if (res.ok) {
        document.querySelector("#input-dir").value ||= input_path;
      }

      if (Array.isArray(res.warnings)) {
        for (const w of res.warnings) {
          appendLog(`[warning] ${w.code ?? "warning"}: ${w.message ?? ""}`);
        }
      }
      if (Array.isArray(res.errors)) {
        for (const e of res.errors) {
          appendLog(`[error] ${e.code ?? "error"}: ${e.message ?? ""}`);
        }
      }

      setStartAllowed(false);
    } catch (e) {
      setScanStatus("error");
      setScanMessage("error", String(e));
      appendLog(`[error] scan failed: ${String(e)}`);
      setStartAllowed(false);
    }
  });

  document.querySelector("#confirm-color-mode").addEventListener("click", async () => {
    if (!lastScanResult) return;
    const selected = document.querySelector("#color-mode-select").value;
    confirmedColorMode = selected || null;
    if (confirmedColorMode) {
      document.querySelector("#scan-color-mode").textContent = confirmedColorMode;
      document.querySelector("#input-dir").value ||= document.querySelector("#scan-input-dir").value;
      appendLog(`[ui] confirmed color_mode=${confirmedColorMode}`);
    }
    setStartAllowed(false);
  });

  await listen(EVENTS.RUNNER_LINE, (event) => {
    appendLog(event.payload.line);

    // Try parse JSON events to capture run_start metadata.
    try {
      const obj = JSON.parse(event.payload.line);
      if (obj && obj.type === "run_start") {
        const runId = obj.run_id;
        const runDir = obj.paths && obj.paths.run_dir;
        setCurrentRun(runId, runDir);
        refreshCurrentStatus();
      }
    } catch {
      // ignore
    }
  });

  await listen(EVENTS.RUNNER_STDERR, (event) => {
    appendLog(`[stderr] ${event.payload.line}`);
  });

  await listen(EVENTS.RUNNER_STARTED, (event) => {
    appendLog(`[runner] started pid=${event.payload.pid}`);
    setConfigEditable(true);
    setStartAllowed(true);
  });

  await listen(EVENTS.RUNNER_EXIT, (event) => {
    const code = event.payload.exit_code;
    appendLog(`[runner] exit code=${code === null ? "null" : String(code)}`);
    setButtons(false);
    setStatus("idle");
    setConfigEditable(false);
    setStartAllowed(false);
  });
});
