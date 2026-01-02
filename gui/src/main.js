import { IPC, EVENTS } from "./constants.js";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

function appendLog(text) {
  const el = document.querySelector("#log");
  el.textContent += text + "\n";
  el.scrollTop = el.scrollHeight;
}

function setStatus(text) {
  document.querySelector("#run-status").textContent = text;
}

function setButtons(running) {
  document.querySelector("#start-run").disabled = running;
  document.querySelector("#stop-run").disabled = !running;
}

function getFormValues() {
  return {
    working_dir: document.querySelector("#working-dir").value,
    config_path: document.querySelector("#config-path").value,
    input_dir: document.querySelector("#input-dir").value,
    runs_dir: document.querySelector("#runs-dir").value,
    pattern: document.querySelector("#pattern").value,
    dry_run: document.querySelector("#dry-run").checked,
  };
}

async function startRun() {
  const args = getFormValues();
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
  setStatus("stopping...");
  appendLog("[ui] stop_run");
  try {
    const res = await invoke(IPC.STOP_RUN);
    setButtons(false);
    setStatus(res.stopped ? "stopped" : "idle");
  } catch (e) {
    setStatus("error");
    appendLog(`[error] ${String(e)}`);
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  document.querySelector("#working-dir").value ||= "..";
  document.querySelector("#config-path").value ||= "tile_compile.yaml";
  document.querySelector("#runs-dir").value ||= "runs";
  document.querySelector("#pattern").value ||= "*.fit*";

  setButtons(false);
  setStatus("idle");

  document.querySelector("#start-run").addEventListener("click", startRun);
  document.querySelector("#stop-run").addEventListener("click", stopRun);

  await listen(EVENTS.RUNNER_LINE, (event) => {
    appendLog(event.payload.line);
  });

  await listen(EVENTS.RUNNER_STDERR, (event) => {
    appendLog(`[stderr] ${event.payload.line}`);
  });

  await listen(EVENTS.RUNNER_STARTED, (event) => {
    appendLog(`[runner] started pid=${event.payload.pid}`);
  });

  await listen(EVENTS.RUNNER_EXIT, (event) => {
    const code = event.payload.exit_code;
    appendLog(`[runner] exit code=${code === null ? "null" : String(code)}`);
    setButtons(false);
    setStatus("idle");
  });
});
