import { ApiClient } from "./api.js";

const VIEW_DEFS = [
  ["dashboard", "Dashboard"],
  ["input", "Input & Scan"],
  ["params", "Parameter Studio"],
  ["run", "Run Monitor"],
  ["history", "History + Tools"],
  ["astrometry", "Astrometry"],
  ["pcc", "PCC"],
  ["log", "Live Log"],
];

const state = {
  api: new ApiClient(localStorage.getItem("gui2.backendBase") || "/"),
  locale: "de",
  currentRunId: "",
  currentRunSocket: null,
  jobPollTimer: null,
};

const $ = (id) => document.getElementById(id);

function appendLine(el, value) {
  if (!el) return;
  const line = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  const old = el.textContent ? `${el.textContent}\n` : "";
  const merged = `${old}${line}`.split("\n");
  el.textContent = merged.slice(-250).join("\n");
}

function setPre(el, value) {
  if (!el) return;
  el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function setStatus(text, ok = true) {
  const el = $("backend-status");
  el.textContent = text;
  el.style.background = ok ? "var(--ok)" : "var(--warn)";
}

function installNav() {
  const nav = $("nav");
  nav.innerHTML = "";
  for (const [id, label] of VIEW_DEFS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.dataset.view = id;
    btn.addEventListener("click", () => showView(id));
    nav.appendChild(btn);
  }
  showView("dashboard");
}

function showView(id) {
  document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === id));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${id}`));
  if (id === "dashboard") void refreshDashboard();
  if (id === "history" || id === "log") void refreshHistory();
}

async function refreshDashboard() {
  const kpi = $("dashboard-kpis");
  const guardrailsOut = $("dashboard-guardrails");
  const runsOut = $("dashboard-runs");

  try {
    const [appState, guardrails, quality, runs] = await Promise.all([
      state.api.get("/api/app/state"),
      state.api.get("/api/guardrails"),
      state.api.get("/api/scan/quality"),
      state.api.get("/api/runs"),
    ]);

    setStatus("online", true);
    state.currentRunId = appState?.project?.current_run_id || state.currentRunId;

    const cards = [
      ["Scan Quality", quality?.score ?? "-"],
      ["Guardrails", guardrails?.status ?? "-"],
      ["Runs", runs?.total ?? 0],
    ];
    kpi.innerHTML = cards
      .map(([k, v]) => `<div class="kpi"><div>${k}</div><div class="v">${v}</div></div>`)
      .join("");

    guardrailsOut.innerHTML = (guardrails?.checks || [])
      .map((c) => `<div class="chip">${c.status || "?"}: ${c.label || c.id}</div>`)
      .join(" ");

    runsOut.innerHTML = (runs?.items || [])
      .slice(0, 12)
      .map((r) => `<div>${r.run_id} · ${r.status}</div>`)
      .join("");

    $("run-ready").textContent = `Run Ready: ${guardrails?.status || "?"}`;
  } catch (err) {
    setStatus("offline", false);
    setPre(guardrailsOut, err.payload || err.message);
  }
}

function installBackendControls() {
  $("backend-base").value = state.api.baseUrl || "/";
  $("backend-apply").addEventListener("click", async () => {
    state.api.setBase($("backend-base").value);
    localStorage.setItem("gui2.backendBase", state.api.baseUrl || "/");
    await refreshDashboard();
  });

  $("locale-de").addEventListener("click", () => (state.locale = "de"));
  $("locale-en").addEventListener("click", () => (state.locale = "en"));
}

function startJobPolling() {
  if (state.jobPollTimer) clearInterval(state.jobPollTimer);
  state.jobPollTimer = setInterval(() => {
    void refreshHistory();
  }, 2000);
}

async function pollJob(jobId, outEl, onDone) {
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    const job = await state.api.get(`/api/jobs/${jobId}`);
    setPre(outEl, job);
    if (["ok", "error", "cancelled"].includes(job.state)) {
      onDone?.(job);
      return job;
    }
    await new Promise((r) => setTimeout(r, 800));
  }
  throw new Error(`Job timeout: ${jobId}`);
}

function installScan() {
  $("scan-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const out = $("scan-job");
    try {
      const payload = {
        input_path: $("scan-input-path").value,
        frames_min: Number($("scan-frames-min").value || 1),
        with_checksums: $("scan-checksums").checked,
      };
      const accepted = await state.api.post("/api/scan", payload);
      setPre(out, accepted);
      await pollJob(accepted.job_id, out);
      await refreshDashboard();
    } catch (err) {
      setPre(out, err.payload || err.message);
    }
  });
}

async function refreshPresets() {
  const sel = $("preset-select");
  const data = await state.api.get("/api/config/presets");
  sel.innerHTML = "";
  for (const item of data.items || []) {
    const opt = document.createElement("option");
    opt.value = item.path;
    opt.textContent = item.name;
    sel.appendChild(opt);
  }
}

async function refreshRevisions() {
  const out = $("config-revisions");
  const data = await state.api.get("/api/config/revisions");
  out.innerHTML = "";
  for (const rev of data.items || []) {
    const row = document.createElement("div");
    row.className = "actions";
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = `Restore ${rev.revision_id}`;
    b.addEventListener("click", async () => {
      await state.api.post(`/api/config/revisions/${rev.revision_id}/restore`, {});
      await refreshRevisions();
    });
    row.append(`${rev.revision_id} (${rev.source}) `, b);
    out.appendChild(row);
  }
}

function installConfig() {
  const yamlEl = $("config-yaml");
  const validationEl = $("config-validation");

  $("config-load").addEventListener("click", async () => {
    try {
      const data = await state.api.get("/api/config/current");
      yamlEl.value = data.config || "";
    } catch (err) {
      setPre(validationEl, err.payload || err.message);
    }
  });

  $("config-validate").addEventListener("click", async () => {
    try {
      const data = await state.api.post("/api/config/validate", { yaml: yamlEl.value });
      setPre(validationEl, data);
    } catch (err) {
      setPre(validationEl, err.payload || err.message);
    }
  });

  $("config-save").addEventListener("click", async () => {
    try {
      const data = await state.api.post("/api/config/save", { yaml: yamlEl.value });
      setPre(validationEl, data);
      await refreshRevisions();
    } catch (err) {
      setPre(validationEl, err.payload || err.message);
    }
  });

  $("preset-apply").addEventListener("click", async () => {
    try {
      const data = await state.api.post("/api/config/presets/apply", { path: $("preset-select").value });
      yamlEl.value = data.config || "";
    } catch (err) {
      setPre(validationEl, err.payload || err.message);
    }
  });
}

function installRunMonitor() {
  const statusOut = $("run-status-out");
  const streamOut = $("run-stream");

  $("run-start").addEventListener("click", async () => {
    try {
      const accepted = await state.api.post("/api/runs/start", {
        input_dir: $("run-input-dir").value,
        run_id: $("run-name").value,
        runs_dir: $("run-runs-dir").value,
      });
      state.currentRunId = accepted.run_id;
      $("resume-run-id").value = accepted.run_id;
      setPre(statusOut, accepted);
      await pollJob(accepted.job_id, statusOut);
      await refreshHistory();
    } catch (err) {
      setPre(statusOut, err.payload || err.message);
    }
  });

  $("run-status").addEventListener("click", async () => {
    try {
      const rid = $("resume-run-id").value || state.currentRunId;
      const data = await state.api.get(`/api/runs/${rid}/status`);
      setPre(statusOut, data);
    } catch (err) {
      setPre(statusOut, err.payload || err.message);
    }
  });

  $("run-stop").addEventListener("click", async () => {
    try {
      const rid = $("resume-run-id").value || state.currentRunId;
      const data = await state.api.post(`/api/runs/${rid}/stop`, {});
      setPre(statusOut, data);
    } catch (err) {
      setPre(statusOut, err.payload || err.message);
    }
  });

  $("run-resume").addEventListener("click", async () => {
    try {
      const rid = $("resume-run-id").value;
      const accepted = await state.api.post(`/api/runs/${rid}/resume`, {
        from_phase: $("resume-phase").value,
        config_revision_id: $("resume-revision").value,
        run_dir: $("resume-run-dir").value,
      });
      setPre(statusOut, accepted);
      await pollJob(accepted.job_id, statusOut);
    } catch (err) {
      setPre(statusOut, err.payload || err.message);
    }
  });

  $("run-connect-ws").addEventListener("click", () => {
    const rid = $("resume-run-id").value;
    if (state.currentRunSocket) state.currentRunSocket.close();
    streamOut.textContent = "";
    state.currentRunSocket = state.api.ws(
      `/api/ws/runs/${rid}`,
      (ev) => appendLine(streamOut, ev),
      (err) => appendLine(streamOut, { ws_error: String(err) }),
    );
  });
}

async function refreshHistory() {
  const runsOut = $("history-runs");
  const jobsOut = $("jobs-list");
  const uiEventsOut = $("ui-events");
  const liveLog = $("live-log");

  try {
    const [runs, jobs, uiEvents] = await Promise.all([
      state.api.get("/api/runs"),
      state.api.get("/api/jobs"),
      state.api.get("/api/app/ui-events"),
    ]);

    runsOut.innerHTML = "";
    for (const item of runs.items || []) {
      const row = document.createElement("div");
      row.className = "actions";
      const setCurrent = document.createElement("button");
      setCurrent.type = "button";
      setCurrent.textContent = "Set Current";
      setCurrent.addEventListener("click", async () => {
        await state.api.post(`/api/runs/${item.run_id}/set-current`, {});
      });
      const status = document.createElement("button");
      status.type = "button";
      status.textContent = "Status";
      status.addEventListener("click", async () => {
        const st = await state.api.get(`/api/runs/${item.run_id}/status`);
        setPre($("run-status-out"), st);
        showView("run");
      });
      row.append(`${item.run_id} · ${item.status} `, setCurrent, status);
      runsOut.appendChild(row);
    }

    jobsOut.innerHTML = (jobs.items || [])
      .slice(0, 25)
      .map((j) => `${j.job_id} · ${j.type} · ${j.state}`)
      .join("<br>");

    uiEventsOut.innerHTML = (uiEvents.items || [])
      .slice(-25)
      .map((e) => `${e.seq}: ${e.event}`)
      .join("<br>");

    setPre(liveLog, (jobs.items || []).slice(0, 30));
  } catch (err) {
    setPre(jobsOut, err.payload || err.message);
  }
}

function installHistory() {
  $("history-refresh").addEventListener("click", () => void refreshHistory());
  $("jobs-refresh").addEventListener("click", () => void refreshHistory());
}

function installAstrometry() {
  const out = $("astrometry-out");

  $("astrometry-detect").addEventListener("click", async () => {
    try {
      const data = await state.api.post("/api/tools/astrometry/detect", {
        astap_cli: $("astap-cli").value || undefined,
        astap_data_dir: $("astap-data-dir").value,
        catalog_dir: $("astap-catalog-dir").value,
      });
      setPre(out, data);
    } catch (err) {
      setPre(out, err.payload || err.message);
    }
  });

  $("astrometry-install").addEventListener("click", async () => {
    try {
      const accepted = await state.api.post("/api/tools/astrometry/install-cli", {
        astap_data_dir: $("astap-data-dir").value,
      });
      setPre(out, accepted);
      await pollJob(accepted.job_id, out);
    } catch (err) {
      setPre(out, err.payload || err.message);
    }
  });

  $("astrometry-download-catalog").addEventListener("click", async () => {
    try {
      const accepted = await state.api.post("/api/tools/astrometry/catalog/download", {
        astap_data_dir: $("astap-data-dir").value,
        catalog_id: "d50",
      });
      setPre(out, accepted);
      await pollJob(accepted.job_id, out);
    } catch (err) {
      setPre(out, err.payload || err.message);
    }
  });

  $("astrometry-solve").addEventListener("click", async () => {
    try {
      const accepted = await state.api.post("/api/tools/astrometry/solve", {
        solve_file: $("astrometry-solve-file").value,
        astap_cli: $("astap-cli").value || undefined,
        astap_data_dir: $("astap-data-dir").value,
        search_radius_deg: Number($("astrometry-radius").value || 180),
      });
      setPre(out, accepted);
      await pollJob(accepted.job_id, out);
    } catch (err) {
      setPre(out, err.payload || err.message);
    }
  });
}

function installPcc() {
  const out = $("pcc-out");

  $("pcc-status").addEventListener("click", async () => {
    try {
      const dir = encodeURIComponent($("pcc-catalog-dir").value);
      const data = await state.api.get(`/api/tools/pcc/siril/status?catalog_dir=${dir}`);
      setPre(out, data);
    } catch (err) {
      setPre(out, err.payload || err.message);
    }
  });

  $("pcc-download").addEventListener("click", async () => {
    try {
      const accepted = await state.api.post("/api/tools/pcc/siril/download-missing", {
        catalog_dir: $("pcc-catalog-dir").value,
      });
      setPre(out, accepted);
      await pollJob(accepted.job_id, out);
    } catch (err) {
      setPre(out, err.payload || err.message);
    }
  });

  $("pcc-check-online").addEventListener("click", async () => {
    try {
      const data = await state.api.post("/api/tools/pcc/check-online", {});
      setPre(out, data);
    } catch (err) {
      setPre(out, err.payload || err.message);
    }
  });

  $("pcc-run").addEventListener("click", async () => {
    try {
      const accepted = await state.api.post("/api/tools/pcc/run", {
        input_rgb: $("pcc-input-rgb").value,
        output_rgb: $("pcc-output-rgb").value,
        r: Number($("pcc-r").value || 1),
        g: Number($("pcc-g").value || 1),
        b: Number($("pcc-b").value || 1),
      });
      setPre(out, accepted);
      await pollJob(accepted.job_id, out);
    } catch (err) {
      setPre(out, err.payload || err.message);
    }
  });
}

async function boot() {
  installNav();
  installBackendControls();
  installScan();
  installConfig();
  installRunMonitor();
  installHistory();
  installAstrometry();
  installPcc();
  startJobPolling();

  await refreshDashboard();
  try {
    await refreshPresets();
    await refreshRevisions();
  } catch {
    // ignored; surface via button actions
  }
}

void boot();
