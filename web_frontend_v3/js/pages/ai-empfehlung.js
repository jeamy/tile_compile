// js/pages/ai-empfehlung.js – Sub-Tab: AI Empfehlung (innerhalb Parameter)

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { getAiState, setAiState, getAiFormData, setAiFormData } from "../state/ai-state.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";

export function createAiEmpfehlungPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  const fd = getAiFormData();
  const aiState = getAiState();

  // Scan context
  const scanCtx = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.scan_context", "Scan-Kontext (auto aus Scan)")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.mount", "Mount")),
        el("select", { class: "tc-select", onchange: (e) => setAiFormData({ mount: e.target.value }) },
          ...["EQ", "Tracker", "Alt/Az"].map(v => el("option", { value: v, ...(fd.mount === v ? { selected: true } : {}) }, v)),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.object_type", "Objekt")),
        el("select", { class: "tc-select", onchange: (e) => setAiFormData({ object_type: e.target.value }) },
          ...["Galaxie", "Nebel", "Sternhaufen", "Sterne"].map(v => el("option", { value: v, ...(fd.object_type === v ? { selected: true } : {}) }, v)),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.camera", "Kamera")),
        el("select", { class: "tc-select", onchange: (e) => setAiFormData({ camera: e.target.value }) },
          ...["Consumer OSC", "Mono CMOS", "CCD"].map(v => el("option", { value: v, ...(fd.camera === v ? { selected: true } : {}) }, v)),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.calibration", "Kalibrierung")),
        el("div", { class: "tc-flex tc-gap-3" },
          el("label", { class: "tc-checkbox" }, el("input", { type: "checkbox", checked: fd.calibration_darks, onchange: (e) => setAiFormData({ calibration_darks: e.target.checked }) }), "Darks"),
          el("label", { class: "tc-checkbox" }, el("input", { type: "checkbox", checked: fd.calibration_flats, onchange: (e) => setAiFormData({ calibration_flats: e.target.checked }) }), "Flats"),
          el("label", { class: "tc-checkbox" }, el("input", { type: "checkbox", checked: fd.calibration_bias, onchange: (e) => setAiFormData({ calibration_bias: e.target.checked }) }), "Bias"),
        ),
      ),
    ),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.field.notes", "Notizen")),
      el("input", { type: "text", class: "tc-input", value: fd.notes, placeholder: "Guiding 0.8\", M31, alt-az test", oninput: (e) => setAiFormData({ notes: e.target.value }) }),
    ),
  );

  // Model & API Key
  const modelCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.model_api", "Modell & API-Key")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.provider", "Provider")),
        el("select", { class: "tc-select", id: "ai-provider", onchange: (e) => { setAiFormData({ provider: e.target.value }); filterModelsByProvider(e.target.value); } },
          ...["anthropic", "openai"].map(v => el("option", { value: v, ...(fd.provider === v ? { selected: true } : {}) }, v)),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.model", "Modell")),
        el("select", { class: "tc-select", id: "ai-model", onchange: (e) => setAiFormData({ model: e.target.value }) },
          el("option", { value: fd.model || "", selected: true }, fd.model || t("ui.placeholder.select_model", "Modell wählen")),
        ),
      ),
    ),
    el("div", { class: "tc-mt-2 tc-flex tc-items-center tc-gap-2" },
      el("input", { type: "password", class: "tc-input", style: { flex: "1 1 auto", minWidth: "0" }, value: fd.apiKey, placeholder: "API-Key", id: "ai-apikey", oninput: (e) => setAiFormData({ apiKey: e.target.value }) }),
      el("button", { class: "tc-btn tc-btn-sm", style: { flexShrink: "0" }, onclick: () => saveApiKey() }, t("ui.button.save_key", "Key speichern")),
      el("span", { class: "tc-badge tc-badge-success", style: { flexShrink: "0", whiteSpace: "nowrap" }, id: "ai-key-status" }, "\u2713 gespeichert"),
    ),
    el("div", { class: "tc-mt-2" },
      el("span", { class: "tc-text-sm tc-text-muted", id: "ai-model-status" }, t("ui.state.model_loading", "Modelle werden geladen...")),
    ),
  );

  // Actions
  const historySelect = el("select", { class: "tc-select", style: { width: "200px" }, id: "ai-history-select",
    onchange: (e) => loadSavedAnalysis(e.target.value),
  },
    el("option", { value: "" }, t("ui.placeholder.saved_analyses", "Gespeicherte Analysen")),
  );

  const actions = el("div", { class: "tc-flex tc-gap-3 tc-flex-wrap" },
    el("button", { class: "tc-btn tc-btn-primary", onclick: () => createAnalysis() }, t("ui.button.create_analysis", "KI-Analyse erstellen")),
    el("button", { class: "tc-btn", onclick: () => createAnalysis(true) }, t("ui.button.reanalyze", "Neu analysieren (Cache ignorieren)")),
    historySelect,
  );

  // Recommendations
  const recs = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.recommendations", "Empfehlungen")),
    el("div", { class: "tc-flex-col tc-gap-3", id: "ai-recommendations" },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_analysis", "Noch keine KI-Analyse erstellt.")),
    ),
  );

  // Apply actions
  const applyBar = el("div", { class: "tc-flex tc-gap-3" },
    el("button", { class: "tc-btn tc-btn-primary", onclick: () => applyRecommendations() }, t("ui.button.apply_selected", "Ausgewaehlte anwenden")),
    el("button", { class: "tc-btn", onclick: () => applyRecommendations(true) }, t("ui.button.apply_all", "Alle anwenden")),
    el("button", { class: "tc-btn", onclick: () => discardRecommendations() }, t("ui.button.discard", "Verwerfen")),
  );

  // AI traffic (collapsible)
  const traffic = el("div", { class: "tc-accordion", id: "ai-traffic" },
    el("div", {
      class: "tc-accordion-header",
      onclick: () => traffic.classList.toggle("open"),
    }, "\u25b8 " + t("ui.title.ai_traffic", "KI-Datenverkehr")),
    el("div", { class: "tc-accordion-body" },
      el("div", { class: "tc-log-viewer", id: "ai-traffic-log" },
        el("div", { class: "tc-text-muted" }, t("ui.state.no_traffic", "Keine Daten")),
      ),
    ),
  );

  page.append(scanCtx, modelCard, actions, recs, applyBar, traffic);

  // Load models from backend
  loadModels();

  // Restore loading state if analysis is in progress
  if (aiState.loading) {
    const recsContainer = document.getElementById("ai-recommendations");
    if (recsContainer) {
      recsContainer.innerHTML = "";
      recsContainer.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.toast.analysis_creating", "KI-Analyse wird erstellt...") + " (Tab-Wechsel OK, läuft im Hintergrund)"));
    }
  }

  // Load saved analyses history and restore current analysis
  loadAnalysisHistory();
  if (aiState.currentAnalysis?.recommendations) {
    renderRecommendations(aiState.currentAnalysis.recommendations);
  }
  if (aiState.trafficLog?.length > 0) {
    renderTrafficLog(aiState.trafficLog);
  }

  return page;
}

let _allModels = [];

async function loadModels() {
  const statusEl = document.getElementById("ai-model-status");
  const modelSelect = document.getElementById("ai-model");
  try {
    const models = await api.get(API_ENDPOINTS.ai.models);
    if (models?.available === false) {
      if (statusEl) statusEl.textContent = t("ui.state.model_unavailable", "Sidecar nicht erreichbar");
      return;
    }
    const providers = Array.isArray(models?.providers) ? models.providers : [];
    _allModels = [];
    for (const p of providers) {
      const providerName = String(p?.provider || "").trim();
      const modelList = Array.isArray(p?.models) ? p.models : [];
      for (const m of modelList) {
        const id = m?.id || m?.name || "";
        const label = m?.label || m?.name || id;
        if (id) _allModels.push({ value: `${providerName}/${id}`, label: `${providerName}: ${label}`, provider: providerName });
      }
    }
    const providerCount = providers.length;
    const modelCount = _allModels.length;
    if (statusEl) statusEl.textContent = t("ui.state.model_loaded", "Modelle geladen") + ` (${providerCount} Provider, ${modelCount} Modelle)`;
    filterModelsByProvider(document.getElementById("ai-provider")?.value || "");
  } catch (e) {
    if (statusEl) statusEl.textContent = t("ui.state.model_load_failed", "Modelle laden fehlgeschlagen") + `: ${e.message}`;
  }
}

function filterModelsByProvider(provider) {
  const select = document.getElementById("ai-model");
  if (!select) return;
  const fd = getAiFormData();
  const currentModel = fd.model || "";
  select.innerHTML = "";
  select.appendChild(el("option", { value: "" }, t("ui.placeholder.select_model", "Modell wählen")));
  const filtered = provider ? _allModels.filter(m => m.provider === provider) : _allModels;
  let foundCurrent = false;
  for (const m of filtered) {
    const selected = m.value === currentModel;
    if (selected) foundCurrent = true;
    select.appendChild(el("option", { value: m.value, ...(selected ? { selected: true } : {}) }, m.label));
  }
  if (currentModel && !foundCurrent) {
    select.appendChild(el("option", { value: currentModel, selected: true }, `${currentModel} (gespeichert)`));
  }
}

async function saveApiKey() {
  const provider = document.getElementById("ai-provider")?.value || "";
  const key = document.getElementById("ai-apikey")?.value || "";
  if (!provider || !key) return;
  try {
    await api.post(API_ENDPOINTS.ai.authProvider(provider), { api_key: key });
    toastSuccess(t("ui.toast.key_saved", "API-Key gespeichert"));
  } catch (e) {
    toastError(t("ui.toast.key_save_failed", "Key speichern fehlgeschlagen"), e.message);
  }
}

async function createAnalysis(force = false) {
  setAiState({ loading: true });
  try {
    toast(t("ui.toast.analysis_creating", "KI-Analyse wird erstellt..."), "", "info");
    addTrafficEntry({ type: "request", text: `POST /api/scan/analysis { force: ${force} }` });

    // Fetch latest scan result
    addTrafficEntry({ type: "info", text: "Fetching latest scan result..." });
    const latestScan = await api.get(API_ENDPOINTS.scan.latest);
    const hasScan = latestScan && (latestScan.has_scan || (latestScan.frames_detected && latestScan.frames_detected > 0));
    if (!hasScan) {
      addTrafficEntry({ type: "error", text: "No scan result available" });
      setAiState({ loading: false });
      toastError(t("ui.toast.analysis_failed", "Analyse fehlgeschlagen"), t("ui.state.no_scan", "Kein Scan vorhanden. Bitte zuerst einen Scan durchfuehren."));
      return;
    }
    addTrafficEntry({ type: "info", text: `Scan: ${latestScan.frames_detected || 0} frames, ${latestScan.color_mode || "unknown"}` });

    // Compute scan metrics (image statistics like FWHM, star count)
    let scanMetrics = null;
    try {
      addTrafficEntry({ type: "info", text: "Computing image statistics..." });
      const metricsStart = await api.post(API_ENDPOINTS.scan.metrics, { input_path: latestScan.input_path || latestScan.input_dirs?.[0] || "" });
      if (metricsStart?.job_id) {
        scanMetrics = await pollJobResult(metricsStart.job_id);
        if (scanMetrics) {
          const sampled = scanMetrics.sample_count ?? "?";
          const total = scanMetrics.frames_total ?? latestScan.frames_detected ?? "?";
          addTrafficEntry({ type: "info", text: `Image statistics: sampled=${sampled}/${total}` });
        }
      }
    } catch (metricsErr) {
      addTrafficEntry({ type: "warn", text: `Image statistics failed: ${metricsErr.message}` });
    }

    // Fetch config schema and base config
    let configSchema = null;
    let baseConfig = null;
    try {
      configSchema = await api.get(API_ENDPOINTS.config.schema);
    } catch {}
    try {
      const configResp = await api.get(API_ENDPOINTS.config.current);
      const yamlText = configResp?.config || configResp?.yaml || (typeof configResp === "string" ? configResp : "");
      const { parseYaml } = await import("../utils/yaml-parse.js");
      baseConfig = parseYaml(yamlText);
    } catch {}

    // Build payload like gui2 does
    const fd = getAiFormData();
    const payload = {
      force,
      scan_result: latestScan,
      model: fd.model || undefined,
    };
    if (scanMetrics) payload.scan_metrics = scanMetrics;
    if (configSchema) payload.config_schema = configSchema;
    if (baseConfig) payload.base_config = baseConfig;

    addTrafficEntry({ type: "request", text: `POST /api/scan/analysis with scan_result, ${scanMetrics ? "scan_metrics, " : ""}${configSchema ? "config_schema, " : ""}${baseConfig ? "base_config" : ""}` });

    const result = await api.post(API_ENDPOINTS.scan.analysis, payload, { timeoutMs: 600000 });
    const recs = result?.validated_updates || result?.updates || result?.recommendations || [];
    if (recs.length > 0) {
      addTrafficEntry({ type: "response", text: `Received ${recs.length} recommendations (cached: ${result.from_cache ?? false})` });
      if (result.summary) {
        addTrafficEntry({ type: "summary", text: result.summary });
      }
      renderRecommendations(recs);
      setAiState({ currentAnalysis: result, loading: false });
      toastSuccess(t("ui.toast.analysis_done", "Analyse erstellt"));
    } else {
      addTrafficEntry({ type: "error", text: `No recommendations in response. Keys: ${Object.keys(result || {}).join(", ")}` });
      setAiState({ loading: false });
      toastError(t("ui.toast.analysis_failed", "Analyse fehlgeschlagen"), "No recommendations returned");
    }
  } catch (e) {
    addTrafficEntry({ type: "error", text: e.message });
    setAiState({ loading: false });
    toastError(t("ui.toast.analysis_failed", "Analyse fehlgeschlagen"), e.message);
  }
}

async function pollJobResult(jobId, timeoutMs = 600000) {
  const maxAttempts = Math.ceil(timeoutMs / 2000);
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 2000));
    const job = await api.get(API_ENDPOINTS.scan.jobStatus(jobId));
    const state = job?.state;
    if (state === "done" || state === "completed" || state === "ok") {
      return job?.data?.result || job?.data || null;
    }
    if (state === "error" || state === "failed") {
      throw new Error(job?.error || "Job failed");
    }
  }
  throw new Error("Job timeout");
}

function renderRecommendations(recs) {
  const container = document.getElementById("ai-recommendations");
  if (!container) return;
  container.innerHTML = "";
  for (const rec of recs) {
    const item = el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
      el("label", { class: "tc-checkbox" },
        el("input", { type: "checkbox", checked: rec.selected !== false,
          onchange: (e) => { rec.selected = e.target.checked; },
        }),
        el("span", { class: "tc-mono tc-text-sm" }, rec.path || rec.id || ""),
      ),
      el("div", { class: "tc-mt-2 tc-text-sm" },
        el("span", { class: "tc-text-muted" }, t("ui.label.current", "Aktuell") + ": "),
        el("span", {}, String(rec.current_value ?? rec.current ?? "")),
      ),
      el("div", { class: "tc-text-sm" },
        el("span", { class: "tc-text-muted" }, t("ui.label.recommended", "Empfohlen") + ": "),
        el("span", {}, String(rec.value ?? rec.recommended ?? rec.recommended_value ?? "")),
      ),
      rec.confidence != null ? el("div", { class: "tc-text-sm tc-text-muted" }, `Confidence: ${Math.round(rec.confidence * 100)}%`) : null,
      rec.reason ? el("div", { class: "tc-mt-1 tc-text-sm tc-text-muted" }, rec.reason) : null,
    );
    container.appendChild(item);
  }
}

async function applyRecommendations(all = false) {
  const { currentAnalysis } = getAiState();
  if (!currentAnalysis?.recommendations) return;
  const selected = all
    ? currentAnalysis.recommendations
    : currentAnalysis.recommendations.filter(r => r.selected !== false);
  try {
    await api.post(API_ENDPOINTS.scan.analysisApply, { recommendations: selected });
    toastSuccess(t("ui.toast.applied", "Empfehlungen angewendet"));
  } catch (e) {
    toastError(t("ui.toast.apply_failed", "Anwenden fehlgeschlagen"), e.message);
  }
}

function discardRecommendations() {
  setAiState({ currentAnalysis: null });
  const container = document.getElementById("ai-recommendations");
  if (container) {
    container.innerHTML = "";
    container.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_analysis", "Noch keine KI-Analyse erstellt.")));
  }
}

async function loadAnalysisHistory() {
  try {
    const result = await api.get(API_ENDPOINTS.scan.analysisHistory);
    const items = result?.items || [];
    const select = document.getElementById("ai-history-select");
    if (!select) return;
    select.innerHTML = "";
    select.appendChild(el("option", { value: "" }, t("ui.placeholder.saved_analyses", "Gespeicherte Analysen")));
    for (const item of items) {
      const label = `${item.persisted_at || ""} - ${(item.summary || "").substring(0, 60)}`;
      select.appendChild(el("option", { value: item.filename || item.analysis_id }, label));
    }
  } catch {
    // silently fail
  }
}

async function loadSavedAnalysis(analysisId) {
  if (!analysisId) return;
  try {
    toast(t("ui.toast.loading_analysis", "Analyse wird geladen..."), "", "info");
    const result = await api.get(API_ENDPOINTS.scan.analysisHistoryItem(analysisId));
    if (result?.recommendations) {
      renderRecommendations(result.recommendations);
      setAiState({ currentAnalysis: result });
      toastSuccess(t("ui.toast.analysis_loaded", "Analyse geladen"));
    } else if (result?.has_analysis === false) {
      toastError(t("ui.toast.analysis_load_failed", "Analyse laden fehlgeschlagen"), "Analysis not found");
    }
  } catch (e) {
    toastError(t("ui.toast.analysis_load_failed", "Analyse laden fehlgeschlagen"), e.message);
  }
}

function addTrafficEntry(entry) {
  const { trafficLog } = getAiState();
  const updated = [...(trafficLog || []), { ...entry, ts: new Date().toISOString() }];
  setAiState({ trafficLog: updated.slice(-100) });
  renderTrafficLog(updated.slice(-100));
}

function renderTrafficLog(log) {
  const container = document.getElementById("ai-traffic-log");
  if (!container) return;
  container.innerHTML = "";
  if (!log || log.length === 0) {
    container.appendChild(el("div", { class: "tc-text-muted" }, t("ui.state.no_traffic", "Keine Daten")));
    return;
  }
  for (const entry of log) {
    const cls = entry.type === "error" ? "tc-text-error" : entry.type === "summary" ? "tc-text-muted" : entry.type === "warn" ? "tc-text-warning" : "";
    container.appendChild(el("div", { class: `tc-text-sm tc-mono ${cls}` }, `[${entry.ts?.substring(11, 19) || ""}] ${entry.text}`));
  }
}
