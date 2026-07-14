// js/pages/ai-empfehlung.js – Sub-Tab: AI Empfehlung (innerhalb Parameter)

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { getAiState, setAiState, getAiFormData, setAiFormData, onAiChange } from "../state/ai-state.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { pollJob } from "../utils/poll.js";
import { createYamlDiff } from "../components/yaml-diff.js";

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
          el("label", { class: "tc-checkbox" }, el("input", { type: "checkbox", checked: fd.calibration_darks, onchange: (e) => setAiFormData({ calibration_darks: e.target.checked }) }), t("ui.label.darks", "Darks")),
          el("label", { class: "tc-checkbox" }, el("input", { type: "checkbox", checked: fd.calibration_flats, onchange: (e) => setAiFormData({ calibration_flats: e.target.checked }) }), t("ui.label.flats", "Flats")),
          el("label", { class: "tc-checkbox" }, el("input", { type: "checkbox", checked: fd.calibration_bias, onchange: (e) => setAiFormData({ calibration_bias: e.target.checked }) }), t("ui.label.bias", "Bias")),
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
      el("span", { class: "tc-badge tc-badge-success", style: { flexShrink: "0", whiteSpace: "nowrap" }, id: "ai-key-status" }, "\u2713 " + t("ui.state.saved", "gespeichert")),
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
  const learnMemory = el("label", { class: "tc-checkbox" },
    el("input", { type: "checkbox", id: "ai-learn-memory" }),
    el("span", {}, t("ui.label.pi_learn_memory", "Aus dieser Optimierung lernen")),
  );
  const applyBar = el("div", { class: "tc-flex tc-gap-3" },
    learnMemory,
    el("button", { class: "tc-btn", onclick: () => previewPiActionPlan() }, t("ui.button.pi_preview", "PI Preview")),
    el("button", { class: "tc-btn", id: "ai-pi-apply", disabled: true, onclick: () => applyPiActionPlan() }, t("ui.button.pi_apply", "PI anwenden")),
    el("button", { class: "tc-btn tc-btn-primary", onclick: () => applyRecommendations() }, t("ui.button.apply_selected", "Ausgewaehlte anwenden")),
    el("button", { class: "tc-btn", onclick: () => applyRecommendations(true) }, t("ui.button.apply_all", "Alle anwenden")),
    el("button", { class: "tc-btn", onclick: () => discardRecommendations() }, t("ui.button.discard", "Verwerfen")),
  );

  const piPreview = el("div", { class: "tc-card", id: "ai-pi-preview" },
    el("div", { class: "tc-card-title" }, t("ui.title.pi_preview", "PI Preview")),
    el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_preview", "Noch keine Preview.")),
  );

  const piMemories = el("div", { class: "tc-card", id: "ai-pi-memories" },
    el("div", { class: "tc-card-title" }, t("ui.title.pi_memories", "PI Memories")),
    el("div", { class: "tc-flex tc-gap-2 tc-items-center tc-mb-2" },
      el("select", { class: "tc-select", id: "ai-pi-memory-filter", onchange: () => loadPiMemories() },
        el("option", { value: "candidate" }, t("ui.pi.status.candidate", "Candidate")),
        el("option", { value: "accepted" }, t("ui.pi.status.accepted", "Accepted")),
        el("option", { value: "rejected" }, t("ui.pi.status.rejected", "Rejected")),
        el("option", { value: "deprecated" }, t("ui.pi.status.deprecated", "Deprecated")),
        el("option", { value: "all" }, t("ui.option.all", "Alle")),
      ),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => loadPiMemories() }, t("ui.button.refresh", "Aktualisieren")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => exportPiMemories() }, t("ui.button.export", "Export")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => importPiMemories() }, t("ui.button.import", "Import")),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => dedupePiMemories() }, t("ui.button.dedupe", "Dedupe")),
      el("span", { class: "tc-text-muted tc-text-sm", id: "ai-pi-memory-status" }, t("ui.state.not_loaded", "nicht geladen")),
    ),
    el("div", { class: "tc-flex-col tc-gap-2", id: "ai-pi-memory-list" }),
  );

  const piAudit = el("div", { class: "tc-card", id: "ai-pi-audit" },
    el("div", { class: "tc-card-title" }, t("ui.title.pi_audit", "PI Audit")),
    el("div", { class: "tc-flex tc-gap-2 tc-items-center tc-mb-2" },
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => loadPiAudit() }, t("ui.button.refresh", "Aktualisieren")),
      el("span", { class: "tc-text-muted tc-text-sm", id: "ai-pi-audit-status" }, t("ui.state.not_loaded", "nicht geladen")),
    ),
    el("div", { class: "tc-flex-col tc-gap-2", id: "ai-pi-audit-list" }),
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

  page.append(scanCtx, modelCard, actions, recs, applyBar, piPreview, piMemories, piAudit, traffic);

  // Load models from backend
  loadModels();
  loadPiContext();
  loadPiMemories();
  loadPiAudit();

  // Restore loading state if analysis is in progress
  if (aiState.loading) {
    const recsContainer = document.getElementById("ai-recommendations");
    if (recsContainer) {
      recsContainer.innerHTML = "";
      recsContainer.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.toast.analysis_creating", "KI-Analyse wird erstellt...") + " " + t("ui.state.background_tab_ok", "(Tab-Wechsel OK, läuft im Hintergrund)")));
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

  // Live-update: subscribe to ai-state changes so traffic log and
  // recommendations update in real-time even after tab switches.
  // The subscription is cleaned up when the page is destroyed (clear).
  _aiUnsub?.();
  _aiUnsub = onAiChange((state) => {
    if (state.trafficLog) {
      renderTrafficLog(state.trafficLog);
    }
    if (state.loading) {
      const recsContainer = document.getElementById("ai-recommendations");
      if (recsContainer && recsContainer.children.length === 0) {
        recsContainer.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.toast.analysis_creating", "KI-Analyse wird erstellt...") + " " + t("ui.state.background_running", "(läuft im Hintergrund)")));
      }
    }
    if (state.currentAnalysis?.recommendations) {
      renderRecommendations(state.currentAnalysis.recommendations);
    }
  });

  return page;
}

let _allModels = [];
let _aiUnsub = null;
let _lastPiPreview = null;

async function loadPiContext() {
  try {
    await api.get(API_ENDPOINTS.pi.context);
  } catch {
    // PI context is optional for the AI tab.
  }
}

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
    select.appendChild(el("option", { value: currentModel, selected: true }, `${currentModel} ${t("ui.state.saved_parenthetical", "(gespeichert)")}`));
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
  setAiState({ loading: true, trafficLog: [], currentAnalysis: null });
  renderTrafficLog([]);
  const recsContainer = document.getElementById("ai-recommendations");
  if (recsContainer) {
    recsContainer.innerHTML = "";
    recsContainer.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.toast.analysis_creating", "KI-Analyse wird erstellt...")));
  }
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

    // Check cache first when force=false
    if (!force) {
      try {
        const cached = await api.get(API_ENDPOINTS.scan.analysisLatest);
        if (cached && cached.has_analysis && cached.recommendations?.length > 0) {
          addTrafficEntry({ type: "response", text: `Cache hit: ${cached.recommendations.length} recommendations` });
          renderRecommendations(cached.recommendations);
          setAiState({ currentAnalysis: cached, loading: false });
          toastSuccess(t("ui.toast.analysis_done", "Analyse aus Cache geladen"));
          return;
        }
      } catch {}
      addTrafficEntry({ type: "info", text: "No cached analysis, starting new..." });
    }

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
      force: true,
      scan_result: latestScan,
      model: fd.model || undefined,
    };
    if (scanMetrics) payload.scan_metrics = scanMetrics;
    if (configSchema) payload.config_schema = configSchema;
    if (baseConfig) payload.base_config = baseConfig;

    addTrafficEntry({ type: "request", text: `POST /api/scan/analysis with scan_result, ${scanMetrics ? "scan_metrics, " : ""}${configSchema ? "config_schema, " : ""}${baseConfig ? "base_config" : ""}` });

    addTrafficEntry({ type: "progress", text: "Initialisierung... 5%" });
    addTrafficEntry({ type: "progress", text: "Prompt wird gebaut... 10%" });
    addTrafficEntry({ type: "progress", text: "Warte auf KI-Antwort... 15%" });

    const result = await api.post(API_ENDPOINTS.scan.analysis, payload, { timeoutMs: 600000 });
    const recs = result?.validated_updates || result?.updates || result?.recommendations || [];
    addTrafficEntry({ type: "progress", text: "Antwort wird verarbeitet... 90%" });
    if (recs.length > 0) {
      addTrafficEntry({ type: "response", text: `Received ${recs.length} recommendations (cached: ${result.from_cache ?? false})` });
      if (result.summary) {
        addTrafficEntry({ type: "summary", text: result.summary });
      }
      addTrafficEntry({ type: "progress", text: "Analyse abgeschlossen 100%" });
      renderRecommendations(recs);
      setAiState({ currentAnalysis: result, loading: false });
      toastSuccess(t("ui.toast.analysis_done", "Analyse erstellt"));
    } else {
      addTrafficEntry({ type: "error", text: `${t("ui.error.no_recommendations_keys", "No recommendations in response. Keys:")} ${Object.keys(result || {}).join(", ")}` });
      setAiState({ loading: false });
      toastError(t("ui.toast.analysis_failed", "Analyse fehlgeschlagen"), t("ui.error.no_recommendations_returned", "No recommendations returned"));
    }
  } catch (e) {
    addTrafficEntry({ type: "error", text: e.message });
    setAiState({ loading: false });
    toastError(t("ui.toast.analysis_failed", "Analyse fehlgeschlagen"), e.message);
  }
}

async function pollJobResult(jobId, timeoutMs = 600000) {
  return pollJob(jobId, {
    endpoint: API_ENDPOINTS.scan.jobStatus,
    timeoutMs,
    onDone: (job) => job?.data?.result || job?.data || null,
  });
}

function renderRecommendations(recs) {
  const container = document.getElementById("ai-recommendations");
  if (!container) return;
  container.innerHTML = "";
  for (const rec of recs) {
    const item = el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
      el("label", { class: "tc-checkbox" },
        el("input", { type: "checkbox", checked: rec.selected !== false,
          onchange: (e) => {
            rec.selected = e.target.checked;
            clearPiPreview();
          },
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

function selectedRecommendations(all = false) {
  const { currentAnalysis } = getAiState();
  const recs = currentAnalysis?.validated_updates || currentAnalysis?.updates || currentAnalysis?.recommendations || [];
  return all ? recs : recs.filter(r => r.selected !== false);
}

function actionPlanFromCurrentAnalysis(selected) {
  const { currentAnalysis } = getAiState();
  if (!currentAnalysis) return null;
  if (currentAnalysis.action_plan && Array.isArray(currentAnalysis.action_plan.actions)) {
    const selectedPaths = new Set(selected.map((r) => String(r.path || "")).filter(Boolean));
    const plan = JSON.parse(JSON.stringify(currentAnalysis.action_plan));
    plan.actions = plan.actions.flatMap((action) => {
      if (action?.type === "config.set") return selectedPaths.has(String(action.path || "")) ? [action] : [];
      if (action?.type === "config.patch" && Array.isArray(action.updates)) {
        const updates = action.updates.filter((u) => selectedPaths.has(String(u?.path || "")));
        return updates.length ? [{ ...action, updates }] : [];
      }
      return [action];
    });
    return plan;
  }
  const actions = selected
    .filter((rec) => rec?.path && Object.prototype.hasOwnProperty.call(rec, "value"))
    .map((rec, index) => ({
      id: `gui3_ai_update_${index + 1}`,
      type: "config.set",
      path: String(rec.path || ""),
      value: rec.value,
      rationale: String(rec.reason || rec.rationale || "validated AI recommendation"),
    }));
  if (!actions.length) return null;
  return {
    schema_version: "pi.action-plan.v1",
    goal: String(currentAnalysis.summary || "Apply selected AI recommendations"),
    confidence: Number.isFinite(Number(currentAnalysis.confidence)) ? Number(currentAnalysis.confidence) : 0,
    actions,
    post_conditions: [{ type: "config.valid" }],
    warnings: [],
  };
}

function clearPiPreview() {
  _lastPiPreview = null;
  const applyButton = document.getElementById("ai-pi-apply");
  if (applyButton) applyButton.disabled = true;
}

function renderPiPreview(preview) {
  const container = document.getElementById("ai-pi-preview");
  if (!container) return;
  container.innerHTML = "";
  const valid = Boolean(preview?.config_valid);
  container.appendChild(el("div", { class: "tc-card-title" }, t("ui.title.pi_preview", "PI Preview")));
  container.appendChild(el("div", { class: valid ? "tc-text-success tc-text-sm" : "tc-text-error tc-text-sm" },
    valid ? t("ui.state.config_valid", "Config gültig") : t("ui.state.config_invalid", "Config ungültig"),
  ));
  container.appendChild(createYamlDiff(preview?.base_config || {}, preview?.patched_config || {}));
}

async function previewPiActionPlan() {
  const selected = selectedRecommendations(false);
  if (!selected.length) {
    toastError(t("ui.toast.preview_failed", "Preview fehlgeschlagen"), t("ui.state.no_selection", "Keine Empfehlung ausgewählt."));
    return;
  }
  const plan = actionPlanFromCurrentAnalysis(selected);
  if (!plan) {
    toastError(t("ui.toast.preview_failed", "Preview fehlgeschlagen"), t("ui.error.no_action_plan", "No action plan available"));
    return;
  }
  try {
    const currentConfig = await api.get(API_ENDPOINTS.config.current);
    const result = await api.post(API_ENDPOINTS.pi.actionPlanPreview, {
      plan,
      yaml: currentConfig?.config || "",
    });
    _lastPiPreview = { plan, preview: result.preview };
    renderPiPreview(result.preview);
    const applyButton = document.getElementById("ai-pi-apply");
    if (applyButton) applyButton.disabled = !result.preview?.config_valid;
    toastSuccess(t("ui.toast.preview_done", "Preview erstellt"));
  } catch (e) {
    clearPiPreview();
    toastError(t("ui.toast.preview_failed", "Preview fehlgeschlagen"), e.message);
  }
}

async function applyPiActionPlan() {
  if (!_lastPiPreview?.plan || !_lastPiPreview?.preview?.config_valid) {
    toastError(t("ui.toast.apply_failed", "Anwenden fehlgeschlagen"), t("ui.state.preview_required", "Erst PI Preview ausführen."));
    return;
  }
  if (!window.confirm(t("ui.confirm.pi_apply", "PI Preview als neue Config-Revision speichern?"))) return;
  try {
    const result = await api.post(API_ENDPOINTS.pi.actionPlanApply, {
      plan: _lastPiPreview.plan,
      confirmed: true,
      expected_patched_yaml: _lastPiPreview.preview.patched_yaml || "",
      base_config: _lastPiPreview.preview.base_config || {},
    });
    clearPiPreview();
    toastSuccess(`${t("ui.toast.applied", "Angewendet")} ${result?.revision_id || ""}`.trim());
    await loadPiAudit();
  } catch (e) {
    toastError(t("ui.toast.apply_failed", "Anwenden fehlgeschlagen"), e.message);
  }
}

async function applyRecommendations(all = false) {
  const { currentAnalysis } = getAiState();
  if (!currentAnalysis) return;
  const selected = selectedRecommendations(all);
  if (!selected.length) return;
  try {
    const learn = Boolean(document.getElementById("ai-learn-memory")?.checked);
    const result = await api.post(API_ENDPOINTS.scan.analysisApply, {
      analysis_id: currentAnalysis.analysis_id || currentAnalysis.job_id || "",
      selected_paths: selected.map((r) => r.path).filter(Boolean),
      recommendations: selected,
      learn,
    });
    const memory = result?.memory?.created ? `, ${t("ui.pi.memory", "Memory")} ${result.memory.memory_id}` : "";
    toastSuccess(`${t("ui.toast.applied", "Empfehlungen angewendet")}${memory}`);
    await loadPiMemories();
    await loadPiAudit();
  } catch (e) {
    toastError(t("ui.toast.apply_failed", "Anwenden fehlgeschlagen"), e.message);
  }
}

function summarizeMemory(memory) {
  const paths = [];
  const visit = (value) => {
    if (!value || typeof value !== "object") return;
    if (Array.isArray(value)) return value.forEach(visit);
    if (typeof value.path === "string") paths.push(value.path);
    Object.values(value).forEach(visit);
  };
  visit(memory);
  return `${memory?.type || "memory"} | ${memory?.status || "candidate"}${paths.length ? ` | ${Array.from(new Set(paths)).slice(0, 4).join(", ")}` : ""}`;
}

function memoryStatusLabel(memory) {
  const status = String(memory?.status || "candidate");
  const reviewed = memory?.review?.reviewed_at ? ` · ${memory.review.reviewed_at}` : "";
  if (status === "accepted") return `${t("ui.pi.status.accepted", "Accepted")}${reviewed}`;
  if (status === "rejected") return `${t("ui.pi.status.rejected", "Rejected")}${reviewed}`;
  if (status === "deprecated") return `${t("ui.pi.status.deprecated", "Deprecated")}${reviewed}`;
  return t("ui.pi.status.candidate", "Candidate");
}

function memoryOutcomeSummary(memory) {
  const outcome = memory?.outcome;
  if (!outcome || typeof outcome !== "object") return "";
  const valid = outcome.validation_valid === true
    ? t("ui.state.valid", "valid")
    : outcome.validation_valid === false ? t("ui.state.invalid", "invalid") : t("ui.state.unknown", "unknown");
  const count = Number.isFinite(Number(outcome.applied_count)) ? Number(outcome.applied_count) : 0;
  const paths = Array.isArray(outcome.applied_paths) ? outcome.applied_paths.slice(0, 3).join(", ") : "";
  return `${t("ui.pi.outcome", "Outcome")}: ${valid}, ${t("ui.pi.update_count", "{count} Updates", { count })}${paths ? ` · ${paths}` : ""}`;
}

async function loadPiMemories() {
  const filter = document.getElementById("ai-pi-memory-filter")?.value || "candidate";
  const status = document.getElementById("ai-pi-memory-status");
  const list = document.getElementById("ai-pi-memory-list");
  if (status) status.textContent = t("ui.state.loading", "Lädt...");
  try {
    const payload = await api.get(`${API_ENDPOINTS.pi.memories}?limit=100&status=${encodeURIComponent(filter)}`);
    if (list) {
      list.innerHTML = "";
      const items = Array.isArray(payload?.items) ? payload.items : [];
      for (const memory of items) {
        const id = memory.memory_id || "";
        const statusName = String(memory.status || "candidate");
        const canReview = statusName === "candidate" || statusName === "accepted";
        list.appendChild(el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
          el("div", { class: "tc-flex tc-justify-between tc-gap-2" },
            el("div", { class: "tc-mono tc-text-sm" }, id || "-"),
            el("span", { class: "tc-badge" }, memoryStatusLabel(memory)),
          ),
          el("div", { class: "tc-text-sm tc-text-muted" }, summarizeMemory(memory)),
          memoryOutcomeSummary(memory) ? el("div", { class: "tc-text-sm tc-text-muted" }, memoryOutcomeSummary(memory)) : null,
          memory?.review?.note ? el("div", { class: "tc-text-sm" }, memory.review.note) : null,
          el("div", { class: "tc-flex tc-gap-2 tc-mt-2" },
            canReview && statusName !== "accepted" ? el("button", { class: "tc-btn tc-btn-sm", onclick: () => reviewPiMemory(id, "accepted") }, t("ui.button.accept", "Accept")) : null,
            canReview ? el("button", { class: "tc-btn tc-btn-sm", onclick: () => reviewPiMemory(id, "rejected") }, t("ui.button.reject", "Reject")) : null,
            canReview ? el("button", { class: "tc-btn tc-btn-sm", onclick: () => reviewPiMemory(id, "deprecated") }, t("ui.button.deprecate", "Deprecate")) : null,
          ),
        ));
      }
      if (!items.length) list.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_data", "Keine Daten")));
    }
    if (status) status.textContent = t("ui.pi.memory_count", "{count} Memories", { count: payload?.count || 0 });
  } catch (e) {
    if (status) status.textContent = e.message;
  }
}

async function reviewPiMemory(memoryId, reviewStatus) {
  if (!memoryId) return;
  try {
    await api.post(API_ENDPOINTS.pi.memoryReview(memoryId), {
      status: reviewStatus,
      reviewer: "gui3",
    });
    await loadPiMemories();
    await loadPiAudit();
    toastSuccess(t("ui.toast.saved", "Gespeichert"));
  } catch (e) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
  }
}

async function exportPiMemories() {
  try {
    const bundle = await api.get(`${API_ENDPOINTS.pi.memoriesExport}?privacy=metadata_only&include_reviews=1`);
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tile_compile_pi_memories_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toastSuccess(t("ui.toast.saved", "Gespeichert"));
  } catch (e) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
  }
}

async function importPiMemories() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "application/json,.json";
  input.style.display = "none";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) {
      input.remove();
      return;
    }
    try {
      const text = await file.text();
      const bundle = JSON.parse(text);
      const result = await api.post(API_ENDPOINTS.pi.memoriesImport, { bundle, dry_run: false });
      await loadPiMemories();
      await loadPiAudit();
      toastSuccess(t("ui.pi.import_result", "Import: {count} Memories", { count: result.imported_memories || 0 }));
    } catch (e) {
      toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
    } finally {
      input.remove();
    }
  };
  document.body.appendChild(input);
  input.click();
}

async function dedupePiMemories() {
  if (!window.confirm(t("ui.confirm.pi_dedupe", "PI Memories deduplizieren?"))) return;
  try {
    const result = await api.post(API_ENDPOINTS.pi.memoriesDedupe, { dry_run: false });
    await loadPiMemories();
    await loadPiAudit();
    toastSuccess(t("ui.pi.dedupe_result", "Dedupe: {count} entfernt", { count: result.removed_count || 0 }));
  } catch (e) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
  }
}

function summarizeAuditItem(item) {
  if (item?.audit_type === "memory_review") {
    return `${item.memory_id || "-"} · ${item.status || "-"} · ${item.review?.reviewer || "user"}`;
  }
  return `${item?.event || item?.audit_type || t("ui.title.pi_audit", "PI Audit")} · ${item?.source || ""}`;
}

async function loadPiAudit() {
  const status = document.getElementById("ai-pi-audit-status");
  const list = document.getElementById("ai-pi-audit-list");
  if (status) status.textContent = t("ui.state.loading", "Lädt...");
  try {
    const payload = await api.get(`${API_ENDPOINTS.pi.audit}?limit=200`);
    if (list) {
      list.innerHTML = "";
      const items = Array.isArray(payload?.items) ? payload.items.slice(-50).reverse() : [];
      for (const item of items) {
        list.appendChild(el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
          el("div", { class: "tc-flex tc-justify-between tc-gap-2" },
            el("div", { class: "tc-text-sm" }, summarizeAuditItem(item)),
            el("span", { class: "tc-badge" }, item.audit_type || t("ui.title.pi_audit", "PI Audit")),
          ),
          el("div", { class: "tc-text-sm tc-text-muted" }, item.ts || item.review?.reviewed_at || item.created_at || ""),
        ));
      }
      if (!items.length) list.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_data", "Keine Daten")));
    }
    if (status) status.textContent = t("ui.pi.audit_count", "{count} Audit Items", { count: payload?.count || 0 });
  } catch (e) {
    if (status) status.textContent = e.message;
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
      toastError(t("ui.toast.analysis_load_failed", "Analyse laden fehlgeschlagen"), t("ui.error.analysis_not_found", "Analysis not found"));
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
