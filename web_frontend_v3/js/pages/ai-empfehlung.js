// js/pages/ai-empfehlung.js – Sub-Tab: AI Empfehlung (innerhalb Parameter)

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { getAiState, setAiState, getAiFormData, setAiFormData, onAiChange } from "../state/ai-state.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";
import { pollJob } from "../utils/poll.js";
import { createYamlDiff } from "../components/yaml-diff.js";
import { getScanData, getQueueItems, getCalValues } from "./input-scan.js";
import { getScanState } from "../state/scan-state.js";

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
        el("select", { class: "tc-select", id: "ai-mount", title: t("ui.tooltip.ai.mount", "Montierungstyp fuer die Analyse einordnen."), onchange: (e) => updateAiUiConfig({ mount: e.target.value }) },
          ...["EQ", "Tracker", "Alt/Az"].map(v => el("option", { value: v, ...(fd.mount === v ? { selected: true } : {}) }, v)),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.object_type", "Objekt")),
        el("select", { class: "tc-select", id: "ai-object-type", title: t("ui.tooltip.ai.object_type", "Objekttyp hilft der KI, typische Artefakte richtig zu bewerten."), onchange: (e) => updateAiUiConfig({ object_type: e.target.value }) },
          ...["Galaxie", "Nebel", "Sternhaufen", "Sterne"].map(v => el("option", { value: v, ...(fd.object_type === v ? { selected: true } : {}) }, v)),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.camera", "Kamera")),
        el("select", { class: "tc-select", id: "ai-camera", title: t("ui.tooltip.ai.camera", "Kameratyp beeinflusst Empfehlungen fuer Farbe, Debayer und Rauschen."), onchange: (e) => updateAiUiConfig({ camera: e.target.value }) },
          ...["Consumer OSC", "Mono CMOS", "CCD"].map(v => el("option", { value: v, ...(fd.camera === v ? { selected: true } : {}) }, v)),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.calibration", "Kalibrierung")),
        el("div", { class: "tc-flex tc-gap-3" },
          el("label", { class: "tc-checkbox", title: t("ui.tooltip.ai.darks", "Aktivieren, wenn Dark-Frames in der Kalibrierung genutzt wurden.") }, el("input", { type: "checkbox", id: "ai-calibration-darks", checked: fd.calibration_darks, title: t("ui.tooltip.ai.darks", "Aktivieren, wenn Dark-Frames in der Kalibrierung genutzt wurden."), onchange: (e) => updateAiUiConfig({ calibration_darks: e.target.checked }) }), t("ui.label.darks", "Darks")),
          el("label", { class: "tc-checkbox", title: t("ui.tooltip.ai.flats", "Aktivieren, wenn Flats gegen Vignettierung/Staub genutzt wurden.") }, el("input", { type: "checkbox", id: "ai-calibration-flats", checked: fd.calibration_flats, title: t("ui.tooltip.ai.flats", "Aktivieren, wenn Flats gegen Vignettierung/Staub genutzt wurden."), onchange: (e) => updateAiUiConfig({ calibration_flats: e.target.checked }) }), t("ui.label.flats", "Flats")),
          el("label", { class: "tc-checkbox", title: t("ui.tooltip.ai.bias", "Aktivieren, wenn Bias-Frames in der Kalibrierung genutzt wurden.") }, el("input", { type: "checkbox", id: "ai-calibration-bias", checked: fd.calibration_bias, title: t("ui.tooltip.ai.bias", "Aktivieren, wenn Bias-Frames in der Kalibrierung genutzt wurden."), onchange: (e) => updateAiUiConfig({ calibration_bias: e.target.checked }) }), t("ui.label.bias", "Bias")),
        ),
      ),
    ),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.field.notes", "Notizen")),
      el("input", { type: "text", class: "tc-input", id: "ai-notes", value: fd.notes, title: t("ui.tooltip.ai.notes", "Freitext fuer Besonderheiten, die nicht automatisch aus dem Scan erkennbar sind."), placeholder: "Guiding 0.8\", M31, alt-az test", oninput: (e) => updateAiUiConfig({ notes: e.target.value }, true) }),
    ),
  );

  // Model & API Key
  const modelCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.model_api", "Modell & API-Key")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.provider", "Provider")),
        el("select", { class: "tc-select", id: "ai-provider", title: t("ui.tooltip.ai.provider", "AI-Anbieter fuer Analyse und Empfehlungen."), onchange: (e) => onAiProviderChange(e.target.value) },
          ...["anthropic", "openai"].map(v => el("option", { value: v, ...(fd.provider === v ? { selected: true } : {}) }, v)),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.model", "Modell")),
        el("select", { class: "tc-select", id: "ai-model", title: t("ui.tooltip.ai.model", "Modell, das die Scan-Daten und Config bewertet."), onchange: (e) => onAiModelChange(e.target.value) },
          el("option", { value: fd.model || "", selected: true }, fd.model || t("ui.placeholder.select_model", "Modell wählen")),
        ),
      ),
    ),
    el("div", { class: "tc-mt-2 tc-flex tc-items-center tc-gap-2" },
      el("input", { type: "password", class: "tc-input", style: { flex: "1 1 auto", minWidth: "0" }, value: fd.apiKey, title: t("ui.tooltip.ai.api_key", "API-Key fuer den ausgewaehlten Provider. Er wird nicht in Tile-Compile-Configs gespeichert."), placeholder: "API-Key", id: "ai-apikey", oninput: (e) => setAiFormData({ apiKey: e.target.value }) }),
      el("button", { class: "tc-btn", style: { flexShrink: "0" }, title: t("ui.tooltip.ai.save_key", "Speichert den Key im PI AuthStorage fuer diesen Provider."), onclick: () => saveApiKey() }, t("ui.button.save_key", "Key speichern")),
      el("span", { class: "tc-badge tc-badge-success tc-hidden", style: { flexShrink: "0", whiteSpace: "nowrap" }, id: "ai-key-status" }, "\u2713 " + t("ui.state.saved", "gespeichert")),
    ),
    el("div", { class: "tc-mt-2" },
      el("span", { class: "tc-text-sm tc-text-muted", id: "ai-model-status" }, t("ui.state.model_loading", "Modelle werden geladen...")),
    ),
    el("div", { class: "tc-text-sm tc-text-muted", id: "ai-pi-version-status" }, t("ui.state.not_loaded", "nicht geladen")),
    el("div", { class: "tc-mt-2 tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.label.image_capability", "Bildfähigkeit")),
        el("select", {
          class: "tc-select",
          id: "ai-vision-override",
          title: t("ui.tooltip.ai.vision_override", "Automatische Bildfähigkeits-Erkennung verwenden oder für dieses Modell manuell überschreiben."),
          onchange: () => saveVisionOverride(),
        },
          el("option", { value: "" }, t("ui.option.auto_detect", "Automatisch")),
          el("option", { value: "true" }, t("ui.option.force_enabled", "Ja erzwingen")),
          el("option", { value: "false" }, t("ui.option.force_disabled", "Nein erzwingen")),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.label.vision_status", "Vision-Status")),
        el("div", { class: "tc-text-sm tc-text-muted", id: "ai-vision-status" }, t("ui.state.not_loaded", "nicht geladen")),
      ),
    ),
    el("div", { class: "tc-mt-2 tc-flex tc-items-center tc-gap-2" },
      el("div", { class: "tc-text-sm tc-text-muted", id: "ai-account-status", style: { flex: "1 1 auto", minWidth: "0" } },
        t("ui.state.account_loading", "Kontostatus wird geladen..."),
      ),
      el("button", { class: "tc-btn", style: { flexShrink: "0" }, title: t("ui.tooltip.ai.refresh_account_status", "Prueft Provider-Key und ausgewaehltes Modell ueber den PI Sidecar."), onclick: () => refreshAiProviderStatus() },
        t("ui.button.refresh_account_status", "Status abrufen"),
      ),
    ),
  );

  // Actions
  const historySelect = el("select", { class: "tc-select", style: { width: "200px" }, id: "ai-history-select", title: t("ui.tooltip.ai.saved_analyses", "Laedt eine gespeicherte Analyse aus dem Verlauf."),
    onchange: (e) => loadSavedAnalysis(e.target.value),
  },
    el("option", { value: "" }, t("ui.placeholder.saved_analyses", "Gespeicherte Analysen")),
  );

  const actions = el("div", { class: "tc-flex tc-gap-3 tc-flex-wrap" },
    el("button", { class: "tc-btn tc-btn-primary", title: t("ui.tooltip.ai.create_analysis", "Analysiert den letzten Scan oder startet automatisch einen Scan, wenn ein Input-Ordner gesetzt ist."), onclick: () => createAnalysis() }, t("ui.button.create_analysis", "KI-Analyse erstellen")),
    el("button", { class: "tc-btn", title: t("ui.tooltip.ai.reanalyze", "Erstellt eine neue Analyse und ignoriert gespeicherte Analyse-Ergebnisse."), onclick: () => createAnalysis(true) }, t("ui.button.reanalyze", "Neu analysieren (Cache ignorieren)")),
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
  const learnMemory = el("label", { class: "tc-checkbox", title: t("ui.tooltip.ai.learn_memory", "Speichert angewendete Optimierungen als reviewbare PI Memory-Kandidaten.") },
    el("input", { type: "checkbox", id: "ai-learn-memory", title: t("ui.tooltip.ai.learn_memory", "Speichert angewendete Optimierungen als reviewbare PI Memory-Kandidaten.") }),
    el("span", {}, t("ui.label.pi_learn_memory", "Lernkandidat speichern")),
  );
  const applyBar = el("div", { class: "tc-flex tc-gap-3" },
    learnMemory,
    el("button", { class: "tc-btn", title: t("ui.tooltip.ai.pi_preview", "Zeigt geplante Config-Aenderungen als validierten YAML-Diff, ohne zu speichern."), onclick: () => previewPiActionPlan() }, t("ui.button.pi_preview", "PI Preview")),
    el("button", { class: "tc-btn", id: "ai-pi-apply", disabled: true, title: t("ui.tooltip.ai.pi_apply", "Speichert die letzte gueltige PI Preview als neue Config-Revision."), onclick: () => applyPiActionPlan() }, t("ui.button.pi_apply", "PI anwenden")),
    el("button", { class: "tc-btn tc-btn-primary", title: t("ui.tooltip.ai.apply_selected", "Wendet nur markierte Empfehlungen auf die Config an."), onclick: () => applyRecommendations() }, t("ui.button.apply_selected", "Ausgewaehlte anwenden")),
    el("button", { class: "tc-btn", title: t("ui.tooltip.ai.apply_all", "Wendet alle Empfehlungen aus der aktuellen Analyse an."), onclick: () => applyRecommendations(true) }, t("ui.button.apply_all", "Alle anwenden")),
    el("button", { class: "tc-btn", title: t("ui.tooltip.ai.discard", "Verwirft die aktuelle Analyseanzeige ohne Config-Aenderung."), onclick: () => discardRecommendations() }, t("ui.button.discard", "Verwerfen")),
  );

  const piPreview = el("div", { class: "tc-card", id: "ai-pi-preview" },
    el("div", { class: "tc-card-title" }, t("ui.title.pi_preview", "PI Preview")),
    el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_preview", "Noch keine Preview.")),
  );

  const piStorage = el("div", { class: "tc-card", id: "ai-pi-storage" },
    el("div", { class: "tc-card-title" }, t("ui.title.pi_storage", "PI Storage")),
    el("div", { class: "tc-flex tc-items-center tc-gap-2" },
      el("input", {
        type: "text",
        class: "tc-input tc-mono",
        id: "ai-pi-storage-dir",
        style: { flex: "1 1 auto", minWidth: "0" },
        title: t("ui.tooltip.ai.pi_storage_dir", "Ordner fuer PI Memories, Reviews, Audit-Kontext und kuenftige PI-Objekte."),
        placeholder: t("ui.placeholder.pi_storage_dir", "z.B. /media/tc_500/.pi_memory"),
      }),
      el("button", {
        class: "tc-btn",
        style: { flexShrink: "0" },
        title: t("ui.tooltip.ai.save_pi_storage", "Speichert diesen PI-Speicherort persistent."),
        onclick: () => savePiStorage(),
      }, t("ui.button.save", "Speichern")),
      el("button", {
        class: "tc-btn",
        style: { flexShrink: "0" },
        title: t("ui.tooltip.ai.refresh_pi_storage", "Laedt den aktiven PI-Speicherort neu."),
        onclick: () => loadPiStorage(),
      }, t("ui.button.refresh", "Aktualisieren")),
    ),
    el("div", { class: "tc-text-muted tc-text-sm tc-mt-2", id: "ai-pi-storage-status" }, t("ui.state.not_loaded", "nicht geladen")),
  );

  const piMemories = el("div", { class: "tc-card", id: "ai-pi-memories" },
    el("div", { class: "tc-card-title" }, t("ui.title.pi_memories", "PI Memories")),
    el("div", { class: "tc-flex tc-gap-2 tc-items-center tc-mb-2" },
      el("select", { class: "tc-select", id: "ai-pi-memory-filter", title: t("ui.tooltip.ai.memory_filter", "Filtert PI Memories nach Review-Status."), onchange: () => loadPiMemories() },
        el("option", { value: "candidate" }, t("ui.pi.status.candidate", "Candidate")),
        el("option", { value: "promotable" }, t("ui.pi.status.promotable", "Promotable")),
        el("option", { value: "accepted" }, t("ui.pi.status.accepted", "Accepted")),
        el("option", { value: "rejected" }, t("ui.pi.status.rejected", "Rejected")),
        el("option", { value: "deprecated" }, t("ui.pi.status.deprecated", "Deprecated")),
        el("option", { value: "all" }, t("ui.option.all", "Alle")),
      ),
      el("button", { class: "tc-btn tc-btn-sm", title: t("ui.tooltip.ai.refresh_memories", "Laedt die PI Memory-Liste neu."), onclick: () => loadPiMemories() }, t("ui.button.refresh", "Aktualisieren")),
      el("button", { class: "tc-btn tc-btn-sm", title: t("ui.tooltip.ai.export_memories", "Exportiert PI Memories ohne Bilddaten."), onclick: () => exportPiMemories() }, t("ui.button.export", "Export")),
      el("button", { class: "tc-btn tc-btn-sm", title: t("ui.tooltip.ai.import_memories", "Importiert ein PI Memory-Bundle."), onclick: () => importPiMemories() }, t("ui.button.import", "Import")),
      el("button", { class: "tc-btn tc-btn-sm", title: t("ui.tooltip.ai.dedupe_memories", "Entfernt doppelte PI Memory-Kandidaten nach Bestaetigung."), onclick: () => dedupePiMemories() }, t("ui.button.dedupe", "Dedupe")),
      el("span", { class: "tc-text-muted tc-text-sm", id: "ai-pi-memory-status" }, t("ui.state.not_loaded", "nicht geladen")),
    ),
    el("div", { class: "tc-text-muted tc-text-sm tc-mono tc-mb-2", id: "ai-pi-memory-dir" }, ""),
    el("div", { class: "tc-flex-col tc-gap-2", id: "ai-pi-memory-list" }),
  );

  const piAudit = el("div", { class: "tc-card", id: "ai-pi-audit" },
    el("div", { class: "tc-card-title" }, t("ui.title.pi_audit", "PI Audit")),
    el("div", { class: "tc-flex tc-gap-2 tc-items-center tc-mb-2" },
      el("button", { class: "tc-btn tc-btn-sm", title: t("ui.tooltip.ai.refresh_audit", "Laedt PI Audit-Eintraege neu."), onclick: () => loadPiAudit() }, t("ui.button.refresh", "Aktualisieren")),
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
      el("div", { class: "tc-flex tc-gap-2 tc-items-center tc-mb-2" },
        el("button", {
          class: "tc-btn tc-btn-sm",
          title: t("ui.tooltip.ai.refresh_traffic", "Laedt den persistenten PI/KI-Traffic-Log aus dem Sidecar."),
          onclick: () => loadPersistentTrafficLog(),
        }, t("ui.button.refresh", "Aktualisieren")),
        el("span", { class: "tc-text-muted tc-text-sm", id: "ai-traffic-status" }, t("ui.state.not_loaded", "nicht geladen")),
      ),
      el("div", { class: "tc-log-viewer", id: "ai-traffic-log" },
        el("div", { class: "tc-text-muted" }, t("ui.state.no_traffic", "Keine Daten")),
      ),
    ),
  );

  page.append(scanCtx, modelCard, actions, recs, applyBar, piPreview, piStorage, piMemories, piAudit, traffic);

  // Load persisted provider/model before loading the model registry.
  loadAiConfig().finally(() => loadModels());
  loadPiContext();
  loadPiStorage();
  loadPiMemories();
  loadPiAudit();
  loadAiAccountStatus(fd.provider);

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
  if (analysisMatchesCurrentScope(aiState.currentAnalysis)) {
    renderRecommendations(analysisRecommendations(aiState.currentAnalysis));
  } else if (aiState.currentAnalysis) {
    setAiState({ currentAnalysis: null });
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
    if (analysisMatchesCurrentScope(state.currentAnalysis)) {
      renderRecommendations(analysisRecommendations(state.currentAnalysis));
    }
  });

  return page;
}

let _allModels = [];
let _aiUnsub = null;
let _lastPiPreview = null;
let _aiUiPersistTimer = null;

function persistentAiUiFromForm(form = getAiFormData()) {
  const scanData = getScanData();
  return {
    mount: String(form.mount || "EQ"),
    object_type: String(form.object_type || "Galaxie"),
    target_name: String(scanData.object_name || ""),
    camera: String(form.camera || "Consumer OSC"),
    calibration_darks: Boolean(form.calibration_darks),
    calibration_flats: Boolean(form.calibration_flats),
    calibration_bias: Boolean(form.calibration_bias),
    notes: String(form.notes || ""),
  };
}

function normalizedObjectName() {
  return String(getScanData().object_name || "").trim().replace(/\s+/g, " ").toUpperCase();
}

function scanScopeFromScan(scan) {
  const source = scan && typeof scan === "object" ? scan : {};
  return {
    input_path: String(source.input_path || source.input_dir || source.input_dirs?.[0] || getScanData().input_dir || "").trim(),
    frame_count: Number(source.frames_detected || source.frames_total || source.frame_count || 0) || 0,
    object_name: normalizedObjectName(),
  };
}

function analysisScopeFromAnalysis(analysis) {
  if (!analysis || typeof analysis !== "object") return null;
  const meta = analysis.scan_metadata && typeof analysis.scan_metadata === "object"
    ? analysis.scan_metadata
    : analysis.analysis_context?.scan_metadata || {};
  const session = analysis.analysis_context?.session_context || {};
  return {
    input_path: String(meta.input_path || "").trim(),
    frame_count: Number(meta.frame_count || meta.frames_detected || meta.frames_total || 0) || 0,
    object_name: String(meta.object_name || meta.target || meta.object || session.target_name || analysis.analysis_scope?.object_name || "")
      .trim().replace(/\s+/g, " ").toUpperCase(),
  };
}

function sameAnalysisScope(analysis, scan) {
  const a = analysisScopeFromAnalysis(analysis);
  if (!a) return false;
  const b = scanScopeFromScan(scan);
  if (a.input_path && b.input_path && a.input_path !== b.input_path) return false;
  if (a.frame_count && b.frame_count && a.frame_count !== b.frame_count) return false;
  if (a.object_name || b.object_name) return a.object_name === b.object_name;
  return Boolean(a.input_path || a.frame_count);
}

function analysisMatchesCurrentScope(analysis) {
  const recs = analysisRecommendations(analysis);
  if (!analysis || !recs.length) return false;
  return sameAnalysisScope(analysis, getScanState().lastScan || {});
}

function analysisRecommendations(analysis) {
  return analysis?.validated_updates || analysis?.updates || analysis?.recommendations || [];
}

function syncAiControlsFromForm() {
  const fd = getAiFormData();
  const setValue = (id, value) => {
    const node = document.getElementById(id);
    if (node) node.value = value;
  };
  const setChecked = (id, value) => {
    const node = document.getElementById(id);
    if (node) node.checked = Boolean(value);
  };
  setValue("ai-mount", fd.mount || "EQ");
  setValue("ai-object-type", fd.object_type || "Galaxie");
  setValue("ai-camera", fd.camera || "Consumer OSC");
  setValue("ai-notes", fd.notes || "");
  setChecked("ai-calibration-darks", fd.calibration_darks);
  setChecked("ai-calibration-flats", fd.calibration_flats);
  setChecked("ai-calibration-bias", fd.calibration_bias);
}

async function loadPiContext() {
  try {
    await api.get(API_ENDPOINTS.pi.context);
  } catch {
    // PI context is optional for the AI tab.
  }
}

function ensureProviderOption(provider) {
  const select = document.getElementById("ai-provider");
  if (!select || !provider) return;
  const exists = Array.from(select.options).some(option => option.value === provider);
  if (!exists) select.appendChild(el("option", { value: provider }, provider));
}

function renderProviderOptions(providers) {
  const select = document.getElementById("ai-provider");
  if (!select) return;
  const fd = getAiFormData();
  const current = select.value || fd.provider || "";
  const names = new Set(["anthropic", "openai"]);
  for (const p of Array.isArray(providers) ? providers : []) {
    const providerName = String(p?.provider || "").trim();
    if (providerName) names.add(providerName);
  }
  if (current) names.add(current);
  select.innerHTML = "";
  for (const provider of Array.from(names).sort()) {
    select.appendChild(el("option", { value: provider, ...(provider === current ? { selected: true } : {}) }, provider));
  }
  if (current) select.value = current;
}

async function loadAiConfig() {
  try {
    const config = await api.get(API_ENDPOINTS.ai.config);
    if (!config || config.available === false) return;
    const provider = String(config.provider || "").trim();
    const model = String(config.model || "").trim();
    const ui = config.ui && typeof config.ui === "object" ? config.ui : {};
    setAiState({ config });
    const patch = {};
    if (provider) patch.provider = provider;
    if (model) patch.model = model;
    for (const key of ["mount", "object_type", "camera", "notes"]) {
      if (typeof ui[key] === "string") patch[key] = ui[key];
    }
    for (const key of ["calibration_darks", "calibration_flats", "calibration_bias"]) {
      if (typeof ui[key] === "boolean") patch[key] = ui[key];
    }
    if (Object.keys(patch).length > 0) setAiFormData(patch);
    syncAiControlsFromForm();
    ensureProviderOption(provider);
    const providerSelect = document.getElementById("ai-provider");
    const modelSelect = document.getElementById("ai-model");
    if (providerSelect && provider) providerSelect.value = provider;
    if (modelSelect && model) {
      modelSelect.innerHTML = "";
      modelSelect.appendChild(el("option", { value: model, selected: true }, model));
    }
  } catch {
    // Defaults from local UI state remain usable when config loading fails.
  }
}

async function persistAiUiConfig(uiPatch = {}) {
  const ui = { ...persistentAiUiFromForm(), ...uiPatch };
  try {
    const saved = await api.patch(API_ENDPOINTS.ai.config, { ui });
    setAiState({ config: saved });
    return saved;
  } catch (e) {
    toastError(t("ui.toast.ai_config_save_failed", "AI-Auswahl speichern fehlgeschlagen"), e.message);
    return null;
  }
}

function updateAiUiConfig(patch, debounce = false) {
  setAiFormData(patch);
  const ui = persistentAiUiFromForm({ ...getAiFormData(), ...patch });
  if (_aiUiPersistTimer) {
    clearTimeout(_aiUiPersistTimer);
    _aiUiPersistTimer = null;
  }
  if (debounce) {
    _aiUiPersistTimer = setTimeout(() => {
      _aiUiPersistTimer = null;
      persistAiUiConfig(ui);
    }, 500);
  } else {
    persistAiUiConfig(ui);
  }
}

async function persistAiProviderModelConfig(patch) {
  const clean = {};
  if (Object.prototype.hasOwnProperty.call(patch, "provider")) clean.provider = String(patch.provider || "").trim();
  if (Object.prototype.hasOwnProperty.call(patch, "model")) clean.model = String(patch.model || "").trim();
  try {
    const saved = await api.patch(API_ENDPOINTS.ai.config, clean);
    setAiState({ config: saved });
    return saved;
  } catch (e) {
    toastError(t("ui.toast.ai_config_save_failed", "AI-Auswahl speichern fehlgeschlagen"), e.message);
    return null;
  }
}

async function onAiProviderChange(provider) {
  const normalized = String(provider || "").trim();
  setAiFormData({ provider: normalized, model: "" });
  const modelSelect = document.getElementById("ai-model");
  if (modelSelect) modelSelect.value = "";
  filterModelsByProvider(normalized);
  loadAiAccountStatus(normalized);
  await persistAiProviderModelConfig({ provider: normalized, model: "" });
}

async function onAiModelChange(model) {
  const normalized = String(model || "").trim();
  const provider = document.getElementById("ai-provider")?.value || getAiFormData().provider || "";
  setAiFormData({ provider, model: normalized });
  renderCachedVisionCapability();
  await persistAiProviderModelConfig({ provider, model: normalized });
}

async function loadModels() {
  const statusEl = document.getElementById("ai-model-status");
  const piVersionEl = document.getElementById("ai-pi-version-status");
  const modelSelect = document.getElementById("ai-model");
  try {
    const models = await api.get(API_ENDPOINTS.ai.models);
    if (models?.available === false) {
      if (statusEl) statusEl.textContent = t("ui.state.model_unavailable", "Sidecar nicht erreichbar");
      if (piVersionEl) piVersionEl.textContent = "";
      return;
    }
    const providers = Array.isArray(models?.providers) ? models.providers : [];
    renderProviderOptions(providers);
    _allModels = [];
    for (const p of providers) {
      const providerName = String(p?.provider || "").trim();
      const modelList = Array.isArray(p?.models) ? p.models : [];
      for (const m of modelList) {
        const id = m?.id || m?.name || "";
        const label = m?.label || m?.name || id;
        if (id) _allModels.push({ value: `${providerName}/${id}`, label: `${providerName}: ${label}`, provider: providerName, capabilities: m.capabilities || null });
      }
    }
    const providerCount = providers.length;
    const modelCount = _allModels.length;
    if (statusEl) statusEl.textContent = t("ui.state.model_loaded", "Modelle geladen") + ` (${providerCount} Provider, ${modelCount} Modelle)`;
    if (piVersionEl) piVersionEl.textContent = piVersionStatusText(models?.pi || null);
    filterModelsByProvider(document.getElementById("ai-provider")?.value || "");
    loadAiAccountStatus(document.getElementById("ai-provider")?.value || "");
  } catch (e) {
    if (statusEl) statusEl.textContent = t("ui.state.model_load_failed", "Modelle laden fehlgeschlagen") + `: ${e.message}`;
    if (piVersionEl) piVersionEl.textContent = "";
  }
  renderCachedVisionCapability();
}

function piVersionStatusText(pi) {
  if (!pi) return "";
  const current = String(pi.current || "");
  const latest = String(pi.latest || "");
  const base = current
    ? t("ui.ai.pi_version_current", "PI-Version: {version}", { version: current })
    : t("ui.ai.pi_version_unknown", "PI-Version: unbekannt");
  if (pi.status === "current" && latest) {
    return `${base} · ${t("ui.ai.pi_version_latest_current", "aktuell")}`;
  }
  if (pi.status === "update_available" && latest) {
    return `${base} · ${t("ui.ai.pi_version_update_available", "Update verfügbar: {version}", { version: latest })}`;
  }
  if (pi.status === "not_installed") {
    return t("ui.ai.pi_version_not_installed", "PI-Paket nicht gefunden");
  }
  return `${base} · ${t("ui.ai.pi_version_latest_unknown", "Latest-Check nicht verfügbar")}`;
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
  renderCachedVisionCapability();
}

function selectedModelCapabilities() {
  const model = document.getElementById("ai-model")?.value || getAiState().aiFormData?.model || "";
  if (!model) return null;
  return _allModels.find(m => m.value === model)?.capabilities || null;
}

function visionStatusText(capabilities) {
  if (!capabilities) return t("ui.state.not_loaded", "nicht geladen");
  const support = capabilities.supports_images
    ? t("ui.state.vision_supported", "Bildfähig")
    : t("ui.state.vision_not_supported", "Nicht bildfähig");
  const sourceMap = {
    override: t("ui.ai.vision_source_override", "Override"),
    live_probe: t("ui.ai.vision_source_live", "Live-Test"),
    registry: t("ui.ai.vision_source_registry", "Registry"),
    heuristic: t("ui.ai.vision_source_heuristic", "Heuristik"),
    unknown: t("ui.ai.vision_source_unknown", "unbekannt"),
  };
  const source = sourceMap[capabilities.source] || capabilities.source || sourceMap.unknown;
  const tested = capabilities.source === "live_probe" && capabilities.live?.tested_at ? ` · ${capabilities.live.tested_at}` : "";
  const error = capabilities.live?.error && capabilities.live?.status === "error" ? ` · ${capabilities.live.error}` : "";
  return `${support} (${source})${tested}${error}`;
}

function setVisionOverrideControl(capabilities) {
  const select = document.getElementById("ai-vision-override");
  if (!select) return;
  const override = capabilities?.override;
  select.value = override === true ? "true" : override === false ? "false" : "";
}

function renderVisionCapability(capabilities) {
  const status = document.getElementById("ai-vision-status");
  if (status) status.textContent = visionStatusText(capabilities);
  setVisionOverrideControl(capabilities);
}

function renderCachedVisionCapability() {
  renderVisionCapability(selectedModelCapabilities());
}

function updateCachedVisionCapability(model, capabilities) {
  const item = _allModels.find(m => m.value === model);
  if (item) item.capabilities = capabilities;
  renderVisionCapability(capabilities);
}

function currentVisionOverrideValue() {
  const raw = document.getElementById("ai-vision-override")?.value ?? "";
  if (raw === "true") return true;
  if (raw === "false") return false;
  return null;
}

async function saveVisionOverride() {
  const model = document.getElementById("ai-model")?.value || "";
  if (!model) return;
  try {
    const result = await api.post(API_ENDPOINTS.ai.test, {
      model,
      vision_override: currentVisionOverrideValue(),
    });
    updateCachedVisionCapability(model, result?.capabilities || null);
    toastSuccess(t("ui.toast.vision_override_saved", "Bildfähigkeits-Override gespeichert"));
  } catch (e) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
  }
}

function renderAiAccountStatus(payload, modelCheck = null) {
  const container = document.getElementById("ai-account-status");
  const keyStatus = document.getElementById("ai-key-status");
  if (!container) return;
  container.innerHTML = "";
  const info = payload?.selected || null;
  if (!info) {
    if (keyStatus) keyStatus.classList.add("tc-hidden");
    container.appendChild(el("span", {}, t("ui.state.account_unavailable", "Kontostatus nicht verfügbar.")));
    return;
  }
  if (keyStatus) {
    keyStatus.classList.toggle("tc-hidden", info.auth_source !== "storage");
    keyStatus.textContent = "\u2713 " + t("ui.state.saved", "gespeichert");
  }
  const authSourceNode = info.auth_source === "env"
    ? el("strong", {}, t("ui.ai.auth_env", "Key aus .env"))
    : info.auth_source === "storage"
      ? el("span", {}, t("ui.ai.auth_storage", "Key gespeichert"))
      : el("span", {}, t("ui.ai.auth_missing", "Kein Key gefunden"));
  const billingState = info.credit_query_supported
    ? t("ui.ai.billing_supported", "Credit-/Abo-Abfrage unterstützt")
    : t("ui.ai.billing_not_supported", "Credit-/Abo-Abfrage nicht automatisch verfügbar");
  container.appendChild(el("div", {},
    el("span", {}, `${t("ui.ai.account_status", "API-Key Status")}: `),
    authSourceNode,
    el("span", {}, ` · ${billingState}`),
  ));
  if (modelCheck) {
    const ok = Boolean(modelCheck.ok);
    container.appendChild(el("div", { class: ok ? "tc-text-success" : "tc-text-error" },
      `${t("ui.ai.model_status", "Modellstatus")}: ${ok ? t("ui.state.model_available", "Status: Modell verfügbar") : t("ui.ai.model_unavailable_selected", "Ausgewähltes Modell nicht verfügbar")}`,
    ));
    if (modelCheck.capabilities) {
      updateCachedVisionCapability(modelCheck.model, modelCheck.capabilities);
      container.appendChild(el("div", { class: modelCheck.capabilities.supports_images ? "tc-text-success" : "tc-text-muted" },
        `${t("ui.label.vision_status", "Vision-Status")}: ${visionStatusText(modelCheck.capabilities)}`,
      ));
    }
  }
  container.appendChild(el("div", {}, t("ui.ai.billing_message", "Kontostand und Abo-Details bitte im Anbieter-Portal prüfen.")));
  if (info.billing_url) {
    container.appendChild(el("a", {
      href: String(info.billing_url),
      target: "_blank",
      rel: "noopener noreferrer",
    }, t("ui.ai.open_billing", "Anbieter-Portal öffnen")));
  }
}

async function loadAiAccountStatus(provider = "") {
  const container = document.getElementById("ai-account-status");
  const selectedProvider = provider || document.getElementById("ai-provider")?.value || "";
  if (container) container.textContent = t("ui.state.account_loading", "Kontostatus wird geladen...");
  try {
    const payload = await api.get(API_ENDPOINTS.ai.account(selectedProvider));
    if (payload?.available === false) {
      if (container) container.textContent = t("ui.state.account_unavailable", "Kontostatus nicht verfügbar.");
      return;
    }
    renderAiAccountStatus(payload);
  } catch (e) {
    if (container) container.textContent = `${t("ui.state.account_load_failed", "Kontostatus laden fehlgeschlagen")}: ${e.message}`;
  }
}

async function refreshAiProviderStatus() {
  const provider = document.getElementById("ai-provider")?.value || "";
  const model = document.getElementById("ai-model")?.value || "";
  const container = document.getElementById("ai-account-status");
  if (!provider || !model) {
    toastError(t("ui.state.account_load_failed", "Kontostatus laden fehlgeschlagen"), t("ui.ai.provider_model_required", "Bitte Provider und Modell auswählen."));
    return;
  }
  if (container) container.textContent = t("ui.state.account_loading", "Kontostatus wird geladen...");
  try {
    const [account, modelCheck] = await Promise.all([
      api.get(API_ENDPOINTS.ai.account(provider)),
      api.post(API_ENDPOINTS.ai.test, {
        model,
        vision_probe: true,
        vision_override: currentVisionOverrideValue(),
      }).catch((e) => ({
        ok: false,
        model,
        error: e.message,
      })),
    ]);
    if (account?.available === false) {
      if (container) container.textContent = t("ui.state.account_unavailable", "Kontostatus nicht verfügbar.");
      return;
    }
    renderAiAccountStatus(account, modelCheck);
    toastSuccess(t("ui.toast.account_status_loaded", "Status abgerufen"));
  } catch (e) {
    if (container) container.textContent = `${t("ui.state.account_load_failed", "Kontostatus laden fehlgeschlagen")}: ${e.message}`;
    toastError(t("ui.state.account_load_failed", "Kontostatus laden fehlgeschlagen"), e.message);
  }
}

async function saveApiKey() {
  const provider = document.getElementById("ai-provider")?.value || "";
  const key = document.getElementById("ai-apikey")?.value || "";
  if (!provider || !key) return;
  try {
    await api.post(API_ENDPOINTS.ai.authProvider(provider), { api_key: key });
    toastSuccess(t("ui.toast.key_saved", "API-Key gespeichert"));
    await loadAiAccountStatus(provider);
  } catch (e) {
    toastError(t("ui.toast.key_save_failed", "Key speichern fehlgeschlagen"), e.message);
  }
}

async function autoScanForAnalysis() {
  const sd = getScanData();
  const inputDir = String(sd.input_dir || "").trim();
  if (!inputDir) {
    throw new Error(t("ui.error.no_input_dir_for_scan", "Kein Eingabeordner festgelegt. Bitte zuerst unter Input & Scan konfigurieren."));
  }
  toast(t("ui.toast.scan_starting", "Scan wird gestartet..."), "", "info");
  addTrafficEntry({ type: "request", text: `POST /api/scan auto input_dir=${inputDir}` });
  const payload = {
    input_dir: inputDir,
    pattern: sd.pattern || "*.fits",
    runs_dir: sd.runs_dir || "",
    run_name: sd.run_name || "",
    object_name: sd.object_name || "",
    target: sd.object_name || "",
    color_mode: sd.color_mode || "OSC",
    frame_min: Number(sd.frame_min) || 1,
    frames_min: Number(sd.frame_min) || 1,
    max_frames: Number(sd.max_frames) || 0,
    sort: sd.sort || "numeric",
    with_checksums: Boolean(sd.with_checksums),
    queue: getQueueItems(),
    calibration: getCalValues(),
  };
  const jobStart = await api.post(API_ENDPOINTS.scan.root, payload);
  if (!jobStart?.job_id) {
    throw new Error(t("ui.error.no_job_id", "Backend hat keine job_id zurückgegeben."));
  }
  addTrafficEntry({ type: "progress", text: `${t("ui.toast.scan_starting", "Scan wird gestartet...")} ${jobStart.job_id}` });
  const scan = await pollJob(jobStart.job_id, {
    endpoint: API_ENDPOINTS.scan.jobStatus,
    intervalMs: 1000,
    timeoutMs: 120000,
    errorLabel: "Scan",
    onDone: async () => api.get(API_ENDPOINTS.scan.latest),
  });
  const autoHasScan = scan && (scan.has_scan || (scan.frames_detected && scan.frames_detected > 0));
  if (!autoHasScan) {
    throw new Error(t("ui.state.no_scan", "Kein Scan vorhanden. Bitte zuerst einen Scan durchfuehren."));
  }
  const frameCount = scan.frames_detected || scan.frame_count || 0;
  addTrafficEntry({ type: "info", text: `Auto scan completed: ${frameCount} frames` });
  toastSuccess(t("ui.toast.scan_completed", "Scan abgeschlossen"), `${frameCount} ${t("ui.label.frames_detected", "Frames erkannt")}`);
  return scan;
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
    let latestScan = await api.get(API_ENDPOINTS.scan.latest);
    const hasScan = latestScan && (latestScan.has_scan || (latestScan.frames_detected && latestScan.frames_detected > 0));
    if (!hasScan) {
      addTrafficEntry({ type: "info", text: "No scan result available, checking configured input directory..." });
      latestScan = await autoScanForAnalysis();
    }
    const objectName = String(getScanData().object_name || "").trim();
    if (objectName) {
      latestScan = { ...latestScan, object_name: objectName, target: objectName };
    }
    addTrafficEntry({ type: "info", text: `Scan: ${latestScan.frames_detected || 0} frames, ${latestScan.color_mode || "unknown"}` });

    // Check cache first when force=false
    if (!force) {
      try {
        const cached = await api.get(API_ENDPOINTS.scan.analysisLatest);
        if (cached && cached.has_analysis && cached.recommendations?.length > 0 && sameAnalysisScope(cached, latestScan)) {
          addTrafficEntry({ type: "response", text: `Cache hit: ${cached.recommendations.length} recommendations` });
          renderRecommendations(cached.recommendations);
          setAiState({ currentAnalysis: cached, loading: false });
          toastSuccess(t("ui.toast.analysis_done", "Analyse aus Cache geladen"));
          return;
        } else if (cached && cached.has_analysis && cached.recommendations?.length > 0) {
          addTrafficEntry({ type: "info", text: "Cached analysis belongs to a different scan/object; ignoring it." });
        }
      } catch {}
      addTrafficEntry({ type: "info", text: "No cached analysis, starting new..." });
    }

    // Compute scan metrics (image statistics like FWHM, star count)
    let scanMetrics = null;
    try {
      addTrafficEntry({ type: "info", text: "Loading image statistics..." });
      const metricsStart = await api.post(API_ENDPOINTS.scan.metrics, {
        input_path: latestScan.input_path || latestScan.input_dirs?.[0] || "",
        object_name: objectName || latestScan.object_name || latestScan.target || "",
        target: objectName || latestScan.object_name || latestScan.target || "",
        frame_count: latestScan.frames_detected || latestScan.frames_total || latestScan.frame_count || 0,
      });
      if (metricsStart?.cached && metricsStart?.result) {
        scanMetrics = metricsStart.result;
        const sampled = scanMetrics.sample_count ?? "?";
        const total = scanMetrics.frames_total ?? latestScan.frames_detected ?? "?";
        addTrafficEntry({ type: "response", text: `Image statistics cache hit: sampled=${sampled}/${total}` });
      } else if (metricsStart?.job_id) {
        addTrafficEntry({ type: "info", text: "Computing image statistics..." });
        scanMetrics = await pollJobResult(metricsStart.job_id);
        if (scanMetrics) {
          const sampled = scanMetrics.sample_count ?? "?";
          const total = scanMetrics.frames_total ?? latestScan.frames_detected ?? "?";
          addTrafficEntry({ type: "info", text: `Image statistics computed: sampled=${sampled}/${total}` });
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
      session_context: {
        mount_type: fd.mount || undefined,
        target_name: objectName || undefined,
        target_type: fd.object_type || undefined,
        camera_type: fd.camera || undefined,
        calibration_darks: Boolean(fd.calibration_darks),
        calibration_flats: Boolean(fd.calibration_flats),
        calibration_bias: Boolean(fd.calibration_bias),
        notes: fd.notes || undefined,
      },
    };
    if (scanMetrics) payload.scan_metrics = scanMetrics;
    if (configSchema) payload.config_schema = configSchema;
    if (baseConfig) payload.base_config = baseConfig;

    addTrafficEntry({ type: "request", text: `POST /api/scan/analysis with scan_result, ${scanMetrics ? "scan_metrics, " : ""}${configSchema ? "config_schema, " : ""}${baseConfig ? "base_config" : ""}` });

    addTrafficEntry({ type: "progress", text: "Initialisierung... 5%" });
    addTrafficEntry({ type: "progress", text: "Prompt wird gebaut... 10%" });
    addTrafficEntry({ type: "progress", text: "Warte auf KI-Antwort... 15%" });

    const result = await api.post(API_ENDPOINTS.scan.analysis, payload, { timeoutMs: 1200000 });
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
          title: t("ui.tooltip.ai.recommendation_select", "Legt fest, ob diese Empfehlung angewendet wird."),
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
  if (status === "promotable") return `${t("ui.pi.status.promotable", "Promotable")}${reviewed}`;
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

function memoryContextSummary(memory) {
  const ctx = memory?.context_signature;
  if (!ctx || typeof ctx !== "object") return "";
  const target = ctx.target || {};
  const acquisition = ctx.acquisition || {};
  const mount = ctx.mount || {};
  const pipeline = ctx.pipeline || {};
  const parts = [
    target.object_name || target.object_type,
    acquisition.camera_name || acquisition.camera_type || acquisition.color_mode,
    Array.isArray(acquisition.filters) ? acquisition.filters.slice(0, 3).join("+") : "",
    mount.type,
    Array.isArray(pipeline.affected_paths) ? pipeline.affected_paths.slice(0, 3).join(", ") : "",
  ].filter(Boolean);
  return parts.length ? `${t("ui.pi.context_signature", "Kontext")}: ${parts.join(" · ")}` : "";
}

function memoryScopeSummary(memory) {
  const scope = memory?.scope;
  if (!scope || typeof scope !== "object") return "";
  const applies = Array.isArray(scope.applies_when) ? scope.applies_when.slice(0, 2).join("; ") : "";
  const avoids = Array.isArray(scope.does_not_apply_when) ? scope.does_not_apply_when.slice(0, 2).join("; ") : "";
  const confidence = scope.confidence !== undefined ? ` · ${t("ui.label.confidence", "Confidence")}: ${scope.confidence}` : "";
  const text = [applies ? `${t("ui.pi.applies_when", "Gilt wenn")}: ${applies}` : "", avoids ? `${t("ui.pi.does_not_apply_when", "Gilt nicht wenn")}: ${avoids}` : ""].filter(Boolean).join(" · ");
  return text ? `${text}${confidence}` : "";
}

function memoryEvidenceSummary(memory) {
  const evidence = memory?.evidence;
  if (!evidence || typeof evidence !== "object") return "";
  const refs = Array.isArray(evidence.run_refs) ? evidence.run_refs.length : 0;
  const human = evidence.human_feedback ? t("ui.pi.human_feedback", "Nutzerfeedback") : "";
  const validation = evidence.validation ? t("ui.pi.validation_evidence", "Validierung") : "";
  const parts = [refs ? `${refs} ${t("ui.pi.run_refs", "Run-Referenzen")}` : "", human, validation].filter(Boolean);
  return parts.length ? `${t("ui.pi.evidence", "Evidenz")}: ${parts.join(" · ")}` : "";
}

async function loadPiStorage() {
  const input = document.getElementById("ai-pi-storage-dir");
  const status = document.getElementById("ai-pi-storage-status");
  if (status) status.textContent = t("ui.state.loading", "Lädt...");
  try {
    const payload = await api.get(API_ENDPOINTS.pi.storage);
    if (input) input.value = payload?.storage_dir || "";
    if (status) {
      const configured = payload?.configured
        ? t("ui.pi.storage_configured", "persistent gespeichert")
        : t("ui.pi.storage_default", "Default");
      status.textContent = `${configured}: ${payload?.storage_dir || ""}`;
    }
    return payload;
  } catch (e) {
    if (status) status.textContent = e.message;
    return null;
  }
}

async function savePiStorage() {
  const input = document.getElementById("ai-pi-storage-dir");
  const raw = String(input?.value || "").trim();
  if (!raw) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), t("ui.error.pi_storage_required", "PI-Speicherort fehlt."));
    return;
  }
  try {
    const payload = await api.post(API_ENDPOINTS.pi.storage, { storage_dir: raw });
    if (input) input.value = payload?.storage_dir || raw;
    await loadPiMemories();
    await loadPiAudit();
    toastSuccess(t("ui.toast.saved", "Gespeichert"));
  } catch (e) {
    toastError(t("ui.toast.save_failed", "Speichern fehlgeschlagen"), e.message);
  }
}

async function loadPiMemories() {
  const filter = document.getElementById("ai-pi-memory-filter")?.value || "candidate";
  const status = document.getElementById("ai-pi-memory-status");
  const list = document.getElementById("ai-pi-memory-list");
  const dir = document.getElementById("ai-pi-memory-dir");
  if (status) status.textContent = t("ui.state.loading", "Lädt...");
  try {
    const payload = await api.get(`${API_ENDPOINTS.pi.memories}?limit=100&status=${encodeURIComponent(filter)}`);
    if (dir) {
      dir.textContent = payload?.memory_dir ? `${t("ui.pi.memory_store", "Memory Store")}: ${payload.memory_dir}` : "";
    }
    if (list) {
      list.innerHTML = "";
      const items = Array.isArray(payload?.items) ? payload.items : [];
      for (const memory of items) {
        const id = memory.memory_id || "";
        const statusName = String(memory.status || "candidate");
        const canReview = statusName === "candidate" || statusName === "promotable" || statusName === "accepted";
        list.appendChild(el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
          el("div", { class: "tc-flex tc-justify-between tc-gap-2" },
            el("div", { class: "tc-mono tc-text-sm" }, id || "-"),
            el("span", { class: "tc-badge" }, memoryStatusLabel(memory)),
          ),
          el("div", { class: "tc-text-sm tc-text-muted" }, summarizeMemory(memory)),
          memoryContextSummary(memory) ? el("div", { class: "tc-text-sm tc-text-muted" }, memoryContextSummary(memory)) : null,
          memoryScopeSummary(memory) ? el("div", { class: "tc-text-sm tc-text-muted" }, memoryScopeSummary(memory)) : null,
          memoryEvidenceSummary(memory) ? el("div", { class: "tc-text-sm tc-text-muted" }, memoryEvidenceSummary(memory)) : null,
          memoryOutcomeSummary(memory) ? el("div", { class: "tc-text-sm tc-text-muted" }, memoryOutcomeSummary(memory)) : null,
          memory?.review?.note ? el("div", { class: "tc-text-sm" }, memory.review.note) : null,
          el("div", { class: "tc-flex tc-gap-2 tc-mt-2" },
            canReview && statusName !== "accepted" ? el("button", { class: "tc-btn tc-btn-sm", title: t("ui.tooltip.ai.memory_accept", "Markiert diese Erfahrung als nuetzlich fuer spaetere Sessions."), onclick: () => reviewPiMemory(id, "accepted") }, t("ui.button.accept", "Accept")) : null,
            canReview ? el("button", { class: "tc-btn tc-btn-sm", title: t("ui.tooltip.ai.memory_reject", "Markiert diese Erfahrung als nicht hilfreich."), onclick: () => reviewPiMemory(id, "rejected") }, t("ui.button.reject", "Reject")) : null,
            canReview ? el("button", { class: "tc-btn tc-btn-sm", title: t("ui.tooltip.ai.memory_deprecate", "Markiert diese Erfahrung als ueberholt."), onclick: () => reviewPiMemory(id, "deprecated") }, t("ui.button.deprecate", "Deprecate")) : null,
            canReview ? el("button", { class: "tc-btn tc-btn-sm", title: t("ui.tooltip.ai.memory_scope", "Bearbeitet, fuer welche Kontexte diese Memory gilt oder nicht gilt."), onclick: () => editPiMemoryScope(memory) }, t("ui.button.edit_scope", "Scope")) : null,
          ),
        ));
      }
      if (!items.length) {
        const legacyCount = Number(payload?.legacy_ignored_count || 0);
        const message = legacyCount > 0
          ? t("ui.pi.no_current_memories_legacy_ignored", "Keine aktuellen v2-Memories. {count} alte v1-Memories werden ignoriert.", { count: legacyCount })
          : t("ui.pi.no_current_memories", "Keine aktuellen v2-Memories. Neue Einträge entstehen erst, wenn ein Lernkandidat gespeichert wird.");
        list.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, message));
        if (payload?.memory_file) {
          list.appendChild(el("div", { class: "tc-text-muted tc-text-sm tc-mono" }, payload.memory_file));
        }
      }
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

function splitScopeLines(value) {
  return String(value || "")
    .split(/\n|;/)
    .map((item) => item.trim())
    .filter(Boolean);
}

async function editPiMemoryScope(memory) {
  const memoryId = memory?.memory_id || "";
  if (!memoryId) return;
  const scope = memory?.scope && typeof memory.scope === "object" ? memory.scope : {};
  const appliesDefault = Array.isArray(scope.applies_when) ? scope.applies_when.join("; ") : "";
  const avoidsDefault = Array.isArray(scope.does_not_apply_when) ? scope.does_not_apply_when.join("; ") : "";
  const appliesRaw = window.prompt(t("ui.prompt.pi_scope_applies", "Gilt wenn (mit Semikolon trennen)"), appliesDefault);
  if (appliesRaw === null) return;
  const avoidsRaw = window.prompt(t("ui.prompt.pi_scope_avoids", "Gilt nicht wenn (mit Semikolon trennen)"), avoidsDefault);
  if (avoidsRaw === null) return;
  const confidenceRaw = window.prompt(t("ui.prompt.pi_scope_confidence", "Confidence 0..1"), scope.confidence ?? "");
  if (confidenceRaw === null) return;
  const confidence = Number(confidenceRaw);
  const nextScope = {
    applies_when: splitScopeLines(appliesRaw),
    does_not_apply_when: splitScopeLines(avoidsRaw),
  };
  if (Number.isFinite(confidence)) nextScope.confidence = Math.max(0, Math.min(1, confidence));
  const currentStatus = String(memory.status || "");
  const reviewStatus = ["accepted", "promotable", "rejected", "deprecated"].includes(currentStatus)
    ? currentStatus
    : "promotable";
  try {
    await api.post(API_ENDPOINTS.pi.memoryReview(memoryId), {
      status: reviewStatus,
      reviewer: "gui3",
      note: t("ui.pi.scope_updated", "Scope updated in GUI3"),
      scope: nextScope,
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
  if (item?.audit_type === "memory_candidate") {
    return `${item.memory_id || "-"} · ${item.status || "candidate"} · ${item.source || "memory"}`;
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

async function loadPersistentTrafficLog() {
  const status = document.getElementById("ai-traffic-status");
  if (status) status.textContent = t("ui.state.loading", "Lädt...");
  try {
    const payload = await api.get(API_ENDPOINTS.ai.traffic(500));
    const items = Array.isArray(payload?.items) ? payload.items : [];
    renderTrafficLog(items.map((line) => ({ type: "log", text: line, ts: "" })));
    if (status) {
      const enabled = payload?.enabled === false ? t("ui.state.disabled", "deaktiviert") : t("ui.state.enabled", "aktiv");
      status.textContent = `${enabled} · ${t("ui.pi.traffic_count", "{count} Zeilen", { count: payload?.count || items.length })} · ${payload?.path || ""}`;
    }
  } catch (e) {
    if (status) status.textContent = e.message;
    toastError(t("ui.toast.traffic_load_failed", "Traffic-Log laden fehlgeschlagen"), e.message);
  }
}
