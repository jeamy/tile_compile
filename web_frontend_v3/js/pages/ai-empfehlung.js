// js/pages/ai-empfehlung.js – Sub-Tab: AI Empfehlung (innerhalb Parameter)

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { getAiState, setAiState } from "../state/ai-state.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toast, toastSuccess, toastError } from "../components/toast.js";

export function createAiEmpfehlungPage() {
  const page = el("div", { class: "tc-flex-col tc-gap-4" });

  // Scan context
  const scanCtx = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.scan_context", "Scan-Kontext (auto aus Scan)")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.mount", "Mount")),
        el("select", { class: "tc-select" },
          el("option", {}, "EQ"),
          el("option", {}, "Tracker"),
          el("option", {}, "Alt/Az"),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.object_type", "Objekt")),
        el("select", { class: "tc-select" },
          el("option", {}, "Galaxie"),
          el("option", {}, "Nebel"),
          el("option", {}, "Sternhaufen"),
          el("option", {}, "Sterne"),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.camera", "Kamera")),
        el("select", { class: "tc-select" },
          el("option", {}, "Consumer OSC"),
          el("option", {}, "Mono CMOS"),
          el("option", {}, "CCD"),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.calibration", "Kalibrierung")),
        el("div", { class: "tc-flex tc-gap-3" },
          el("label", { class: "tc-checkbox" }, el("input", { type: "checkbox" }), "Darks"),
          el("label", { class: "tc-checkbox" }, el("input", { type: "checkbox" }), "Flats"),
          el("label", { class: "tc-checkbox" }, el("input", { type: "checkbox" }), "Bias"),
        ),
      ),
    ),
    el("div", { class: "tc-mt-2" },
      el("label", { class: "tc-label" }, t("ui.field.notes", "Notizen")),
      el("input", { type: "text", class: "tc-input", placeholder: "Guiding 0.8\", M31, alt-az test" }),
    ),
  );

  // Model & API Key
  const modelCard = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.model_api", "Modell & API-Key")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.provider", "Provider")),
        el("select", { class: "tc-select", id: "ai-provider" },
          el("option", { value: "anthropic" }, "anthropic"),
          el("option", { value: "openai" }, "openai"),
        ),
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.field.model", "Modell")),
        el("select", { class: "tc-select", id: "ai-model" },
          el("option", {}, "claude-sonnet-4-20250514"),
        ),
      ),
    ),
    el("div", { class: "tc-mt-2 tc-flex tc-items-center tc-gap-2" },
      el("input", { type: "password", class: "tc-input", placeholder: "API-Key", id: "ai-apikey" }),
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => saveApiKey() }, t("ui.button.save_key", "Key speichern")),
      el("span", { class: "tc-badge tc-badge-success", id: "ai-key-status" }, "\u2713 gespeichert"),
    ),
    el("div", { class: "tc-mt-2" },
      el("span", { class: "tc-text-sm tc-text-muted" }, t("ui.state.model_available", "Status: Modell verfuegbar")),
    ),
  );

  // Actions
  const actions = el("div", { class: "tc-flex tc-gap-3 tc-flex-wrap" },
    el("button", { class: "tc-btn tc-btn-primary", onclick: () => createAnalysis() }, t("ui.button.create_analysis", "KI-Analyse erstellen")),
    el("button", { class: "tc-btn", onclick: () => createAnalysis(true) }, t("ui.button.reanalyze", "Neu analysieren (Cache ignorieren)")),
    el("select", { class: "tc-select", style: { width: "200px" } },
      el("option", {}, t("ui.placeholder.saved_analyses", "Gespeicherte Analysen")),
    ),
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
    }, "\u25b8 " + t("ui.title.ai_traffic", "KI-Datenverkehr (ausgeblendet)")),
    el("div", { class: "tc-accordion-body" },
      el("div", { class: "tc-log-viewer", id: "ai-traffic-log" },
        el("div", { class: "tc-text-muted" }, t("ui.state.no_traffic", "Keine Daten")),
      ),
    ),
  );

  page.append(scanCtx, modelCard, actions, recs, applyBar, traffic);
  return page;
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
  try {
    toast(t("ui.toast.analysis_creating", "KI-Analyse wird erstellt..."), "", "info");
    const result = await api.post(API_ENDPOINTS.scan.analysis, { force });
    if (result?.recommendations) {
      renderRecommendations(result.recommendations);
      setAiState({ currentAnalysis: result });
      toastSuccess(t("ui.toast.analysis_done", "Analyse erstellt"));
    }
  } catch (e) {
    toastError(t("ui.toast.analysis_failed", "Analyse fehlgeschlagen"), e.message);
  }
}

function renderRecommendations(recs) {
  const container = document.getElementById("ai-recommendations");
  if (!container) return;
  container.innerHTML = "";
  for (const rec of recs) {
    const item = el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
      el("label", { class: "tc-checkbox" },
        el("input", { type: "checkbox" }),
        el("span", { class: "tc-mono tc-text-sm" }, rec.path || rec.key || ""),
      ),
      el("div", { class: "tc-mt-2 tc-text-sm" },
        el("span", { class: "tc-text-muted" }, t("ui.label.current", "Aktuell") + ": "),
        el("span", {}, String(rec.current ?? "")),
      ),
      el("div", { class: "tc-text-sm" },
        el("span", { class: "tc-text-muted" }, t("ui.label.recommended", "Empfohlen") + ": "),
        el("span", {}, String(rec.recommended ?? rec.value ?? "")),
      ),
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
