// js/pages/jev-empfehlung.js – Jev (second, independent recommendation source)
//
// Two pieces, both independent of the PI provider slot in ai-empfehlung.js:
//   createJevSettingsCard()      Tools -> AI & API: mode, experimental flag, write-only API key
//   createJevEmpfehlungPage()    Parameter tab "Jev-Empfehlungen": request, compare current vs proposed, apply to draft
// The proposal is always shown next to the current config value. Applying only changes the config DRAFT;
// nothing is saved and no run is started from here.

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { toastError, toastSuccess } from "../components/toast.js";
import { getConfigState, setConfigState, deepClone } from "../state/config-state.js";
import { getUiState, setUiState } from "../state/ui-state.js";
import { parseYaml, stringifyYaml } from "../utils/yaml-parse.js";
import { pollJob } from "../utils/poll.js";
import { getScanData } from "./input-scan.js";

const POLL_MS = 1500;
const POLL_TIMEOUT_MS = 90000;

function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function currentDraftYaml() {
  const s = getConfigState();
  if (s.draftYaml && s.draftYaml.trim()) return s.draftYaml;
  return s.draft ? stringifyYaml(s.draft) : "";
}

// ---------------------------------------------------------------------------------------------
// Settings card (Tools -> AI & API)
// ---------------------------------------------------------------------------------------------

export function createJevSettingsCard() {
  const statusLine = el("div", { class: "tc-text-sm tc-text-muted", id: "jev-status" }, t("ui.jev.status_loading", "Jev-Status wird geladen..."));
  const modeSelect = el("select", { class: "tc-select", id: "jev-mode", title: t("ui.jev.tooltip.mode", "off: keine Anfragen. shadow: Anfragen werden aufgezeichnet, aber nie als Vorschlag angezeigt. suggest: geprüfte Vorschläge werden angezeigt.") },
    el("option", { value: "off" }, t("ui.jev.mode.off", "Aus (keine Anfragen)")),
    el("option", { value: "shadow" }, t("ui.jev.mode.shadow", "Shadow (nur aufzeichnen)")),
    el("option", { value: "suggest" }, t("ui.jev.mode.suggest", "Vorschlagen")),
  );
  const experimental = el("input", { type: "checkbox", id: "jev-experimental" });
  const keyInput = el("input", { type: "password", class: "tc-input", id: "jev-apikey", autocomplete: "off", placeholder: t("ui.jev.key_placeholder", "OpenRouter-Key für Jev"), title: t("ui.jev.tooltip.key", "Eigener Key nur für Jev, getrennt vom Key der KI-Karte. Wird nie angezeigt oder zurückgegeben.") });
  const model = el("span", { class: "tc-text-sm tc-text-muted", id: "jev-model" }, "—");

  async function refresh() {
    try {
      const s = await api.get(API_ENDPOINTS.decisions.status);
      if (!s?.available) {
        statusLine.textContent = t("ui.jev.status_unreachable", "PI-Sidecar nicht erreichbar");
        return;
      }
      modeSelect.value = s.mode || "off";
      experimental.checked = Boolean(s.allow_experimental_suggestions);
      model.textContent = s.model || "—";
      statusLine.textContent = `${t("ui.jev.status_key", "API-Key")}: ${s.has_api_key ? t("ui.jev.key_present", "vorhanden") : t("ui.jev.key_missing", "fehlt")} · ${t("ui.jev.status_endpoint", "Endpunkt")}: ${s.endpoint_host || "—"}`;
    } catch (e) {
      statusLine.textContent = t("ui.jev.status_unreachable", "PI-Sidecar nicht erreichbar");
    }
  }

  async function save(patch) {
    try {
      const s = await api.post(API_ENDPOINTS.decisions.settings, patch);
      toastSuccess(t("ui.jev.saved", "Jev-Einstellungen gespeichert"));
      keyInput.value = "";
      await refresh();
      return s;
    } catch (e) {
      toastError(t("ui.jev.save_failed", "Jev-Einstellungen speichern fehlgeschlagen"), e.message);
      return null;
    }
  }

  const card = el("div", { class: "tc-card tc-jev", id: "jev-settings-card" },
    el("div", { class: "tc-card-title" }, t("ui.jev.title", "Jev (Decisions API)")),
    el("div", { class: "tc-text-sm tc-text-muted tc-mb-2" }, t("ui.jev.intro", "Zweite, unabhängige Empfehlungsquelle. Läuft getrennt von der KI-Karte oben; Umschalten dort ändert Jev nicht.")),
    el("div", { class: "tc-grid-2" },
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.jev.mode", "Betriebsmodus")),
        modeSelect,
      ),
      el("div", {},
        el("label", { class: "tc-label" }, t("ui.jev.model", "Modell (fest gepinnt)")),
        model,
      ),
    ),
    el("div", { class: "tc-mt-2 tc-jev-row" },
      experimental,
      el("label", { for: "jev-experimental", class: "tc-text-sm" }, t("ui.jev.allow_experimental", "Experimentelle Kandidaten anzeigen (nicht validiert)")),
    ),
    el("div", { class: "tc-mt-2 tc-jev-row" },
      el("div", { class: "tc-jev-grow" }, keyInput),
      el("button", { class: "tc-btn", onclick: () => { if (keyInput.value.trim()) save({ api_key: keyInput.value.trim() }); } }, t("ui.jev.save_key", "Key speichern")),
      el("button", { class: "tc-btn", onclick: () => save({ api_key: "" }) }, t("ui.jev.remove_key", "Key entfernen")),
    ),
    el("div", { class: "tc-mt-2 tc-jev-row" },
      el("button", { class: "tc-btn tc-btn-primary", onclick: () => save({ mode: modeSelect.value, allow_experimental_suggestions: experimental.checked }) }, t("ui.jev.save_settings", "Modus speichern")),
      statusLine,
    ),
  );
  refresh();
  return card;
}

// ---------------------------------------------------------------------------------------------
// Recommendation page (Parameter tab)
// ---------------------------------------------------------------------------------------------

const STATUS_TEXT = {
  no_change: ["ui.jev.result.no_change", "Keine Änderung empfohlen"],
  abstain: ["ui.jev.result.abstain", "Enthaltung: die Belege reichen für keine Empfehlung"],
  unavailable: ["ui.jev.result.unavailable", "Nicht verfügbar"],
  rejected: ["ui.jev.result.rejected", "Verworfen (Prüfung nicht bestanden)"],
  validated: ["ui.jev.result.validated", "Vorschlag geprüft, noch nicht übernommen"],
  presented: ["ui.jev.result.validated", "Vorschlag geprüft, noch nicht übernommen"],
  applied_to_draft: ["ui.jev.result.applied", "In den Entwurf übernommen (nicht gespeichert)"],
  stale: ["ui.jev.result.stale", "Veraltet: Entwurf oder Scan haben sich geändert"],
};

export function createJevEmpfehlungPage({ onDraftApplied } = {}) {
  const resultBox = el("div", { class: "tc-mt-4", id: "jev-result" });
  const requestBtn = el("button", { class: "tc-btn tc-btn-primary", id: "jev-request", onclick: () => requestAdvice() }, t("ui.jev.request", "Jev-Empfehlung anfordern"));
  // The object class is a statement by the user; the scan never infers it. Candidates that need it abstain without it.
  const objectClassSelect = el("select", { class: "tc-select", id: "jev-object-class", title: t("ui.jev.tooltip.object_class", "Ihre Angabe zum Objekt. Manche Empfehlungen (z. B. Denoise, Hintergrund) hängen davon ab; ohne Angabe enthalten sie sich.") },
    el("option", { value: "" }, t("ui.jev.object_class.none", "Nicht angegeben")),
    el("option", { value: "compact" }, t("ui.jev.object_class.compact", "Kompakt (z. B. Galaxie)")),
    el("option", { value: "diffuse" }, t("ui.jev.object_class.diffuse", "Diffus (z. B. großer Nebel)")),
    el("option", { value: "star_field" }, t("ui.jev.object_class.star_field", "Sternfeld")),
  );
  objectClassSelect.value = getUiState().jevObjectClass || "";
  objectClassSelect.addEventListener("change", () => setUiState({ jevObjectClass: objectClassSelect.value }));
  function sessionContext() {
    const v = objectClassSelect.value;
    return v ? { object_class: { value: v, source: "user" } } : {};
  }
  const page = el("div", { class: "tc-flex-col tc-gap-4 tc-jev" },
    el("div", { class: "tc-card tc-jev" },
      el("div", { class: "tc-card-title" }, t("ui.jev.page_title", "Jev-Empfehlungen")),
      el("div", { class: "tc-text-sm tc-text-muted tc-mb-2" }, t("ui.jev.page_intro", "Prüft den aktuellen Config-Entwurf gegen die Scan-Statistiken. Vorschläge sind experimentell: es gibt noch keinen Nachweis, dass sie Ergebnisse verbessern. Übernehmen ändert nur den Entwurf.")),
      el("div", { class: "tc-flex tc-items-center tc-gap-2 tc-flex-wrap" },
        el("label", { class: "tc-text-sm", for: "jev-object-class" }, t("ui.jev.object_class.label", "Objektklasse")),
        objectClassSelect, requestBtn),
      resultBox,
    ),
  );

  let pollToken = 0;

  function renderMessage(text, cls = "tc-text-muted") {
    resultBox.replaceChildren(el("div", { class: `tc-text-sm ${cls}` }, text));
  }

  function badge(text, kind) { return el("span", { class: `tc-badge tc-badge-${kind}`, style: { marginRight: "6px" } }, text); }

  function render(view) {
    const nodes = [];
    const proposal = view.proposal || {};
    const status = proposal.status || "unavailable";
    const [key, fallback] = STATUS_TEXT[status] || STATUS_TEXT.unavailable;
    const head = el("div", { class: "tc-flex tc-items-center tc-gap-2 tc-flex-wrap" },
      el("strong", {}, t(key, fallback)),
      view.mode ? badge(`${t("ui.jev.mode", "Betriebsmodus")}: ${view.mode}`, "info") : null,
      view.evidence?.experimental ? badge(t("ui.jev.experimental", "experimentell"), "warning") : null,
    );
    nodes.push(head);
    if (view.shadow) nodes.push(el("div", { class: "tc-text-sm tc-text-muted tc-mt-2" }, t("ui.jev.shadow_note", "Shadow-Modus: die Antwort wurde nur aufgezeichnet, es gibt keinen anwendbaren Vorschlag.")));
    if (view.synthetic_baseline && !view.model_called)
      nodes.push(el("div", { class: "tc-text-sm tc-text-muted tc-mt-2" }, t("ui.jev.no_model_call", "Das Modell wurde nicht befragt: es war kein Kandidat mit Änderung anwendbar (siehe Ausschlussgründe).")));
    const reasons = (proposal.reason_codes || []).filter((r) => r !== "validated");
    if (reasons.length) nodes.push(el("div", { class: "tc-text-sm tc-text-muted tc-mt-2" }, `${t("ui.jev.reasons", "Gründe")}: ${reasons.join(", ")}`));

    if (view.comparison?.length) {
      const rows = view.comparison.map((c) => el("tr", {},
        el("td", {}, c.path), el("td", {}, fmt(c.current)), el("td", {}, c.changed ? el("strong", {}, fmt(c.proposed)) : fmt(c.proposed))));
      nodes.push(el("table", { class: "tc-frame-table tc-mt-2" },
        el("thead", {}, el("tr", {}, el("th", {}, t("ui.jev.col.path", "Parameter")), el("th", {}, t("ui.jev.col.current", "Aktuell")), el("th", {}, t("ui.jev.col.proposed", "Vorgeschlagen")))),
        el("tbody", {}, ...rows)));
    }
    const params = view.evidence?.rationale?.params;
    if (params && Object.keys(params).length) {
      nodes.push(el("div", { class: "tc-text-sm tc-mt-2" }, `${t("ui.jev.evidence", "Belege (gemessen)")}: ` +
        Object.entries(params).map(([k, v]) => `${k} = ${typeof v === "number" ? v.toFixed(4) : v}`).join(", ")));
      nodes.push(el("div", { class: "tc-text-sm tc-text-muted" }, t("ui.jev.evidence_caveat", "Die Belege beschreiben die Scan-Statistik, nicht den Nutzen für das Ergebnis.")));
    }
    if (view.excluded?.length || view.offered?.length) {
      nodes.push(el("details", { class: "tc-mt-2" },
        el("summary", { class: "tc-text-sm" }, t("ui.jev.details", "Angebotene und ausgeschlossene Kandidaten")),
        el("div", { class: "tc-text-sm" }, `${t("ui.jev.offered", "Angeboten")}: ${(view.offered || []).map((o) => o.candidate_id).join(", ") || "—"}`),
        ...(view.excluded || []).map((x) => el("div", { class: "tc-text-sm tc-text-muted" }, `${x.candidate_id}: ${(x.reasons || []).join(", ")}`)),
      ));
    }
    if (status === "validated" || status === "presented") {
      nodes.push(el("div", { class: "tc-mt-2 tc-flex tc-gap-2" },
        el("button", { class: "tc-btn tc-btn-primary", onclick: () => applyToDraft(view.proposal_id) }, t("ui.jev.apply", "In Entwurf übernehmen"))));
    }
    resultBox.replaceChildren(...nodes.filter(Boolean));
  }

  async function poll(id) {
    const token = ++pollToken;
    const started = Date.now();
    while (token === pollToken) {
      let view;
      try {
        view = await api.get(API_ENDPOINTS.decisions.byId(id));
      } catch (e) {
        if (e.status === 404) { setUiState({ jevProposalId: "" }); renderMessage(t("ui.jev.gone", "Der Vorschlag existiert nicht mehr.")); return; }
        renderMessage(`${t("ui.jev.error", "Fehler")}: ${e.message}`, "tc-text-error");
        return;
      }
      if (view.state === "running") {
        renderMessage(t("ui.jev.running", "Jev wird befragt..."));
        if (Date.now() - started > POLL_TIMEOUT_MS) { renderMessage(t("ui.jev.timeout", "Zeitüberschreitung beim Warten auf die Beratung.")); return; }
        await new Promise((r) => setTimeout(r, POLL_MS));
        continue;
      }
      if (view.state === "failed") { renderMessage(`${t("ui.jev.error", "Fehler")}: ${view.error || "?"}`, "tc-text-error"); return; }
      render(view);
      return;
    }
  }

  async function computeScanMetrics() {
    const scan = await api.get(API_ENDPOINTS.scan.latest);
    const objectName = String(getScanData().object_name || scan?.object_name || scan?.target || "").trim();
    const started = await api.post(API_ENDPOINTS.scan.metrics, {
      input_path: scan?.input_path || scan?.input_dirs?.[0] || "",
      object_name: objectName,
      target: objectName,
      frame_count: scan?.frames_detected || scan?.frames_total || scan?.frame_count || 0,
    });
    if (started?.cached && started?.result) return;
    if (!started?.job_id) throw new Error(t("ui.jev.metrics_failed", "Bildstatistik konnte nicht berechnet werden."));
    await pollJob(started.job_id, { endpoint: API_ENDPOINTS.scan.jobStatus, timeoutMs: 600000, onDone: (job) => job?.data?.result || job?.data || null });
  }

  async function requestAdvice() {
    const yaml = currentDraftYaml();
    if (!yaml.trim()) { toastError(t("ui.jev.request_failed", "Anfrage fehlgeschlagen"), t("ui.jev.no_draft", "Kein Config-Entwurf geladen.")); return; }
    requestBtn.disabled = true;
    try {
      const send = () => api.post(API_ENDPOINTS.decisions.advice, { yaml, locked_paths: [], session_context: sessionContext() });
      let r;
      try {
        r = await send();
      } catch (e) {
        // The image statistics are computed on demand, like the AI card does, instead of asking the user to do it first.
        if (e.payload?.code !== "NO_SCAN_METRICS") throw e;
        renderMessage(t("ui.jev.computing_metrics", "Bildstatistik wird berechnet..."));
        await computeScanMetrics();
        r = await send();
      }
      setUiState({ jevProposalId: r.proposal_id });
      await poll(r.proposal_id);
    } catch (e) {
      const code = e.payload?.code;
      const hint = code === "NO_SCAN" ? t("ui.jev.need_scan", "Zuerst einen Scan ausführen.")
        : code === "NO_SCAN_METRICS" ? t("ui.jev.need_metrics", "Zuerst die Scan-Metriken berechnen.") : e.message;
      toastError(t("ui.jev.request_failed", "Anfrage fehlgeschlagen"), hint);
    } finally {
      requestBtn.disabled = false;
    }
  }

  async function applyToDraft(id) {
    try {
      const r = await api.post(API_ENDPOINTS.decisions.apply(id), { yaml: currentDraftYaml(), locked_paths: [], session_context: sessionContext() });
      const parsed = parseYaml(r.patched_yaml);
      setConfigState({ draft: deepClone(parsed), draftYaml: r.patched_yaml, dirty: true,
        jevAppliedProposalId: id, jevSavedProposalId: "" });
      onDraftApplied?.();
      toastSuccess(t("ui.jev.applied_toast", "Vorschlag in den Entwurf übernommen (nicht gespeichert)"));
      await poll(id);
    } catch (e) {
      const code = e.payload?.code;
      if (code === "PROPOSAL_STALE" || code === "DRAFT_CHANGED") {
        renderMessage(t("ui.jev.result.stale", "Veraltet: Entwurf oder Scan haben sich geändert") + " — " + t("ui.jev.request_again", "bitte neu anfordern."), "tc-text-error");
      } else {
        toastError(t("ui.jev.apply_failed", "Übernehmen fehlgeschlagen"), `${e.message}${e.payload?.reasons ? ": " + e.payload.reasons.join(", ") : ""}`);
      }
    }
  }

  // Restore after reload/navigation: the proposal state lives on the backend, only its id is remembered here.
  const remembered = getUiState().jevProposalId;
  if (remembered) poll(remembered);
  return page;
}
