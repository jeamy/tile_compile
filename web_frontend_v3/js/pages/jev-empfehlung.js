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
import { autoScanForAnalysis } from "./ai-empfehlung.js";
import { createJevTrafficPanel } from "../components/jev-traffic.js";

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

// Plain-language text for the machine reason codes of a proposal or an excluded candidate; unknown codes are shown as is.
const REASON_TEXT = {
  "provider:mode_off": ["ui.jev.reason.mode_off", "Jev ist ausgeschaltet. Oben den Betriebsmodus auf \u201eVorschlagen\u201c stellen."],
  "provider:no_api_key": ["ui.jev.reason.no_api_key", "Es ist kein API-Schl\u00fcssel f\u00fcr Jev hinterlegt."],
  "provider:sidecar_unreachable": ["ui.jev.reason.sidecar_unreachable", "Der PI-Sidecar ist nicht erreichbar."],
  provider_unavailable: ["ui.jev.reason.provider_unavailable", "Jev wurde nicht befragt."],
  validation_not_run: ["ui.jev.reason.validation_not_run", "Es wurde nichts gepr\u00fcft, weil keine Antwort von Jev vorlag."],
  experimental_not_enabled: ["ui.jev.reason.experimental_not_enabled", "Experimenteller Kandidat: oben \u201eExperimentelle Kandidaten anzeigen\u201c aktivieren."],
  already_active: ["ui.jev.reason.already_active", "Ist im Entwurf bereits so eingestellt."],
  candidate_rejected: ["ui.jev.reason.candidate_rejected", "Verworfen: die bisherigen Tests zeigen keinen Nutzen."],
  policy_thresholds_not_frozen: ["ui.jev.reason.thresholds_not_frozen", "F\u00fcr diesen Kandidaten sind noch keine Freigabeschwellen festgelegt."],
  path_locked: ["ui.jev.reason.path_locked", "Der Parameter ist gesperrt."],
};

function reasonText(code) {
  const c = String(code);
  const m = /^precondition_failed:config_equals:(.+?)=(.+)$/.exec(c);
  if (m) return t("ui.jev.reason.config_equals", "Nur gepr\u00fcft, wenn {path} = {value} ist; der Entwurf hat einen anderen Wert.", { path: m[1], value: m[2] });
  const entry = REASON_TEXT[c];
  return entry ? t(entry[0], entry[1]) : c;
}

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
  const traffic = createJevTrafficPanel();
  traffic.refresh();
  const requestBtn = el("button", { class: "tc-btn tc-btn-primary", id: "jev-request", onclick: () => requestAdvice() }, t("ui.jev.request", "Jev-Empfehlung anfordern"));
  const rescanBtn = el("button", { class: "tc-btn", id: "jev-rescan", title: t("ui.jev.tooltip.rescan", "F\u00fchrt den Scan mit den aktuellen Einstellungen unter Input & Scan erneut aus und berechnet die Bildstatistik neu."), onclick: () => rescan() }, t("ui.jev.rescan", "Scan neu starten"));
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
        objectClassSelect, rescanBtn, requestBtn),
      resultBox,
      el("div", { class: "tc-mt-4" }, traffic.element),
    ),
  );

  let pollToken = 0;

  function renderMessage(text, cls = "tc-text-muted") {
    resultBox.replaceChildren(el("div", { class: `tc-text-sm ${cls}` }, text));
  }

  function badge(text, kind) { return el("span", { class: `tc-badge tc-badge-${kind}`, style: { marginRight: "6px" } }, text); }

  // One recommendation (one candidate group) as a card. Only a validated proposal with a change can be selected.
  const selected = new Set();
  let currentViews = [];

  function isApplicable(view) {
    const status = view?.proposal?.status;
    return (status === "validated" || status === "presented") && (view.comparison || []).some((c) => c.changed);
  }

  function renderCard(view) {
    const nodes = [];
    const proposal = view.proposal || {};
    const status = proposal.status || "unavailable";
    const [key, fallback] = STATUS_TEXT[status] || STATUS_TEXT.unavailable;
    const applicable = isApplicable(view);
    const title = proposal.candidate_id && proposal.candidate_id !== "keep_current" ? proposal.candidate_id : (view.group || t("ui.jev.no_change_group", "keine Änderung"));
    const head = el("div", { class: "tc-flex tc-items-center tc-gap-2 tc-flex-wrap" },
      applicable ? el("input", { type: "checkbox", checked: selected.has(view.proposal_id), "aria-label": title,
        onchange: (e) => { if (e.target.checked) selected.add(view.proposal_id); else selected.delete(view.proposal_id); updateActions(); } }) : null,
      el("strong", {}, title),
      el("span", { class: "tc-text-sm tc-text-muted" }, t(key, fallback)),
      view.mode ? badge(`${t("ui.jev.mode", "Betriebsmodus")}: ${view.mode}`, "info") : null,
      view.evidence?.experimental ? badge(t("ui.jev.experimental", "experimentell"), "warning") : null,
    );
    nodes.push(head);
    if (view.shadow) nodes.push(el("div", { class: "tc-text-sm tc-text-muted tc-mt-2" }, t("ui.jev.shadow_note", "Shadow-Modus: die Antwort wurde nur aufgezeichnet, es gibt keinen anwendbaren Vorschlag.")));
    if (view.synthetic_baseline && !view.model_called)
      nodes.push(el("div", { class: "tc-text-sm tc-text-muted tc-mt-2" }, t("ui.jev.no_model_call", "Das Modell wurde nicht befragt: es war kein Kandidat mit Änderung anwendbar (siehe Ausschlussgründe).")));
    const reasons = (proposal.reason_codes || []).filter((r) => r !== "validated");
    if (reasons.length) nodes.push(el("div", { class: "tc-text-sm tc-text-muted tc-mt-2" }, `${t("ui.jev.reasons", "Gründe")}: ${reasons.map(reasonText).join(" ")}`));
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
    return el("div", { class: "tc-card tc-mt-2" }, ...nodes.filter(Boolean));
  }

  const selectedBtn = el("button", { class: "tc-btn tc-btn-primary", id: "jev-apply-selected", onclick: () => applySelected(false) }, t("ui.jev.apply_selected", "Ausgewählte anwenden"));
  const allBtn = el("button", { class: "tc-btn", id: "jev-apply-all", onclick: () => applySelected(true) }, t("ui.jev.apply_all", "Alle anwenden"));
  function updateActions() {
    const applicable = currentViews.filter(isApplicable).map((v) => v.proposal_id);
    selectedBtn.disabled = ![...selected].some((id) => applicable.includes(id));
    allBtn.disabled = applicable.length === 0;
  }

  function renderAll(views) {
    currentViews = views;
    const applicable = views.filter(isApplicable);
    for (const v of applicable) if (!selected.has(v.proposal_id) && !v._seen) selected.add(v.proposal_id);
    for (const v of views) v._seen = true;
    const nodes = [];
    const actionable = views.filter((v) => isApplicable(v) || ["applied_to_draft", "stale"].includes(v.proposal?.status));
    const unavailable = views.some((v) => v.proposal?.status === "unavailable");
    if (unavailable) nodes.push(el("div", { class: "tc-text-sm tc-mt-2 tc-text-error" }, t("ui.jev.not_asked", "Jev wurde nicht befragt, es gibt daher keine Empfehlung (Gr\u00fcnde unten).")));
    else if (!actionable.length) nodes.push(el("div", { class: "tc-text-sm tc-mt-2" }, t("ui.jev.result.no_change", "Keine Änderung empfohlen")));
    for (const v of views) nodes.push(renderCard(v));
    if (applicable.length)
      nodes.push(el("div", { class: "tc-mt-2 tc-flex tc-gap-2" }, selectedBtn, allBtn));
    const seenEx = new Set();
    const excluded = [];
    for (const v of views) for (const x of v.excluded || []) if (!seenEx.has(x.candidate_id)) { seenEx.add(x.candidate_id); excluded.push(x); }
    if (excluded.length) {
      nodes.push(el("details", { class: "tc-mt-2" },
        el("summary", { class: "tc-text-sm" }, t("ui.jev.details", "Angebotene und ausgeschlossene Kandidaten")),
        ...excluded.map((x) => el("div", { class: "tc-text-sm tc-text-muted" }, `${x.candidate_id}: ${(x.reasons || []).map(reasonText).join(" ")}`))));
    }
    resultBox.replaceChildren(...nodes);
    updateActions();
  }

  // Waits for every proposal of a request; a failed or vanished one is shown as a placeholder view, the others still render.
  async function pollAll(ids) {
    const token = ++pollToken;
    const started = Date.now();
    const views = new Array(ids.length).fill(null);
    while (token === pollToken) {
      let running = false;
      await Promise.all(ids.map(async (id, k) => {
        if (views[k] && views[k].state !== "running") return;
        try {
          views[k] = await api.get(API_ENDPOINTS.decisions.byId(id));
        } catch (e) {
          views[k] = e.status === 404 ? { state: "done", proposal_id: id, proposal: { status: "unavailable", reason_codes: ["proposal_gone"] } }
            : { state: "done", proposal_id: id, proposal: { status: "unavailable", reason_codes: [e.message] } };
        }
        if (views[k].state === "failed") views[k] = { state: "done", proposal_id: id, proposal: { status: "unavailable", reason_codes: [views[k].error || "failed"] } };
        if (views[k].state === "running") running = true;
      }));
      if (token !== pollToken) return;
      if (!running) { renderAll(views); return; }
      renderMessage(t("ui.jev.running", "Jev wird befragt..."));
      if (Date.now() - started > POLL_TIMEOUT_MS) { renderMessage(t("ui.jev.timeout", "Zeitüberschreitung beim Warten auf die Beratung.")); return; }
      await new Promise((r) => setTimeout(r, POLL_MS));
    }
  }

  async function computeScanMetrics(force = false) {
    const scan = await api.get(API_ENDPOINTS.scan.latest);
    const objectName = String(getScanData().object_name || scan?.object_name || scan?.target || "").trim();
    const started = await api.post(API_ENDPOINTS.scan.metrics, {
      input_path: scan?.input_path || scan?.input_dirs?.[0] || "",
      object_name: objectName,
      target: objectName,
      frame_count: scan?.frames_detected || scan?.frames_total || scan?.frame_count || 0,
      force,
    });
    if (started?.cached && started?.result) return;
    if (!started?.job_id) throw new Error(t("ui.jev.metrics_failed", "Bildstatistik konnte nicht berechnet werden."));
    await pollJob(started.job_id, { endpoint: API_ENDPOINTS.scan.jobStatus, timeoutMs: 600000, onDone: (job) => job?.data?.result || job?.data || null });
  }

  // A new scan also makes the stored proposal stale (the backend compares scan identity on apply), so the result box is cleared.
  async function rescan() {
    rescanBtn.disabled = true;
    requestBtn.disabled = true;
    try {
      renderMessage(t("ui.jev.scanning", "Scan wird ausgef\u00fchrt..."));
      await autoScanForAnalysis();
      renderMessage(t("ui.jev.computing_metrics", "Bildstatistik wird berechnet..."));
      await computeScanMetrics(true);
      renderMessage(t("ui.jev.rescan_done", "Scan und Bildstatistik sind aktuell. Empfehlung neu anfordern."));
    } catch (e) {
      toastError(t("ui.jev.rescan_failed", "Scan fehlgeschlagen"), e.message);
      renderMessage(`${t("ui.jev.error", "Fehler")}: ${e.message}`, "tc-text-error");
    } finally {
      rescanBtn.disabled = false;
      requestBtn.disabled = false;
    }
  }

  async function requestAdvice() {
    const yaml = currentDraftYaml();
    if (!yaml.trim()) { toastError(t("ui.jev.request_failed", "Anfrage fehlgeschlagen"), t("ui.jev.no_draft", "Kein Config-Entwurf geladen.")); return; }
    requestBtn.disabled = true;
    try {
      const send = () => api.post(API_ENDPOINTS.decisions.advice, { yaml, locked_paths: [], session_context: sessionContext() });
      // Scan and image statistics are prepared on demand, like the AI card does, instead of asking the user to do it first.
      let r;
      for (let attempt = 0; ; ++attempt) {
        try {
          r = await send();
          break;
        } catch (e) {
          const code = e.payload?.code;
          if (attempt >= 2 || (code !== "NO_SCAN" && code !== "NO_SCAN_METRICS")) throw e;
          if (code === "NO_SCAN") {
            renderMessage(t("ui.jev.scanning", "Scan wird ausgef\u00fchrt..."));
            await autoScanForAnalysis();
          } else {
            renderMessage(t("ui.jev.computing_metrics", "Bildstatistik wird berechnet..."));
            await computeScanMetrics();
          }
        }
      }
      const ids = Array.isArray(r.proposal_ids) && r.proposal_ids.length ? r.proposal_ids : [r.proposal_id];
      selected.clear();
      setUiState({ jevProposalIds: ids });
      await pollAll(ids);
    } catch (e) {
      const code = e.payload?.code;
      const hint = code === "NO_SCAN" ? t("ui.jev.need_scan", "Zuerst einen Scan ausführen.")
        : code === "NO_SCAN_METRICS" ? t("ui.jev.need_metrics", "Zuerst die Scan-Metriken berechnen.") : e.message;
      toastError(t("ui.jev.request_failed", "Anfrage fehlgeschlagen"), hint);
    } finally {
      requestBtn.disabled = false;
      traffic.refresh();
    }
  }

  async function applySelected(all) {
    const ids = currentViews.filter(isApplicable).map((v) => v.proposal_id).filter((id) => all || selected.has(id));
    if (!ids.length) return;
    try {
      const r = await api.post(API_ENDPOINTS.decisions.applyBatch, { proposal_ids: ids, yaml: currentDraftYaml(), locked_paths: [], session_context: sessionContext() });
      const parsed = parseYaml(r.patched_yaml);
      // The save links ONE proposal to the config revision; the others of the batch are listed on it (batch_proposal_ids).
      setConfigState({ draft: deepClone(parsed), draftYaml: r.patched_yaml, dirty: true,
        jevAppliedProposalId: ids[0], jevSavedProposalId: "" });
      onDraftApplied?.();
      toastSuccess(t("ui.jev.applied_toast", "Vorschlag in den Entwurf übernommen (nicht gespeichert)"));
      await pollAll(getUiState().jevProposalIds || ids);
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
  const remembered = getUiState().jevProposalIds;
  if (Array.isArray(remembered) && remembered.length) pollAll(remembered);
  return page;
}
