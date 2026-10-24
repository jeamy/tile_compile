import { el, clear } from "../utils/dom.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { t } from "../i18n/i18n.js";

let catalogRequest = null;

async function loadCatalog() {
  if (!catalogRequest) {
    catalogRequest = api.get(API_ENDPOINTS.pi.reasonCodes).then((catalog) => {
      if (!catalog || typeof catalog.catalog_version !== "string" || !catalog.domains || typeof catalog.domains !== "object") {
        throw new Error("Invalid reason-code catalog");
      }
      return catalog;
    }).catch((error) => {
      catalogRequest = null;
      throw error;
    });
  }
  return catalogRequest;
}

function normalizedValue(value = {}) {
  return {
    reason_codes: [...new Set(Array.isArray(value.reason_codes) ? value.reason_codes.filter(code => typeof code === "string") : [])],
    catalog_version: typeof value.catalog_version === "string" ? value.catalog_version : "",
    text: typeof value.text === "string" ? value.text : "",
    no_reason_given: value.no_reason_given === true,
  };
}

// Collects an optional user rationale in the decision-record wire shape. The caller
// attaches it to an explicit action; this component does not write records itself.
export function createReasonPicker({ domain, value, disabled = false, onChange = () => {} } = {}) {
  const current = normalizedValue(value);
  const chips = el("div", { class: "tc-reason-picker__chips", "aria-live": "polite" },
    el("span", { class: "tc-text-muted tc-text-sm" }, t("ui.reason_picker.loading", "Gruende werden geladen...")));
  const text = el("textarea", {
    class: "tc-input tc-reason-picker__text", rows: 2, maxlength: 1000,
    placeholder: t("ui.reason_picker.text_placeholder", "Optionaler Hinweis"),
    "aria-label": t("ui.reason_picker.text", "Eigener Grund"), value: current.text,
  });
  const noReason = el("input", { type: "checkbox", checked: current.no_reason_given,
    "aria-label": t("ui.reason_picker.no_reason", "Keinen Grund angeben") });
  const noReasonLabel = el("label", { class: "tc-checkbox" }, noReason,
    el("span", {}, t("ui.reason_picker.no_reason", "Keinen Grund angeben")));
  const root = el("fieldset", { class: "tc-reason-picker", disabled },
    el("legend", { class: "tc-label" }, t("ui.reason_picker.title", "Warum?")),
    chips, text, noReasonLabel);

  let catalogVersion = current.catalog_version;
  let codeInputs = [];
  const emit = () => onChange(getValue());
  const syncNoReason = () => {
    const hasRationale = codeInputs.some(input => input.checked) || text.value.trim().length > 0;
    if (hasRationale) noReason.checked = false;
    noReason.disabled = hasRationale;
  };
  const getValue = () => ({
    reason_codes: codeInputs.filter(input => input.checked).map(input => input.value),
    catalog_version: catalogVersion,
    text: text.value.trim(),
    no_reason_given: noReason.checked,
  });

  text.addEventListener("input", () => { syncNoReason(); emit(); });
  noReason.addEventListener("change", () => {
    if (noReason.checked) {
      for (const input of codeInputs) input.checked = false;
      text.value = "";
    }
    syncNoReason(); emit();
  });

  loadCatalog().then((catalog) => {
    const entries = Array.isArray(catalog.domains[domain]) ? catalog.domains[domain] : [];
    catalogVersion = catalog.catalog_version;
    codeInputs = entries.filter(entry => entry && typeof entry.code === "string" && typeof entry.label_key === "string")
      .map((entry) => {
        const input = el("input", { type: "checkbox", value: entry.code, checked: current.reason_codes.includes(entry.code) });
        input.addEventListener("change", () => { syncNoReason(); emit(); });
        return input;
      });
    clear(chips);
    if (!codeInputs.length) {
      chips.append(el("span", { class: "tc-text-muted tc-text-sm" }, t("ui.reason_picker.empty", "Keine Gründe verfügbar.")));
    } else {
      for (let index = 0; index < codeInputs.length; ++index) {
        const entry = entries.filter(item => item && typeof item.code === "string" && typeof item.label_key === "string")[index];
        chips.append(el("label", { class: "tc-checkbox tc-reason-picker__chip" }, codeInputs[index], el("span", {}, t(entry.label_key, entry.code))));
      }
    }
    syncNoReason();
  }).catch(() => {
    clear(chips);
    chips.append(el("span", { class: "tc-text-error tc-text-sm" }, t("ui.reason_picker.unavailable", "Gründe sind derzeit nicht verfügbar.")));
  });

  return { element: root, getValue, setDisabled: (next) => { root.disabled = Boolean(next); } };
}
