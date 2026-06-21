// js/components/explain-panel.js – Explain-Panel für Parameter

import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";

export function createExplainPanel() {
  const wrapper = el("div", { class: "tc-card tc-param-explain" },
    el("div", { class: "tc-card-title" }, t("ui.title.explain", "Explain")),
    el("div", { class: "tc-flex-col tc-gap-3", id: "explain-body" },
      el("div", { class: "tc-text-muted tc-text-sm" },
        t("ui.state.select_param", "W\u00e4hle einen Parameter, um Erkl\u00e4rungen zu sehen."),
      ),
    ),
  );
  return wrapper;
}

export function updateExplainPanel(entry) {
  const body = document.getElementById("explain-body");
  if (!body) return;
  clear(body);

  if (!entry) {
    body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" },
      t("ui.state.select_param", "W\u00e4hle einen Parameter, um Erkl\u00e4rungen zu sehen."),
    ));
    return;
  }

  const rangeStr = entry.range || (entry.minimum !== undefined || entry.maximum !== undefined
    ? `${entry.minimum ?? ""}..${entry.maximum ?? ""}`
    : "");

  const fields = [
    { label: t("ui.label.label", "Label"), value: entry.label || entry.path || "" },
    { label: t("ui.label.category", "Kategorie"), value: entry.category || "" },
    { label: t("ui.label.type", "Typ"), value: entry.type || "" },
    { label: t("ui.label.default", "Default"), value: entry.default ?? entry.defaultValue ?? "" },
    { label: t("ui.label.range", "Range"), value: rangeStr },
  ];

  for (const f of fields) {
    if (f.value === "" || f.value === undefined || f.value === null) continue;
    body.appendChild(el("div", {},
      el("div", { class: "tc-label" }, f.label),
      el("div", { class: "tc-text-sm tc-mono" }, String(f.value)),
    ));
  }

  if (entry.deprecated) {
    body.appendChild(el("div", { class: "tc-mt-1" },
      el("span", { class: "tc-badge", style: { background: "var(--error-bg)", color: "var(--error)", padding: "2px 8px", borderRadius: "4px", fontSize: "var(--text-xs)" } },
        t("ui.label.deprecated", "deprecated")),
    ));
  }

  if (entry.description) {
    body.appendChild(el("div", { class: "tc-mt-2" },
      el("div", { class: "tc-label" }, t("ui.label.description", "Beschreibung")),
      el("div", { class: "tc-text-sm tc-text-muted", style: { lineHeight: "1.5" } }, entry.description),
    ));
  }

  if (entry.enum && entry.enum.length > 0) {
    body.appendChild(el("div", { class: "tc-mt-2" },
      el("div", { class: "tc-label" }, t("ui.label.options", "Optionen")),
      el("div", { class: "tc-text-sm tc-mono" }, entry.enum.join(", ")),
    ));
  }
}
