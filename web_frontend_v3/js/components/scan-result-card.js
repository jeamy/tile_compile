// js/components/scan-result-card.js – Scan-Ergebnis Anzeige

import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";

export function createScanResultCard(result) {
  if (!result) {
    return el("div", { class: "tc-card" },
      el("div", { class: "tc-card-title" }, t("ui.title.scan_results", "Scan-Ergebnis")),
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_scan", "Kein Scan durchgefuehrt.")),
    );
  }

  const card = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.scan_results", "Scan-Ergebnis")),
    el("div", { class: "tc-grid-2" },
      statItem(t("ui.label.frame_count", "Frames"), result.frame_count ?? result.total ?? 0),
      statItem(t("ui.label.color_mode", "Farbmodus"), result.color_mode || result.colour_mode || "\u2014"),
      statItem(t("ui.label.bayer_pattern", "Bayer"), result.bayer_pattern || "\u2014"),
      statItem(t("ui.label.dimensions", "Dimension"), result.dimensions || result.size || "\u2014"),
      statItem(t("ui.label.bit_depth", "Bittiefe"), result.bit_depth || "\u2014"),
      statItem(t("ui.label.filter", "Filter"), result.filter || "\u2014"),
    ),
  );

  if (result.calibration_status) {
    card.appendChild(el("div", { class: "tc-mt-2" },
      el("div", { class: "tc-label" }, t("ui.label.cal_status", "Kalibrierung")),
      el("div", { class: "tc-flex tc-gap-2" },
        calBadge("Bias", result.calibration_status.bias),
        calBadge("Dark", result.calibration_status.dark),
        calBadge("Flat", result.calibration_status.flat),
      ),
    ));
  }

  if (result.guardrails) {
    card.appendChild(el("div", { class: "tc-mt-2" },
      el("div", { class: "tc-label" }, t("ui.label.guardrails", "Guardrails")),
      el("div", { class: "tc-flex tc-gap-2" },
        ...result.guardrails.map(g => guardrailBadge(g)),
      ),
    ));
  }

  return card;
}

function statItem(label, value) {
  return el("div", {},
    el("div", { class: "tc-label" }, label),
    el("div", { class: "tc-text-sm tc-mono" }, String(value)),
  );
}

function calBadge(label, status) {
  const cls = status === "ok" ? "tc-badge-success" : status === "error" ? "tc-badge-error" : "tc-badge-warning";
  return el("span", { class: `tc-badge ${cls}` }, `${label}: ${status || "\u2014"}`);
}

function guardrailBadge(g) {
  const cls = g.status === "ok" ? "tc-badge-success" : g.status === "error" ? "tc-badge-error" : "tc-badge-warning";
  return el("span", { class: `tc-badge ${cls}` }, `${g.name}: ${g.status}`);
}
