// js/components/scan-result-card.js – Scan-Ergebnis Anzeige

import { el, statItem } from "../utils/dom.js";
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
      statItem(t("ui.label.frame_count", "Frames"), result.frames_detected ?? result.frame_count ?? result.total ?? 0),
      statItem(t("ui.label.color_mode", "Farbmodus"), result.color_mode || result.colour_mode || "\u2014"),
      statItem(t("ui.label.bayer_pattern", "Bayer"), result.bayer_pattern || "\u2014"),
      statItem(t("ui.label.dimensions", "Dimension"), result.image_width && result.image_height ? `${result.image_width}\u00d7${result.image_height}` : (result.dimensions || result.size || "\u2014")),
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

  const frames = Array.isArray(result.frames) ? result.frames : [];
  const framesTotal = result.frames_total || frames.length;
  const framesTruncated = result.frames_truncated || false;
  if (frames.length > 0) {
    card.appendChild(createFrameListSection(frames, framesTotal, framesTruncated));
  }

  return card;
}

function createFrameListSection(frames, total, truncated) {
  const toggleBtn = el("button", {
    class: "tc-btn tc-btn-sm tc-mt-2",
    onclick: () => {
      const body = document.getElementById("scan-frame-list-body");
      if (!body) return;
      const isOpen = body.style.display !== "none";
      body.style.display = isOpen ? "none" : "";
      toggleBtn.textContent = isOpen
        ? `\u25b6 ${t("ui.button.show_frames", "Frames anzeigen")} (${total})`
        : `\u25bc ${t("ui.button.hide_frames", "Frames ausblenden")} (${total})`;
    },
  }, `\u25b6 ${t("ui.button.show_frames", "Frames anzeigen")} (${total})`);

  const tableWrap = el("div", {
    id: "scan-frame-list-body",
    style: { display: "none", "max-height": "400px", overflow: "auto", "margin-top": "var(--space-2)" },
  });

  const suffix = truncated ? ` / ${total}` : "";
  tableWrap.appendChild(el("div", { class: "tc-text-muted tc-text-sm tc-mb-1" },
    `${frames.length}${suffix ? ` ${t("ui.label.of", "von")} ${suffix}` : ""} ${t("ui.label.frames_shown", "Frames")}`));

  const table = el("table", { class: "tc-frame-table" },
    el("thead", {},
      el("tr", {},
        el("th", {}, "#"),
        el("th", {}, t("ui.label.file_name", "Dateiname")),
        el("th", {}, t("ui.label.camera", "Kamera")),
        el("th", {}, t("ui.label.exposure", "Belichtung")),
        el("th", {}, t("ui.label.gain", "Gain")),
        el("th", {}, t("ui.label.temp", "Temp")),
      ),
    ),
    el("tbody", {},
      ...frames.map((f, i) => el("tr", {},
        el("td", { class: "tc-mono tc-text-sm" }, String(i)),
        el("td", { class: "tc-mono tc-text-sm", title: f.abs_path || "" }, f.file_name || f.abs_path || "\u2014"),
        el("td", { class: "tc-text-sm" }, f.camera || "\u2014"),
        el("td", { class: "tc-text-sm" }, f.exposure_seconds != null ? `${f.exposure_seconds}s` : "\u2014"),
        el("td", { class: "tc-text-sm" }, f.gain != null ? String(f.gain) : "\u2014"),
        el("td", { class: "tc-text-sm" }, f.temperature_c != null ? `${f.temperature_c}\u00b0C` : "\u2014"),
      )),
    ),
  );
  tableWrap.appendChild(table);

  return el("div", { class: "tc-mt-2" }, toggleBtn, tableWrap);
}

function calBadge(label, status) {
  const cls = status === "ok" ? "tc-badge-success" : status === "error" ? "tc-badge-error" : "tc-badge-warning";
  return el("span", { class: `tc-badge ${cls}` }, `${label}: ${status || "\u2014"}`);
}

function guardrailBadge(g) {
  const cls = g.status === "ok" ? "tc-badge-success" : g.status === "error" ? "tc-badge-error" : "tc-badge-warning";
  return el("span", { class: `tc-badge ${cls}` }, `${g.name}: ${g.status}`);
}
