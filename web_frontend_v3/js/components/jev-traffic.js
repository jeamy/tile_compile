import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";

// Jev traffic: the requests sent to and the answers received from the provider (one JSON line per event), read from the
// sidecar's Jev log. Same layout as the AI traffic panel. The log is redacted when written (no key, no paths).

export function createJevTrafficPanel(id = "jev-traffic") {
  const arrow = el("span", { "aria-hidden": "true" }, "▾");
  const status = el("span", { class: "tc-text-muted tc-text-sm", id: `${id}-status` }, t("ui.state.not_loaded", "nicht geladen"));
  const viewer = el("div", { class: "tc-log-viewer", id, style: { maxHeight: "260px" } },
    el("div", { class: "tc-text-muted" }, t("ui.state.no_traffic", "Keine Daten")));

  async function refresh() {
    status.textContent = t("ui.state.loading", "Lädt...");
    try {
      const payload = await api.get(API_ENDPOINTS.decisions.log(500));
      const items = Array.isArray(payload?.items) ? payload.items : [];
      clear(viewer);
      if (!items.length) viewer.append(el("div", { class: "tc-text-muted" }, t("ui.state.no_traffic", "Keine Daten")));
      for (const line of items) viewer.append(el("div", { class: "tc-text-sm tc-mono" }, String(line)));
      viewer.scrollTop = viewer.scrollHeight;
      const enabled = payload?.enabled === false ? t("ui.state.disabled", "deaktiviert") : t("ui.state.enabled", "aktiv");
      status.textContent = `${enabled} · ${t("ui.pi.traffic_count", "{count} Zeilen", { count: payload?.count ?? items.length })}`;
    } catch (error) {
      status.textContent = error?.message || String(error);
    }
  }

  const body = el("div", { class: "tc-accordion-body" },
    el("div", { class: "tc-flex tc-gap-2 tc-items-center tc-mb-2" },
      el("button", { class: "tc-btn tc-btn-sm", onclick: () => refresh(), title: t("ui.jev.tooltip.refresh_traffic", "Lädt das Jev-Protokoll (Anfragen und Antworten) aus dem Sidecar.") }, t("ui.button.refresh", "Aktualisieren")),
      status),
    viewer);
  const panel = el("div", { class: "tc-accordion open", id: `${id}-panel` });
  const header = el("div", { class: "tc-accordion-header", role: "button", tabindex: "0", "aria-expanded": "true", onclick: toggle,
    onkeydown: (event) => { if (event.key !== "Enter" && event.key !== " ") return; event.preventDefault(); toggle(); } },
    arrow, " " + t("ui.jev.traffic_title", "Jev-Datenverkehr"));
  function toggle() {
    const open = panel.classList.toggle("open");
    arrow.textContent = open ? "▾" : "▸";
    header.setAttribute("aria-expanded", String(open));
  }
  panel.append(header, body);
  return { element: panel, refresh };
}
