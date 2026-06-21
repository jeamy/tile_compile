// js/components/header.js – Top-Bar mit Logo, Tabs, Status, Locale, Theme-Toggle

import { el } from "../utils/dom.js";
import { getUiState, setUiState } from "../state/ui-state.js";
import { t } from "../i18n/i18n.js";

export function createHeader(onTabChange) {
  const ui = getUiState();

  const header = el("header", { class: "tc-header" },
    el("div", { class: "tc-header-logo" }, "Tile Compile"),
    el("div", { class: "tc-header-tabs", id: "header-tabs" },
      createTab("processing", t("ui.tab.processing", "Processing"), ui.activeTab === "processing", onTabChange),
      createTab("tools", t("ui.tab.tools", "Tools"), ui.activeTab === "tools", onTabChange),
      createTab("history", t("ui.tab.history", "History"), ui.activeTab === "history", onTabChange),
    ),
    el("div", { class: "tc-header-right" },
      el("div", { class: "tc-header-status" },
        el("span", { class: "tc-badge tc-badge-success" }, "\u25cf " + t("ui.badge.run_ready", "Run ready")),
        el("span", { class: "tc-badge tc-badge-success" }, "\u25cf " + t("ui.badge.guardrails_ok", "Guardrails OK")),
      ),
      el("div", { class: "tc-locale-switch" },
        el("button", {
          class: `tc-locale-btn${ui.locale === "de" ? " active" : ""}`,
          onclick: () => {
            if (getUiState().locale !== "de") {
              setUiState({ locale: "de" });
              window.location.reload();
            }
          },
        }, "DE"),
        el("span", { class: "tc-locale-sep" }, "|"),
        el("button", {
          class: `tc-locale-btn${ui.locale === "en" ? " active" : ""}`,
          onclick: () => {
            if (getUiState().locale !== "en") {
              setUiState({ locale: "en" });
              window.location.reload();
            }
          },
        }, "EN"),
      ),
      el("button", {
        class: "tc-theme-toggle",
        title: t("ui.theme.toggle", "Theme wechseln"),
        onclick: () => toggleTheme(),
      }, getUiState().theme === "dark" ? "\u2600" : "\u263d"),
    ),
  );

  return header;
}

export function updateActiveTab(tabId) {
  const bar = document.getElementById("header-tabs");
  if (!bar) return;
  for (const btn of bar.querySelectorAll(".tc-tab")) {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
  }
}

function createTab(id, label, active, onTabChange) {
  return el("button", {
    class: `tc-tab${active ? " active" : ""}`,
    "data-tab": id,
    onclick: () => {
      updateActiveTab(id);
      setUiState({ activeTab: id });
      onTabChange(id);
    },
  }, label);
}

function toggleTheme() {
  const current = getUiState().theme;
  const next = current === "dark" ? "light" : "dark";
  setUiState({ theme: next });
  document.documentElement.setAttribute("data-theme", next);
  const btn = document.querySelector(".tc-theme-toggle");
  if (btn) btn.textContent = next === "dark" ? "\u2600" : "\u263d";
}
