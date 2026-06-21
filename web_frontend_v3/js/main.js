// js/main.js – Entry point: Initialisierung, Tab-Routing

import { initUiState, getUiState, setUiState } from "./state/ui-state.js";
import { setApi } from "./state/store.js";
import { api } from "./api/client.js";
import { loadLocale, t } from "./i18n/i18n.js";
import { createHeader, updateActiveTab } from "./components/header.js";
import { createSubTabs } from "./components/sub-tabs.js";
import { createGuardrailBadges } from "./components/guardrail-badges.js";
import { toast } from "./components/toast.js";
import { el, clear } from "./utils/dom.js";

import { createProcessingPage } from "./pages/processing.js";
import { createToolsPage } from "./pages/tools.js";
import { createHistoryPage } from "./pages/history.js";

setApi(api);

const SUB_TABS = {
  processing: [
    { id: "input-scan", label: t("ui.subtab.input_scan", "Input & Scan") },
    { id: "parameter", label: t("ui.subtab.parameter", "Parameter") },
    { id: "run-monitor", label: t("ui.subtab.run_monitor", "Run Monitor") },
  ],
  tools: [
    { id: "raw-stack", label: t("ui.subtab.raw_stack", "Raw Stack") },
    { id: "astrometry", label: t("ui.subtab.astrometry", "Astrometry") },
    { id: "pcc", label: t("ui.subtab.pcc", "PCC") },
  ],
  history: [
    { id: "run-history", label: t("ui.subtab.run_history", "Run History") },
  ],
};

const PAGES = {
  processing: createProcessingPage,
  tools: createToolsPage,
  history: createHistoryPage,
};

let currentTab = null;
let currentSubTab = null;
let contentRoot = null;

async function init() {
  initUiState();
  const ui = getUiState();
  await loadLocale(ui.locale);

  // Apply saved theme
  if (ui.theme) {
    document.documentElement.setAttribute("data-theme", ui.theme);
  }

  const appRoot = document.getElementById("app-root");
  clear(appRoot);

  const header = createHeader(navigateToTab);
  appRoot.appendChild(header);

  const subTabBar = el("div", { class: "tc-subtab-bar", id: "subtab-bar" });
  appRoot.appendChild(subTabBar);

  contentRoot = el("div", { class: "tc-content", id: "content" });
  appRoot.appendChild(contentRoot);

  const footer = el("footer", { class: "tc-footer" },
    el("span", {}, t("ui.footer.default", "Bereit")),
  );
  appRoot.appendChild(footer);

  const tab = ui.activeTab || "processing";
  navigateToTab(tab);

  setupKeyboardShortcuts();
}

function navigateToTab(tab) {
  if (!PAGES[tab]) tab = "processing";
  currentTab = tab;
  setUiState({ activeTab: tab });
  updateActiveTab(tab);

  renderSubTabs(tab);
  renderContent(tab);
}

function renderSubTabs(tab) {
  const bar = document.getElementById("subtab-bar");
  clear(bar);

  const tabs = SUB_TABS[tab];
  if (!tabs || tabs.length === 0) return;

  const subTabs = createSubTabs(tabs, tab, (subId) => {
    currentSubTab = subId;
    renderContent(tab, subId);
  });
  bar.appendChild(subTabs);

  if (tab === "processing") {
    bar.appendChild(createGuardrailBadges());
  }
}

function renderContent(tab, subId) {
  if (!contentRoot) return;
  clear(contentRoot);

  const ui = getUiState();
  if (!subId) {
    subId = ui.activeSubTab[tab] || SUB_TABS[tab]?.[0]?.id;
  }
  currentSubTab = subId;

  const pageFactory = PAGES[tab];
  if (pageFactory) {
    const page = pageFactory(subId);
    if (page) contentRoot.appendChild(page);
  }
}

window.addEventListener("hashchange", () => {
  const hash = window.location.hash.slice(1);
  if (hash && PAGES[hash]) {
    navigateToTab(hash);
  }
});

window.addEventListener("tc-subtab-change", () => {
  const ui = getUiState();
  const tab = ui.activeTab || "processing";
  renderSubTabs(tab);
  renderContent(tab);
});

function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    // Skip if typing in input/textarea/select
    const tag = (e.target.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag)) return;

    // Ctrl/Cmd+Shift+D: Toggle dark mode
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "D") {
      e.preventDefault();
      const current = getUiState().theme || "light";
      const next = current === "dark" ? "light" : "dark";
      setUiState({ theme: next });
      document.documentElement.setAttribute("data-theme", next);
      return;
    }

    // Ctrl/Cmd+Shift+L: Toggle locale
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "L") {
      e.preventDefault();
      const newLocale = getUiState().locale === "de" ? "en" : "de";
      setUiState({ locale: newLocale });
      window.location.reload();
      return;
    }

    // 1/2/3: Switch main tabs
    if (e.key === "1" && !e.ctrlKey && !e.metaKey) {
      navigateToTab("processing");
    } else if (e.key === "2" && !e.ctrlKey && !e.metaKey) {
      navigateToTab("tools");
    } else if (e.key === "3" && !e.ctrlKey && !e.metaKey) {
      navigateToTab("history");
    }

    // Arrow Left/Right: Switch sub-tabs
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const ui = getUiState();
      const tab = ui.activeTab || "processing";
      const subs = SUB_TABS[tab];
      if (!subs || subs.length < 2) return;
      const currentIdx = subs.findIndex(s => s.id === currentSubTab);
      if (currentIdx < 0) return;
      const nextIdx = e.key === "ArrowRight"
        ? Math.min(currentIdx + 1, subs.length - 1)
        : Math.max(currentIdx - 1, 0);
      if (nextIdx !== currentIdx) {
        currentSubTab = subs[nextIdx].id;
        renderContent(tab, subs[nextIdx].id);
      }
    }
  });
}

// Global error handler
window.addEventListener("error", (e) => {
  console.error("Global error:", e.error || e.message);
  toast("Error", String(e.message || "Unknown error"), "error");
});

window.addEventListener("unhandledrejection", (e) => {
  console.error("Unhandled rejection:", e.reason);
  toast("Error", String(e.reason?.message || e.reason || "Unhandled promise rejection"), "error");
});

init().catch((e) => {
  console.error("Init failed:", e);
  toast("Initialization failed", String(e.message || e), "error");
});
