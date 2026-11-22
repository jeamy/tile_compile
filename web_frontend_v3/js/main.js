// js/main.js – Entry point: Initialisierung, Tab-Routing

import { initUiState, getUiState, setUiState } from "./state/ui-state.js";
import { loadLocale, t } from "./i18n/i18n.js";
import { createHeader, updateActiveTab } from "./components/header.js";
import { createSubTabs } from "./components/sub-tabs.js";
import { createGuardrailBadges } from "./components/guardrail-badges.js";
import { refreshGuardrails } from "./services/guardrail-service.js";
import { toast } from "./components/toast.js";
import { el, clear } from "./utils/dom.js";
import { onAiChange } from "./state/ai-state.js";
import { createAssistantDock } from "./assistant/dock.js";
import { runPageUnmount } from "./utils/page-lifecycle.js";

import { createProcessingPage } from "./pages/processing.js";
import { createToolsPage } from "./pages/tools.js";
import { createHistoryPage } from "./pages/history.js";

// Labels are looked up when the bar is drawn: at import time the locale is not
// loaded yet and t() would only return the fallback text.
const SUB_TAB_DEFS = {
  processing: [
    ["input-scan", "ui.subtab.input_scan", "Input & Scan"],
    ["parameter", "ui.subtab.parameter", "Parameter"],
    ["run-monitor", "ui.subtab.run_monitor", "Run Monitor"],
  ],
  tools: [
    ["raw-stack", "ui.subtab.raw_stack", "Raw Stack"],
    ["astrometry", "ui.subtab.astrometry", "Astrometry"],
    ["pcc", "ui.subtab.pcc", "PCC"],
    ["ai-settings", "ui.subtab.ai_settings", "AI & API"],
  ],
  history: [
    ["run-history", "ui.subtab.run_history", "Run History"],
  ],
};

function subTabsFor(tab) {
  return (SUB_TAB_DEFS[tab] || []).map(([id, key, fallback]) => ({ id, label: t(key, fallback) }));
}

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
  appRoot.appendChild(createAssistantDock());

  const footer = el("footer", { class: "tc-footer" },
    el("span", {}, t("ui.footer.default", "Bereit")),
  );
  appRoot.appendChild(footer);

  const tab = ui.activeTab || "processing";
  navigateToTab(tab);

  setupKeyboardShortcuts();

  // Global AI analysis state indicator: update sub-tab badge when
  // analysis is running, so the user sees it even on other tabs.
  onAiChange((aiState) => {
    const subTabBtn = document.querySelector('[data-subtab="parameter"]');
    if (!subTabBtn) return;
    const existing = subTabBtn.querySelector('.tc-badge-running');
    if (aiState.loading && !existing) {
      const badge = el('span', { class: 'tc-badge-running', style: { marginLeft: '6px', fontSize: '10px', color: 'var(--accent)' } }, '\u25cf');
      subTabBtn.appendChild(badge);
    } else if (!aiState.loading && existing) {
      existing.remove();
    }
  });
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

  const tabs = subTabsFor(tab);
  if (tabs.length === 0) return;

  const subTabs = createSubTabs(tabs, tab, (subId) => {
    currentSubTab = subId;
    renderContent(tab, subId);
  });
  bar.appendChild(subTabs);

  if (tab === "processing") {
    bar.appendChild(createGuardrailBadges());
    refreshGuardrails();
  }
}

function renderContent(tab, subId) {
  if (!contentRoot) return;
  runPageUnmount();
  clear(contentRoot);

  const ui = getUiState();
  if (!subId) {
    subId = ui.activeSubTab[tab] || subTabsFor(tab)[0]?.id;
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
  // goToSubTab already rendered this tab; a hashchange for it would mount the
  // page a second time.
  if (hash && PAGES[hash] && hash !== currentTab) {
    navigateToTab(hash);
  }
});

window.addEventListener("tc-subtab-change", () => {
  const ui = getUiState();
  navigateToTab(ui.activeTab || "processing");
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
      const subs = SUB_TAB_DEFS[tab];
      if (!subs || subs.length < 2) return;
      const currentIdx = subs.findIndex(([id]) => id === currentSubTab);
      if (currentIdx < 0) return;
      const nextIdx = e.key === "ArrowRight"
        ? Math.min(currentIdx + 1, subs.length - 1)
        : Math.max(currentIdx - 1, 0);
      if (nextIdx !== currentIdx) {
        // Same path as a click: remember the sub-tab and redraw the bar so its
        // active style follows.
        const nextId = subs[nextIdx][0];
        setUiState({ activeSubTab: { ...ui.activeSubTab, [tab]: nextId } });
        renderSubTabs(tab);
        renderContent(tab, nextId);
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
