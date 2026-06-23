// js/state/ui-state.js – Locale, Theme, activeTab, activeSubTab

import { getStore } from "./store.js";

const DEFAULT = {
  locale: "de",
  theme: "dark",
  activeTab: "processing",
  activeSubTab: {},
  paramView: "parameter",
  selectedCategory: "all",
  selectedSituations: [],
};

const store = getStore("ui-state", DEFAULT);

export function getUiState() {
  return store.getState();
}

export function setUiState(patch) {
  store.setState(patch);
  if (patch.theme) {
    document.documentElement.setAttribute("data-theme", patch.theme);
  }
  if (patch.locale) {
    document.documentElement.setAttribute("lang", patch.locale);
  }
}

export function initUiState() {
  const state = store.getState();
  document.documentElement.setAttribute("data-theme", state.theme);
  document.documentElement.setAttribute("lang", state.locale);
}
