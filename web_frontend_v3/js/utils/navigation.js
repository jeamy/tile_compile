// js/utils/navigation.js – Shared sub-tab navigation helper

import { getUiState, setUiState } from "../state/ui-state.js";

// main.js renders on "tc-subtab-change" right away and ignores the hashchange
// that follows for the tab it already shows, so the page mounts once.
export function goToSubTab(parentTab, subId) {
  const ui = getUiState();
  setUiState({ activeTab: parentTab, activeSubTab: { ...ui.activeSubTab, [parentTab]: subId } });
  window.dispatchEvent(new Event("tc-subtab-change"));
  if (window.location.hash !== `#${parentTab}`) window.location.hash = `#${parentTab}`;
}
