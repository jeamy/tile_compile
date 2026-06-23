// js/utils/navigation.js – Shared sub-tab navigation helper

import { getUiState, setUiState } from "../state/ui-state.js";

export function goToSubTab(parentTab, subId) {
  const ui = getUiState();
  setUiState({ activeSubTab: { ...ui.activeSubTab, [parentTab]: subId } });
  window.location.hash = `#${parentTab}`;
  window.dispatchEvent(new Event("tc-subtab-change"));
}
