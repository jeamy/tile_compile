// js/components/sub-tabs.js – Sub-Tab-Leiste (kontextabhängig)

import { el, clear } from "../utils/dom.js";
import { getUiState, setUiState } from "../state/ui-state.js";

export function createSubTabs(tabs, parentTab, onSubTabChange) {
  const ui = getUiState();
  const activeSub = ui.activeSubTab[parentTab] || tabs[0]?.id;

  const bar = el("div", { class: "tc-subtab-list tc-tabs", role: "tablist" });

  for (const tab of tabs) {
    const isActive = tab.id === activeSub;
    const btn = el("button", {
      class: `tc-tab${isActive ? " active" : ""}`,
      "data-subtab": tab.id,
      role: "tab",
      "aria-selected": isActive ? "true" : "false",
      onclick: () => {
        setUiState({
          activeSubTab: { ...ui.activeSubTab, [parentTab]: tab.id },
        });
        onSubTabChange(tab.id);
        updateActive(bar, tab.id);
      },
    }, tab.label);
    bar.appendChild(btn);
  }

  return bar;
}

function updateActive(bar, activeId) {
  for (const btn of bar.querySelectorAll(".tc-tab")) {
    const active = btn.getAttribute("data-subtab") === activeId;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  }
}
