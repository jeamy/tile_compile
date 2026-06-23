// js/pages/history.js – Tab 3: Container + Sub-Tab-Orchestrierung

import { createRunHistoryPage } from "./run-history.js";

export function createHistoryPage(subTab) {
  switch (subTab) {
    case "run-history":
      return createRunHistoryPage();
    default:
      return createRunHistoryPage();
  }
}
