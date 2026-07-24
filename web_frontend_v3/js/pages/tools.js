// js/pages/tools.js – Tab 2: Container + Sub-Tab-Orchestrierung

import { createRawStackPage } from "./raw-stack.js";
import { createAstrometryPage } from "./astrometry.js";
import { createPccPage } from "./pcc.js";
import { createAiModelSettingsPage } from "./ai-empfehlung.js";

export function createToolsPage(subTab) {
  switch (subTab) {
    case "raw-stack":
      return createRawStackPage();
    case "astrometry":
      return createAstrometryPage();
    case "pcc":
      return createPccPage();
    case "ai-settings":
      return createAiModelSettingsPage();
    default:
      return createRawStackPage();
  }
}
