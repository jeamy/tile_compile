// js/pages/tools.js – Tab 2: Container + Sub-Tab-Orchestrierung

import { el } from "../utils/dom.js";
import { createRawStackPage } from "./raw-stack.js";
import { createAstrometryPage } from "./astrometry.js";
import { createPccPage } from "./pcc.js";

export function createToolsPage(subTab) {
  switch (subTab) {
    case "raw-stack":
      return createRawStackPage();
    case "astrometry":
      return createAstrometryPage();
    case "pcc":
      return createPccPage();
    default:
      return createRawStackPage();
  }
}
