// js/pages/processing.js – Tab 1: Container + Sub-Tab-Orchestrierung

import { el } from "../utils/dom.js";
import { createInputScanPage } from "./input-scan.js";
import { createParameterPage } from "./parameter.js";
import { createRunMonitorPage } from "./run-monitor.js";

export function createProcessingPage(subTab) {
  switch (subTab) {
    case "input-scan":
      return createInputScanPage();
    case "parameter":
      return createParameterPage();
    case "run-monitor":
      return createRunMonitorPage();
    default:
      return createInputScanPage();
  }
}
