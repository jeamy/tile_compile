/**
 * Central utility functions for the tile_compile frontend.
 * Replaces duplicate implementations across shell.js, parameter-studio-page.js, etc.
 */

/**
 * Escapes HTML special characters to prevent XSS.
 * @param {string} text - Text to escape
 * @returns {string} Escaped HTML
 */
export function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/**
 * Storage keys used across the application.
 * Centralized to prevent typos and duplicates.
 */
export const STORAGE_KEYS = {
  locale: "gui2.locale",
  backendBase: "gui2.backendBase",
  configYamlDraft: "gui2.configYamlDraft",
  configValidationState: "gui2.configValidationState",
  parameterDirtyState: "gui2.parameterDirtyState",
  historyCurrentRunId: "gui2.history.currentRunId",
  lastInputDirs: "gui2.lastInputDirs",
  presetsDir: "gui2.presetsDir",
  calibrationPathCache: "gui2.calibrationPathCache",
  lastScanColorMode: "gui2.lastScanColorMode",
  dashboardRunsDir: "gui2.run.runsDir",
  dashboardRunName: "gui2.run.runName",
  dashboardQueue: "gui2.dashboard.queueDraft",
  dashboardPreset: "gui2.dashboard.presetPath",
  parameterPreset: "gui2.parameter.presetPath",
  wizardRunsDir: "gui2.run.runsDir",
  wizardRunName: "gui2.run.runName",
  wizardQueue: "gui2.wizard.queueDraft",
  wizardPreset: "gui2.wizard.presetPath",
  historySelectedRunId: "gui2.history.selectedRunId",
  historyCompareRunId: "gui2.history.compareRunId",
  liveFilter: "gui2.live.filter",
  helpState: "gui2.help.state",
};

/**
 * Gets the active locale from window object or localStorage.
 * @returns {string} "en" or "de"
 */
export function getActiveLocale() {
  return String(window.GUI2_LOCALE || localStorage.getItem(STORAGE_KEYS.locale) || "de").toLowerCase() === "en"
    ? "en"
    : "de";
}

/**
 * Retrieves a localized message.
 * @param {string} key - Message key
 * @param {string} deFallback - German fallback
 * @param {string} [enFallback] - English fallback (defaults to deFallback)
 * @returns {string} Localized message
 */
export function getMessage(key, deFallback, enFallback = deFallback) {
  const locale = getActiveLocale();
  const fallback = locale === "en" ? enFallback : deFallback;
  const msg = window.GUI2_LOCALE_MESSAGES?.[key];
  return typeof msg === "string" && msg ? msg : fallback;
}

/**
 * Safely parses JSON from localStorage.
 * @param {string} key - localStorage key
 * @param {any} defaultValue - Default value if parsing fails
 * @returns {any} Parsed value or default
 */
export function getStorageJson(key, defaultValue = {}) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "{}");
    return parsed && typeof parsed === "object" ? parsed : defaultValue;
  } catch {
    return defaultValue;
  }
}

/**
 * Safely saves JSON to localStorage.
 * @param {string} key - localStorage key
 * @param {any} value - Value to store
 */
export function setStorageJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Silently fail if storage is full
  }
}

/**
 * Humanizes a control ID for display.
 * @param {string} controlId - Control ID
 * @returns {string} Humanized text
 */
export function humanizeControlId(controlId) {
  return String(controlId || "")
    .replace(/[._]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Gets label text for a form control element.
 * @param {Element} el - Form element
 * @returns {string} Label text or empty string
 */
export function getLabelTextForControl(el) {
  if (!el || !(el instanceof Element)) return "";
  const rowLabel = el.closest(".ps-row")?.querySelector("label");
  if (rowLabel) return (rowLabel.textContent || "").replace(/\s+/g, " ").trim();
  if (el.id) {
    try {
      const linked = document.querySelector(`label[for='${el.id}']`);
      if (linked) return (linked.textContent || "").replace(/\s+/g, " ").trim();
    } catch {
      // ignore invalid selectors
    }
  }
  return "";
}
