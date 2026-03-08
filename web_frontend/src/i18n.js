let messages = {};
let currentLocale = "de";

const CONTROL_TEXT_KEYS = {
  "dashboard.quick.start_wizard": "ui.button.start_wizard",
  "history.refresh": "ui.button.refresh",
  "history.set_current": "ui.button.set_current",
  "monitor.start": "ui.button.run_start",
  "monitor.stop": "ui.button.stop",
  "monitor.stats.generate": "monitor.stats.generate",
  "monitor.stats.open_folder": "monitor.stats.open_folder",
  "monitor.report": "ui.button.report",
  "monitor.open_run_folder": "ui.button.open_run_folder",
  "monitor.resume": "ui.button.resume",
  "monitor.resume.restore_revision": "ui.button.restore_config_revision",
  "input_scan.scan_run": "ui.button.scan_refresh",
  "wizard.nav.back": "ui.button.back",
  "wizard.nav.next": "ui.button.next",
  "wizard.start": "ui.button.run_start",
  "wizard.situation.apply": "ui.button.situation",
  "parameter.preset_apply": "ui.button.preset",
  "parameter.yaml_sync": "ui.button.yaml_sync",
  "parameter.validate": "ui.button.validate",
  "parameter.reset_default": "ui.button.reset_default",
  "parameter.review_changes": "ui.button.review_changes",
  "parameter.save": "ui.button.save",
  "tools.astrometry.detect": "ui.button.detect_astap",
  "tools.astrometry.install_cli": "ui.button.install_astap_cli",
  "tools.astrometry.download_catalog": "ui.button.download_catalog",
  "tools.astrometry.cancel_download": "ui.button.cancel",
  "tools.astrometry.solve": "ui.button.solve",
  "tools.astrometry.save_solved": "ui.button.save_solved",
  "tools.pcc.download_missing": "ui.button.download_missing",
  "tools.pcc.cancel_download": "ui.button.cancel",
  "tools.pcc.check_online": "ui.button.check_online",
  "tools.pcc.run": "ui.button.run_pcc",
  "tools.pcc.save_corrected": "ui.button.save_corrected",
};

const CONTROL_TITLE_KEYS = {
  "nav.dashboard": "ui.tooltip.shell.nav_dashboard",
  "nav.input_scan": "ui.tooltip.shell.nav_input_scan",
  "nav.parameter_studio": "ui.tooltip.shell.nav_parameter_studio",
  "nav.assumptions": "ui.tooltip.shell.nav_assumptions",
  "nav.run_monitor": "ui.tooltip.shell.nav_run_monitor",
  "nav.history_tools": "ui.tooltip.shell.nav_history_tools",
  "nav.astrometry": "ui.tooltip.shell.nav_astrometry",
  "nav.pcc": "ui.tooltip.shell.nav_pcc",
  "nav.live_log": "ui.tooltip.shell.nav_live_log",
  "dashboard.quick.start_wizard": "ui.tooltip.dashboard.start_wizard",
  "history.refresh": "ui.tooltip.history.refresh",
  "history.set_current": "ui.tooltip.history.set_current",
  "monitor.stop": "ui.tooltip.monitor.stop",
  "monitor.stats.generate": "ui.tooltip.monitor.stats_generate",
  "monitor.stats.open_folder": "ui.tooltip.monitor.stats_open_folder",
  "monitor.report": "ui.tooltip.monitor.report",
  "monitor.open_run_folder": "ui.tooltip.monitor.open_run_folder",
  "monitor.resume": "ui.tooltip.monitor.resume",
  "monitor.resume.restore_revision": "ui.tooltip.monitor.restore_config_revision",
  "input_scan.input_dirs": "ui.tooltip.input_scan.input_dirs",
  "input_scan.pattern": "ui.tooltip.input_scan.pattern",
  "input_scan.max_frames": "ui.tooltip.input_scan.max_frames",
  "input_scan.sort": "ui.tooltip.input_scan.sort",
  "input_scan.color_mode_confirm": "ui.tooltip.input_scan.color_mode_confirm",
  "input_scan.bayer_pattern": "ui.tooltip.input_scan.bayer_pattern",
  "input_scan.with_checksums": "ui.tooltip.input_scan.with_checksums",
  "input_scan.scan_run": "ui.tooltip.input_scan.scan_run",
  "wizard.nav.back": "ui.tooltip.wizard.nav_back",
  "wizard.nav.next": "ui.tooltip.wizard.nav_next",
  "wizard.input.runs_dir": "ui.tooltip.wizard.runs_dir",
  "wizard.input.run_name": "ui.tooltip.wizard.run_name",
  "wizard.situation.apply": "ui.tooltip.wizard.situation_apply",
  "wizard.start": "ui.tooltip.wizard.start",
  "tools.astrometry.binary": "ui.tooltip.tools.astrometry_binary",
  "tools.astrometry.data_dir": "ui.tooltip.tools.astrometry_data_dir",
  "tools.astrometry.browse_binary": "ui.tooltip.tools.astrometry_browse_binary",
  "tools.astrometry.browse_data_dir": "ui.tooltip.tools.astrometry_browse_data_dir",
  "tools.astrometry.file": "ui.tooltip.tools.astrometry_file",
  "tools.astrometry.browse_file": "ui.tooltip.tools.astrometry_browse_file",
  "tools.astrometry.detect": "ui.tooltip.tools.astrometry_detect",
  "tools.astrometry.download_catalog": "ui.tooltip.tools.astrometry_download_catalog",
  "tools.astrometry.solve": "ui.tooltip.tools.astrometry_solve",
  "tools.astrometry.save_solved": "ui.tooltip.tools.astrometry_save_solved",
  "tools.pcc.rgb_fits": "ui.tooltip.tools.pcc_rgb_fits",
  "tools.pcc.wcs_file": "ui.tooltip.tools.pcc_wcs_file",
  "tools.pcc.browse_rgb": "ui.tooltip.tools.pcc_browse_rgb",
  "tools.pcc.browse_wcs": "ui.tooltip.tools.pcc_browse_wcs",
  "tools.pcc.source": "ui.tooltip.tools.pcc_source",
  "tools.pcc.siril_catalog_dir": "ui.tooltip.tools.pcc_siril_catalog_dir",
  "tools.pcc.browse_catalog_dir": "ui.tooltip.tools.pcc_browse_catalog_dir",
  "tools.pcc.download_missing": "ui.tooltip.tools.pcc_download_missing",
  "tools.pcc.check_online": "ui.tooltip.tools.pcc_check_online",
  "tools.pcc.run": "ui.tooltip.tools.pcc_run",
  "tools.pcc.save_corrected": "ui.tooltip.tools.pcc_save_corrected",
};

const PAGE_BINDINGS = {
  shared: [
    { selector: ".footer-note", key: "ui.footer.default", all: false },
  ],
  "index.html": [
    { selector: "document:title", key: "page.dashboard.title" },
    { selector: "main > .intro", key: "page.dashboard.intro" },
    { selector: ".app-content > h2", key: "page.dashboard.heading" },
    { selector: ".app-content > .ps-sub", key: "page.dashboard.sub" },
    { selector: "#dashboard-kpi-scan-quality > div:first-child", key: "page.dashboard.kpi.frames_detected" },
    { selector: "#dashboard-kpi-open-warnings > div:first-child", key: "page.dashboard.kpi.scan_quality" },
    { selector: "#dashboard-kpi-guardrail-warnings > div:first-child", key: "page.dashboard.kpi.open_warnings" },
    { selector: "div[style*='Letzter Lauf']", key: "page.dashboard.kpi.last_run", matchText: "Letzter Lauf" },
    { selector: ".ps-section div[style*='Letzter Input-Scan']", key: "page.dashboard.last_scan" },
    { selector: ".ps-section div[style*='Guided Run']", key: "page.dashboard.guided_run" },
    { selector: ".ps-section div[style*='1) Input/Run-Ziel']", key: "page.dashboard.guided_steps" },
    { selector: "#dashboard-guided-mode-simple", key: "ui.mode.simple" },
    { selector: "#dashboard-guided-mode-advanced", key: "ui.mode.advanced" },
    { selector: "#dashboard-guided-open-wizard", key: "page.dashboard.wizard_page" },
    { selector: "#dashboard-label-input-dirs", key: "ui.field.input_dirs" },
    { selector: "#dashboard-label-runs-dir", key: "ui.field.runs_dir" },
    { selector: "#dashboard-label-color-mode", key: "ui.field.color_mode" },
    { selector: "#dashboard-label-preset", key: "ui.field.preset" },
    { selector: "#dashboard-label-run-name", key: "ui.field.run_name" },
    { selector: "#dashboard-label-run-path", key: "ui.field.run_path_preview" },
    { selector: ".footer-note", key: "page.dashboard.footer" },
  ],
  "input-scan.html": [
    { selector: "document:title", key: "page.input_scan.title" },
    { selector: "main > .intro", key: "page.input_scan.intro" },
    { selector: ".app-content > h2", key: "page.input_scan.heading" },
    { selector: ".app-content > .ps-sub", key: "page.input_scan.sub" },
    { selector: ".ps-section-title", key: "page.input_scan.queue_title", index: 0 },
    { selector: ".ps-section-title", key: "page.input_scan.calibration_title", index: 1 },
    { selector: ".ps-result-title", key: "ui.panel.scan_results" },
    { selector: "#btn-scan", key: "page.input_scan.scan_button" },
    { selector: "#scan-note", key: "page.input_scan.footer" },
  ],
  "wizard.html": [
    { selector: "document:title", key: "page.wizard.title" },
    { selector: "main .intro", key: "page.wizard.intro" },
    { selector: ".ps-section-title", key: "page.wizard.step1", index: 0 },
    { selector: ".ps-section-title", key: "page.wizard.step2", index: 1 },
    { selector: ".ps-section-title", key: "page.wizard.step3", index: 2 },
    { selector: ".ps-section-title", key: "page.wizard.step4", index: 4 },
    { selector: ".ps-section-title", key: "page.wizard.impl_chain", index: 5 },
    { selector: ".ps-section-title", key: "page.wizard.resume_context", index: 6 },
    { selector: "#btn-scan", key: "page.wizard.scan_button" },
    { selector: "#wizard-nav-next", key: "page.wizard.next_validation" },
    { selector: "#wizard-nav-back", key: "ui.button.back" },
    { selector: "#wizard-start", key: "ui.button.run_start" },
    { selector: "#wizard-situation-apply", key: "page.wizard.apply_scenario" },
    { selector: "#wizard-validation-result .ps-result-title", key: "page.wizard.validation" },
    { selector: "#scan-note", key: "page.wizard.footer" },
  ],
  "run-monitor.html": [
    { selector: "document:title", key: "page.run_monitor.title" },
    { selector: "main > .intro", key: "page.run_monitor.intro" },
    { selector: ".app-content > h2", key: "ui.nav.run_monitor" },
    { selector: ".app-content > .ps-sub", key: "page.run_monitor.sub" },
    { selector: ".ps-section-title", key: "page.run_monitor.pipeline_phases", index: 0 },
    { selector: "#monitor-start", key: "ui.button.run_start" },
    { selector: "#monitor-stop", key: "ui.button.stop" },
    { selector: ".ps-section-title", key: "ui.nav.live_log", index: 1 },
    { selector: ".ps-section-title", key: "page.run_monitor.stats", index: 2 },
    { selector: "#monitor-stats-generate", key: "monitor.stats.generate" },
    { selector: "#monitor-stats-open-folder", key: "monitor.stats.open_folder" },
    { selector: ".ps-section-title", key: "page.run_monitor.resume_revision", index: 3 },
    { selector: "#monitor-resume", key: "page.run_monitor.resume_start" },
    { selector: "#monitor-resume-restore-revision", key: "ui.button.restore_config_revision" },
    { selector: ".ps-section-title", key: "page.run_monitor.artifacts", index: 4 },
    { selector: "#monitor-report", key: "ui.button.report" },
    { selector: "#monitor-open-run-folder", key: "ui.button.open_run_folder" },
    { selector: ".footer-note", key: "page.run_monitor.footer" },
  ],
  "assumptions.html": [
    { selector: "document:title", key: "page.assumptions.title" },
    { selector: "main > .intro", key: "page.assumptions.intro" },
    { selector: ".app-content > h2", key: "ui.nav.assumptions" },
    { selector: ".app-content > .ps-sub", key: "page.assumptions.sub" },
    { selector: ".ps-info-title", key: "page.assumptions.current_mode" },
    { selector: "#asmpt-note", key: "page.assumptions.footer" },
  ],
  "history-tools.html": [
    { selector: "document:title", key: "page.history.title" },
    { selector: "main > .intro", key: "page.history.intro" },
    { selector: ".app-content > h2", key: "page.history.heading" },
    { selector: ".app-content > .ps-sub", key: "page.history.sub" },
    { selector: ".ps-section-title", key: "page.history.run_history", index: 0 },
    { selector: "#history-refresh", key: "ui.button.refresh" },
    { selector: "#history-set-current", key: "ui.button.set_current" },
    { selector: "#history-open-report", key: "page.history.open_report" },
    { selector: "#history-delete-run", key: "page.history.delete_entry" },
    { selector: ".ps-section-title", key: "page.history.selected_run", index: 1 },
    { selector: ".ps-section-title", key: "page.history.compare_run", index: 2 },
    { selector: "#history-compare-use-current", key: "page.history.use_current_compare" },
    { selector: "#history-compare-clear", key: "page.history.clear_compare" },
    { selector: ".footer-note", key: "page.history.footer" },
  ],
  "astrometry.html": [
    { selector: "document:title", key: "page.astrometry.title" },
    { selector: "main > .intro", key: "page.astrometry.intro" },
    { selector: ".app-content > h2", key: "ui.nav.astrometry" },
    { selector: ".app-content > .ps-sub", key: "page.astrometry.sub" },
    { selector: ".ps-section-title", key: "page.astrometry.setup", index: 0 },
    { selector: ".ps-section-title", key: "page.astrometry.star_database", index: 1 },
    { selector: ".ps-section-title", key: "page.astrometry.plate_solve", index: 2 },
    { selector: ".ps-section-title", key: "page.astrometry.log", index: 3 },
    { selector: ".footer-note", key: "page.astrometry.footer" },
  ],
  "pcc.html": [
    { selector: "document:title", key: "page.pcc.title" },
    { selector: "main > .intro", key: "page.pcc.intro" },
    { selector: ".app-content > h2", key: "ui.nav.pcc" },
    { selector: ".app-content > .ps-sub", key: "page.pcc.sub" },
    { selector: ".ps-section-title", key: "page.pcc.input", index: 0 },
    { selector: ".ps-section-title", key: "page.pcc.catalog_source", index: 1 },
    { selector: ".ps-section-title", key: "page.pcc.parameters", index: 2 },
    { selector: ".ps-section-title", key: "page.pcc.result_log", index: 3 },
    { selector: ".footer-note", key: "page.pcc.footer" },
  ],
  "live-log.html": [
    { selector: "document:title", key: "page.live_log.title" },
    { selector: "main > .intro", key: "page.live_log.intro" },
    { selector: ".app-content > h2", key: "ui.nav.live_log" },
    { selector: ".app-content > .ps-sub", key: "page.live_log.sub" },
    { selector: ".ps-btn.ps-btn-secondary", key: "page.live_log.filter_all", index: 0 },
    { selector: ".ps-btn.ps-btn-secondary", key: "page.live_log.filter_info", index: 1 },
    { selector: ".ps-btn.ps-btn-secondary", key: "page.live_log.filter_warning", index: 2 },
    { selector: ".ps-btn.ps-btn-secondary", key: "page.live_log.filter_error", index: 3 },
    { selector: ".ps-btn.ps-btn-secondary", key: "page.live_log.clear", index: 4 },
    { selector: ".ps-actions .ps-btn", key: "page.live_log.export" },
    { selector: ".footer-note", key: "page.live_log.footer" },
  ],
  "parameter-studio.html": [
    { selector: "document:title", key: "page.parameter_studio.title" },
    { selector: "main > .intro", key: "page.parameter_studio.intro" },
    { selector: ".app-content > h2", key: "ui.nav.parameter_studio" },
    { selector: ".app-content > .ps-sub", key: "page.parameter_studio.sub" },
    { selector: ".ps-section-title", key: "page.parameter_studio.categories", index: 0 },
    { selector: ".ps-section-title", key: "page.parameter_studio.search_actions", index: 1 },
    { selector: ".ps-section-title", key: "ui.panel.parameter_full_editor", index: 2 },
    { selector: ".footer-note", key: "page.parameter_studio.footer" },
  ],
};

function pageName() {
  return (window.location.pathname.split("/").pop() || "index.html").toLowerCase();
}

async function fetchMessages(locale) {
  const normalized = String(locale || "de").toLowerCase() === "en" ? "en" : "de";
  const response = await fetch(`i18n/${normalized}.json`);
  if (!response.ok) {
    throw new Error(`Failed to load locale ${normalized}`);
  }
  return response.json();
}

function textFor(key, fallback = "") {
  return messages[key] ?? fallback ?? key;
}

function setNodeText(node, value) {
  if (!node) return;
  node.textContent = value;
}

function findNode(selector, index, matchText) {
  if (selector === "document:title") return document.querySelector("title");
  const nodes = Array.from(document.querySelectorAll(selector));
  if (typeof matchText === "string") {
    return nodes.find((node) => String(node.textContent || "").includes(matchText)) || null;
  }
  if (typeof index === "number") return nodes[index] || null;
  return nodes[0] || null;
}

function applyBinding(binding) {
  const node = findNode(binding.selector, binding.index, binding.matchText);
  if (!node) return;
  const value = textFor(binding.key, node.textContent || "");
  if (binding.selector === "document:title") {
    document.title = value;
    return;
  }
  if (binding.attr === "placeholder") {
    node.setAttribute("placeholder", value);
    return;
  }
  if (binding.attr === "title") {
    node.setAttribute("title", value);
    return;
  }
  setNodeText(node, value);
}

function applyDataAttributes(root = document) {
  root.querySelectorAll("[data-control]").forEach((node) => {
    const control = node.getAttribute("data-control") || "";
    if (!node.hasAttribute("data-i18n") && CONTROL_TEXT_KEYS[control] && /^(button|a|span)$/i.test(node.tagName)) {
      setNodeText(node, textFor(CONTROL_TEXT_KEYS[control], node.textContent || ""));
    }
    if (!node.hasAttribute("data-i18n-title") && CONTROL_TITLE_KEYS[control]) {
      node.setAttribute("title", textFor(CONTROL_TITLE_KEYS[control], node.getAttribute("title") || ""));
    }
  });
  root.querySelectorAll("[data-i18n]").forEach((node) => {
    setNodeText(node, textFor(node.getAttribute("data-i18n"), node.textContent || ""));
  });
  root.querySelectorAll("[data-i18n-title]").forEach((node) => {
    node.setAttribute("title", textFor(node.getAttribute("data-i18n-title"), node.getAttribute("title") || ""));
  });
  root.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.setAttribute("placeholder", textFor(node.getAttribute("data-i18n-placeholder"), node.getAttribute("placeholder") || ""));
  });
}

function applyPageBindings() {
  const currentPage = pageName();
  const bindings = [...(PAGE_BINDINGS.shared || []), ...(PAGE_BINDINGS[currentPage] || [])];
  bindings.forEach(applyBinding);
}

function dispatchLocaleChanged() {
  window.GUI2_LOCALE = currentLocale;
  window.GUI2_LOCALE_MESSAGES = { ...messages };
  document.dispatchEvent(new CustomEvent("gui2:locale-changed", { detail: { locale: currentLocale, messages } }));
}

export function t(key, fallback = "") {
  return textFor(key, fallback);
}

export function getLocaleMessages() {
  return { ...messages };
}

export async function applyLocaleMessages(locale) {
  currentLocale = String(locale || "de").toLowerCase() === "en" ? "en" : "de";
  try {
    messages = await fetchMessages(currentLocale);
  } catch {
    messages = {};
  }
  applyDataAttributes(document);
  applyPageBindings();
  dispatchLocaleChanged();
  return messages;
}

export function applyTranslations(root = document) {
  applyDataAttributes(root);
  applyPageBindings();
}
