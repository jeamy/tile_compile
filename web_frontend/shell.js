(function renderSharedShell() {
  const HEADER_NAV_ITEMS = [
    { key: "dashboard", href: "index.html", label: "Dashboard", control: "nav.dashboard" },
    { key: "input_scan", href: "input-scan.html", label: "Input&Scan", control: "nav.input_scan" },
    { key: "parameter_studio", href: "parameter-studio.html", label: "Parameter Studio", control: "nav.parameter_studio" },
    { key: "assumptions", href: "assumptions.html", label: "Assumptions", control: "nav.assumptions" },
    { key: "run_monitor", href: "run-monitor.html", label: "Run Monitor", control: "nav.run_monitor" },
    { key: "history_tools", href: "history-tools.html", label: "History+Tools", control: "nav.history_tools" },
    { key: "astrometry", href: "astrometry.html", label: "Astrometry", control: "nav.astrometry" },
    { key: "pcc", href: "pcc.html", label: "PCC", control: "nav.pcc" },
    { key: "live_log", href: "live-log.html", label: "Live Log", control: "nav.live_log" },
    { key: "wizard", href: "wizard.html", label: "Wizard", control: "nav.wizard" },
  ];

  const SIDEBAR_ITEMS = [
    { key: "dashboard", href: "index.html", label: "Dashboard", control: "nav.dashboard" },
    { key: "input_scan", href: "input-scan.html", label: "Input & Scan", control: "nav.input_scan" },
    { key: "parameter_studio", href: "parameter-studio.html", label: "Parameter Studio", control: "nav.parameter_studio" },
    { key: "assumptions", href: "assumptions.html", label: "Assumptions", control: "nav.assumptions" },
    { key: "run_monitor", href: "run-monitor.html", label: "Run Monitor", control: "nav.run_monitor" },
    { key: "history_tools", href: "history-tools.html", label: "History + Tools", control: "nav.history_tools" },
    { key: "astrometry", href: "astrometry.html", label: "Astrometry", control: "nav.astrometry" },
    { key: "pcc", href: "pcc.html", label: "PCC", control: "nav.pcc" },
    { key: "live_log", href: "live-log.html", label: "Live Log", control: "nav.live_log" },
  ];

  function currentPageKey() {
    const file = (window.location.pathname.split("/").pop() || "index.html").toLowerCase();
    if (file === "index.html" || file === "dashboard.html") return "dashboard";
    if (file === "input-scan.html") return "input_scan";
    if (file === "parameter-studio.html") return "parameter_studio";
    if (file === "assumptions.html") return "assumptions";
    if (file === "run-monitor.html") return "run_monitor";
    if (file === "history-tools.html") return "history_tools";
    if (file === "astrometry.html") return "astrometry";
    if (file === "pcc.html") return "pcc";
    if (file === "live-log.html") return "live_log";
    if (file === "wizard.html") return "wizard";
    return "dashboard";
  }

  function navLink(item, activeKey) {
    const classes = item.key === activeKey ? " class=\"active\"" : "";
    const dataControl = item.control ? ` data-control=\"${item.control}\"` : "";
    return `<a${classes}${dataControl} href=\"${item.href}\">${item.label}</a>`;
  }

  function sidebarLink(item, activeKey) {
    const classes = item.key === activeKey ? "ps-item ps-active" : "ps-item";
    const dataControl = item.control ? ` data-control=\"${item.control}\"` : "";
    return `<a class=\"${classes}\"${dataControl} href=\"${item.href}\">${item.label}</a>`;
  }

  function renderHeader(activeKey) {
    const nav = HEADER_NAV_ITEMS.map((item) => navLink(item, activeKey)).join("\n    ");
    return `
  <nav class=\"pill-nav\">\n    ${nav}\n  </nav>\n\n  <div class=\"pill-subbar\">\n    <button id=\"locale-de\" data-control=\"locale.de\" class=\"locale-btn active\" title=\"Sprache auf Deutsch setzen.\">DE</button>\n    <button id=\"locale-en\" data-control=\"locale.en\" class=\"locale-btn\" title=\"Language to English.\">EN</button>\n    <span id=\"status-run-ready\" data-control=\"status.run_ready\" class=\"run-ready-chip\" title=\"Readiness-Guardrails anzeigen.\">Run Ready</span>\n  </div>\n`;
  }

  function renderSidebar(activeKey) {
    const links = SIDEBAR_ITEMS.map((item) => sidebarLink(item, activeKey)).join("\n        ");
    const quickStart =
      activeKey === "dashboard"
        ? "\n        <div style=\"margin-top:auto;padding:16px;\">\n          <a id=\"dashboard-quick-start-wizard\" data-control=\"dashboard.quick.start_wizard\" class=\"ps-btn\" href=\"wizard.html\" style=\"display:block;text-align:center;padding:10px;\" title=\"Neuen Guided Run starten.\">Start Wizard</a>\n        </div>"
        : "";
    return `<div class=\"ps-label\">Navigation</div>\n        ${links}${quickStart}`;
  }

  const activeKey = currentPageKey();

  const headerHost = document.querySelector("[data-shell-header]");
  if (headerHost) {
    headerHost.outerHTML = renderHeader(activeKey);
  }

  const sidebars = Array.from(document.querySelectorAll("[data-shell-sidebar]"));
  sidebars.forEach((host) => {
    host.innerHTML = renderSidebar(activeKey);
  });
})();
