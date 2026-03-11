(function renderSharedShell() {
  const HELP_STATE_KEY = "gui2.helpWindowState";
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
    const dataControl = item.control ? ` data-control="${item.control}"` : "";
    const dataI18n = item.control ? ` data-i18n="ui.${item.control}"` : "";
    const dataTitle = item.control ? ` data-i18n-title="ui.tooltip.shell.${item.control.replaceAll(".", "_")}"` : "";
    return `<a${classes}${dataControl}${dataI18n}${dataTitle} href="${item.href}">${item.label}</a>`;
  }

  function activeLocale() {
    return String(window.GUI2_LOCALE || localStorage.getItem("gui2.locale") || "de").toLowerCase() === "en" ? "en" : "de";
  }

  function message(key, deFallback, enFallback = deFallback) {
    const locale = activeLocale();
    const fallback = locale === "en" ? enFallback : deFallback;
    const msg = window.GUI2_LOCALE_MESSAGES?.[key];
    return typeof msg === "string" && msg ? msg : fallback;
  }

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function readHelpState() {
    try {
      const parsed = JSON.parse(localStorage.getItem(HELP_STATE_KEY) || "{}");
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      return {};
    }
  }

  function writeHelpState(patch) {
    const current = readHelpState();
    localStorage.setItem(HELP_STATE_KEY, JSON.stringify({ ...current, ...patch }));
  }

  function helpMarkup() {
    const sections = [
      {
        title: message("help.workflow.step2.title", "Schritt 2: Eingabedaten scannen", "Step 2: Scan the input frames"),
        lead: message(
          "help.workflow.step2.lead",
          "Öffne Input & Scan oder Wizard Schritt 1 und starte mit einer sauberen Eingangskontrolle.",
          "Open Input & Scan or Wizard step 1 and start with a clean intake check.",
        ),
        bullets: [
          message(
            "help.workflow.step2.bullet1",
            "Wähle einen absoluten Eingabeordner für OSC oder eine serielle MONO-Queue mit genau einem Eingabeordner pro Filter.",
            "Choose one absolute input directory for OSC or a serial MONO queue with exactly one input directory per filter.",
          ),
          message(
            "help.workflow.step2.bullet2",
            "Starte den Scan und prüfe, ob die erkannte Frame-Anzahl plausibel ist und keine unerwarteten Dateien im Datensatz liegen.",
            "Run the scan and verify that the detected frame count is plausible and that no unexpected files are mixed into the dataset.",
          ),
          message(
            "help.workflow.step2.bullet3",
            "Prüfe den erkannten Farbmodus, die Bildgröße und bei OSC das Bayer-Pattern. Bestätige den Farbmodus, wenn der Datensatz nicht eindeutig ist.",
            "Verify the detected color mode, image size and, for OSC, the Bayer pattern. Confirm the color mode manually if the dataset is ambiguous.",
          ),
          message(
            "help.workflow.step2.bullet4",
            "Wenn du MONO verarbeitest, kontrolliere die Queue-Reihenfolge, optionale Pattern und Run-Labels pro Filter vor dem Start.",
            "If you process MONO data, review queue order, optional patterns and run labels per filter before starting.",
          ),
        ],
      },
      {
        title: message("help.workflow.step3.title", "Schritt 3: Kalibrierung prüfen", "Step 3: Review calibration inputs"),
        lead: message(
          "help.workflow.step3.lead",
          "Bias, Dark und Flat brauchst du nur, wenn die Lights noch nicht kalibriert sind.",
          "Bias, dark and flat are only needed if the light frames are not already calibrated.",
        ),
        bullets: [
          message(
            "help.workflow.step3.bullet1",
            "Aktiviere Bias, Dark und Flat nur für die Komponenten, die dein Datensatz wirklich benötigt.",
            "Enable bias, dark and flat only for the components your dataset actually needs.",
          ),
          message(
            "help.workflow.step3.bullet2",
            "Wähle je nach Setup entweder ein Verzeichnis mit Rohkalibrierdaten oder direkt eine Master-Datei.",
            "Depending on your setup, choose either a directory with raw calibration frames or a master calibration file directly.",
          ),
          message(
            "help.workflow.step3.bullet3",
            "Lass die Schalter deaktiviert, wenn die Lights bereits kalibriert sind oder du die Kalibrierung bewusst ausserhalb der Pipeline gemacht hast.",
            "Keep the switches disabled if the lights are already calibrated or if you intentionally calibrated outside the pipeline.",
          ),
          message(
            "help.workflow.step3.bullet4",
            "Achte auf lesbare absolute Pfade und darauf, dass Filter-/Belichtungslogik zu deinen Lights passt.",
            "Use readable absolute paths and make sure the filter and exposure logic matches your light frames.",
          ),
        ],
      },
      {
        title: message("help.workflow.step4.title", "Schritt 4: Parameter anpassen und validieren", "Step 4: Adjust and validate parameters"),
        lead: message(
          "help.workflow.step4.lead",
            "Das Parameter Studio ist der zentrale Ort für Presets, Feintuning und die letzte fachliche Kontrolle.",
          "Parameter Studio is the central place for presets, fine tuning and the final technical review.",
        ),
        bullets: [
          message(
            "help.workflow.step4.bullet1",
            "Starte mit einem passenden Preset oder mit dem aktuellen Konfigurationsstand und bearbeite die Parameter abschnittsweise.",
            "Start from a matching preset or from the current configuration and adjust parameters section by section.",
          ),
          message(
            "help.workflow.step4.bullet2",
            "Nutze Suche und Explain-Panel, um Wirkung, Wertebereich und Risiko einer Einstellung schnell nachzuvollziehen.",
            "Use search and the Explain panel to quickly understand effect, value range and risk of a setting.",
          ),
          message(
            "help.workflow.step4.bullet3",
            "Prüfe für problematische Datensätze besonders Registrierung, Runtime-Limits sowie BGE- und PCC-Parameter.",
            "For difficult datasets, pay special attention to registration, runtime limits and the BGE/PCC parameters.",
          ),
          message(
            "help.workflow.step4.bullet4",
            "Führe vor dem Run immer eine Validierung aus. Fehler blockieren den Start, Warnungen solltest du bewusst akzeptieren können.",
            "Always run validation before starting. Errors should block the run; warnings should only remain if you accept them deliberately.",
          ),
        ],
      },
      {
        title: message("help.workflow.step5.title", "Schritt 5: Run starten", "Step 5: Start a run"),
        lead: message(
          "help.workflow.step5.lead",
          "Der Run kann aus Dashboard, Wizard oder den dedizierten Start-Controls ausgelost werden.",
          "A run can be launched from Dashboard, Wizard or the dedicated start controls.",
        ),
        bullets: [
          message(
            "help.workflow.step5.bullet1",
            "Setze Ausgabeordner und Run-Name so, dass der Ergebnisordner später eindeutig dem Datensatz zugeordnet werden kann.",
            "Set output folder and run name so the result directory can later be clearly mapped to the dataset.",
          ),
          message(
            "help.workflow.step5.bullet2",
            "Kontrolliere vor dem Start die Guardrails und den Validierungsstatus. Ein roter Zustand sollte vor dem Start behoben werden.",
            "Check guardrails and validation status before starting. A red state should be resolved before launch.",
          ),
          message(
            "help.workflow.step5.bullet3",
            "Starte den Run und wechsle anschließend in den Run Monitor, um Phasen, Logs, Stats, Artefakte und Resume-Ziele zu verfolgen.",
            "Start the run and then move to Run Monitor to track phases, logs, stats, artifacts and resume targets.",
          ),
        ],
      },
      {
        title: message("help.workflow.step6.title", "Schritt 6: Ergebnisse und Report prüfen", "Step 6: Review outputs and report"),
        lead: message(
          "help.workflow.step6.lead",
          "Wenn alle Phasen beendet sind, findest du im Ausgabeordner die erzeugten Dateien, Zwischenergebnisse und den optionalen Report.",
          "Once all phases are complete, the output directory contains the generated files, intermediate results and the optional report.",
        ),
        bullets: [
          message(
            "help.workflow.step6.bullet1",
            "Typische Hauptresultate sind Dateien wie stack*, solve*, bge* oder pcc*. Sie stehen für den gestackten Zustand, Plate-Solve/WCS-Ergebnisse, Hintergrundkorrektur und Farbkalibrierung.",
            "Typical top-level results are files such as stack*, solve*, bge* or pcc*. They represent the stacked result, plate-solve/WCS outputs, background correction and color calibration.",
          ),
          message(
            "help.workflow.step6.bullet2",
            "Im Unterordner outputs liegen in der Regel die eigentlichen Bildprodukte der Pipeline. Dort findest du die finalen oder nahezu finalen FITS-Dateien in den wichtigsten Verarbeitungsschritten.",
            "The outputs subfolder usually contains the actual image products of the pipeline. That is where you find the final or near-final FITS files for the key processing stages.",
          ),
          message(
            "help.workflow.step6.bullet3",
            "Im Unterordner artifacts liegen Diagnose- und Nebenprodukte wie Logs, Statistiken, JSON-Zusammenfassungen und – wenn du ihn erzeugst – der HTML-Report. Wenn du einen Report erstellen möchtest, kannst du ihn nach dem Lauf generieren; danach findest du ihn im artifacts-Ordner.",
            "The artifacts subfolder contains diagnostics and side products such as logs, statistics, JSON summaries and, if you generate it, the HTML report. If you want a report, generate it after the run; it will then appear in the artifacts folder.",
          ),
          message(
            "help.workflow.step6.bullet4",
            "registered oder ähnliche Unterordner enthalten – je nach Konfiguration – registrierte Zwischenstände, Debug-Ausgaben oder abgeleitete Zwischenprodukte. Diese Dateien helfen vor allem bei Analyse, Vergleich und Fehlersuche.",
            "registered and similar subfolders may contain registered intermediates, debug outputs or derived intermediate products, depending on configuration. These files are mainly useful for analysis, comparison and troubleshooting.",
          ),
          message(
            "help.workflow.step6.bullet5",
            "Wenn du nur das Endergebnis beurteilen willst, beginne mit den finalen Stack-/Solve-/BGE-/PCC-Dateien. Wenn du verstehen willst, warum ein Ergebnis gut oder schlecht wurde, arbeite dich zusätzlich durch artifacts, Logs und den Report.",
            "If you only want to assess the final result, start with the final stack/solve/BGE/PCC files. If you want to understand why the result turned out good or bad, also work through artifacts, logs and the report.",
          ),
        ],
      },
    ];
    return `
      <div class="help-window-copy">
        <p class="help-window-intro">${escapeHtml(message(
          "help.workflow.subtitle",
          "Hilfe für den Standardablauf. Du kannst es während der Dateneingabe offen lassen.",
          "Help for the standard sequence. You can leave it open while entering data.",
        ))}</p>
        ${sections.map((section) => `
          <section class="help-window-section">
            <h4>${escapeHtml(section.title)}</h4>
            <p>${escapeHtml(section.lead)}</p>
            <ul>
              ${section.bullets.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
            </ul>
          </section>
        `).join("")}
      </div>
    `;
  }

  function sidebarLink(item, activeKey) {
    const classes = item.key === activeKey ? "ps-item ps-active" : "ps-item";
    const dataControl = item.control ? ` data-control="${item.control}"` : "";
    const dataI18n = item.control ? ` data-i18n="ui.${item.control}"` : "";
    const dataTitle = item.control ? ` data-i18n-title="ui.tooltip.shell.${item.control.replaceAll(".", "_")}"` : "";
    return `<a class="${classes}"${dataControl}${dataI18n}${dataTitle} href="${item.href}">${item.label}</a>`;
  }

  function renderHeader(activeKey) {
    const nav = HEADER_NAV_ITEMS.map((item) => navLink(item, activeKey)).join("\n    ");
    return `
  <nav class="pill-nav">\n    ${nav}\n    <div class="pill-nav-tools">\n      <button id="shell-help-toggle" class="shell-help-toggle" type="button" data-i18n-title="ui.tooltip.shell.help_open" title="Arbeitsablauf-Hilfe öffnen." aria-label="Open help">?</button>\n    </div>\n  </nav>\n\n  <div class="pill-subbar">\n    <button id="locale-de" data-control="locale.de" data-i18n="ui.locale.de" data-i18n-title="ui.tooltip.shell.locale_de" class="locale-btn active" title="Sprache auf Deutsch setzen.">DE</button>\n    <button id="locale-en" data-control="locale.en" data-i18n="ui.locale.en" data-i18n-title="ui.tooltip.shell.locale_en" class="locale-btn" title="Language to English.">EN</button>\n    <div class="pill-status-group">\n      <span id="status-run-ready" data-control="status.run_ready" data-i18n-title="ui.tooltip.shell.run_ready" class="shell-status-chip shell-status-chip-check" title="Run-Status anzeigen.">${escapeHtml(message("ui.status.run_ready_check", "Start prüfen", "Run check"))}</span>\n      <span id="status-guardrail" data-control="status.guardrail" data-i18n-title="ui.tooltip.shell.guardrail_status" class="shell-status-chip shell-status-chip-check" title="Guardrail-Status anzeigen.">${escapeHtml(message("ui.status.guardrail_check", "Guardrails: prüfen", "Guardrails: check"))}</span>\n    </div>\n  </div>\n  <aside id="shell-help-window" class="help-window" hidden>\n    <div id="shell-help-drag-handle" class="help-window-header">\n      <div class="help-window-title-wrap">\n        <div id="shell-help-title" class="help-window-title">${escapeHtml(message("help.workflow.title", "Hilfe", "Help"))}</div>\n           </div>\n      <button id="shell-help-close" class="help-window-close" type="button" data-i18n-title="ui.tooltip.shell.help_close" title="Hilfe schließen." aria-label="Close help">x</button>\n    </div>\n    <div id="shell-help-content" class="help-window-content">${helpMarkup()}</div>\n  </aside>\n`;
  }

  function renderSidebar(activeKey) {
    const links = SIDEBAR_ITEMS.map((item) => sidebarLink(item, activeKey)).join("\n        ");
    const quickStart =
      activeKey === "dashboard"
        ? "\n        <div style=\"margin-top:auto;padding:16px;\">\n          <a id=\"dashboard-quick-start-wizard\" data-control=\"dashboard.quick.start_wizard\" data-i18n=\"ui.button.start_wizard\" data-i18n-title=\"ui.tooltip.dashboard.start_wizard\" class=\"ps-btn\" href=\"wizard.html\" style=\"display:block;text-align:center;padding:10px;\" title=\"Neuen Guided Run starten.\">Start Wizard</a>\n        </div>"
        : "";
    return `<div class="ps-label" data-i18n="page.shared.navigation">Navigation</div>\n        ${links}${quickStart}`;
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

  const helpToggle = document.getElementById("shell-help-toggle");
  const helpWindow = document.getElementById("shell-help-window");
  const helpClose = document.getElementById("shell-help-close");
  const helpContent = document.getElementById("shell-help-content");
  const helpTitle = document.getElementById("shell-help-title");
  const helpSubtitle = document.getElementById("shell-help-subtitle");
  const dragHandle = document.getElementById("shell-help-drag-handle");
  let dragState = null;

  function applyHelpCopy() {
    if (helpTitle) helpTitle.textContent = message("help.workflow.title", "Hilfe", "Help");
    if (helpContent) helpContent.innerHTML = helpMarkup();
  }

  function clampHelpWindow() {
    if (!helpWindow) return;
    const rect = helpWindow.getBoundingClientRect();
    const maxLeft = Math.max(8, window.innerWidth - rect.width - 8);
    const maxTop = Math.max(8, window.innerHeight - 64);
    const left = Math.min(Math.max(8, rect.left), maxLeft);
    const top = Math.min(Math.max(8, rect.top), maxTop);
    helpWindow.style.left = `${left}px`;
    helpWindow.style.top = `${top}px`;
  }

  function placeHelpWindowFromState() {
    if (!helpWindow) return;
    const state = readHelpState();
    const width = Number(state.width);
    const height = Number(state.height);
    helpWindow.style.left = `${Number.isFinite(Number(state.left)) ? Number(state.left) : Math.max(24, window.innerWidth - 500)}px`;
    helpWindow.style.top = `${Number.isFinite(Number(state.top)) ? Number(state.top) : 108}px`;
    if (Number.isFinite(width) && width >= 320) helpWindow.style.width = `${width}px`;
    if (Number.isFinite(height) && height >= 260) helpWindow.style.height = `${height}px`;
    clampHelpWindow();
  }

  function openHelpWindow() {
    if (!helpWindow) return;
    helpWindow.hidden = false;
    placeHelpWindowFromState();
    writeHelpState({ open: true });
  }

  function closeHelpWindow() {
    if (!helpWindow) return;
    helpWindow.hidden = true;
    writeHelpState({ open: false });
  }

  function persistHelpWindowGeometry() {
    if (!helpWindow) return;
    const rect = helpWindow.getBoundingClientRect();
    writeHelpState({
      left: Math.round(rect.left),
      top: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    });
  }

  helpToggle?.addEventListener("click", () => {
    if (helpWindow?.hidden) openHelpWindow();
    else closeHelpWindow();
  });
  helpClose?.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    closeHelpWindow();
  });
  helpClose?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    closeHelpWindow();
  });

  dragHandle?.addEventListener("pointerdown", (event) => {
    if (!helpWindow || event.target instanceof Element && event.target.closest("#shell-help-close")) return;
    const rect = helpWindow.getBoundingClientRect();
    dragState = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
      pointerId: event.pointerId,
    };
    dragHandle.setPointerCapture(event.pointerId);
    helpWindow.classList.add("dragging");
  });

  dragHandle?.addEventListener("pointermove", (event) => {
    if (!dragState || dragState.pointerId !== event.pointerId || !helpWindow) return;
    const maxLeft = Math.max(8, window.innerWidth - helpWindow.offsetWidth - 8);
    const maxTop = Math.max(8, window.innerHeight - 64);
    const left = Math.min(Math.max(8, event.clientX - dragState.offsetX), maxLeft);
    const top = Math.min(Math.max(8, event.clientY - dragState.offsetY), maxTop);
    helpWindow.style.left = `${left}px`;
    helpWindow.style.top = `${top}px`;
  });

  function stopDragging(event) {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    dragState = null;
    helpWindow?.classList.remove("dragging");
    persistHelpWindowGeometry();
  }

  dragHandle?.addEventListener("pointerup", stopDragging);
  dragHandle?.addEventListener("pointercancel", stopDragging);
  window.addEventListener("resize", () => {
    clampHelpWindow();
    if (helpWindow && !helpWindow.hidden) persistHelpWindowGeometry();
  });
  helpWindow?.addEventListener("mouseup", persistHelpWindowGeometry);

  document.addEventListener("gui2:locale-changed", applyHelpCopy);
  applyHelpCopy();
  if (readHelpState().open) openHelpWindow();
})();
