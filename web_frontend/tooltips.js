document.addEventListener("DOMContentLoaded", () => {
  function humanizeControlId(controlId) {
    return String(controlId || "")
      .replace(/[._]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function labelTextForControl(el) {
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

  function tooltipFromControlId(el, controlId) {
    const c = String(controlId || "").trim().toLowerCase();
    if (!c) return "";
    if (c.startsWith("nav.")) return `Navigiert zu: ${humanizeControlId(controlId)}.`;
    if (c.includes(".browse_") || c.endsWith(".browse")) {
      return "Pfad auswählen (Doppelklick öffnet Verzeichnisse).";
    }
    if (c.includes(".scan_run") || c.includes(".scan_refresh")) return "Scan starten und Ergebnisse aktualisieren.";
    if (c.includes(".run_start")) return "Run mit aktuellen Eingaben starten.";
    if (c.includes(".resume")) return "Resume mit ausgewählter Phase ausführen.";
    if (c.includes(".set_current")) return "Ausgewählten Run als aktuellen Run setzen.";
    if (c.includes(".preset")) return "Preset auswählen oder anwenden.";
    if (c.includes(".validate")) return "Konfiguration validieren.";
    if (c.includes(".save")) return "Änderungen speichern.";
    if (c.includes(".open_report") || c.endsWith(".report")) return "Report anzeigen.";
    if (c.includes(".color_mode")) return "Farbmodus setzen oder bestätigen.";
    if (c.includes(".input_dirs")) return "Eingabeverzeichnisse setzen.";
    if (c.includes(".runs_dir")) return "Ausgabeverzeichnis für Runs setzen.";
    if (c.includes(".queue")) return "Wert für die MONO Filter-Queue setzen.";
    if (c.includes(".phase.progress")) return "Fortschritt der Pipeline-Phase in Prozent.";
    if (c.includes(".phase.")) return "Pipeline-Phase auswählen.";
    if (c.startsWith("tools.astrometry.")) return "Astrometry-Toolkonfiguration oder Aktion.";
    if (c.startsWith("tools.pcc.")) return "PCC-Toolkonfiguration oder Aktion.";
    if (c.startsWith("parameter.value.")) {
      const path = String(el.closest(".ps-dyn-row")?.getAttribute("data-path") || "").trim();
      return path ? `Parameterwert bearbeiten: ${path}` : "Parameterwert bearbeiten.";
    }
    const label = labelTextForControl(el);
    if (label) return `Feld bearbeiten: ${label}.`;
    return `Steuerelement: ${humanizeControlId(controlId)}.`;
  }

  function applyFallbackTooltips(root = document) {
    const selector = "a, button, input, select, textarea, [role='button']";
    const controls = [];
    if (root instanceof Element && root.matches(selector)) controls.push(root);
    controls.push(...root.querySelectorAll(selector));
    controls.forEach((el) => {
      const title = (el.getAttribute("title") || "").trim();
      if (title) return;

      const explicitTip = (el.getAttribute("data-tooltip") || "").trim();
      if (explicitTip) {
        el.setAttribute("title", explicitTip);
        return;
      }

      const controlId = (el.getAttribute("data-control") || "").trim();
      if (controlId) {
        el.setAttribute("title", tooltipFromControlId(el, controlId));
        return;
      }

      const aria = (el.getAttribute("aria-label") || "").trim();
      if (aria) {
        el.setAttribute("title", aria);
        return;
      }

      const placeholder = (el.getAttribute("placeholder") || "").trim();
      if (placeholder) {
        el.setAttribute("title", "Eingabefeld: " + placeholder);
        return;
      }

      const label = labelTextForControl(el);
      if (label) {
        el.setAttribute("title", `Feld bearbeiten: ${label}.`);
        return;
      }

      const text = (el.textContent || "").replace(/\s+/g, " ").trim();
      if (text) {
        el.setAttribute("title", text);
        return;
      }

      el.setAttribute("title", "Interaktives Element");
    });
  }

  function isAbsolutePath(value) {
    const s = String(value || "").trim();
    return s.startsWith("/") || /^[A-Za-z]:[\\/]/.test(s) || s.startsWith("\\\\");
  }

  applyFallbackTooltips(document);
  const tooltipObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (!(node instanceof Element)) return;
        applyFallbackTooltips(node);
      });
    });
  });
  tooltipObserver.observe(document.body, { childList: true, subtree: true });

  const browseTargetMap = {
    "tools.astrometry.browse_binary": "tools-astrometry-bin",
    "tools.astrometry.browse_data_dir": "tools-astrometry-data-dir",
    "tools.astrometry.browse_file": "tools-astrometry-file",
    "tools.pcc.browse_rgb": "tools-pcc-rgb",
    "tools.pcc.browse_wcs": "tools-pcc-wcs",
    "tools.pcc.browse_catalog_dir": "tools-pcc-catalog-dir",
  };

  function resolveBrowseTarget(button) {
    const explicitTargetId = (button.getAttribute("data-target-id") || "").trim();
    if (explicitTargetId) {
      const explicitTarget = document.getElementById(explicitTargetId);
      if (explicitTarget) return explicitTarget;
    }

    const controlId = (button.getAttribute("data-control") || "").trim();
    if (controlId && browseTargetMap[controlId]) {
      const mappedTarget = document.getElementById(browseTargetMap[controlId]);
      if (mappedTarget) return mappedTarget;
    }

    const cluster = button.closest(".ps-inline-cluster");
    if (cluster) {
      const clusterInput = cluster.querySelector("input[type='text']");
      if (clusterInput) return clusterInput;
    }

    const row = button.closest(".ps-row");
    if (row) {
      const rowInput = row.querySelector("input[type='text']");
      if (rowInput) return rowInput;
    }

    const actions = button.closest(".ps-actions");
    if (actions) {
      let prev = actions.previousElementSibling;
      while (prev) {
        const prevInput = prev.querySelector("input[type='text']");
        if (prevInput) return prevInput;
        prev = prev.previousElementSibling;
      }
    }

    return null;
  }

  async function apiGet(path) {
    const resp = await fetch(path, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
    let payload = null;
    try {
      payload = await resp.json();
    } catch {
      payload = null;
    }
    if (!resp.ok) {
      const msg = payload?.error?.message || `HTTP ${resp.status}`;
      const err = new Error(msg);
      err.status = resp.status;
      err.payload = payload;
      throw err;
    }
    return payload;
  }

  async function apiPost(path, body) {
    const resp = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    let payload = null;
    try {
      payload = await resp.json();
    } catch {
      payload = null;
    }
    if (!resp.ok) {
      const msg = payload?.error?.message || `HTTP ${resp.status}`;
      const err = new Error(msg);
      err.status = resp.status;
      err.payload = payload;
      throw err;
    }
    return payload;
  }

  function inferBrowseMode(button, target) {
    const controlId = (button.getAttribute("data-control") || "").trim().toLowerCase();
    const targetId = String(target?.id || "").toLowerCase();
    const rowSelectText = String(button.closest(".ps-row")?.querySelector("select")?.value || "").trim().toLowerCase();
    if (
      rowSelectText.includes("datei") ||
      controlId.includes("browse_file") ||
      controlId.includes("browse_rgb") ||
      controlId.includes("browse_wcs") ||
      controlId.includes("browse_binary") ||
      targetId.includes("file") ||
      targetId.includes("rgb") ||
      targetId.includes("wcs") ||
      targetId.includes("bin")
    ) {
      return "file";
    }
    return "dir";
  }

  function ensurePathPickerElements() {
    let overlay = document.getElementById("path-picker-overlay");
    if (overlay) return overlay;

    overlay = document.createElement("div");
    overlay.id = "path-picker-overlay";
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(15,23,42,0.35)";
    overlay.style.display = "none";
    overlay.style.zIndex = "2000";
    overlay.innerHTML = [
      "<div id='path-picker-dialog' style='width:min(920px,92vw);max-height:80vh;overflow:hidden;margin:6vh auto;background:#fff;border:1px solid #c8d4df;border-radius:12px;display:flex;flex-direction:column;'>",
      "  <div style='padding:12px 14px;border-bottom:1px solid #d9e3ec;display:flex;gap:8px;align-items:center;'>",
      "    <strong style='font-size:15px;'>Pfad auswählen</strong>",
      "    <span id='path-picker-mode' style='margin-left:auto;font-size:12px;color:#64748b;'></span>",
      "  </div>",
      "  <div style='padding:10px 14px;border-bottom:1px solid #d9e3ec;display:flex;gap:8px;align-items:center;'>",
      "    <input id='path-picker-current' type='text' style='flex:1;min-width:0;padding:7px 10px;border:1px solid #c8d4df;border-radius:8px;'>",
      "    <button id='path-picker-go' type='button' style='padding:6px 10px;border:1px solid #c8d4df;border-radius:8px;background:#fff;cursor:pointer;'>Go</button>",
      "  </div>",
      "  <div id='path-picker-list' style='padding:6px 10px;overflow:auto;min-height:280px;'></div>",
      "  <div style='padding:12px 14px;border-top:1px solid #d9e3ec;display:flex;gap:8px;justify-content:flex-end;'>",
      "    <button id='path-picker-cancel' type='button' style='padding:8px 12px;border:1px solid #c8d4df;border-radius:8px;background:#fff;cursor:pointer;'>Abbrechen</button>",
      "    <button id='path-picker-select' type='button' style='padding:8px 14px;border:1px solid #0f8fa0;border-radius:8px;background:#0f8fa0;color:#fff;cursor:pointer;'>Auswählen</button>",
      "  </div>",
      "</div>",
    ].join("");
    document.body.appendChild(overlay);
    return overlay;
  }

  function renderPickerEntries(listEl, data, mode, onOpenDir, onPickFile, onPickPath) {
    listEl.innerHTML = "";
    (data?.items || []).forEach((item) => {
      const row = document.createElement("div");
      row.style.cssText =
        "display:flex;align-items:center;gap:8px;padding:7px 6px;border-bottom:1px solid #eef3f7;border-radius:6px;";
      const name = document.createElement("span");
      name.textContent = `${item.type === "dir" ? "📁" : "📄"} ${item.name}`;
      name.style.cssText = "flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      row.appendChild(name);
      if (item.type === "dir") {
        row.style.cursor = "pointer";
        row.title = "Doppelklick: Verzeichnis öffnen";
        row.addEventListener("click", () => onPickPath(item.path));
        row.addEventListener("dblclick", () => onOpenDir(item.path));
      } else if (mode === "file") {
        row.style.cursor = "pointer";
        row.title = "Doppelklick: Datei auswählen";
        row.addEventListener("click", () => onPickPath(item.path));
        row.addEventListener("dblclick", () => onPickFile(item.path));
      }
      listEl.appendChild(row);
    });
  }

  async function pickPathValue(currentValue, mode) {
    const overlay = ensurePathPickerElements();
    const modeEl = document.getElementById("path-picker-mode");
    const currentEl = document.getElementById("path-picker-current");
    const listEl = document.getElementById("path-picker-list");
    const goBtn = document.getElementById("path-picker-go");
    const cancelBtn = document.getElementById("path-picker-cancel");
    const selectBtn = document.getElementById("path-picker-select");

    modeEl.textContent = mode === "file" ? "Dateiauswahl" : "Verzeichnisauswahl";
    selectBtn.textContent = mode === "file" ? "Datei übernehmen" : "Verzeichnis übernehmen";

    let resolvePromise;
    const done = (value) => {
      overlay.style.display = "none";
      resolvePromise(value);
    };

    let currentPath = String(currentValue || "").trim();
    let selectedFile = "";

    async function openPath(path, allowGrant = true) {
      const p = String(path || "").trim();
      const query = new URLSearchParams({
        path: p,
        include_files: mode === "file" ? "1" : "0",
      });
      let data;
      try {
        data = await apiGet(`/api/fs/list?${query.toString()}`);
      } catch (err) {
        const code = err?.payload?.error?.code || "";
        if (allowGrant && code === "PATH_NOT_ALLOWED" && isAbsolutePath(p)) {
          const allow = window.confirm(
            `Pfad ist aktuell nicht freigegeben:\n${p}\n\nSoll dieser Pfad für die aktuelle Sitzung freigegeben werden?`,
          );
          if (allow) {
            await apiPost("/api/fs/grant-root", { path: p });
            return openPath(p, false);
          }
        }
        throw err;
      }
      currentPath = String(data.path || "");
      selectedFile = "";
      currentEl.value = currentPath;
      renderPickerEntries(
        listEl,
        data,
        mode,
        (dirPath) => void openPath(dirPath),
        (filePath) => {
          selectedFile = filePath;
          currentEl.value = filePath;
        },
        (pickedPath) => {
          currentEl.value = pickedPath;
        },
      );
    }

    const roots = await apiGet("/api/fs/roots");
    const fallbackRoot = String(roots.default_path || roots.items?.[0] || "");
    if (!currentPath) currentPath = fallbackRoot;
    if (!currentPath) throw new Error("Keine erlaubten Root-Pfade verfügbar");

    try {
      await openPath(currentPath);
    } catch (err) {
      if (!fallbackRoot || currentPath === fallbackRoot) {
        throw err;
      }
      await openPath(fallbackRoot);
    }

    const onCancel = () => done(null);
    const onSelect = () => {
      const candidate = String(currentEl.value || "").trim();
      if (!candidate) return;
      done(candidate);
    };
    const onGo = () => {
      const typed = String(currentEl.value || "").trim();
      if (typed) void openPath(typed);
    };
    const onOverlay = (ev) => {
      if (ev.target === overlay) done(null);
    };

    cancelBtn.addEventListener("click", onCancel, { once: true });
    selectBtn.addEventListener("click", onSelect, { once: true });
    goBtn.addEventListener("click", onGo);
    overlay.addEventListener("click", onOverlay);
    overlay.style.display = "block";

    return new Promise((resolve) => {
      resolvePromise = resolve;
    }).finally(() => {
      goBtn.removeEventListener("click", onGo);
      overlay.removeEventListener("click", onOverlay);
    });
  }

  const browseButtons = Array.from(document.querySelectorAll("button")).filter((btn) => {
    const txt = (btn.textContent || "").replace(/\s+/g, " ").trim().toLowerCase();
    const controlId = (btn.getAttribute("data-control") || "").trim();
    return txt.startsWith("browse") || controlId.includes(".browse_");
  });

  browseButtons.forEach((button) => {
    if (button.dataset.dirPickerBound === "1") return;
    button.dataset.dirPickerBound = "1";
    button.addEventListener("click", async (ev) => {
      const target = resolveBrowseTarget(button);
      if (!target) return;
      ev.preventDefault();
      try {
        const mode = inferBrowseMode(button, target);
        const pickedValue = await pickPathValue(target.value || "", mode);
        if (!pickedValue) return;
        if (!isAbsolutePath(pickedValue)) {
          window.alert("Bitte absoluten Vollpfad auswählen.");
          return;
        }
        target.value = pickedValue;
        target.dispatchEvent(new Event("input", { bubbles: true }));
      } catch (err) {
        window.alert(`Browse fehlgeschlagen: ${err?.message || err}`);
      }
    });
  });
});
