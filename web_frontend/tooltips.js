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

  function normalizePathSeparators(value) {
    return String(value || "").replace(/\\/g, "/");
  }

  function i18nText(key, fallback) {
    return String(window.GUI2_LOCALE_MESSAGES?.[key] || fallback || "");
  }

  function joinPath(basePath, childPath) {
    const base = normalizePathSeparators(basePath).replace(/\/+$/, "");
    const child = normalizePathSeparators(childPath).replace(/^\/+/, "");
    if (!base) return child ? `/${child}` : "/";
    if (!child) return base || "/";
    return `${base}/${child}`;
  }

  function splitPathParts(value) {
    const raw = normalizePathSeparators(value).trim().replace(/\/+$/, "");
    if (!raw) return { dir: "", base: "" };
    const slash = raw.lastIndexOf("/");
    if (slash < 0) return { dir: "", base: raw };
    if (slash === 0) return { dir: "/", base: raw.slice(1) };
    return { dir: raw.slice(0, slash), base: raw.slice(slash + 1) };
  }

  function normalizeAbsolutePath(value) {
    const raw = normalizePathSeparators(value).trim();
    if (!raw) return "";
    const isUnc = raw.startsWith("//");
    const prefixMatch = raw.match(/^[A-Za-z]:/);
    const prefix = prefixMatch ? prefixMatch[0] : isUnc ? "//" : raw.startsWith("/") ? "/" : "";
    const withoutPrefix = prefix ? raw.slice(prefix.length) : raw;
    const parts = withoutPrefix.split("/").filter(Boolean);
    const normalized = [];
    parts.forEach((part) => {
      if (part === ".") return;
      if (part === "..") {
        if (normalized.length > 0 && normalized[normalized.length - 1] !== "..") normalized.pop();
        else if (!prefix) normalized.push("..");
        return;
      }
      normalized.push(part);
    });
    const joined = normalized.join("/");
    if (prefix === "/") return joined ? `/${joined}` : "/";
    if (prefix === "//") return joined ? `//${joined}` : "//";
    if (prefixMatch) return joined ? `${prefix}/${joined}` : `${prefix}/`;
    return joined;
  }

  function resolvePickerInputPath(typedValue, currentPath) {
    const typed = String(typedValue || "").trim();
    if (!typed) return "";
    if (isAbsolutePath(typed)) return normalizeAbsolutePath(typed);
    return normalizeAbsolutePath(joinPath(currentPath || "/", typed));
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
    const rowSelect = button.closest(".ps-row")?.querySelector("select");
    const rowSelectText = String(
      rowSelect?.selectedOptions?.[0]?.textContent || rowSelect?.value || "",
    ).trim().toLowerCase();
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
      "  <div id='path-picker-save-row' style='display:none;padding:10px 14px;border-bottom:1px solid #d9e3ec;gap:8px;align-items:center;'>",
      "    <label id='path-picker-filename-label' for='path-picker-filename' style='font-size:13px;color:#475569;white-space:nowrap;'>Dateiname</label>",
      "    <input id='path-picker-filename' type='text' style='flex:1;min-width:0;padding:7px 10px;border:1px solid #c8d4df;border-radius:8px;'>",
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
    const fileSelectable = mode === "file" || mode === "save-file";
    if (data?.parent) {
      const parentRow = document.createElement("div");
      parentRow.style.cssText =
        "display:flex;align-items:center;gap:8px;padding:7px 6px;border-bottom:1px solid #d7e2eb;border-radius:6px;font-weight:600;";
      parentRow.style.cursor = "pointer";
      parentRow.title = "Ins Elternverzeichnis wechseln";
      const parentName = document.createElement("span");
      parentName.textContent = "📁 ..";
      parentName.style.cssText = "flex:1;min-width:0;";
      parentRow.appendChild(parentName);
      parentRow.addEventListener("click", () => onPickPath(data.parent));
      parentRow.addEventListener("dblclick", () => onOpenDir(data.parent));
      listEl.appendChild(parentRow);
    }
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
      } else if (fileSelectable) {
        row.style.cursor = "pointer";
        row.title = mode === "save-file" ? "Klick: Datei zum Überschreiben auswählen" : "Doppelklick: Datei auswählen";
        row.addEventListener("click", () => onPickPath(item.path));
        row.addEventListener("dblclick", () => onPickFile(item.path));
      }
      listEl.appendChild(row);
    });
  }

  async function pickPathValue(currentValue, modeOrOptions) {
    const options =
      typeof modeOrOptions === "string"
        ? { mode: modeOrOptions }
        : modeOrOptions && typeof modeOrOptions === "object"
          ? { ...modeOrOptions }
          : { mode: "dir" };
    const mode = String(options.mode || "dir").trim();
    const isFileMode = mode === "file" || mode === "save-file";
    const isSaveFileMode = mode === "save-file";
    const overlay = ensurePathPickerElements();
    const modeEl = document.getElementById("path-picker-mode");
    const currentEl = document.getElementById("path-picker-current");
    const saveRowEl = document.getElementById("path-picker-save-row");
    const fileNameLabelEl = document.getElementById("path-picker-filename-label");
    const fileNameEl = document.getElementById("path-picker-filename");
    const listEl = document.getElementById("path-picker-list");
    const goBtn = document.getElementById("path-picker-go");
    const cancelBtn = document.getElementById("path-picker-cancel");
    const selectBtn = document.getElementById("path-picker-select");

    modeEl.textContent = isSaveFileMode
      ? i18nText("ui.dialog.save_file", "Datei speichern unter")
      : isFileMode
        ? i18nText("ui.dialog.pick_file", "Dateiauswahl")
        : i18nText("ui.dialog.pick_directory", "Verzeichnisauswahl");
    selectBtn.textContent = isSaveFileMode
      ? i18nText("ui.button.save_file", "Datei speichern")
      : isFileMode
        ? i18nText("ui.button.use_file", "Datei übernehmen")
        : i18nText("ui.button.use_directory", "Verzeichnis übernehmen");
    cancelBtn.textContent = i18nText("ui.button.cancel", "Abbrechen");
    goBtn.textContent = i18nText("ui.button.go", "Go");
    fileNameLabelEl.textContent = i18nText("ui.field.file_name", "Dateiname");
    saveRowEl.style.display = isSaveFileMode ? "flex" : "none";

    let resolvePromise;
    const done = (value) => {
      overlay.style.display = "none";
      resolvePromise(value);
    };

    let currentPath = String(currentValue || "").trim();
    let selectedFile = "";

    if (isSaveFileMode) {
      const initialName = String(options.defaultFileName || "").trim();
      const parts = splitPathParts(currentPath);
      if (parts.dir && parts.base) {
        currentPath = parts.dir;
        fileNameEl.value = parts.base;
      } else {
        fileNameEl.value = initialName;
      }
    } else {
      fileNameEl.value = "";
    }

    async function openPath(path, allowGrant = true) {
      const p = String(path || "").trim();
      const query = new URLSearchParams({
        path: p,
        include_files: isFileMode ? "1" : "0",
      });
      let data;
      try {
        data = await apiGet(`/api/fs/list?${query.toString()}`);
      } catch (err) {
        const code = err?.payload?.error?.code || "";
        if (isSaveFileMode && code === "NOT_A_DIRECTORY" && isAbsolutePath(p)) {
          const parentPath = resolvePickerInputPath("..", p);
          if (parentPath && parentPath !== p) {
            await openPath(parentPath, allowGrant);
            const parts = splitPathParts(p);
            if (parts.base) fileNameEl.value = parts.base;
            return;
          }
        }
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
          if (isSaveFileMode) {
            const parts = splitPathParts(filePath);
            currentPath = parts.dir || currentPath;
            currentEl.value = currentPath;
            fileNameEl.value = parts.base || fileNameEl.value;
          } else {
            currentEl.value = filePath;
          }
        },
        (pickedPath) => {
          if (isSaveFileMode) {
            const parts = splitPathParts(pickedPath);
            if (parts.dir) {
              currentPath = parts.dir;
              currentEl.value = currentPath;
              if (parts.base) fileNameEl.value = parts.base;
            } else {
              currentEl.value = pickedPath;
            }
          } else {
            currentEl.value = pickedPath;
          }
        },
      );
      if (isSaveFileMode && isAbsolutePath(p) && p !== currentPath) {
        const parts = splitPathParts(p);
        if (parts.dir) currentEl.value = parts.dir;
        if (parts.base) fileNameEl.value = parts.base;
      }
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
      let candidate;
      if (isSaveFileMode) {
        const rawDir = String(currentEl.value || "").trim();
        const rawFile = String(fileNameEl.value || "").trim();
        const parts = splitPathParts(rawDir);
        const baseDir = rawFile ? rawDir : parts.dir || rawDir;
        const finalFile = rawFile || parts.base;
        candidate = joinPath(baseDir, finalFile);
      } else {
        candidate = String(currentEl.value || "").trim();
      }
      if (!candidate) return;
      done(candidate);
    };
    const onGo = () => {
      const resolved = resolvePickerInputPath(currentEl.value, currentPath);
      if (resolved) void openPath(resolved);
    };
    const onCurrentKeydown = (ev) => {
      if (ev.key !== "Enter") return;
      ev.preventDefault();
      onGo();
    };
    const onFileNameKeydown = (ev) => {
      if (ev.key !== "Enter") return;
      ev.preventDefault();
      onSelect();
    };
    const onOverlay = (ev) => {
      if (ev.target === overlay) done(null);
    };

    cancelBtn.addEventListener("click", onCancel, { once: true });
    selectBtn.addEventListener("click", onSelect, { once: true });
    goBtn.addEventListener("click", onGo);
    currentEl.addEventListener("keydown", onCurrentKeydown);
    fileNameEl.addEventListener("keydown", onFileNameKeydown);
    overlay.addEventListener("click", onOverlay);
    overlay.style.display = "block";

    return new Promise((resolve) => {
      resolvePromise = resolve;
    }).finally(() => {
      goBtn.removeEventListener("click", onGo);
      currentEl.removeEventListener("keydown", onCurrentKeydown);
      fileNameEl.removeEventListener("keydown", onFileNameKeydown);
      overlay.removeEventListener("click", onOverlay);
    });
  }

  window.gui2PickPathValue = pickPathValue;

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
