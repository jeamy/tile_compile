// js/components/path-picker-modal.js – File/Directory picker modal using /api/fs/list

import { el, clear } from "../utils/dom.js";
import { api } from "../api/client.js";
import { t } from "../i18n/i18n.js";

export async function promptGrantRoot(path, _allowedRoots) {
  return new Promise((resolve) => {
    const rootToGrant = path.replace(/\/[^/]*$/, "") || path;

    const overlay = el("div", { class: "tc-modal-overlay" });

    const modal = el("div", { class: "tc-modal" },
      el("div", { class: "tc-modal-header" },
        el("span", { class: "tc-modal-title" },
          t("ui.grant_root.title", "Pfad nicht erlaubt")),
        el("button", { class: "tc-btn tc-btn-sm", onclick: () => close(false) }, "✕"),
      ),
      el("div", { class: "tc-modal-body" },
        el("p", { style: "font-size: var(--text-sm); margin: 0 0 0.5em;" },
          t("ui.grant_root.message",
            "Der Pfad liegt außerhalb der erlaubten Verzeichnisse. Soll er für diese Sitzung freigegeben werden?")),
        el("code", { class: "tc-code-block" }, rootToGrant),
      ),
      el("div", { class: "tc-modal-footer" },
        el("button", { class: "tc-btn", onclick: () => close(false) },
          t("ui.button.cancel", "Abbrechen")),
        el("button", { class: "tc-btn tc-btn-primary", onclick: () => grant() },
          t("ui.grant_root.allow", "Erlauben")),
      ),
    );

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    function close(result) {
      overlay.remove();
      resolve(result);
    }

    async function grant() {
      try {
        await api.post("/api/fs/grant-root", { path: rootToGrant });
        close(true);
      } catch (e) {
        close(false);
      }
    }
  });
}

export function openPathPicker(opts = {}) {
  const {
    mode = "dir",
    initialPath = "",
    filter = "",
    onSelect = null,
  } = opts;

  return new Promise((resolve) => {
    let currentPath = initialPath || "";
    let selectedPath = "";

    const overlay = el("div", {
      class: "tc-modal-overlay",
      onclick: (e) => { if (e.target === overlay) close(null); },
    });

    const modal = el("div", { class: "tc-modal tc-path-picker" },
      el("div", { class: "tc-modal-header" },
        el("span", { class: "tc-modal-title" },
          mode === "dir" ? t("ui.picker.select_dir", "Verzeichnis wählen") : t("ui.picker.select_file", "Datei wählen")),
        el("button", { class: "tc-btn tc-btn-sm", onclick: () => close(null) }, "✕"),
      ),
      el("div", { class: "tc-path-picker-cwd" },
        el("input", {
          type: "text",
          class: "tc-input",
          id: "path-picker-cwd",
          value: currentPath,
          oninput: (e) => { currentPath = e.target.value; },
        }),
        el("button", {
          class: "tc-btn tc-btn-sm",
          onclick: () => loadListing(currentPath),
        }, t("ui.picker.go", "Go")),
      ),
      el("div", { class: "tc-path-picker-list tc-scroll", id: "path-picker-list" },
        el("div", { class: "tc-text-muted" }, t("ui.state.loading", "Lädt...")),
      ),
      el("div", { class: "tc-modal-footer" },
        el("button", { class: "tc-btn", onclick: () => close(null) }, t("ui.button.cancel", "Abbrechen")),
        el("button", {
          class: "tc-btn tc-btn-primary",
          id: "path-picker-confirm",
          onclick: () => close(selectedPath || currentPath),
        }, t("ui.button.confirm", "Bestätigen")),
      ),
    );

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    function close(result) {
      overlay.remove();
      if (onSelect) onSelect(result);
      resolve(result);
    }

    async function loadListing(path) {
      const listEl = modal.querySelector("#path-picker-list");
      clear(listEl);
      listEl.appendChild(el("div", { class: "tc-text-muted" }, t("ui.state.loading", "Lädt...")));

      try {
        const params = new URLSearchParams();
        if (path) params.set("path", path);
        params.set("include_files", mode === "file" ? "1" : "0");
        let data;
        try {
          data = await api.get(`/api/fs/list?${params}`);
        } catch (firstErr) {
          if (firstErr.status === 403 &&
              firstErr.payload?.code === "PATH_NOT_ALLOWED") {
            const granted = await promptGrantRoot(path, firstErr.payload?.details?.allowed_roots);
            if (granted) {
              data = await api.get(`/api/fs/list?${params}`);
            } else {
              throw firstErr;
            }
          } else if (firstErr.status === 400 && path) {
            const parent = path.replace(/\/[^/]+$/, "");
            if (parent && parent !== path) {
              currentPath = parent;
              updateCwd();
              if (mode === "file") selectedPath = path;
              const retryParams = new URLSearchParams();
              retryParams.set("path", parent);
              retryParams.set("include_files", mode === "file" ? "1" : "0");
              data = await api.get(`/api/fs/list?${retryParams}`);
            } else {
              throw firstErr;
            }
          } else {
            throw firstErr;
          }
        }
        clear(listEl);

        const parentPath = data?.parent || "";
        if (parentPath) {
          listEl.appendChild(pickerRow("📁 ..", parentPath, true, () => {
            currentPath = parentPath;
            updateCwd();
            loadListing(parentPath);
          }));
        }

        const items = data?.items || [];
        for (const item of items) {
          const isDir = item.type === "dir" || item.is_dir;
          const name = item.name || item.label || "";
          const fullPath = item.path || (path ? `${path}/${name}` : name);

          if (mode === "file" && !isDir) {
            if (filter && !matchesFilter(name, filter)) continue;
            listEl.appendChild(pickerRow(`📄 ${name}`, fullPath, false, () => {
              selectedPath = fullPath;
              modal.querySelectorAll(".tc-path-picker-row").forEach(r => r.classList.remove("active"));
              const last = listEl.querySelector(".tc-path-picker-row:last-child");
            }, true));
          } else if (isDir) {
            listEl.appendChild(pickerRow(`📁 ${name}`, fullPath, true, () => {
              currentPath = fullPath;
              updateCwd();
              loadListing(fullPath);
            }, mode === "dir"));
          }
        }

        if (items.length === 0 && !parentPath) {
          listEl.appendChild(el("div", { class: "tc-text-muted" }, t("ui.state.empty", "Leer")));
        }
      } catch (e) {
        clear(listEl);
        listEl.appendChild(el("div", { class: "tc-text-error" }, e.message || "Error"));
      }
    }

    function pickerRow(label, path, isDir, onClick, selectable = false) {
      const row = el("div", {
        class: "tc-path-picker-row",
        onclick: (e) => {
          if (selectable) {
            modal.querySelectorAll(".tc-path-picker-row").forEach(r => r.classList.remove("active"));
            row.classList.add("active");
            selectedPath = path;
          }
          if (isDir && e.detail === 2) {
            onClick();
          } else if (!selectable) {
            onClick();
          }
        },
        ondblclick: () => { if (isDir) onClick(); },
      }, label);
      row.dataset.path = path;
      return row;
    }

    function updateCwd() {
      const cwdInput = modal.querySelector("#path-picker-cwd");
      if (cwdInput) cwdInput.value = currentPath;
    }

    function matchesFilter(name, filter) {
      if (!filter) return true;
      const patterns = filter.split(";").map(p => p.trim().toLowerCase());
      const lower = name.toLowerCase();
      return patterns.some(p => {
        if (p.startsWith("*.")) return lower.endsWith(p.slice(1));
        return lower === p;
      });
    }

    // Initial load
    if (currentPath) {
      loadListing(currentPath);
    } else {
      // Load roots first
      api.get("/api/fs/roots").then(data => {
        const roots = data?.items || data?.roots || [];
        const listEl = modal.querySelector("#path-picker-list");
        clear(listEl);
        for (const root of roots) {
          const rootPath = typeof root === "string" ? root : (root.path || root.name || "");
          const rootLabel = typeof root === "string" ? root : (root.label || root.name || root.path || "");
          listEl.appendChild(pickerRow(`📁 ${rootLabel}`, rootPath, true, () => {
            currentPath = rootPath;
            updateCwd();
            loadListing(rootPath);
          }));
        }
      }).catch(() => {
        loadListing("");
      });
    }
  });
}
