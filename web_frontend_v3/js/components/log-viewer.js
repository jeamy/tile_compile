// js/components/log-viewer.js – Virtualisierter Log-Viewer mit Filter/Search/Pause

import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { LOG_LEVELS, normalizeLogLevel } from "../utils/log.js";

const MAX_LINES = 10000;
const SEARCH_DEBOUNCE_MS = 150;

export function createLogViewer() {
  let lines = [];
  let lineKeys = new Set();
  let paused = false;
  let filterLevels = new Set(LOG_LEVELS);
  let searchText = "";
  let searchTimer = null;

  const wrapper = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title tc-flex tc-justify-between tc-items-center" },
      el("span", {}, t("ui.title.live_log", "Live Log")),
      el("div", { class: "tc-flex tc-gap-2 tc-items-center" },
        levelFilterButton("All", ""),
        levelFilterButton("INFO", "INFO"),
        levelFilterButton("WARN", "WARN"),
        levelFilterButton("ERROR", "ERROR"),
        levelFilterButton("DEBUG", "DEBUG"),
        el("button", {
          class: "tc-btn tc-btn-sm",
          id: "log-pause-btn",
          onclick: () => {
            paused = !paused;
            document.getElementById("log-pause-btn").textContent = paused ? "\u25b6" : "\u23f8";
            if (!paused) render();
          },
        }, "\u23f8"),
        el("button", {
          class: "tc-btn tc-btn-sm",
          onclick: () => exportLog(),
        }, "\u2b07"),
      ),
    ),
    el("input", {
      type: "text", class: "tc-input tc-mb-2", placeholder: t("ui.placeholder.search_log", "Log durchsuchen..."),
      oninput: (e) => {
        const value = e.target.value.toLowerCase();
        clearTimeout(searchTimer);
        searchTimer = setTimeout(() => { searchText = value; render(); }, SEARCH_DEBOUNCE_MS);
      },
    }),
    el("div", { class: "tc-log-viewer", id: "log-viewer-body" }),
  );

  updateFilterButtonStates();

  function levelFilterButton(level, colorKey) {
    const btn = el("button", {
      class: `tc-btn tc-btn-sm tc-log-filter-${colorKey || "all"}`,
      "data-level": level,
      onclick: () => {
        if (level === "All") {
          filterLevels = new Set(LOG_LEVELS);
        } else {
          if (filterLevels.has(level)) filterLevels.delete(level);
          else filterLevels.add(level);
        }
        updateFilterButtonStates();
        render();
      },
    }, level);
    return btn;
  }

  function updateFilterButtonStates() {
    const buttons = wrapper.querySelectorAll("[data-level]");
    buttons.forEach(btn => {
      const level = btn.dataset.level;
      const colorKey = btn.className.match(/tc-log-filter-(\w+)/)?.[1] || "all";
      let active;
      if (level === "All") {
        active = filterLevels.size === LOG_LEVELS.length;
      } else {
        active = filterLevels.has(level);
      }
      btn.classList.toggle("tc-log-filter-active", active);
      btn.classList.toggle("tc-log-filter-inactive", !active);
    });
  }

  function matches(line) {
    if (!filterLevels.has(line.level)) return false;
    if (searchText && !line.text.toLowerCase().includes(searchText)) return false;
    return true;
  }

  function getFilteredLines() {
    return lines.filter(matches);
  }

  function render() {
    const body = wrapper.querySelector("#log-viewer-body");
    if (!body) return;
    clear(body);
    const visible = getFilteredLines().slice(-MAX_LINES);
    for (const line of visible) {
      body.appendChild(logLineEl(line));
    }
    if (!paused) body.scrollTop = body.scrollHeight;
  }

  // Append only the new lines that pass the filter instead of rebuilding the
  // whole list per line; the DOM keeps at most MAX_LINES rows.
  function appendRendered(added) {
    const body = wrapper.querySelector("#log-viewer-body");
    if (!body) return;
    for (const line of added) {
      if (matches(line)) body.appendChild(logLineEl(line));
    }
    while (body.childNodes.length > MAX_LINES) body.removeChild(body.firstChild);
    if (!paused) body.scrollTop = body.scrollHeight;
  }

  function push(time, level, text, dedupeKey) {
    const normalized = normalizeLogLevel(level);
    const key = dedupeKey || `${time}|${normalized}|${text}`;
    if (lineKeys.has(key)) return null;
    lineKeys.add(key);
    const line = { time, level: normalized, text: String(text ?? ""), key };
    lines.push(line);
    return line;
  }

  function trimHistory() {
    if (lines.length <= MAX_LINES * 2) return false;
    lines = lines.slice(-MAX_LINES);
    lineKeys = new Set(lines.map(l => l.key));
    return true;
  }

  function settle(added) {
    if (paused || added.length === 0) return;
    if (trimHistory()) render();
    else appendRendered(added);
  }

  function addLine(time, level, text, dedupeKey = "") {
    const line = push(time, level, text, dedupeKey);
    if (line) settle([line]);
    else trimHistory();
  }

  function addLines(newLines) {
    const added = [];
    for (const l of newLines) {
      const line = push(l.time, l.level, l.text, l.key || "");
      if (line) added.push(line);
    }
    if (added.length) settle(added);
  }

  function clearLines() {
    lines = [];
    lineKeys = new Set();
    render();
  }

  function exportLog() {
    const text = lines.map(l => `${l.time} ${l.level} ${l.text}`).join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "run-log.txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  return { wrapper, addLine, addLines, clearLines, render };
}

function logLineEl(line) {
  return el("div", { class: "tc-log-line" },
    el("span", { class: "tc-log-time" }, line.time),
    el("span", { class: `tc-log-level tc-log-level-${line.level}` }, line.level),
    el("span", { class: "tc-log-msg" }, line.text),
  );
}
