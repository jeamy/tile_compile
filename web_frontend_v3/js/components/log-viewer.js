// js/components/log-viewer.js – Virtualisierter Log-Viewer mit Filter/Search/Pause

import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { formatLogLevel } from "../utils/log.js";

const MAX_LINES = 10000;
const RENDER_BATCH = 50;

export function createLogViewer() {
  let lines = [];
  let lineKeys = new Set();
  let paused = false;
  let filterLevels = new Set(["INFO", "WARN", "ERROR", "DEBUG", "TRACE"]);
  let searchText = "";

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
      oninput: (e) => { searchText = e.target.value.toLowerCase(); render(); },
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
          filterLevels = new Set(["INFO", "WARN", "ERROR", "DEBUG", "TRACE"]);
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
        active = filterLevels.size === 5;
      } else {
        active = filterLevels.has(level);
      }
      btn.classList.toggle("tc-log-filter-active", active);
      btn.classList.toggle("tc-log-filter-inactive", !active);
    });
  }

  function getFilteredLines() {
    return lines.filter(l => {
      if (!filterLevels.has(l.level)) return false;
      if (searchText && !l.text.toLowerCase().includes(searchText)) return false;
      return true;
    });
  }

  function render() {
    const body = wrapper.querySelector("#log-viewer-body");
    if (!body) return;
    clear(body);
    const filtered = getFilteredLines();
    const visible = filtered.slice(-MAX_LINES);
    for (const line of visible) {
      body.appendChild(logLineEl(line));
    }
    if (!paused) body.scrollTop = body.scrollHeight;
  }

  function addLine(time, level, text) {
    const key = `${time}|${level}|${text}`;
    if (lineKeys.has(key)) return;
    lineKeys.add(key);
    lines.push({ time, level, text });
    if (lines.length > MAX_LINES * 2) {
      lines = lines.slice(-MAX_LINES);
      lineKeys = new Set(lines.map(l => `${l.time}|${l.level}|${l.text}`));
    }
    if (!paused) render();
  }

  function addLines(newLines) {
    for (const l of newLines) {
      const key = `${l.time}|${l.level}|${l.text}`;
      if (lineKeys.has(key)) continue;
      lineKeys.add(key);
      lines.push(l);
    }
    if (lines.length > MAX_LINES * 2) {
      lines = lines.slice(-MAX_LINES);
      lineKeys = new Set(lines.map(l => `${l.time}|${l.level}|${l.text}`));
    }
    if (!paused) render();
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
