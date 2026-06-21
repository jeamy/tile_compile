// js/components/queue-editor.js – Run-Queue Editor (MONO Filter-Queue)

import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { toast, toastError } from "./toast.js";

const FILTER_PRESETS = ["", "OSC", "L", "R", "G", "B", "Ha", "OIII", "SII"];

export function createQueueEditor({ items = [], onChange, currentInputDir = "" } = {}) {
  const wrapper = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.queue", "Run-Queue")),
    el("div", { class: "tc-queue-body", id: "queue-body" }),
    el("div", { class: "tc-mt-2" },
      el("button", {
        class: "tc-btn tc-btn-sm",
        onclick: () => addRow(),
      }, "+ " + t("ui.button.add_queue_row", "Eintrag hinzufuegen")),
    ),
  );

  function renderRows(rows) {
    const body = wrapper.querySelector("#queue-body");
    clear(body);
    for (const item of rows) {
      body.appendChild(createQueueRow(item, (updated) => {
        const idx = rows.indexOf(item);
        rows[idx] = { ...item, ...updated };
        onChange?.(rows);
      }, () => {
        const idx = rows.indexOf(item);
        if (idx >= 0) {
          rows.splice(idx, 1);
          renderRows(rows);
          onChange?.(rows);
        }
      }));
    }
  }

  function addRow() {
    const inputDir = typeof currentInputDir === "function" ? currentInputDir() : currentInputDir;
    if (!inputDir) {
      toastError(t("ui.toast.queue_no_input_dir", "Kein Eingabeordner ausgew\u00e4hlt / No input directory set"));
      return;
    }
    const alreadyInQueue = items.some(it => it.input_dir === inputDir);
    if (alreadyInQueue) {
      toastError(t("ui.toast.queue_dir_exists", "Verzeichnis bereits in der Queue / Directory already in queue"));
      return;
    }
    const newItem = {
      filter: "",
      input_dir: inputDir,
      pattern: "",
      run_id: "",
      enabled: false,
    };
    items.push(newItem);
    renderRows(items);
    onChange?.(items);
  }

  renderRows(items);
  return wrapper;
}

function createQueueRow(item, onUpdate, onRemove) {
  const row = el("div", { class: "tc-queue-row" });

  // Filter select + custom input
  const filterWrap = el("div", { class: "tc-queue-filter-stack" });
  const filterSelect = el("select", { class: "tc-select", "data-queue-field": "filter-select" },
    ...FILTER_PRESETS.map(f => el("option", { value: f }, f || "-")),
  );
  filterSelect.value = item.filter || "";
  const filterCustom = el("input", {
    type: "text", class: "tc-input", placeholder: "frei",
    "data-queue-field": "filter-custom", value: "",
  });
  if (item.filter && !FILTER_PRESETS.includes(item.filter)) {
    filterCustom.value = item.filter;
  }
  filterSelect.onchange = () => {
    filterCustom.value = "";
    onUpdate({ filter: filterSelect.value });
  };
  filterCustom.oninput = () => {
    if (filterCustom.value) filterSelect.value = "";
    onUpdate({ filter: filterCustom.value || filterSelect.value });
  };
  filterWrap.append(filterSelect, filterCustom);

  // Input dir
  const inputDir = el("input", {
    type: "text", class: "tc-input", placeholder: "/data/...",
    value: item.input_dir || "",
  });
  inputDir.oninput = () => onUpdate({ input_dir: inputDir.value });

  // Pattern
  const pattern = el("input", {
    type: "text", class: "tc-input", value: item.pattern || "", placeholder: "*.fits",
  });
  pattern.oninput = () => onUpdate({ pattern: pattern.value });

  // Run ID / Label
  const runId = el("input", {
    type: "text", class: "tc-input", value: item.run_id || "", placeholder: "M31_L",
  });
  runId.oninput = () => onUpdate({ run_id: runId.value });

  // Enabled toggle
  const toggle = el("label", { class: "tc-checkbox" },
    el("input", {
      type: "checkbox", checked: item.enabled !== false,
      "data-queue-field": "enabled",
      onchange: (e) => onUpdate({ enabled: e.target.checked }),
    }),
    el("span", {}, item.enabled !== false ? "on" : "off"),
  );

  // Remove button – red cross icon
  const removeBtn = el("button", {
    class: "tc-queue-remove",
    title: t("ui.button.remove", "Entfernen"),
    onclick: (e) => {
      e.stopPropagation();
      onRemove?.();
    },
  }, "\u2715");

  row.append(filterWrap, inputDir, pattern, runId, toggle, removeBtn);
  return row;
}
