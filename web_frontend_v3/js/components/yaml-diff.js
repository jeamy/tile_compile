// js/components/yaml-diff.js – YAML-Diff Anzeige

import { el } from "../utils/dom.js";
import { formatYamlDiff } from "../utils/yaml.js";
import { t } from "../i18n/i18n.js";

export function createYamlDiff(before = "", after = "") {
  const wrapper = el("div", { class: "tc-card", style: { background: "var(--bg)" } },
    el("div", { class: "tc-card-title" }, t("ui.title.yaml_diff", "YAML Diff")),
    el("div", { class: "tc-mono tc-text-xs", id: "yaml-diff-body" },
      renderDiff(before, after),
    ),
  );
  return wrapper;
}

export function updateYamlDiff(before, after) {
  const body = document.getElementById("yaml-diff-body");
  if (!body) return;
  body.innerHTML = "";
  body.appendChild(renderDiff(before, after));
}

function renderDiff(before, after) {
  const diff = formatYamlDiff(before, after);
  if (diff.length === 0) {
    return el("div", { class: "tc-text-muted" }, t("ui.state.no_changes", "Keine \u00c4nderungen"));
  }

  const container = el("div", { class: "tc-flex-col" });
  for (const line of diff) {
    const cls = line.type === "added" ? "tc-diff-added" :
                line.type === "removed" ? "tc-diff-removed" : "tc-diff-unchanged";
    const prefix = line.type === "added" ? "+ " : line.type === "removed" ? "- " : "  ";
    container.appendChild(el("div", { class: cls }, prefix + line.text));
  }
  return container;
}
