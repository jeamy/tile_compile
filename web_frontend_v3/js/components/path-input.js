// js/components/path-input.js – Input + Browse-Button

import { el } from "../utils/dom.js";
import { openPathPicker } from "./path-picker-modal.js";

export function createPathInput(opts = {}) {
  const {
    label = "",
    value = "",
    placeholder = "",
    mode = "dir",
    filter = "",
    onInput = null,
    onBrowse = null,
  } = opts;

  const input = el("input", {
    type: "text",
    class: "tc-input",
    value,
    placeholder,
    ...(onInput ? { oninput: (e) => onInput(e.target.value) } : {}),
  });

  const browseBtn = el("button", {
    class: "tc-btn tc-btn-sm",
    onclick: async () => {
      if (onBrowse) {
        onBrowse(input);
      } else {
        const chosen = await openPathPicker({
          mode,
          initialPath: input.value || "",
          filter,
        });
        if (chosen) {
          input.value = chosen;
          if (onInput) onInput(chosen);
          input.dispatchEvent(new Event("input"));
        }
      }
    },
  }, "...");

  const group = el("div", { class: "tc-input-group" }, input, browseBtn);

  if (label) {
    return el("div", {},
      el("label", { class: "tc-label" }, label),
      group,
    );
  }
  return group;
}
