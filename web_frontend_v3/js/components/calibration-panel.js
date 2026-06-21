// js/components/calibration-panel.js – Kalibrierung (Bias/Dark/Flat)

import { el } from "../utils/dom.js";
import { createPathInput } from "./path-input.js";
import { t } from "../i18n/i18n.js";

export function createCalibrationPanel({ values = {}, onChange } = {}) {
  const state = {
    bias_enabled: values.bias_enabled ?? false,
    bias_dir: values.bias_dir || "",
    bias_master: values.bias_master || "",
    dark_enabled: values.dark_enabled ?? false,
    dark_dir: values.dark_dir || "",
    dark_master: values.dark_master || "",
    flat_enabled: values.flat_enabled ?? false,
    flat_dir: values.flat_dir || "",
    flat_master: values.flat_master || "",
  };

  function update(key, val) {
    state[key] = val;
    onChange?.(state);
  }

  const wrapper = el("div", { class: "tc-card" },
    el("div", { class: "tc-card-title" }, t("ui.title.calibration", "Kalibrierung")),
    ...["bias", "dark", "flat"].map(type => createCalRow(type, state, update)),
  );

  return wrapper;
}

function createCalRow(type, state, update) {
  const enabledKey = `${type}_enabled`;
  const dirKey = `${type}_dir`;
  const masterKey = `${type}_master`;
  const label = type.charAt(0).toUpperCase() + type.slice(1);

  const row = el("div", { class: "tc-cal-row" });

  const checkbox = el("input", {
    type: "checkbox",
    checked: state[enabledKey],
    onchange: (e) => {
      update(enabledKey, e.target.checked);
      row.classList.toggle("tc-cal-disabled", !e.target.checked);
    },
  });

  const dirInput = createPathInput({
    label: t(`ui.field.${type}_dir`, `${label} Ordner`),
    value: state[dirKey],
    onInput: (v) => update(dirKey, v),
  });

  const masterInput = createPathInput({
    label: t(`ui.field.${type}_master`, `${label} Master`),
    mode: "file",
    filter: "*.fits;*.fit;*.fts",
    value: state[masterKey],
    onInput: (v) => update(masterKey, v),
  });

  row.append(
    el("label", { class: "tc-checkbox" }, checkbox, label),
    dirInput,
    masterInput,
  );

  if (!state[enabledKey]) row.classList.add("tc-cal-disabled");
  return row;
}
