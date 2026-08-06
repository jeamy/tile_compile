// js/components/calibration-panel.js – Kalibrierung (Bias/Dark/Flat)

import { el } from "../utils/dom.js";
import { createPathInput } from "./path-input.js";
import { t } from "../i18n/i18n.js";

export function createCalibrationPanel({ values = {}, onChange } = {}) {
  const sourceFor = (type) => {
    const explicitSource = values[`${type}_source`];
    if (explicitSource === "master" || explicitSource === "dir") return explicitSource;
    const explicitUseMaster = values[`${type}_use_master`];
    if (typeof explicitUseMaster === "boolean") return explicitUseMaster ? "master" : "dir";
    const hasMaster = Boolean((values[`${type}_master`] || "").trim());
    const hasDir = Boolean((values[`${type}_dir`] || "").trim());
    return hasMaster && !hasDir ? "master" : "dir";
  };

  const biasSource = sourceFor("bias");
  const darkSource = sourceFor("dark");
  const flatSource = sourceFor("flat");

  const state = {
    bias_enabled: values.bias_enabled ?? false,
    bias_source: biasSource,
    bias_use_master: biasSource === "master",
    bias_dir: values.bias_dir || "",
    bias_master: values.bias_master || "",
    dark_enabled: values.dark_enabled ?? false,
    dark_source: darkSource,
    dark_use_master: darkSource === "master",
    dark_dir: values.dark_dir || "",
    dark_master: values.dark_master || "",
    flat_enabled: values.flat_enabled ?? false,
    flat_source: flatSource,
    flat_use_master: flatSource === "master",
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
  const sourceKey = `${type}_source`;
  const useMasterKey = `${type}_use_master`;
  const dirKey = `${type}_dir`;
  const masterKey = `${type}_master`;
  const label = type.charAt(0).toUpperCase() + type.slice(1);

  const row = el("div", { class: "tc-cal-row" });
  const setInputEnabled = () => {
    const useMaster = state[sourceKey] === "master";
    const enabled = Boolean(state[enabledKey]);
    for (const input of dirInput.querySelectorAll("input, button")) input.disabled = !enabled || useMaster;
    for (const input of masterInput.querySelectorAll("input, button")) input.disabled = !enabled || !useMaster;
    dirInput.classList.toggle("tc-cal-source-inactive", useMaster);
    masterInput.classList.toggle("tc-cal-source-inactive", !useMaster);
  };

  const checkbox = el("input", {
    type: "checkbox",
    checked: state[enabledKey],
    onchange: (e) => {
      update(enabledKey, e.target.checked);
      row.classList.toggle("tc-cal-disabled", !e.target.checked);
      setInputEnabled();
    },
  });

  const sourceSelect = el("div", {},
    el("label", { class: "tc-label" }, t("ui.field.calibration_source", "Quelle")),
    el("select", {
      class: "tc-select",
      value: state[sourceKey],
      onchange: (e) => {
        const source = e.target.value === "master" ? "master" : "dir";
        update(sourceKey, source);
        update(useMasterKey, source === "master");
        setInputEnabled();
      },
    },
      el("option", { value: "dir", ...(state[sourceKey] === "dir" ? { selected: true } : {}) }, t("ui.option.calibration_source.dir", "Ordner")),
      el("option", { value: "master", ...(state[sourceKey] === "master" ? { selected: true } : {}) }, t("ui.option.calibration_source.master", "Master-Datei")),
    ),
  );

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
    sourceSelect,
    dirInput,
    masterInput,
  );

  if (!state[enabledKey]) row.classList.add("tc-cal-disabled");
  setInputEnabled();
  return row;
}
