// js/components/situation-assistant.js – Situation-Assistant für Parameter-Empfehlungen

import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";

const SCENARIO_DELTAS = {
  altaz: [
    ["registration.allow_rotation", true],
    ["registration.transform_model", "affine"],
    ["registration.star_topk", 180],
    ["registration.reject_shift_px_min", 120],
    ["registration.reject_shift_median_multiplier", 5.0],
  ],
  rotation: [
    ["registration.engine", "robust_phase_ecc"],
    ["registration.allow_rotation", true],
    ["registration.transform_model", "affine"],
    ["registration.star_inlier_tol_px", 4.0],
    ["registration.reject_cc_min_abs", 0.25],
  ],
  bright_stars: [
    ["pcc.mag_bright_limit", 6.0],
    ["pcc.k_max", 2.4],
    ["pcc.sigma_clip", 2.7],
    ["bge.mask.star_dilate_px", 6],
  ],
  few_frames: [
    ["assumptions.frames_reduced_threshold", 200],
    ["assumptions.reduced_mode_skip_clustering", true],
    ["synthetic.frames_min", 4],
    ["synthetic.clustering.cluster_count_range", [3, 10]],
  ],
  gradient: [
    ["bge.enabled", true],
    ["bge.fit.method", "rbf"],
    ["bge.fit.rbf_lambda", "1e-2"],
    ["bge.sample_estimator", "quantile"],
    ["bge.sample_quantile", 0.15],
    ["bge.structure_thresh_percentile", 0.8],
  ],
  dithered: [
    ["registration.reject_shift_px_min", 50],
    ["registration.star_inlier_tol_px", 5.0],
  ],
  low_snr: [
    ["registration.reject_cc_min_abs", 0.15],
    ["registration.star_topk", 250],
    ["aqmh.pyramid.w_snr", 1.5],
  ],
  wide_field: [
    ["registration.transform_model", "affine"],
    ["registration.star_topk", 300],
    ["bge.fit.method", "rbf"],
  ],
};

const SCENARIOS = [
  { id: "altaz", label: "Alt/Az", hint: "Starke Feldrotation" },
  { id: "rotation", label: "Rotation", hint: "Frame-Rotation vorhanden" },
  { id: "bright_stars", label: "Bright Stars", hint: "Helle Sterne im Feld" },
  { id: "few_frames", label: "Few Frames", hint: "< 50 Frames" },
  { id: "gradient", label: "Gradient", hint: "Starker Gradient" },
  { id: "dithered", label: "Dithered", hint: "Frames wurden gedithert" },
  { id: "low_snr", label: "Low SNR", hint: "Niedriges Signal-Rausch-Verh\u00e4ltnis" },
  { id: "wide_field", label: "Wide Field", hint: "Weitwinkel-Feld" },
];

export function getScenarioDeltas(scenarioIds) {
  const merged = new Map();
  for (const id of scenarioIds) {
    for (const [path, value] of SCENARIO_DELTAS[id] || []) {
      if (!merged.has(path)) merged.set(path, { values: [], sources: [] });
      const entry = merged.get(path);
      entry.values.push(value);
      entry.sources.push(id);
    }
  }
  return merged;
}

export function createSituationAssistant({ selected = [], onApply, onChange } = {}) {
  const state = new Set(selected);

  const wrapper = el("div", { class: "tc-card", style: { background: "var(--surface-2)" } },
    el("div", { class: "tc-card-title" }, t("ui.title.situation", "Situation")),
    el("div", { class: "tc-flex-col tc-gap-1", id: "scenario-list" },
      ...SCENARIOS.map(s => {
        const cb = el("input", { type: "checkbox", checked: state.has(s.id) });
        cb.onchange = () => {
          if (cb.checked) state.add(s.id);
          else state.delete(s.id);
          renderPreview();
          onChange?.([...state]);
        };
        return el("label", { class: "tc-checkbox", title: s.hint }, cb, s.label);
      }),
    ),
    el("div", { class: "tc-mt-2 tc-flex-col tc-gap-1", id: "situation-preview" },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.select_situation", "Situation auswählen um Änderungen zu sehen")),
    ),
    el("button", {
      class: "tc-btn tc-btn-sm tc-mt-2",
      id: "situation-apply-btn",
      onclick: () => onApply?.([...state]),
    }, t("ui.button.apply", "Apply")),
  );

  function renderPreview() {
    const previewEl = wrapper.querySelector("#situation-preview");
    if (!previewEl) return;
    clear(previewEl);
    if (state.size === 0) {
      previewEl.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.select_situation", "Situation auswählen um Änderungen zu sehen")));
      return;
    }
    const deltas = getScenarioDeltas([...state]);
    if (deltas.size === 0) {
      previewEl.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_changes", "Keine Änderungen")));
      return;
    }
    for (const [path, info] of deltas) {
      const values = info.values;
      const valueText = values.length > 1
        ? `${values.map(formatPreviewVal).join(" | ")} (Conflict)`
        : formatPreviewVal(values[0]);
      previewEl.appendChild(el("div", { class: "tc-text-sm tc-mono" },
        el("span", { class: "tc-diff-added" }, `${path} = ${valueText}`),
      ));
    }
  }

  renderPreview();
  return wrapper;
}

function formatPreviewVal(v) {
  if (v === true) return "true";
  if (v === false) return "false";
  if (v === null) return "null";
  if (Array.isArray(v)) return `[${v.map(formatPreviewVal).join(", ")}]`;
  return String(v);
}

