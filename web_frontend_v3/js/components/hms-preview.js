import { el } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";

const FIELDS = {
  mode: ["ready_to_use", "scientific"], sensor_profile: "rec709", fallback_profile: "rec709",
  adaptive_anchor: true, target_bg: 0.15, protect_b: 6, convergence_power: 3.5,
  log_d_mode: ["auto", "fixed"], fixed_log_d: 2,
  color_strategy: ["auto", "fixed"], fixed_color_strategy: 0,
  color_grip: 1, shadow_convergence: 0, shadow_color_floor: 1, linear_expansion: 0,
};

const SENSOR_PROFILES = [
  ["rec709", "Rec.709"],
  ["Sony IMX571 (ASI2600/QHY268)", "Sony IMX571 (ASI2600/QHY268)"],
  ["Sony IMX455 (ASI6200/QHY600)", "Sony IMX455 (ASI6200/QHY600)"],
  ["Sony IMX410 (ASI2400)", "Sony IMX410 (ASI2400)"],
  ["Sony IMX269 (Altair/ToupTek)", "Sony IMX269 (Altair/ToupTek)"],
  ["Sony IMX294 (ASI294)", "Sony IMX294 (ASI294)"],
  ["Sony IMX533 (ASI533)", "Sony IMX533 (ASI533)"],
  ["Sony IMX676 (ASI676)", "Sony IMX676 (ASI676)"],
  ["Sony IMX585 (ASI585) - STARVIS 2", "Sony IMX585 (ASI585) – STARVIS 2"],
  ["Sony IMX662 (ASI662) - STARVIS 2", "Sony IMX662 (ASI662) – STARVIS 2"],
  ["Sony IMX678 (ASI678) - STARVIS 2", "Sony IMX678 (ASI678) – STARVIS 2"],
  ["Sony IMX415 (DWARF II)", "Sony IMX415 (DWARF II)"],
  ["Sony IMX462 (ASI462)", "Sony IMX462 (ASI462)"],
  ["Sony IMX715 (ASI715)", "Sony IMX715 (ASI715)"],
  ["Sony IMX482 (ASI482)", "Sony IMX482 (ASI482)"],
  ["Sony IMX183 (ASI183)", "Sony IMX183 (ASI183)"],
  ["Sony IMX178 (ASI178)", "Sony IMX178 (ASI178)"],
  ["Sony IMX224 (ASI224)", "Sony IMX224 (ASI224)"],
  ["Canon EOS (Modern - 60D/600D/500D)", "Canon EOS (Modern – 60D/600D/500D)"],
  ["Canon EOS (Legacy - 300D/40D/20D)", "Canon EOS (Legacy – 300D/40D/20D)"],
  ["Nikon DSLR (Modern - D5100/D7200)", "Nikon DSLR (Modern – D5100/D7200)"],
  ["Nikon DSLR (Legacy - D3/D300/D90)", "Nikon DSLR (Legacy – D3/D300/D90)"],
  ["Fujifilm X-Trans 5 HR", "Fujifilm X-Trans 5 HR"],
  ["Panasonic MN34230 (ASI1600)", "Panasonic MN34230 (ASI1600)"],
  ["ZWO Seestar S50", "ZWO Seestar S50"], ["ZWO Seestar S30", "ZWO Seestar S30"],
  ["Narrowband HOO", "Narrowband HOO"], ["Narrowband SHO", "Narrowband SHO"],
];

function scalar(raw, fallback) {
  const value = String(raw || "").trim().replace(/\s+#.*$/, "");
  if (typeof fallback === "boolean") return value === "true" ? true : value === "false" ? false : fallback;
  if (typeof fallback === "number") { const n = Number(value); return Number.isFinite(n) ? n : fallback; }
  return value.replace(/^['"]|['"]$/g, "") || fallback;
}

export function readHmsYaml(yaml) {
  const result = {};
  const lines = String(yaml || "").split(/\r?\n/);
  const start = lines.findIndex((line) => /^hypermetric_stretch:\s*(?:#.*)?$/.test(line));
  if (start < 0) return Object.fromEntries(Object.entries(FIELDS).map(([k, v]) => [k, Array.isArray(v) ? v[0] : v]));
  for (const [key, spec] of Object.entries(FIELDS)) {
    const fallback = Array.isArray(spec) ? spec[0] : spec;
    const match = lines.slice(start + 1).find((line) => new RegExp(`^\\s+${key}:`).test(line));
    result[key] = match ? scalar(match.replace(new RegExp(`^\\s+${key}:\\s*`), ""), fallback) : fallback;
  }
  return result;
}

function yamlValue(value) {
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

export function patchHmsYaml(yaml, params) {
  const lines = String(yaml || "").split(/\r?\n/);
  let start = lines.findIndex((line) => /^hypermetric_stretch:\s*(?:#.*)?$/.test(line));
  if (start < 0) { if (lines.length && lines.at(-1) !== "") lines.push(""); start = lines.length; lines.push("hypermetric_stretch:"); }
  let end = start + 1;
  while (end < lines.length && (lines[end].trim() === "" || /^\s+/.test(lines[end]))) end++;
  for (const [key, value] of Object.entries(params)) {
    const index = lines.findIndex((line, i) => i > start && i < end && new RegExp(`^\\s+${key}:`).test(line));
    if (index >= 0) lines[index] = lines[index].replace(new RegExp(`^(\\s+${key}:\\s*).*$`), (_, prefix) => `${prefix}${yamlValue(value)}`);
    else { lines.splice(end, 0, `  ${key}: ${yamlValue(value)}`); end++; }
  }
  return lines.join("\n");
}

function histogram(canvas, image) {
  const temp = document.createElement("canvas"); temp.width = image.width; temp.height = image.height;
  const ctx = temp.getContext("2d", { willReadFrequently: true }); ctx.drawImage(image, 0, 0);
  const data = ctx.getImageData(0, 0, temp.width, temp.height).data;
  const bins = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  for (let i = 0; i < data.length; i += 4) { bins[0][data[i]]++; bins[1][data[i + 1]]++; bins[2][data[i + 2]]++; }
  const out = canvas.getContext("2d"); out.clearRect(0, 0, canvas.width, canvas.height);
  const max = Math.max(1, ...bins.flatMap((b) => Array.from(b.slice(1, 255))));
  ["#ef4444", "#22c55e", "#3b82f6"].forEach((color, c) => {
    out.strokeStyle = color; out.globalAlpha = .8; out.beginPath();
    for (let x = 0; x < 256; x++) { const y = canvas.height - Math.log1p(bins[c][x]) / Math.log1p(max) * canvas.height; x ? out.lineTo(x, y) : out.moveTo(x, y); }
    out.stroke();
  }); out.globalAlpha = 1;
}

export function openHmsPreview({ runId, runDir, yaml, onApply }) {
  const initial = readHmsYaml(yaml); let generation = 0; let controller = null; let timer = null;
  let image = null, scale = 1, offsetX = 0, offsetY = 0, dragging = false, lastX = 0, lastY = 0;
  const controls = {};
  const canvas = el("canvas", { width: 900, height: 600, style: "width:100%;height:min(62vh,600px);background:#05070a;cursor:grab" });
  const hist = el("canvas", { width: 256, height: 90, style: "width:256px;height:90px;background:#0b1020;border:1px solid var(--tc-border-color,#334155)" });
  const status = el("div", { class: "tc-text-sm tc-text-muted", style: "min-height:1.5em" }, t("ui.message.hms_loading", "Vorschau wird geladen…"));
  const diagnostics = el("div", { class: "tc-text-sm tc-mono" }, "");

  function draw() {
    const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!image) return;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(image, offsetX, offsetY, image.width * scale, image.height * scale);
  }
  function fit() {
    if (!image) return; scale = Math.min(canvas.width / image.width, canvas.height / image.height);
    offsetX = (canvas.width - image.width * scale) / 2; offsetY = (canvas.height - image.height * scale) / 2; draw();
  }
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); if (!image) return; const old = scale; scale = Math.max(.1, Math.min(20, scale * (event.deltaY < 0 ? 1.15 : .87))); const rect = canvas.getBoundingClientRect(); const x = (event.clientX - rect.left) * canvas.width / rect.width; const y = (event.clientY - rect.top) * canvas.height / rect.height; offsetX = x - (x - offsetX) * scale / old; offsetY = y - (y - offsetY) * scale / old; draw(); });
  canvas.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => { if (!dragging) return; const rect = canvas.getBoundingClientRect(); offsetX += (e.clientX - lastX) * canvas.width / rect.width; offsetY += (e.clientY - lastY) * canvas.height / rect.height; lastX = e.clientX; lastY = e.clientY; draw(); });
  canvas.addEventListener("pointerup", () => { dragging = false; }); canvas.addEventListener("dblclick", fit);

  const params = () => Object.fromEntries(Object.entries(controls).map(([key, input]) => {
    if (input.dataset.numeric === "1") {
      const number = Number(input.value);
      const min = Number(input.min); const max = Number(input.max);
      if (!Number.isFinite(number) || number < min || number > max)
        throw new Error(t("ui.error.hms_parameter_range", "{parameter} must be between {min} and {max}")
          .replace("{parameter}", input.dataset.label || key).replace("{min}", input.min).replace("{max}", input.max));
      return [key, number];
    }
    return [key, input.type === "checkbox" ? input.checked : input.value];
  }));
  function updateDependencies() { controls.fixed_log_d.disabled = controls.log_d_mode.value !== "fixed"; controls.fixed_color_strategy.disabled = controls.color_strategy.value !== "fixed"; }
  async function preview() {
    const current = ++generation; controller?.abort(); controller = new AbortController(); status.textContent = t("ui.message.hms_loading", "Vorschau wird geladen…");
    try {
      const response = await fetch(api.httpUrl(API_ENDPOINTS.runs.hmePreview(runId)), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run_dir: runDir, params: params() }), signal: controller.signal });
      if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(error?.error?.message || `HTTP ${response.status}`); }
      const diag = JSON.parse(response.headers.get("X-HMS-Diagnostics") || "{}"); const blob = await response.blob(); const url = URL.createObjectURL(blob); const next = new Image();
      await new Promise((resolve, reject) => { next.onload = resolve; next.onerror = reject; next.src = url; }); URL.revokeObjectURL(url);
      if (current !== generation) return; image = next; fit(); histogram(hist, image);
      diagnostics.textContent = `${t("ui.label.hms_source", "Quelle")}: ${diag.source || "PCC"} · log D ${Number(diag.log_d || 0).toFixed(3)} · ${t("ui.label.hms_anchor", "Anker")} ${Number(diag.anchor || 0).toFixed(4)} · ${t("ui.label.hms_black_clip", "Schwarz")} ${Number(diag.black_clip_percent || 0).toFixed(3)}% · ${t("ui.label.hms_white_clip", "Weiß")} ${Number(diag.white_clip_percent || 0).toFixed(3)}%`;
      status.textContent = "";
    } catch (error) { if (error.name !== "AbortError" && current === generation) status.textContent = error.message; }
  }
  const schedule = () => {
    updateDependencies(); clearTimeout(timer);
    try { params(); status.textContent = ""; timer = setTimeout(preview, 150); }
    catch (error) { status.textContent = error.message; }
  };
  function field(key, label, spec = {}) {
    const value = initial[key]; let input;
    if (typeof value === "boolean") input = el("input", { type: "checkbox", checked: value, onchange: schedule });
    else if (key === "sensor_profile" || key === "fallback_profile") {
      const available = key === "sensor_profile" ? [["auto", t("ui.option.hms_auto", "Automatic")], ...SENSOR_PROFILES] : SENSOR_PROFILES;
      const profiles = available.some(([v]) => v === value) ? available : [[value, value], ...available];
      input = el("select", { class: "tc-select", onchange: schedule }, ...profiles.map(([v, text]) => el("option", { value: v, selected: value === v }, text)));
    } else if (Array.isArray(FIELDS[key])) input = el("select", { class: "tc-select", onchange: schedule }, ...FIELDS[key].map((v) => el("option", { value: v, selected: value === v }, t(`ui.option.hms_${v}`, v))));
    else input = el("input", { class: "tc-input", type: spec.numeric ? "number" : "text", value, min: spec.min, max: spec.max, step: spec.step,
      oninput: schedule,
      onchange: () => {
        if (spec.numeric) {
          const number = Number(input.value);
          input.value = Number.isFinite(number) ? Math.min(spec.max, Math.max(spec.min, number)) : initial[key];
        }
        schedule();
      },
      "data-numeric": spec.numeric ? "1" : "0", "data-label": label });
    const help = t(`param.hypermetric_stretch.${key}.short_help`, "");
    input.title = help;
    controls[key] = input;
    return el("label", { class: "tc-flex-col tc-gap-1", title: help },
      el("span", { class: "tc-label tc-flex tc-gap-1 tc-items-center" }, label, el("span", { title: help, "aria-label": help, style: "cursor:help;opacity:.75" }, "ⓘ")), input);
  }
  const form = el("div", { class: "tc-grid-2 tc-gap-2" },
    field("mode", t("param.hypermetric_stretch.mode.label", "Mode")), field("sensor_profile", t("param.hypermetric_stretch.sensor_profile.label", "Sensor profile")),
    field("fallback_profile", t("param.hypermetric_stretch.fallback_profile.label", "Fallback profile")), field("adaptive_anchor", t("param.hypermetric_stretch.adaptive_anchor.label", "Adaptive anchor")),
    field("target_bg", t("param.hypermetric_stretch.target_bg.label", "Target background"), { numeric:true,min:.05,max:.5,step:.01 }), field("protect_b", t("param.hypermetric_stretch.protect_b.label", "Protect B"), { numeric:true,min:.1,max:15,step:.1 }),
    field("convergence_power", t("param.hypermetric_stretch.convergence_power.label", "Convergence power"), { numeric:true,min:1,max:10,step:.1 }), field("log_d_mode", t("param.hypermetric_stretch.log_d_mode.label", "log D mode")),
    field("fixed_log_d", t("param.hypermetric_stretch.fixed_log_d.label", "Fixed log D"), { numeric:true,min:0,max:7,step:.05 }), field("color_strategy", t("param.hypermetric_stretch.color_strategy.label", "Color strategy")),
    field("fixed_color_strategy", t("param.hypermetric_stretch.fixed_color_strategy.label", "Fixed color strategy"), { numeric:true,min:-1,max:1,step:.01 }), field("color_grip", t("param.hypermetric_stretch.color_grip.label", "Color grip"), { numeric:true,min:0,max:1,step:.05 }),
    field("shadow_convergence", t("param.hypermetric_stretch.shadow_convergence.label", "Shadow convergence"), { numeric:true,min:0,max:3,step:.1 }), field("shadow_color_floor", t("param.hypermetric_stretch.shadow_color_floor.label", "Shadow color floor"), { numeric:true,min:0,max:1,step:.05 }), field("linear_expansion", t("param.hypermetric_stretch.linear_expansion.label", "Linear expansion"), { numeric:true,min:0,max:1,step:.01 })
  );
  const backdrop = el("div", { style: "position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:16px" });
  const modal = el("div", { class: "tc-card", role: "dialog", "aria-modal": "true", style: "width:min(1500px,96vw);max-height:95vh;overflow:auto" });
  const close = () => { generation++; controller?.abort(); clearTimeout(timer); backdrop.remove(); };
  const apply = async () => { const button = modal.querySelector("[data-apply]"); button.disabled = true; try { await onApply(patchHmsYaml(yaml, params())); close(); } catch (error) { status.textContent = error.message; button.disabled = false; } };
  modal.append(
    el("div", { class: "tc-card-title tc-flex tc-justify-between tc-items-center" }, el("span", {}, t("ui.title.hms_preview", "HyperMetric Stretch konfigurieren")), el("button", { class:"tc-btn tc-btn-sm", onclick:close }, "×")),
    el("div", { style:"display:grid;grid-template-columns:minmax(0,2fr) minmax(320px,1fr);gap:16px" }, el("div", {}, canvas, el("div", { class:"tc-mt-2 tc-flex tc-gap-3 tc-items-center tc-flex-wrap" }, hist, diagnostics)), form), status,
    el("div", { class:"tc-mt-3 tc-flex tc-gap-2 tc-justify-between" }, el("button", { class:"tc-btn", onclick:() => { for (const [k,input] of Object.entries(controls)) { if (input.type === "checkbox") input.checked = initial[k]; else input.value = initial[k]; } schedule(); } }, t("ui.button.hms_reset", "Reset")), el("div", { class:"tc-flex tc-gap-2" }, el("button", { class:"tc-btn", onclick:close }, t("ui.button.cancel", "Cancel")), el("button", { class:"tc-btn tc-btn-primary", "data-apply":"1", onclick:apply }, t("ui.button.hms_apply_resume", "Apply & start resume"))))
  );
  backdrop.appendChild(modal); document.body.appendChild(backdrop); updateDependencies(); preview();
}
