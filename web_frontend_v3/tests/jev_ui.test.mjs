import test from "node:test";
import assert from "node:assert/strict";

// Minimal DOM & browser globals shim for testing web_frontend_v3 ES modules in Node.
class MockNode {
  constructor() {
    this.childNodes = [];
    this.parentNode = null;
  }
  get firstChild() {
    return this.childNodes[0] || null;
  }
  append(...nodes) {
    for (const item of nodes) {
      if (item === null || item === undefined) continue;
      const child = item instanceof MockNode ? item : new MockText(String(item));
      if (child.parentNode) child.parentNode.removeChild(child);
      child.parentNode = this;
      this.childNodes.push(child);
    }
  }
  appendChild(child) {
    this.append(child);
    return child;
  }
  removeChild(child) {
    const idx = this.childNodes.indexOf(child);
    if (idx >= 0) {
      this.childNodes.splice(idx, 1);
      child.parentNode = null;
    }
    return child;
  }
  replaceChildren(...nodes) {
    this.childNodes = [];
    this.append(...nodes);
  }
  get textContent() {
    return this.childNodes.map((c) => c.textContent).join("");
  }
  set textContent(val) {
    this.childNodes = val === "" || val === null || val === undefined ? [] : [new MockText(String(val))];
  }
}

class MockText extends MockNode {
  constructor(text) {
    super();
    this._text = String(text);
  }
  get textContent() {
    return this._text;
  }
  set textContent(val) {
    this._text = String(val);
  }
}

class MockClassList {
  constructor(el) {
    this.el = el;
  }
  _set() {
    return new Set(String(this.el.className || "").split(/\s+/).filter(Boolean));
  }
  _write(s) {
    this.el.className = [...s].join(" ");
  }
  add(...tokens) {
    const s = this._set();
    for (const t of tokens) if (t) s.add(t);
    this._write(s);
  }
  remove(...tokens) {
    const s = this._set();
    for (const t of tokens) s.delete(t);
    this._write(s);
  }
  contains(token) {
    return this._set().has(token);
  }
  toggle(token, force) {
    const s = this._set();
    const shouldHave = force !== undefined ? Boolean(force) : !s.has(token);
    if (shouldHave) s.add(token);
    else s.delete(token);
    this._write(s);
    return shouldHave;
  }
}

class MockElement extends MockNode {
  constructor(tagName) {
    super();
    this.tagName = String(tagName).toUpperCase();
    this.className = "";
    this.style = {};
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.classList = new MockClassList(this);
  }
  get id() {
    return this.attributes.get("id") || "";
  }
  set id(val) {
    this.setAttribute("id", val);
  }
  setAttribute(name, value) {
    const str = String(value);
    this.attributes.set(name, str);
    if (name === "checked") this.checked = true;
    if (name === "disabled") this.disabled = true;
    if (name === "value" && !this.value) this.value = str;
    if (name.startsWith("data-")) {
      const camel = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      this.dataset[camel] = str;
    }
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  removeAttribute(name) {
    this.attributes.delete(name);
    if (name === "checked") this.checked = false;
    if (name === "disabled") this.disabled = false;
  }
  addEventListener(type, fn) {
    if (typeof fn !== "function") return;
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatchEvent(event) {
    const ev = event || { type: "" };
    ev.target = ev.target || this;
    ev.currentTarget = this;
    ev.preventDefault = ev.preventDefault || (() => {});
    for (const fn of this.listeners.get(ev.type) || []) {
      fn(ev);
    }
    if (typeof this[`on${ev.type}`] === "function") {
      this[`on${ev.type}`](ev);
    }
    return true;
  }
  click() {
    if (this.disabled) return;
    return this.dispatchEvent({ type: "click" });
  }
  matches(selector) {
    if (selector.startsWith("#")) return this.id === selector.slice(1);
    if (selector.startsWith(".")) return this.classList.contains(selector.slice(1));
    const attrMatch = /^\[([a-zA-Z0-9_-]+)(?:="([^"]*)")?\]$/.exec(selector);
    if (attrMatch) {
      const val = this.getAttribute(attrMatch[1]);
      return attrMatch[2] === undefined ? val !== null : val === attrMatch[2];
    }
    return this.tagName.toLowerCase() === selector.toLowerCase();
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
  querySelectorAll(selector) {
    const out = [];
    const visit = (node) => {
      for (const child of node.childNodes || []) {
        if (child instanceof MockElement) {
          if (child.matches(selector)) out.push(child);
          visit(child);
        }
      }
    };
    visit(this);
    return out;
  }
}

const storageMap = new Map();
globalThis.Node = MockNode;
globalThis.Element = MockElement;
globalThis.HTMLElement = MockElement;
globalThis.CustomEvent = class CustomEvent {
  constructor(type, init = {}) {
    this.type = type;
    this.detail = init.detail;
  }
};
globalThis.localStorage = {
  getItem: (k) => (storageMap.has(k) ? storageMap.get(k) : null),
  setItem: (k, v) => storageMap.set(k, String(v)),
  removeItem: (k) => storageMap.delete(k),
  clear: () => storageMap.clear(),
};
const docBody = new MockElement("body");
const docEl = new MockElement("html");
globalThis.document = {
  body: docBody,
  documentElement: docEl,
  createElement: (tag) => new MockElement(tag),
  createTextNode: (text) => new MockText(text),
  getElementById: (id) => docBody.querySelector(`#${id}`),
  querySelector: (sel) => docBody.querySelector(sel),
  querySelectorAll: (sel) => docBody.querySelectorAll(sel),
};
const windowListeners = new Map();
globalThis.window = {
  location: { protocol: "http:", host: "127.0.0.1:8080", hostname: "127.0.0.1", port: "8080" },
  confirm: () => true,
  addEventListener: (type, fn) => {
    if (!windowListeners.has(type)) windowListeners.set(type, []);
    windowListeners.get(type).push(fn);
  },
  dispatchEvent: (ev) => {
    for (const fn of windowListeners.get(ev.type) || []) fn(ev);
    return true;
  },
};

let fetchHandler = async () => {
  throw new Error("unhandled fetch");
};
globalThis.fetch = async (url, opts = {}) => {
  const result = await fetchHandler(url, opts);
  const status = result?.status ?? 200;
  const bodyObj = result?.body ?? result ?? {};
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => (typeof bodyObj === "string" ? bodyObj : JSON.stringify(bodyObj)),
  };
};

// Dynamic imports after globals are installed.
const { createJevSettingsCard, createJevEmpfehlungPage } = await import("../js/pages/jev-empfehlung.js");
const { createPostRunAdvicePanel } = await import("../js/components/post-run-advice.js");
const { getConfigState, setConfigState } = await import("../js/state/config-state.js");
const { setUiState } = await import("../js/state/ui-state.js");
const { setRunState } = await import("../js/state/run-state.js");
const { setFeatureFlags, getFeatureFlags } = await import("../js/state/feature-flags.js");

const flushPromises = () => new Promise((r) => setTimeout(r, 20));

test("createJevSettingsCard loads status, saves write-only API key, runs provider check, and toggles feature switch", async () => {
  setFeatureFlags({ jevEnabled: true });
  const calls = [];
  let state = {
    available: true,
    mode: "suggest",
    allow_experimental_suggestions: true,
    has_api_key: true,
    model: "openrouter/alpha-jev",
    endpoint_host: "openrouter.ai",
  };
  fetchHandler = async (url, opts) => {
    const body = opts.body ? JSON.parse(opts.body) : undefined;
    calls.push({ method: opts.method || "GET", url, body });
    if (url === "/api/pi/decisions/status") return { status: 200, body: state };
    if (url === "/api/pi/decisions/settings") {
      if (body.mode !== undefined) state.mode = body.mode;
      if (body.allow_experimental_suggestions !== undefined) {
        state.allow_experimental_suggestions = Boolean(body.allow_experimental_suggestions);
      }
      if (body.api_key !== undefined) state.has_api_key = body.api_key.length > 0;
      return { status: 200, body: state };
    }
    if (url === "/api/pi/decisions/test") {
      return { status: 200, body: { ok: true, latency_ms: 42, model_reported: "openrouter/alpha-jev" } };
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  const card = createJevSettingsCard();
  await flushPromises();

  const statusLine = card.querySelector("#jev-status");
  assert.match(statusLine.textContent, /vorhanden/);
  assert.match(statusLine.textContent, /openrouter\.ai/);

  // Write-only API key save clears the input afterwards
  const keyInput = card.querySelector("#jev-apikey");
  keyInput.value = "sk-or-test-key";
  const buttons = card.querySelectorAll("button");
  const saveKeyBtn = buttons.find((b) => b.textContent.includes("Key speichern"));
  assert.ok(saveKeyBtn);
  saveKeyBtn.click();
  await flushPromises();
  assert.equal(keyInput.value, "");
  assert.ok(calls.some((c) => c.url === "/api/pi/decisions/settings" && c.body?.api_key === "sk-or-test-key"));

  // Provider connection check
  const checkBtn = card.querySelector("#jev-check");
  checkBtn.click();
  await flushPromises();
  const checkResult = card.querySelector("#jev-check-result");
  assert.match(checkResult.textContent, /Provider antwortet/);
  assert.match(checkResult.textContent, /42 ms/);

  // Feature switch disables Jev UI
  const toggleInput = card.querySelector("#jev-enabled-switch");
  toggleInput.checked = false;
  toggleInput.dispatchEvent({ type: "change", target: toggleInput });
  assert.equal(getFeatureFlags().jevEnabled, false);
  setFeatureFlags({ jevEnabled: true });
});

test("createJevEmpfehlungPage requests advice, renders candidate comparison and exclusions, and applies batch to draft only", async () => {
  setUiState({ jevProposalIds: [], jevObjectClass: "compact", jevLockedPathsText: "global_metrics.adaptive_weights" });
  setConfigState({
    draft: { reconstruction: { drizzle: { pixfrac: 0.8 } } },
    draftYaml: "reconstruction:\n  drizzle:\n    pixfrac: 0.8\n",
    dirty: false,
    jevAppliedProposalId: "",
  });

  let proposalView = {
    state: "done",
    proposal_id: "prop_pixfrac_1",
    group: "reconstruction_pixfrac",
    mode: "suggest",
    model_called: true,
    proposal: {
      status: "validated",
      candidate_id: "set_pixfrac",
      reason_codes: ["validated"],
    },
    evidence: {
      experimental: true,
      rationale: { params: { usable_frames: 45, median_fwhm_px: 3.125 } },
    },
    comparison: [
      { path: "reconstruction.drizzle.pixfrac", current: 0.8, proposed: 1.0, changed: true },
    ],
    excluded: [
      { candidate_id: "enable_adaptive_weights", reasons: ["candidate_rejected"] },
    ],
  };

  const calls = [];
  fetchHandler = async (url, opts) => {
    const body = opts.body ? JSON.parse(opts.body) : undefined;
    calls.push({ method: opts.method || "GET", url, body });
    if (url.startsWith("/api/pi/decisions/log")) {
      return { status: 200, body: { enabled: true, count: 1, items: ['{"event":"request"}'] } };
    }
    if (url === "/api/scan/decisions") {
      return {
        status: 202,
        body: {
          proposal_id: "prop_pixfrac_1",
          proposal_ids: ["prop_pixfrac_1"],
          analysis_context: { analysis_uid: "analysis_123" },
        },
      };
    }
    if (url === "/api/scan/decisions/prop_pixfrac_1") {
      return { status: 200, body: proposalView };
    }
    if (url === "/api/scan/decisions/apply-batch") {
      proposalView = {
        ...proposalView,
        proposal: { ...proposalView.proposal, status: "applied_to_draft" },
      };
      return {
        status: 200,
        body: {
          patched_yaml: "reconstruction:\n  drizzle:\n    pixfrac: 1.0\n",
          applied_proposal_ids: ["prop_pixfrac_1"],
        },
      };
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  let draftAppliedCalled = false;
  const page = createJevEmpfehlungPage({ onDraftApplied: () => { draftAppliedCalled = true; } });
  await flushPromises();

  const requestBtn = page.querySelector("#jev-request");
  requestBtn.click();
  await flushPromises();

  // Verify request payload included object_class and locked_paths
  const adviceCall = calls.find((c) => c.url === "/api/scan/decisions");
  assert.ok(adviceCall);
  assert.deepEqual(adviceCall.body.locked_paths, ["global_metrics.adaptive_weights"]);
  assert.deepEqual(adviceCall.body.session_context, { object_class: { value: "compact", source: "user" } });

  // Verify rendered card, comparison table, measured evidence, and excluded candidates
  const resultBox = page.querySelector("#jev-result");
  assert.match(resultBox.textContent, /set_pixfrac/);
  assert.match(resultBox.textContent, /reconstruction\.drizzle\.pixfrac/);
  assert.match(resultBox.textContent, /usable_frames = 45/);
  assert.match(resultBox.textContent, /enable_adaptive_weights/);

  // Apply selected updates only the config draft (no config save or run start)
  const applySelectedBtn = page.querySelector("#jev-apply-selected");
  assert.ok(applySelectedBtn);
  assert.equal(applySelectedBtn.disabled, false);
  applySelectedBtn.click();
  await flushPromises();

  assert.equal(draftAppliedCalled, true);
  const cfg = getConfigState();
  assert.equal(cfg.dirty, true);
  assert.equal(cfg.jevAppliedProposalId, "prop_pixfrac_1");
  assert.equal(cfg.draft?.reconstruction?.drizzle?.pixfrac, 1.0);
  assert.ok(!calls.some((c) => c.url.includes("/api/config/save") || c.url.includes("/api/runs/start")));
});

test("createPostRunAdvicePanel renders metric evidence, handles dismissal, and combines resume plan on apply", async () => {
  setRunState({ currentRunId: "run_20261026", currentRunDir: "/tmp/runs/run_20261026" });
  const calls = [];
  fetchHandler = async (url, opts) => {
    const body = opts.body ? JSON.parse(opts.body) : undefined;
    calls.push({ method: opts.method || "GET", url, body });
    if (url === "/api/pi/post-run/advice") {
      return {
        status: 200,
        body: {
          run_uid: "ruid_1",
          event_id: "ev_1",
          model_selection: { status: "ok", mode: "suggest" },
          advice: {
            outcome: "suggest_downstream",
            model_recommendation: "enable_bge_rbf",
            resume_phase_order: ["STRETCH", "PCC", "BGE"],
            findings: [
              {
                code: "bge_background_slope_measured",
                severity: "warning",
                detail: "residual gradient",
                evidence: {
                  comparisons: [
                    {
                      channel: "G",
                      before: { stage: "bge.input", value: 0.0025 },
                      after: { stage: "bge.output", value: 0.0018 },
                      delta: -0.0007,
                    },
                  ],
                },
              },
              {
                code: "hms_curve_clipping_measured",
                severity: "info",
                evidence: {
                  stage: "hypermetric_stretch.curve_output",
                  black_clip_percent: 0.12,
                  white_clip_percent: 0.04,
                },
              },
              {
                code: "post_run_star_shape_measured",
                severity: "info",
                evidence: {
                  median_ratio: 1.01,
                  p90_ratio: 1.04,
                  matched_stars: 84,
                  ci95_star_bootstrap: [0.99, 1.02],
                },
              },
            ],
            suggestions: [
              {
                candidate_id: "enable_bge_rbf",
                experimental: true,
                resume_mode: "resume",
                min_resume_phase: "BGE",
                updates: [{ path: "bge.method", old_value: "classic", value: "rbf" }],
              },
              {
                candidate_id: "hms_soften_stretch",
                experimental: true,
                resume_mode: "resume",
                min_resume_phase: "STRETCH",
                updates: [{ path: "stretch.target_bg", old_value: 0.2, value: 0.15 }],
              },
            ],
            not_measured: ["final_stretched_image_quality"],
          },
          dismissed_candidates: [
            { result: { candidate_id: "old_dismissed_cand", source_patch_id: "patch_0" } },
          ],
        },
      };
    }
    if (url === "/api/pi/post-run/dismissal") {
      return { status: 200, body: { ok: true } };
    }
    if (url.startsWith("/api/runs/") && url.includes("/config")) {
      return {
        status: 200,
        body: { config_yaml: "bge:\n  method: classic\nstretch:\n  target_bg: 0.2\n" },
      };
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  let appliedPlan = null;
  const panel = createPostRunAdvicePanel("post-run-advice", {
    onApply: async (plan) => {
      appliedPlan = plan;
    },
  });

  const reqBtn = panel.querySelector("#post-run-advice-button");
  reqBtn.click();
  await flushPromises();

  const content = panel.querySelector("#post-run-advice-content");
  // Check rendered evidence for BGE slope, HMS clipping, and paired star shape
  assert.ok(content.querySelector('[data-metric-evidence="bge_background_slope_measured"]'));
  assert.ok(content.querySelector('[data-metric-evidence="hms_curve_clipping_measured"]'));
  assert.ok(content.querySelector('[data-metric-evidence="post_run_star_shape_measured"]'));
  assert.match(content.textContent, /84 Sterne/);
  assert.match(content.textContent, /final_stretched_image_quality|Qualität des gestreckten Endbilds/);

  // Apply selected combines resume phases: between STRETCH (index 0) and BGE (index 2), earliest required phase is BGE
  const applyBtn = panel.querySelector("#post-run-advice-apply-selected");
  assert.ok(applyBtn);
  applyBtn.click();
  await flushPromises();

  assert.ok(appliedPlan);
  assert.equal(appliedPlan.resumeMode, "resume");
  assert.equal(appliedPlan.minResumePhase, "BGE");
  assert.deepEqual(appliedPlan.candidateIds, ["enable_bge_rbf", "hms_soften_stretch"]);
  assert.match(appliedPlan.patchedYaml, /method: rbf/);
  assert.match(appliedPlan.patchedYaml, /target_bg: 0\.15/);

  // Dismiss candidate triggers POST /api/pi/decisions/post-run-dismissal
  const dismissBtn = content.querySelectorAll("button").find((b) => b.textContent.includes("Nicht mehr vorschlagen"));
  assert.ok(dismissBtn);
  dismissBtn.click();
  await flushPromises();
  const dismissCall = calls.find((c) => c.url === "/api/pi/post-run/dismissal");
  assert.ok(dismissCall);
  assert.equal(dismissCall.body.candidate_id, "enable_bge_rbf");
  assert.equal(dismissCall.body.dismissed, true);
});
