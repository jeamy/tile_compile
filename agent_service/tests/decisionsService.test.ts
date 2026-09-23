import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import {
  DecisionsRequestError,
  DecisionsService,
  TransportError,
  buildProviderBody,
  decisionsConfigFromEnv,
  isPinnedModel,
  validateProviderResponse,
  validateRequest,
  type DecisionsRequest,
  type TransportRequest,
  type TransportResponse,
} from "../src/services/decisionsService.js";
import { redactTrafficLogText } from "../src/services/trafficLog.js";
import { decisionsSettingsPath, loadDecisionsSettings, saveDecisionsSettings } from "../src/services/decisionsSettings.js";
import os from "node:os";

const FIXTURES = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../web_backend_cpp/config/pi_decisions/fixtures");
const fixture = (name: string) => JSON.parse(fs.readFileSync(path.join(FIXTURES, name), "utf8"));
const KEY = "sk-or-v1-TESTKEY-not-real-0000";

const goodRequest = (over: Partial<DecisionsRequest> = {}): DecisionsRequest => ({
  request_id: "req-1",
  state_hash: "sha256:abc",
  question_set_version: "decision-questions.v1",
  state_projection: { domain: "pre_run", coverage: { detected: 10, measured: 10, read_ok: 10 } },
  allowed_candidates: ["keep_current", "insufficient_evidence", "enable_adaptive_weights"],
  ...over,
});

const ok200 = (body: unknown): TransportResponse => ({ status: 200, headers: {}, bodyText: JSON.stringify(body), truncated: false });
const status = (s: number, headers: Record<string, string> = {}, body: unknown = { error: { message: "x", code: s } }): TransportResponse =>
  ({ status: s, headers, bodyText: JSON.stringify(body), truncated: false });

type Step = TransportResponse | Error | ((req: TransportRequest) => Promise<TransportResponse>);
function fakeTransport(steps: Step[]) {
  const calls: TransportRequest[] = [];
  const fn = async (req: TransportRequest): Promise<TransportResponse> => {
    calls.push(req);
    const step = steps[Math.min(calls.length - 1, steps.length - 1)];
    if (step instanceof Error) throw step;
    if (typeof step === "function") return step(req);
    return step;
  };
  return { fn, calls };
}
const noSleep = () => { const waits: number[] = []; return { waits, fn: async (ms: number) => { waits.push(ms); } }; };

function service(steps: Step[], cfg: Record<string, unknown> = {}, extra: { key?: string | undefined; logs?: string[] } = {}) {
  const t = fakeTransport(steps);
  const s = noSleep();
  const logs = extra.logs ?? [];
  const svc = new DecisionsService({ mode: "suggest", ...cfg } as any, {
    transport: t.fn, sleep: s.fn,
    resolveApiKey: () => ("key" in extra ? extra.key : KEY),
    log: (l) => logs.push(l),
  });
  return { svc, t, sleeps: s.waits, logs };
}

describe("configuration", () => {
  it("refuses model aliases and non-https endpoints", () => {
    assert.throws(() => new DecisionsService({ model: "typesafe/jev-latest" }), /pinned/);
    assert.throws(() => new DecisionsService({ model: "typesafe/jev-preview" }), /pinned/);
    assert.throws(() => new DecisionsService({ model: "jev" }), /pinned/);
    assert.throws(() => new DecisionsService({ endpoint: "http://example.com/x" }), /https/);
    assert.throws(() => new DecisionsService({ mode: "auto" as any }), /PI_DECISIONS_MODE/);
    assert.ok(new DecisionsService({ endpoint: "http://127.0.0.1:9/x" }));
    assert.ok(isPinnedModel("typesafe/jev-1.13") && !isPinnedModel("typesafe/jev-latest"));
  });
  it("defaults to mode off and reads env", () => {
    assert.equal(decisionsConfigFromEnv({}).mode, "off");
    const c = decisionsConfigFromEnv({ PI_DECISIONS_MODE: "SHADOW", PI_DECISIONS_ALLOW_EXPERIMENTAL: "1", PI_DECISIONS_DEADLINE_MS: "5000" });
    assert.equal(c.mode, "shadow");
    assert.equal(c.allowExperimentalSuggestions, true);
    assert.equal(c.deadlineMs, 5000);
    assert.equal(decisionsConfigFromEnv({ PI_DECISIONS_ALLOW_EXPERIMENTAL: "" }).allowExperimentalSuggestions, false);
  });
});

describe("mode and key gating: zero requests", () => {
  it("mode off sends nothing", async () => {
    const { svc, t } = service([ok200(fixture("response_ok.json"))], { mode: "off" });
    const r = await svc.decide(goodRequest());
    assert.deepEqual([r.status, r.error_code], ["unavailable", "mode_off"]);
    assert.equal(t.calls.length, 0);
  });
  it("missing key sends nothing", async () => {
    const { svc, t } = service([ok200(fixture("response_ok.json"))], {}, { key: undefined });
    const r = await svc.decide(goodRequest());
    assert.equal(r.error_code, "no_api_key");
    assert.equal(t.calls.length, 0);
  });
  it("status reports mode/model/host and key presence, never the key", async () => {
    const { svc } = service([]);
    const st = await svc.status();
    assert.deepEqual(st, { mode: "suggest", model: "typesafe/jev-1.13", endpoint_host: "openrouter.ai", has_api_key: true, allow_experimental_suggestions: false });
    assert.ok(!JSON.stringify(st).includes(KEY));
  });
});

describe("request handling", () => {
  it("builds the verified wire body from the application request", async () => {
    const { svc, t } = service([ok200(fixture("response_ok.json"))]);
    await svc.decide(goodRequest());
    const call = t.calls[0];
    assert.equal(call.url, "https://openrouter.ai/api/alpha/decisions");
    assert.equal(call.headers.Authorization, `Bearer ${KEY}`);
    const body = JSON.parse(call.body);
    const ref = fixture("request_ok.json");
    assert.deepEqual(Object.keys(body).sort(), Object.keys(ref).sort());
    assert.equal(body.model, "typesafe/jev-1.13");
    assert.deepEqual(body.state, goodRequest().state_projection);
    const q = body.questions.candidate_selection;
    assert.equal(q.type, "choice");
    assert.deepEqual(Object.keys(q.criteria).sort(), ["enable_adaptive_weights", "insufficient_evidence", "keep_current"]);
    assert.deepEqual(Object.keys(q).sort(), Object.keys(ref.questions.candidate_selection).sort());
  });
  it("rejects malformed requests locally, before any network", async () => {
    const { svc, t } = service([ok200(fixture("response_ok.json"))]);
    const cases: Array<[string, unknown]> = [
      ["body", "x"], ["request_id", goodRequest({ request_id: "" })],
      ["question_set", goodRequest({ question_set_version: "nope" })],
      ["one candidate", goodRequest({ allowed_candidates: ["keep_current"] })],
      ["missing baseline", goodRequest({ allowed_candidates: ["keep_current", "enable_adaptive_weights"] })],
      ["duplicate", goodRequest({ allowed_candidates: ["keep_current", "keep_current", "insufficient_evidence"] })],
      ["bad id", goodRequest({ allowed_candidates: ["keep_current", "insufficient_evidence", "Bad Id"] })],
      ["path leak", goodRequest({ state_projection: { x: "/home/lux/runs" } })],
      ["key leak", goodRequest({ state_projection: { x: "sk-or-v1-abc" } })],
      ["url leak", goodRequest({ state_projection: { x: "https://evil.example" } })],
      ["key name", goodRequest({ state_projection: { api_key: "x" } })],
      ["too large", goodRequest({ state_projection: { blob: "a".repeat(70_000) } })],
    ];
    for (const [name, req] of cases) await assert.rejects(() => svc.decide(req), DecisionsRequestError, name);
    assert.equal(t.calls.length, 0);
    assert.throws(() => validateRequest(null), DecisionsRequestError);
  });
});

describe("provider response validation (fixtures + hostile variants)", () => {
  it("accepts the captured-shape ok fixture and normalizes it", async () => {
    const { svc } = service([ok200(fixture("response_ok.json"))]);
    const r = await svc.decide(goodRequest());
    assert.equal(r.status, "ok");
    assert.equal(r.selection?.candidate_id, "keep_current");
    assert.equal(r.selection?.probabilities.keep_current, 0.61);
    assert.equal(r.selection?.provider_confidence, 0.61);
    assert.equal(r.model_reported, "typesafe/jev-1.13-20260917");
    assert.equal(r.model_requested, "typesafe/jev-1.13");
    assert.equal(r.usage?.cost, 0.0000147);
    assert.match(r.generation_id ?? "", /^gen-dec-/);
    assert.equal(r.request_id, "req-1");
    assert.equal(r.state_hash, "sha256:abc");
  });
  it("rejects an unknown candidate, never mapping it", async () => {
    const { svc } = service([ok200(fixture("response_invalid_unknown_candidate.json"))]);
    const r = await svc.decide(goodRequest());
    assert.deepEqual([r.status, r.error_code, r.selection], ["invalid_response", "unknown_candidate", null]);
  });
  it("rejects a non-numeric probability wholesale", async () => {
    const { svc } = service([ok200(fixture("response_invalid_nan_probability.json"))]);
    const r = await svc.decide(goodRequest());
    assert.deepEqual([r.status, r.error_code], ["invalid_response", "non_finite_probability"]);
  });
  const mutate = (fn: (b: any) => void) => { const b = fixture("response_ok.json"); fn(b); return b; };
  const hostile: Array<[string, (b: any) => void, string]> = [
    ["probability > 1", (b) => { b.answers.candidate_selection.probabilities.keep_current = 1.2; }, "non_finite_probability"],
    ["negative probability", (b) => { b.answers.candidate_selection.probabilities.keep_current = -0.1; }, "non_finite_probability"],
    ["probability for an unrequested id", (b) => { b.answers.candidate_selection.probabilities.disable_bge = 0.1; }, "unknown_candidate"],
    ["confidence out of range", (b) => { b.answers.candidate_selection.confidence = 3; }, "non_finite_confidence"],
    ["extra top-level field", (b) => { b.debug = true; }, "unexpected_field"],
    ["extra answer field", (b) => { b.answers.candidate_selection.config_updates = {}; }, "unexpected_field"],
    ["extra answer", (b) => { b.answers.other = b.answers.candidate_selection; }, "unexpected_answer"],
    ["wrong answer type", (b) => { b.answers.candidate_selection = { type: "noul", noul: 0.5, confidence: 0.5 }; }, "unexpected_field"],
    ["answer missing", (b) => { b.answers = {}; }, "answer_missing"],
    ["answers not object", (b) => { b.answers = []; }, "answers_missing"],
    ["model version drift", (b) => { b.model = "typesafe/jev-2.0-20261001"; }, "model_version_mismatch"],
    ["model missing", (b) => { delete b.model; }, "model_missing"],
    ["usage missing", (b) => { delete b.usage; }, "usage_missing"],
    ["negative cost", (b) => { b.usage.cost = -1; }, "usage_invalid"],
    ["bad generation id", (b) => { b.id = "chatcmpl-1"; }, "generation_id_invalid"],
  ];
  for (const [name, fn, code] of hostile) {
    it(`rejects: ${name}`, async () => {
      const { svc } = service([ok200(mutate(fn))]);
      const r = await svc.decide(goodRequest());
      assert.deepEqual([r.status, r.error_code], ["invalid_response", code], name);
      assert.equal(r.selection, null);
    });
  }
  it("rejects non-JSON, non-object and oversized bodies", async () => {
    let r = await service([{ status: 200, headers: {}, bodyText: "<html>", truncated: false }]).svc.decide(goodRequest());
    assert.deepEqual([r.status, r.error_code], ["invalid_response", "invalid_json"]);
    r = await service([ok200([1, 2])]).svc.decide(goodRequest());
    assert.equal(r.error_code, "response_not_object");
    r = await service([{ status: 200, headers: {}, bodyText: "{", truncated: true }]).svc.decide(goodRequest());
    assert.deepEqual([r.status, r.error_code], ["invalid_response", "response_too_large"]);
  });
  it("validateProviderResponse is usable directly", () => {
    const v = validateProviderResponse(fixture("response_ok.json"), validateRequest(goodRequest()), "typesafe/jev-1.13");
    assert.ok(v.ok);
    assert.equal(buildProviderBody(validateRequest(goodRequest()), "typesafe/jev-1.13").includes(KEY), false);
  });
});

describe("application contract pi.decisions.response.v1", () => {
  const schema = JSON.parse(fs.readFileSync(path.resolve(FIXTURES, "../schemas/pi.decisions.response.v1.schema.json"), "utf8"));
  const conforms = (r: Record<string, any>) => {
    for (const k of schema.required) assert.ok(k in r, `required ${k}`);
    for (const k of Object.keys(r)) assert.ok(k in schema.properties, `unexpected property ${k}`);
    assert.ok(schema.properties.status.enum.includes(r.status));
    if (r.status === "ok") assert.ok(r.selection && r.selection.candidate_id, "ok needs selection");
    else assert.ok(typeof r.error_code === "string", "non-ok needs error_code");
    if (r.selection) for (const k of Object.keys(r.selection)) assert.ok(k in schema.properties.selection.properties, `selection.${k}`);
    if (r.usage) for (const k of Object.keys(r.usage)) assert.ok(k in schema.properties.usage.properties, `usage.${k}`);
  };
  it("every outcome the service can produce conforms", async () => {
    const outcomes: Array<[Step[], Record<string, unknown>?, { key?: string | undefined }?]> = [
      [[ok200(fixture("response_ok.json"))]], [[ok200(fixture("response_invalid_unknown_candidate.json"))]],
      [[status(401)]], [[status(400)]], [[status(429)]], [[status(503)]], [[new TransportError("network")]],
      [[ok200(fixture("response_ok.json"))], { mode: "off" }], [[ok200(fixture("response_ok.json"))], {}, { key: undefined }],
    ];
    for (const [steps, cfg, extra] of outcomes) conforms(await service(steps, cfg, extra).svc.decide(goodRequest()) as any);
  });
});

describe("HTTP errors and retries", () => {
  it("401/403 -> unavailable auth_failed, no retry", async () => {
    for (const s of [401, 403]) {
      const { svc, t } = service([status(s)]);
      const r = await svc.decide(goodRequest());
      assert.deepEqual([r.status, r.error_code, t.calls.length], ["unavailable", "auth_failed", 1]);
    }
  });
  it("400/422 -> invalid_response provider_rejected_request, no retry (our bug, not an outage)", async () => {
    for (const fx of ["error_400_validation.json", "error_400_invalid_model.json", "error_400_wrong_endpoint.json"]) {
      const f = fixture(fx);
      const { svc, t } = service([{ status: f.http_status, headers: {}, bodyText: JSON.stringify(f.body), truncated: false }]);
      const r = await svc.decide(goodRequest());
      assert.deepEqual([r.status, r.error_code, t.calls.length], ["invalid_response", "provider_rejected_request", 1], fx);
    }
    const { svc } = service([status(422)]);
    assert.equal((await svc.decide(goodRequest())).error_code, "provider_rejected_request");
  });
  it("429 with Retry-After: one retry after that wait, then success", async () => {
    const { svc, t, sleeps } = service([status(429, { "retry-after": "2" }), ok200(fixture("response_ok.json"))]);
    const r = await svc.decide(goodRequest());
    assert.equal(r.status, "ok");
    assert.deepEqual([t.calls.length, sleeps], [2, [2000]]);
  });
  it("429 twice -> unavailable rate_limited after exactly one retry", async () => {
    const { svc, t } = service([status(429, { "retry-after": "1" })]);
    const r = await svc.decide(goodRequest());
    assert.deepEqual([r.status, r.error_code, t.calls.length], ["unavailable", "rate_limited", 2]);
  });
  it("Retry-After beyond the remaining budget is not waited for", async () => {
    const { svc, t, sleeps } = service([status(429, { "retry-after": "120" })], { deadlineMs: 5000 });
    const r = await svc.decide(goodRequest());
    assert.deepEqual([r.error_code, t.calls.length, sleeps], ["rate_limited", 1, []]);
  });
  it("5xx retries once with the configured backoff, then gives up", async () => {
    for (const s of [500, 502, 503, 529]) {
      const { svc, t, sleeps } = service([status(s)], { retryBackoffMs: 250 });
      const r = await svc.decide(goodRequest());
      assert.deepEqual([r.status, r.error_code, t.calls.length, sleeps], ["unavailable", "provider_error", 2, [250]], String(s));
    }
    const { svc } = service([status(503), ok200(fixture("response_ok.json"))]);
    assert.equal((await svc.decide(goodRequest())).status, "ok");
  });
  it("other statuses are not retried", async () => {
    const { svc, t } = service([status(404)]);
    const r = await svc.decide(goodRequest());
    assert.deepEqual([r.error_code, t.calls.length], ["http_404", 1]);
  });
  it("network errors retry once, then unavailable network_error", async () => {
    const { svc, t } = service([new TransportError("network", "ECONNRESET")]);
    const r = await svc.decide(goodRequest());
    assert.deepEqual([r.status, r.error_code, t.calls.length], ["unavailable", "network_error", 2]);
  });
  it("maxRetries 0 never retries", async () => {
    const { svc, t } = service([status(503)], { maxRetries: 0 });
    assert.equal((await svc.decide(goodRequest())).error_code, "provider_error");
    assert.equal(t.calls.length, 1);
  });
});

describe("deadline, abort, concurrency", () => {
  const hang = (req: TransportRequest) =>
    new Promise<TransportResponse>((_, reject) => req.signal.addEventListener("abort", () => reject(new TransportError("aborted")), { once: true }));
  it("a hanging provider ends at the deadline with unavailable timeout", async () => {
    const { svc } = service([hang], { deadlineMs: 40 });
    const r = await svc.decide(goodRequest());
    assert.deepEqual([r.status, r.error_code], ["unavailable", "timeout"]);
  });
  it("the caller can abort", async () => {
    const { svc } = service([hang], { deadlineMs: 10_000 });
    const ac = new AbortController();
    const p = svc.decide(goodRequest(), ac.signal);
    setTimeout(() => ac.abort(), 20);
    const r = await p;
    assert.deepEqual([r.status, r.error_code], ["unavailable", "aborted"]);
  });
  it("identical concurrent requests share one provider call, each keeping its own request_id", async () => {
    let release!: () => void;
    const gate = new Promise<void>((res) => { release = res; });
    const { svc, t } = service([async () => { await gate; return ok200(fixture("response_ok.json")); }]);
    const a = svc.decide(goodRequest({ request_id: "A" }));
    const b = svc.decide(goodRequest({ request_id: "B" }));
    await new Promise((r) => setTimeout(r, 10));
    release();
    const [ra, rb] = await Promise.all([a, b]);
    assert.equal(t.calls.length, 1);
    assert.deepEqual([ra.request_id, rb.request_id, ra.status, rb.status], ["A", "B", "ok", "ok"]);
    await svc.decide(goodRequest({ request_id: "C" }));
    assert.equal(t.calls.length, 2, "no result cache: a later identical request is a fresh call");
  });
  it("different states are not deduplicated; concurrency and queue are bounded", async () => {
    let release!: () => void;
    const gate = new Promise<void>((res) => { release = res; });
    const { svc, t } = service([async () => { await gate; return ok200(fixture("response_ok.json")); }], { maxConcurrent: 1, maxQueued: 1 });
    const p1 = svc.decide(goodRequest({ request_id: "1", state_hash: "h1" }));
    const p2 = svc.decide(goodRequest({ request_id: "2", state_hash: "h2" }));  // queued
    const p3 = svc.decide(goodRequest({ request_id: "3", state_hash: "h3" }));  // queue full -> busy
    const r3 = await p3;
    assert.deepEqual([r3.status, r3.error_code], ["unavailable", "busy"]);
    await new Promise((r) => setTimeout(r, 10));
    assert.equal(t.calls.length, 1, "only one call in flight");
    release();
    assert.deepEqual([(await p1).status, (await p2).status], ["ok", "ok"]);
    assert.equal(t.calls.length, 2);
  });
});

describe("secrets and logging", () => {
  it("results, logs and status never contain the key or the state", async () => {
    const logs: string[] = [];
    const { svc } = service([ok200(fixture("response_ok.json"))], {}, { logs });
    const r = await svc.decide(goodRequest({ state_projection: { marker: "STATE-MARKER-42" } }));
    const blob = JSON.stringify([r, logs, await svc.status()]);
    assert.ok(!blob.includes(KEY) && !blob.includes("STATE-MARKER-42"), "no key / state in any output");
    assert.equal(logs.length, 1);
    assert.match(logs[0], /^decisions request_id=req-1 state_hash=sha256:abc candidates=3 status=ok code=- ms=\d+ cost=0\.0000147$/);
  });
  it("failures log a summary too, still without secrets", async () => {
    const logs: string[] = [];
    const { svc } = service([status(401)], {}, { logs });
    await svc.decide(goodRequest());
    assert.match(logs[0], /status=unavailable code=auth_failed/);
    assert.ok(!logs[0].includes(KEY));
  });
  it("the traffic-log redactor now catches JEV_ keys and sk-or tokens", () => {
    const out = redactTrafficLogText(`env JEV_OPENROUTER_API_KEY=${KEY} and header Authorization: Bearer ${KEY} and bare ${KEY}`);
    assert.ok(!out.includes(KEY), out);
    assert.ok(out.includes("JEV_OPENROUTER_API_KEY=<redacted>"));
  });
});

describe("runtime settings (Jev card)", () => {
  it("applySettings switches mode at runtime and validates before mutating", async () => {
    const t = fakeTransport([ok200(fixture("response_ok.json"))]);
    const svc = new DecisionsService({ mode: "off" }, { transport: t.fn, sleep: async () => {}, resolveApiKey: () => KEY });
    assert.equal((await svc.decide(goodRequest())).error_code, "mode_off");
    svc.applySettings({ mode: "shadow" });
    assert.equal((await svc.decide(goodRequest())).status, "ok");
    assert.throws(() => svc.applySettings({ mode: "auto" }), /mode must be/);
    assert.throws(() => svc.applySettings({ apiKey: "has space" }), /single token/);
    assert.equal((await svc.status()).mode, "shadow", "a rejected update leaves the settings intact");
    svc.applySettings({ allowExperimentalSuggestions: true });
    assert.equal((await svc.status()).allow_experimental_suggestions, true);
  });
  it("a stored key wins over the environment and is never exposed", async () => {
    const seen: string[] = [];
    const svc = new DecisionsService({ mode: "suggest" }, {
      transport: async (r) => { seen.push(r.headers.Authorization); return ok200(fixture("response_ok.json")); },
    });
    const prev = process.env.JEV_OPENROUTER_API_KEY;
    process.env.JEV_OPENROUTER_API_KEY = "sk-or-v1-ENVKEY-0000";
    try {
      await svc.decide(goodRequest());
      svc.applySettings({ apiKey: "sk-or-v1-STOREDKEY-1111" });
      await svc.decide(goodRequest({ state_hash: "h2" }));
      svc.applySettings({ apiKey: null });
      await svc.decide(goodRequest({ state_hash: "h3" }));
      assert.deepEqual(seen, ["Bearer sk-or-v1-ENVKEY-0000", "Bearer sk-or-v1-STOREDKEY-1111", "Bearer sk-or-v1-ENVKEY-0000"]);
      svc.applySettings({ apiKey: "sk-or-v1-STOREDKEY-1111" });
      const st = JSON.stringify(await svc.status());
      assert.ok(!st.includes("STOREDKEY") && !st.includes("ENVKEY"), "status never contains a key");
      assert.equal((await svc.status()).has_api_key, true);
    } finally {
      if (prev === undefined) delete process.env.JEV_OPENROUTER_API_KEY; else process.env.JEV_OPENROUTER_API_KEY = prev;
    }
  });
  it("the settings file is 0600, tolerant on read, and filters junk", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "jev-settings-"));
    const file = path.join(dir, "nested", "jev_decisions_settings.json");
    assert.deepEqual(loadDecisionsSettings(file), {}, "missing file -> empty");
    saveDecisionsSettings(file, { mode: "shadow", allow_experimental_suggestions: true, api_key: "sk-or-v1-abc" });
    assert.equal(fs.statSync(file).mode & 0o777, 0o600);
    assert.deepEqual(loadDecisionsSettings(file), { mode: "shadow", allow_experimental_suggestions: true, api_key: "sk-or-v1-abc" });
    fs.writeFileSync(file, JSON.stringify({ mode: "hack", api_key: "has space", allow_experimental_suggestions: "yes", extra: 1 }));
    assert.deepEqual(loadDecisionsSettings(file), {}, "invalid values are dropped, not trusted");
    fs.writeFileSync(file, "not json");
    assert.deepEqual(loadDecisionsSettings(file), {});
    assert.ok(decisionsSettingsPath({ TILE_COMPILE_PI_STORAGE_DIR: "/x" } as any).endsWith("/x/jev_decisions_settings.json"));
    fs.rmSync(dir, { recursive: true });
  });
});
