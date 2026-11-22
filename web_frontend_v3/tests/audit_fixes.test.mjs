// Regression tests for the frontend fixes of the 2026-11-21 code audit:
// WebSocket reconnect, log level normalization, page cleanup hooks and the
// pending draft flush before an outgoing config. No DOM, no network.
import test from "node:test";
import assert from "node:assert/strict";

globalThis.window = { location: { protocol: "http:", host: "localhost:8080", hostname: "localhost", port: "8080" } };
// store.js persists through localStorage; a throwing one must not break it.
globalThis.localStorage = {
  getItem() { throw new Error("storage blocked"); },
  setItem() { throw new Error("storage blocked"); },
};

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances = [];
  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
    this.closeCalls = 0;
    FakeWebSocket.instances.push(this);
  }
  open() { this.readyState = FakeWebSocket.OPEN; this.onopen?.(); }
  close() { this.closeCalls++; this.readyState = FakeWebSocket.CLOSED; }
  // The server or the network drops the connection.
  drop() { this.readyState = FakeWebSocket.CLOSED; this.onclose?.(); }
}
globalThis.WebSocket = FakeWebSocket;

const { connectWebSocket, disconnectWebSocket, onWebSocketMessage } =
  await import("../js/components/ws-manager.js");
const { normalizeLogLevel, LOG_LEVELS } = await import("../js/utils/log.js");
const { onPageUnmount, runPageUnmount } = await import("../js/utils/page-lifecycle.js");
const { setPendingDraftFlush, getOutgoingConfig, setConfigState } =
  await import("../js/state/config-state.js");

test("ws-manager reconnects after a dropped socket instead of keeping the dead one", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  FakeWebSocket.instances.length = 0;
  const events = [];
  const off = onWebSocketMessage((e) => events.push(e.type));
  try {
    connectWebSocket("run-a", false, "/runs/a");
    assert.equal(FakeWebSocket.instances.length, 1);
    const first = FakeWebSocket.instances[0];
    first.open();
    // Same target while the socket is live: no second socket.
    connectWebSocket("run-a", false, "/runs/a");
    assert.equal(FakeWebSocket.instances.length, 1);

    first.drop();
    assert.deepEqual(events, ["ws:open", "ws:close"]);
    t.mock.timers.tick(3000);
    assert.equal(FakeWebSocket.instances.length, 2, "a reconnect opens a new socket");
    assert.equal(FakeWebSocket.instances[1].url, first.url, "same run and run dir");

    // A dead socket for the same target is replaced at once, not kept.
    FakeWebSocket.instances[1].readyState = FakeWebSocket.CLOSED;
    connectWebSocket("run-a", false, "/runs/a");
    assert.equal(FakeWebSocket.instances.length, 3);
  } finally {
    off();
    disconnectWebSocket();
  }
});

test("ws-manager ignores a stale socket after switching runs", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  FakeWebSocket.instances.length = 0;
  const events = [];
  const off = onWebSocketMessage((e) => events.push(e.type));
  try {
    connectWebSocket("run-a");
    const a = FakeWebSocket.instances[0];
    connectWebSocket("run-b");
    assert.equal(a.closeCalls, 1, "the old socket is closed");
    a.drop();
    a.onmessage?.({ data: "{}" });
    assert.deepEqual(events, [], "events of the old socket are dropped");
    t.mock.timers.tick(10000);
    assert.equal(FakeWebSocket.instances.length, 2, "no reconnect for the old run");

    disconnectWebSocket();
    FakeWebSocket.instances[1].drop();
    t.mock.timers.tick(10000);
    assert.equal(FakeWebSocket.instances.length, 2, "no reconnect after disconnect");
  } finally {
    off();
    disconnectWebSocket();
  }
});

test("normalizeLogLevel maps runner spellings onto the filter levels", () => {
  assert.equal(normalizeLogLevel("WARNING"), "WARN");
  assert.equal(normalizeLogLevel("warning"), "WARN");
  assert.equal(normalizeLogLevel("warn"), "WARN");
  assert.equal(normalizeLogLevel("fatal"), "ERROR");
  assert.equal(normalizeLogLevel("CRITICAL"), "ERROR");
  assert.equal(normalizeLogLevel("dbg"), "DEBUG");
  assert.equal(normalizeLogLevel(""), "INFO");
  assert.equal(normalizeLogLevel(undefined), "INFO");
  assert.equal(normalizeLogLevel("chatter"), "INFO");
  for (const level of LOG_LEVELS) assert.equal(normalizeLogLevel(level), level);
});

test("page cleanups run once and a failing one does not stop the rest", (t) => {
  t.mock.method(console, "error", () => {});
  const calls = [];
  onPageUnmount(() => calls.push("a"));
  onPageUnmount(() => { throw new Error("boom"); });
  onPageUnmount(() => calls.push("c"));
  runPageUnmount();
  runPageUnmount();
  assert.deepEqual(calls, ["a", "c"]);
});

test("getOutgoingConfig commits a pending draft edit first", () => {
  setConfigState({ draft: { a: 1 }, draftYaml: "a: 1\n" });
  let flushes = 0;
  setPendingDraftFlush(() => {
    flushes++;
    setConfigState({ draft: { a: 2 }, draftYaml: "a: 2\n", dirty: true });
  });
  const outgoing = getOutgoingConfig();
  assert.equal(flushes, 1);
  assert.deepEqual(outgoing.draft, { a: 2 });
  assert.equal(outgoing.yaml, "a: 2\n");
  getOutgoingConfig();
  assert.equal(flushes, 1, "the flush is consumed");
});

test("preview helpers: YAML scalars and wheel zoom shared by BGE and HMS", async () => {
  const { yamlScalar, yamlScalarText } = await import("../js/utils/yaml-scalar.js");
  assert.equal(yamlScalar(" true # on", false), true);
  assert.equal(yamlScalar("nope", true), true);
  assert.equal(yamlScalar("2.5", 1), 2.5);
  assert.equal(yamlScalar("x", 1), 1);
  assert.equal(yamlScalar("'rbf'", "poly"), "rbf");
  assert.equal(yamlScalar("", "poly"), "poly");
  assert.equal(yamlScalarText(false), "false");
  assert.equal(yamlScalarText(3), "3");

  const { wheelZoom } = await import("../js/utils/canvas-zoom.js");
  const rect = { left: 0, top: 0, width: 100, height: 100 };
  const v = wheelZoom({ scale: 1, offsetX: 0, offsetY: 0 }, { deltaY: -1, clientX: 50, clientY: 50 }, rect, 200, 200);
  assert.equal(v.scale, 1.15);
  // The canvas point under the pointer (100,100) shows the same image point.
  assert.ok(Math.abs((100 - v.offsetX) / v.scale - 100) < 1e-9);
  assert.equal(wheelZoom({ scale: 20, offsetX: 0, offsetY: 0 }, { deltaY: -1, clientX: 0, clientY: 0 }, rect, 100, 100).scale, 20);
});
