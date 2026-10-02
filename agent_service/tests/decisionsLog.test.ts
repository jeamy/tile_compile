import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, it } from "node:test";
import { createJevLogger, jevLogPath, readJevLog } from "../src/services/decisionsLog.js";

function tmp() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "jevlog-"));
  return { dir, file: path.join(dir, "sub", "jev.log"), env: { JEV_DECISIONS_LOG_PATH: path.join(dir, "sub", "jev.log") } as NodeJS.ProcessEnv };
}
const lines = (file: string) => fs.readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l));

describe("Jev log file", () => {
  it("resolves its path from env, then the PI storage dir, then the project runs/.pi_memory", () => {
    assert.equal(jevLogPath({ JEV_DECISIONS_LOG_PATH: "/x/a.log" } as NodeJS.ProcessEnv), path.resolve("/x/a.log"));
    assert.equal(jevLogPath({ TILE_COMPILE_PI_STORAGE_DIR: "/s" } as NodeJS.ProcessEnv), path.join("/s", "jev_decisions.log"));
    assert.ok(jevLogPath({} as NodeJS.ProcessEnv).endsWith(path.join("runs", ".pi_memory", "jev_decisions.log")));
  });
  it("appends one JSON object per line with time, source and pid, creating the directory", () => {
    const { file, env } = tmp();
    const log = createJevLogger("eval", env);
    log({ kind: "request", request_id: "a", body: { state: { x: 1 } } });
    log({ kind: "result", request_id: "a", choice: "keep_current" });
    const rows = lines(file);
    assert.equal(rows.length, 2);
    assert.deepEqual(rows.map((r) => r.kind), ["request", "result"]);
    assert.equal(rows[0].source, "eval");
    assert.equal(rows[0].pid, process.pid);
    assert.ok(!Number.isNaN(Date.parse(rows[0].ts)));
    assert.deepEqual(rows[0].body, { state: { x: 1 } });
  });
  it("two writers (sidecar and eval script) share the file without losing lines", () => {
    const { file, env } = tmp();
    const a = createJevLogger("sidecar", env), b = createJevLogger("eval", env);
    for (let i = 0; i < 20; i++) { a({ kind: "result", n: i }); b({ kind: "result", n: i }); }
    const rows = lines(file);
    assert.equal(rows.length, 40);
    assert.equal(rows.filter((r) => r.source === "sidecar").length, 20);
  });
  it("redacts keys, bearer tokens and project paths in whatever it is given", () => {
    const { file, env } = tmp();
    createJevLogger("eval", env)({ kind: "x", note: "sk-or-v1-SECRETSECRET Bearer abc.def JEV_OPENROUTER_API_KEY=hunter2" });
    const text = fs.readFileSync(file, "utf8");
    assert.ok(!/SECRETSECRET|abc\.def|hunter2/.test(text), text);
  });
  it("can be switched off and never throws when the file is not writable", () => {
    const { file, env } = tmp();
    createJevLogger("eval", { ...env, JEV_LOG: "off" })({ kind: "x" });
    assert.equal(fs.existsSync(file), false, "JEV_LOG=off writes nothing");
    const blocker = path.join(os.tmpdir(), `jevlog-blocker-${process.pid}`);
    fs.writeFileSync(blocker, "not a directory");
    assert.doesNotThrow(() => createJevLogger("eval", { JEV_DECISIONS_LOG_PATH: path.join(blocker, "x.log") } as NodeJS.ProcessEnv)({ kind: "x" }));
    fs.rmSync(blocker);
  });
});

describe("Jev log reader", () => {
  it("returns the newest lines last, honours the limit and reports the total", () => {
    const { env } = tmp();
    const log = createJevLogger("sidecar", env);
    for (let i = 0; i < 30; i++) log({ kind: "result", n: i });
    const r = readJevLog(5, env);
    assert.equal(r.items.length, 5);
    assert.equal(r.count, 30);
    assert.equal(JSON.parse(r.items[4]).n, 29);
    assert.equal(JSON.parse(r.items[0]).n, 25);
    assert.equal(r.enabled, true);
  });
  it("a missing file is an empty result, JEV_LOG=off is reported, a huge limit is capped", () => {
    const { env } = tmp();
    assert.deepEqual(readJevLog(10, env).items, []);
    assert.equal(readJevLog(10, { ...env, JEV_LOG: "off" }).enabled, false);
    const log = createJevLogger("sidecar", env);
    log({ kind: "result" });
    assert.equal(readJevLog(1e9, env).items.length, 1);
    assert.equal(readJevLog(Number.NaN, env).items.length, 1);
  });
  it("reads only the tail of a large file and drops the cut first line", () => {
    const { file, env } = tmp();
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const filler = JSON.stringify({ kind: "x", pad: "y".repeat(1000) });
    const n = 6000;  // ~6 MB > 4 MiB
    fs.writeFileSync(file, Array.from({ length: n }, (_, i) => JSON.stringify({ i, pad: "y".repeat(1000) })).join("\n") + "\n");
    void filler;
    const r = readJevLog(3, env);
    assert.equal(r.items.length, 3);
    assert.equal(JSON.parse(r.items[2]).i, n - 1);
    assert.ok(r.count < n && r.count > 3000);
    for (const line of r.items) JSON.parse(line);
  });
});
