import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  createAgentSession,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
} from "@earendil-works/pi-coding-agent";

test("Pi SDK creates an isolated tool-free sidecar session", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tile-pi-sdk-"));
  try {
    const modelRuntime = await ModelRuntime.create({
      authPath: path.join(dir, "auth.json"),
      modelsPath: null,
      refreshOnCreate: false,
    });
    const model = modelRuntime.getModels()[0];
    assert.ok(model, "Pi must expose at least one built-in model");
    const resourceLoader = new DefaultResourceLoader({ cwd: dir, agentDir: dir });
    await resourceLoader.reload();
    const { session } = await createAgentSession({
      cwd: dir,
      agentDir: dir,
      model,
      modelRuntime,
      sessionManager: SessionManager.inMemory(),
      resourceLoader,
      tools: [],
    });
    try {
      assert.deepEqual(session.getActiveToolNames(), []);
      assert.equal(session.model?.id, model.id);
    } finally {
      session.dispose();
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Pi SDK completes a deterministic tool-call round trip", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "tile-pi-tool-"));
  try {
    const fixture = path.join(dir, "fixture.txt");
    await writeFile(fixture, "pi-tool-round-trip\n", "utf8");
    const modelRuntime = await ModelRuntime.create({
      authPath: path.join(dir, "auth.json"),
      modelsPath: null,
      refreshOnCreate: false,
    });
    const model = modelRuntime.getModels()[0];
    assert.ok(model);
    await modelRuntime.setRuntimeApiKey(model.provider, "pi-compatibility-test-key");
    const resourceLoader = new DefaultResourceLoader({ cwd: dir, agentDir: dir });
    await resourceLoader.reload();
    const { session } = await createAgentSession({
      cwd: dir,
      agentDir: dir,
      model,
      modelRuntime,
      sessionManager: SessionManager.inMemory(),
      resourceLoader,
      tools: ["read"],
    });
    try {
      assert.deepEqual(session.getActiveToolNames(), ["read"]);
      let calls = 0;
      const events: string[] = [];
      const unsubscribe = session.subscribe((event) => { events.push(event.type); });
      session.agent.streamFunction = (requestedModel) => {
        calls += 1;
        const message = {
          role: "assistant",
          content: calls === 1
            ? [{ type: "toolCall", id: "read-1", name: "read", arguments: { path: fixture } }]
            : [{ type: "text", text: "round-trip-complete" }],
          api: requestedModel.api,
          provider: requestedModel.provider,
          model: requestedModel.id,
          usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
          stopReason: calls === 1 ? "toolUse" : "stop",
          timestamp: Date.now(),
        };
        return {
          async *[Symbol.asyncIterator]() { yield { type: "done", message }; },
          async result() { return message; },
        } as any;
      };
      await session.prompt("Read the fixture and respond.");
      unsubscribe();
      assert.equal(calls, 2);
      assert.ok(events.includes("tool_execution_start"));
      assert.ok(events.includes("tool_execution_end"));
      assert.ok(events.includes("agent_end"));
      const results = session.messages.filter((message) => message.role === "toolResult");
      assert.equal(results.length, 1);
      assert.match(JSON.stringify(results[0]), /pi-tool-round-trip/);
    } finally {
      session.dispose();
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
