import assert from "node:assert/strict";
import test from "node:test";
import { sanitizeProviderPayloadForModel } from "../src/services/providerPayload.js";

test("OpenAI reasoning requests omit temperature", () => {
  const payload = { temperature: 0, max_output_tokens: 2048 };
  const result = sanitizeProviderPayloadForModel(payload, {
    provider: "openai", api: "openai-responses", id: "gpt-6.1-sol",
    reasoning: true, maxTokens: 128000,
  }, "test");
  assert.equal(Object.hasOwn(result, "temperature"), false);
  assert.equal(result.max_output_tokens, 2048);
});

test("OpenAI non-reasoning requests retain temperature", () => {
  const result = sanitizeProviderPayloadForModel({ temperature: 0 }, {
    provider: "openai", api: "openai-responses", id: "gpt-4.1-mini", reasoning: false,
  }, "test");
  assert.equal(result.temperature, 0);
});

test("Codex requests omit temperature independent of reasoning metadata", () => {
  const result = sanitizeProviderPayloadForModel({ temperature: 1 }, {
    provider: "openai-codex", api: "openai-codex-responses", id: "gpt-6.1-sol", reasoning: false,
  }, "test");
  assert.equal(Object.hasOwn(result, "temperature"), false);
});
