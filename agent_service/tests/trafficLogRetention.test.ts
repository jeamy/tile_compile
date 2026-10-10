import test from "node:test";
import assert from "node:assert/strict";
import { retainRecentTrafficLogLines } from "../src/services/trafficLog.js";

test("traffic-log retention removes entries at or before the cutoff", () => {
  const cutoff = Date.parse("2026-01-31T00:00:00.000Z");
  const input = [
    "[2026-01-30T23:59:59.999Z] expired\n",
    "[2026-01-31T00:00:00.000Z] boundary\n",
    "[2026-01-31T00:00:00.001Z] current\n",
  ].join("");
  assert.equal(retainRecentTrafficLogLines(input, cutoff), "[2026-01-31T00:00:00.001Z] current\n");
});

test("traffic-log retention discards malformed or missing timestamps", () => {
  const input = "[not-a-date] malformed\nlegacy line without timestamp\n[2026-02-01T00:00:00.000Z] current\n";
  assert.equal(
    retainRecentTrafficLogLines(input, Date.parse("2026-01-31T00:00:00.000Z")),
    "[2026-02-01T00:00:00.000Z] current\n",
  );
});
