// Static browser fixtures only. No backend, sidecar or processing run is started.
// PLAYWRIGHT_MODULE_PATH=/path/to/playwright/index.mjs node web_frontend_v3/tests/run-learning-archive.browser.mjs
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE_PATH || "playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const apiPosts = [];
const errors = [];
let excluded = false;
let listFails = false;
let capabilitiesMissing = false;
let identityReachable = false;
let slowA = false;
let deleted = false;
const snapshots = {
  a: { run_uid: "uid-a", run_id: "run-a", snapshot_id: "snapshot-a", run_key: "/fixture/run-a", stage: "completed", status: "completed",
    captured_at_epoch: 100000, artifacts_state: "deleted", artifacts_reachable: false, excluded_from_learning: false,
    exclusion_code: "", validation_state: "unreviewed", config: { yaml: "data: OSC", effective: { data: "OSC" }, parser_defaults_included: true },
    source: { original_input_dir: "/fixture/lights", calibration: { dark: "<img src=x onerror=alert(1)>" } },
    artifacts: { "stats.json": { frames: 2 } }, phase_events: [], capture_issues: [], missing_optional_files: ["optional.json"],
    light_manifest_available: true, raw_data_stored: false, preview_snapshot_id: "snapshot-a" },
  b: { run_uid: "uid-b", run_id: "run-b", snapshot_id: "snapshot-b", run_key: "/fixture/run-b", stage: "run_start", status: "running",
    captured_at_epoch: 100001, artifacts_state: "available", artifacts_reachable: true, excluded_from_learning: false,
    config: { yaml: "data: MONO", parser_defaults_included: false }, source: {}, artifacts: {}, phase_events: [], capture_issues: [], missing_optional_files: [], raw_data_stored: false },
};
const old = { ...snapshots.a, snapshot_id: "snapshot-old", stage: "run_start", artifacts: { "stats.json": { frames: 1 } } };
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("pageerror", error => errors.push(error.message));
  page.on("dialog", dialog => dialog.accept());
  await page.route("**/*", async route => {
    const req = route.request();
    const url = new URL(req.url());
    let p = decodeURIComponent(url.pathname);
    if (p.startsWith("/api/runs/")) {
      const parts = p.split("/");
      if (parts[3].startsWith("b64_")) parts[3] = Buffer.from(parts[3].slice(4), "base64url").toString();
      p = parts.join("/");
    }
    const json = (body, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (req.method() === "POST") {
      const body = req.postDataJSON(); apiPosts.push({ path: p, body });
      if (p.endsWith("/exclusion")) { excluded = body.excluded; return json({ ok: true }); }
      if (p.endsWith("/relink")) { identityReachable = true; return json({ ok: true, run_uid: "uid-a" }); }
      if (p.endsWith("/delete")) { deleted = true; snapshots.b.artifacts_state = "deleted"; return json({ ok: true, run_uid: "uid-b", learning_retained: true }); }
      throw new Error(`Unexpected mutation: ${p}`);
    }
    if (p === "/api/pi/run-learning/capabilities" && capabilitiesMissing) return json({ message: "old backend" }, 404);
    if (p === "/api/pi/run-learning/capabilities") return json({ schema_version: "pi.run-learning-capabilities.v1", history_summaries: true, snapshot_lookup: true, delete_preserves_learning: true });
    if (p === "/api/pi/run-learning") {
      if (listFails) return json({ message: "fixture unavailable" }, 503);
      return json({ items: [{ ...snapshots.a, excluded_from_learning: excluded }, snapshots.b] });
    }
    if (p === "/api/pi/run-learning/uid-a") {
      if (slowA) await new Promise(resolve => setTimeout(resolve, 150));
      return json({ ...snapshots.a, artifacts_reachable: identityReachable, excluded_from_learning: excluded, exclusion_code: excluded ? "test_run" : "" });
    }
    if (p === "/api/pi/run-learning/uid-b") return json(snapshots.b);
    if (p === "/api/pi/run-contexts/uid-a") return json({ run_keys: identityReachable ? ["/fixture/run-a", "/fixture/moved-a"] : ["/fixture/run-a"] });
    if (p.endsWith("/history")) return json({ items: p.includes("uid-a") ? [snapshots.a, old].map(({ snapshot_id, stage }) => ({ snapshot_id, stage, captured_at_epoch: 100000 })) : [] });
    if (p.endsWith("/snapshots/snapshot-old")) return json(old);
    if (p.endsWith("/snapshots/snapshot-a")) return route.fulfill({ contentType: "application/json", body: JSON.stringify(snapshots.a).replace(/}$/, ',"mtime_file_clock_ticks":-4652672195123456789}') });
    if (p.endsWith("/preview")) return route.fulfill({ contentType: "image/png", body: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6bzYAAAAASUVORK5CYII=", "base64") });
    if (p === "/api/runs") return json(deleted ? [] : [{ run_id: "run-b", path: "/fixture/run-b", status: "completed" }]);
    if (p === "/api/runs/run-b/status") return json({ run_dir: "/fixture/run-b", status: "completed", events: [] });
    if (p.endsWith("/artifacts")) return json({ items: [] });
    if (p.endsWith("/stats/status")) return json({});
    if (p.startsWith("/api/")) return json({ message: "not in fixture" }, 404);
    if (p === "/ui/fixture") return route.fulfill({ contentType: "text/html", body: `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">${["tokens", "base", "layout", "components", "pages"].map(n => `<link rel="stylesheet" href="/ui/css/${n}.css">`).join("")}</head><body><main id="root"></main><script type="module">import {loadLocale} from '/ui/js/i18n/i18n.js'; import {createRunHistoryPage} from '/ui/js/pages/run-history.js'; await loadLocale('de'); document.getElementById('root').append(createRunHistoryPage());</script></body></html>` });
    // Locale loader uses a relative URL beneath /ui/.
    const relative = p.replace(/^\/ui\//, "");
    const file = path.resolve(root, relative);
    assert.ok(file.startsWith(root + path.sep), "static path containment");
    return route.fulfill({ contentType: file.endsWith(".js") ? "text/javascript" : file.endsWith(".json") ? "application/json" : "text/css", body: await fs.readFile(file) });
  });
  await page.goto("https://archive.test/ui/fixture");
  await page.locator('[data-archive-uid="uid-a"]').click();
  await page.locator('[data-archive-detail]').getByText("snapshot-a", { exact: true }).waitFor();
  assert.equal(await page.getByRole("button", { name: "Weitere Archive laden" }).isVisible(), false, "no misleading load-more for short list");
  assert.match(await page.locator('[data-archive-detail]').innerText(), /Ungeprüft/);
  await page.getByText("Lights und Kalibrationsherkunft", { exact: true }).click();
  assert.match(await page.locator('.tc-archive-json').innerText(), /<img src=x onerror=alert\(1\)>/);
  assert.equal(await page.locator('.tc-archive-json img').count(), 0, "metadata is text, not HTML");
  await page.getByLabel("Snapshot-Version").selectOption("snapshot-old");
  await page.locator('[data-archive-detail]').getByText("snapshot-old", { exact: true }).waitFor();
  assert.equal(await page.locator('.tc-archive-preview').count(), 0, "no latest PNG attached to older snapshot");
  await page.getByLabel("Snapshot-Version").selectOption("snapshot-a");
  await page.getByLabel("Ausschlussgrund").selectOption("test_run");
  await page.getByRole("button", { name: "Vom Lernen ausschließen", exact: true }).click();
  await page.getByRole("button", { name: "Ausschluss aufheben", exact: true }).waitFor();
  assert.deepEqual(apiPosts.at(-1), { path: "/api/pi/run-learning/uid-a/exclusion", body: { confirmed: true, excluded: true, reason_code: "test_run" } });
  await page.getByRole("button", { name: "Ausschluss aufheben", exact: true }).click();
  await page.getByRole("button", { name: "Vom Lernen ausschließen", exact: true }).waitFor();
  await page.getByText("Verschobenen Run neu verknüpfen", { exact: true }).first().click();
  await page.getByLabel("Neuer vollständiger Run-Pfad").fill("/fixture/moved-a");
  await page.getByRole("button", { name: "Verschobenen Run neu verknüpfen", exact: true }).click();
  await page.locator('[data-archive-detail]').getByText("Ja", { exact: true }).waitFor();
  assert.deepEqual(apiPosts.at(-1), { path: "/api/pi/run-contexts/uid-a/relink", body: { run_dir: "/fixture/moved-a", confirmed: true } });

  slowA = true;
  await page.locator('[data-archive-uid="uid-b"]').click();
  await page.locator('[data-archive-detail]').getByText("snapshot-b", { exact: true }).waitFor();
  await page.locator('[data-archive-uid="uid-a"]').click();
  await page.locator('[data-archive-uid="uid-b"]').click();
  await page.waitForTimeout(220);
  assert.equal(await page.locator('[data-archive-detail]').getByText("snapshot-a", { exact: true }).count(), 0, "late context response ignored");
  slowA = false;
  await page.locator('[data-archive-uid="uid-a"]').click();
  await page.locator('[data-archive-detail]').getByText("snapshot-a", { exact: true }).waitFor();
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `no horizontal overflow at ${width}px`);
    await page.screenshot({ path: `/tmp/out_archive_ui_${width}.png`, fullPage: true });
  }
  await page.reload();
  await page.locator('[data-archive-detail]').getByText("snapshot-a", { exact: true }).waitFor();
  listFails = true;
  await page.locator('.tc-run-archive').getByRole("button", { name: "Aktualisieren" }).click();
  await page.locator('[data-archive-list]').getByText(/Lerndaten konnten nicht/).waitFor();
  listFails = false;
  await page.locator('.tc-run-archive').getByRole("button", { name: "Aktualisieren" }).click();
  await page.locator('[data-archive-uid="uid-a"]').waitFor();

  const cases = await page.evaluate(async () => {
    const { deleteRunFiles, deletionErrorMessage } = await import('/ui/js/components/run-artifact-actions.js');
    const run = async (answers, code, status = 409, supported = true) => {
      const calls = []; let confirms = 0; let errorText;
      const confirm = () => { confirms++; return answers.shift() ?? false; };
      const client = { get: async () => supported ? { schema_version: "pi.run-learning-capabilities.v1", delete_preserves_learning: true } : {}, post: async (path, body) => {
        calls.push({ path, body });
        if (code && calls.length === 1) { const e = new Error(code); e.status = status; e.payload = { error: true, code, message: code }; throw e; }
        return { ok: true, learning_retained: true };
      } };
      try { await deleteRunFiles('run-b', '/fixture/run-b', { confirm, client }); }
      catch (e) { errorText = deletionErrorMessage(e); }
      return { calls, confirms, errorText };
    };
    return Promise.all([run([false]), run([true]), run([true, false], 'LEARNING_SNAPSHOT_INCOMPLETE'),
      run([true, true], 'LEARNING_SNAPSHOT_INCOMPLETE'), run([true, true], 'RAW_SOURCE_INSIDE_RUN'),
      run([true, true], 'LEARNING_SNAPSHOT_FAILED', 503), run([true, true], 'RUN_ACTIVE'), run([true], null, 409, false)]);
  });
  assert.equal(cases[0].calls.length, 0);
  assert.deepEqual(cases[1].calls[0].body, { run_dir: "/fixture/run-b" });
  assert.equal(cases[2].calls.length, 1);
  assert.deepEqual(cases[3].calls[1].body, { run_dir: "/fixture/run-b", allow_incomplete_snapshot: true });
  for (const c of cases.slice(4, 7)) { assert.equal(c.calls.length, 1); assert.equal(c.confirms, 1); assert.ok(c.errorText && !c.errorText.includes("ui.archive.")); }
  assert.equal(cases[7].calls.length, 0, "older backend cannot execute unprotected file deletion");
  assert.ok(cases[7].errorText.includes("Backend"));
  await page.locator('#run-list .tc-run-item').click();
  await page.getByRole("button", { name: "Run-Dateien löschen — Lerndaten behalten", exact: true }).waitFor();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "file actions fit mobile viewport");
  await page.getByRole("button", { name: "Run-Dateien löschen — Lerndaten behalten", exact: true }).click();
  await page.locator('#run-list').getByText("Keine Runs gefunden", { exact: true }).waitFor();
  await page.locator('[data-archive-detail]').getByText("snapshot-b", { exact: true }).waitFor();
  assert.deepEqual(apiPosts.at(-1), { path: "/api/runs/run-b/delete", body: { run_dir: "/fixture/run-b" } });
  assert.equal(await page.locator('#run-actions button').count(), 0, "stale file actions cleared after deletion");
  await page.evaluate(async () => {
    const { loadLocale } = await import('/ui/js/i18n/i18n.js');
    const { createRunHistoryPage } = await import('/ui/js/pages/run-history.js');
    await loadLocale('en'); document.getElementById('root').replaceChildren(createRunHistoryPage());
  });
  await page.locator('[data-archive-uid="uid-a"]').click();
  await page.getByRole('button', { name: 'Exclude from learning', exact: true }).waitFor();
  assert.ok(!(await page.locator('.tc-run-archive').innerText()).includes('ui.archive.'), 'English archive labels resolve');
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save this snapshot locally as JSON', exact: true }).click();
  const download = await downloaded;
  assert.match(await fs.readFile(await download.path(), 'utf8'), /-4652672195123456789/, '64-bit metadata remains lossless in download');
  capabilitiesMissing = true;
  await page.locator('.tc-run-archive').getByRole('button', { name: 'Refresh' }).click();
  await page.locator('[data-archive-list]').getByText(/running backend does not confirm/).waitFor();
  assert.equal(await page.locator('[data-archive-detail] button').count(), 0, 'unsupported backend clears stale archive controls');
  assert.deepEqual(errors, [], "no browser errors");
  console.log("Archive browser fixtures passed: retrieval/version selection, XSS-safe metadata, exclusion/re-enable, relink, stale responses, reload, retry, delete guards, desktop/mobile.");
} finally { await browser.close(); }
