// Static dock fixture; all API requests intercepted. No services or processing runs.
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE_PATH || "playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const turns = { a: [{ message: "A question", result: { summary: "A answer" } }], b: [] };
const posts = [];
let records = 0;
let delayChat = false;
let missing = false;
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = []; page.on("pageerror", e => errors.push(e.message));
  await page.route("**/*", async route => {
    const req = route.request(); const url = new URL(req.url()); const p = url.pathname;
    const json = body => route.fulfill({ contentType: "application/json", body: JSON.stringify(body) });
    if (req.method() === "POST") {
      const body = req.postDataJSON(); posts.push({ p, body });
      if (p.endsWith("select-context")) return json({ kind: "run", run_uid: body.run_uid, context_id: `run:${body.run_uid}`, run_key: `/fixture/${body.run_uid}`, artifacts_reachable: !(missing && body.run_uid === "a") });
      if (p === "/api/pi/run-chat") {
        if (delayChat) await new Promise(resolve => setTimeout(resolve, 200));
        const result = { summary: `${body.run_uid} reply`, run_uid: body.run_uid, context_id: `run:${body.run_uid}` };
        turns[body.run_uid].push({ message: body.message, result }); return json(result);
      }
      if (p === "/api/pi/post-run/advice") return json({ summary: "Jev choice" });
      throw new Error(`Unexpected mutation ${p}`);
    }
    if (p === "/api/pi/assistant/capabilities") return json({ schema_version: "pi.assistant-capabilities.v1", run_uid_threads: true });
    if (p === "/api/pi/active-context") return json({ context: null });
    if (p === "/api/pi/assistant/run-context") return json({ run_uid: url.searchParams.get('run_id').endsWith('b') ? 'b' : 'a' });
    if (p.startsWith("/api/pi/run-contexts/")) return json({ run_keys: ["/fixture/a"] });
    if (p === "/api/pi/run-chat/history") return json({ turns: turns[url.searchParams.get("run_uid")] || [] });
    if (p === "/api/pi/decision-records") { records++; return json({ items: [] }); }
    if (p === "/fixture") return route.fulfill({ contentType: "text/html", body: `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">${['tokens','base','layout','components'].map(x => `<link rel="stylesheet" href="/ui/css/${x}.css">`).join('')}</head><body><div id="content">Page one</div><div id="host"></div><script type="module">import { loadLocale } from '/ui/js/i18n/i18n.js';import {createAssistantDock} from '/ui/js/assistant/dock.js';await loadLocale('en');document.getElementById('host').append(createAssistantDock());</script></body></html>` });
    const file = path.resolve(root, p.replace(/^\/ui\//, '').replace(/^\/i18n\//, 'i18n/'));
    assert.ok(file.startsWith(root + path.sep));
    return route.fulfill({ contentType: file.endsWith('.js') ? 'text/javascript' : file.endsWith('.json') ? 'application/json' : 'text/css', body: await fs.readFile(file) });
  });
  await page.goto('https://dock.test/fixture');
  await page.locator('#assistant-dock').waitFor();
  const choose = uid => page.evaluate(uid => window.dispatchEvent(new CustomEvent('tc-assistant-context', { detail: { run_uid: uid } })), uid);
  await choose('a'); await page.getByText('A answer', { exact: true }).waitFor();
  assert.equal(posts.filter(x => x.p === '/api/pi/run-chat').length, 0, 'selection does not open chat session');
  assert.equal(records, 0, 'Why is lazy');
  await page.getByText('Why', { exact: true }).first().click();
  await page.getByText(/No permanent decision recording/).waitFor(); assert.equal(records, 1);
  await page.evaluate(() => document.getElementById('content').textContent = 'Page two');
  assert.equal(await page.locator('#assistant-dock').count(), 1, 'one host survives navigation');
  await page.getByLabel('Question about the selected run').fill('message to A'); delayChat = true;
  await page.getByRole('button', { name: 'Ask PI', exact: true }).click();
  await choose('b'); await page.getByText('run:b', { exact: false }).waitFor();
  await page.waitForTimeout(260);
  assert.equal(await page.getByText('a reply', { exact: true }).count(), 0, 'A answer cannot enter B thread');
  await choose('a'); await page.getByText('a reply', { exact: true }).waitFor();
  assert.deepEqual(posts.find(x => x.p === '/api/pi/run-chat').body, { run_id: '/fixture/a', run_uid: 'a', message: 'message to A' });
  await page.getByRole('button', { name: 'Jev: post-run advice' }).click(); await page.getByText('Jev choice', { exact: true }).waitFor();
  await page.getByRole('button', { name: 'Collapse', exact: true }).click();
  assert.equal(await page.locator('#assistant-dock').isVisible(), false);
  await page.getByRole('button', { name: 'PI Assistant', exact: true }).click(); await page.getByText('a reply', { exact: true }).waitFor();
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `dock fits ${width}`);
    await page.screenshot({ path: `/tmp/out_assistant_dock_${width}.png`, fullPage: true });
  }
  await page.reload(); await page.getByText('a reply', { exact: true }).waitFor();
  missing = true;
  await page.getByRole('button', { name: 'Refresh' }).click(); await page.getByText(/Run files are missing/).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Ask PI', exact: true }).isDisabled(), true);
  assert.ok(!(await page.locator('#assistant-dock').innerText()).includes('ui.dock.'));
  assert.deepEqual(errors, []);
  console.log('Dock fixtures passed: single host, UID threads, lazy Why, stale send isolation, Jev card, collapse/reload, missing artifacts, desktop/mobile.');
} finally { await browser.close(); }
