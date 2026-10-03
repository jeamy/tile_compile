// Static fixture using the real Resume handoff listener, no services or runs.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  const requests = [];
  let valid = true, expired = false, source = 'bge:\n  enabled: false\n', wrongUid = false, delayPreview = false;
  await page.route('**/*', async route => {
    const req = route.request(), url = new URL(req.url()), p = url.pathname;
    const json = body => route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) });
    if (p.startsWith('/api/')) {
      requests.push({ p, method: req.method(), body: req.method() === 'POST' ? req.postDataJSON() : null });
      if (p === '/api/pi/assistant/run-context') return json({ run_uid: wrongUid ? 'b' : 'a' });
      if (p.endsWith('/config') && req.method() === 'GET') return json({ config_yaml: source });
      if (p === '/api/pi/action-plans/preview') {
        if (delayPreview) await new Promise(resolve => setTimeout(resolve, 200));
        return json({ preview: { preview_id: 'test_preview', action_plan_id: 'plan1', config_sha256: 'hash1', config_valid: valid, base_yaml: source, patched_yaml: 'bge:\n  enabled: true\n' } });
      }
      if (p === '/api/pi/action-plans/previews/test_preview') return json({ state: expired ? 'expired' : 'pending', action_plan_id: 'plan1', config_sha256: 'hash1' });
      throw new Error(`Unexpected API ${req.method()} ${p}`);
    }
    if (p === '/fixture') return route.fulfill({ contentType: 'text/html', body: `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">${['tokens','base','layout','components'].map(x => `<link rel="stylesheet" href="/ui/css/${x}.css">`).join('')}</head><body><div id="resume-panel"><textarea id="resume-config-yaml"></textarea><div id="resume-hint"></div></div><div id="cards"></div><script type="module">import '/ui/js/pages/run-monitor.js';import {loadLocale} from '/ui/js/i18n/i18n.js';import {setRunState} from '/ui/js/state/run-state.js';import {assistantActionPlans,createAssistantActionCard} from '/ui/js/assistant/action-card.js';await loadLocale('en');setRunState({currentRunDir:'/fixture/a',currentRunId:'a'});window.active=true;window.mount=(readOnly=false,jev=false)=>{document.getElementById('cards').replaceChildren();const result=jev?{advice:{suggestions:[{candidate_id:'rule1',resume_mode:'full_run',updates:[{path:'bge.enabled',value:true}]}]}}:{action_plan:{schema_version:'pi.action-plan.v1',source:'pi.run-chat',mutation_free:true,actions:[{id:'one',type:'config.set',path:'bge.enabled',value:true},{id:'tool',type:'tool.call',tool:'run.start'}]}};for(const plan of assistantActionPlans(result))document.getElementById('cards').append(createAssistantActionCard(plan,{run_uid:'a',run_key:'/fixture/a',context_id:'run:a',readOnly},()=>window.active));};window.mount();</script></body></html>` });
    const file = path.resolve(root, p.replace(/^\/ui\//, '').replace(/^\/i18n\//, 'i18n/'));
    assert.ok(file.startsWith(root + path.sep));
    return route.fulfill({ contentType: file.endsWith('.js') ? 'text/javascript' : file.endsWith('.json') ? 'application/json' : 'text/css', body: await fs.readFile(file) });
  });
  page.on('dialog', d => d.accept());
  await page.goto('https://actions.test/fixture');
  const select = () => page.getByRole('checkbox', { name: 'bge.enabled: true', exact: true }).check();
  const preview = () => page.getByRole('button', { name: 'Validate selection', exact: true }).click();
  const apply = () => page.getByRole('button', { name: 'Transfer to Resume draft', exact: true }).click();
  await page.getByRole('button', { name: 'Validate selection' }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Validate selection' }).isDisabled(), true);
  assert.equal(requests.length, 0, 'rendering is read-only');
  await select(); await preview(); await page.getByText(/Config valid/).waitFor();
  const body = requests.find(x => x.method === 'POST').body;
  assert.equal(body.yaml, source, 'explicit run YAML, never default config');
  assert.equal(body.plan.actions.length, 1, 'tool/run actions cannot enter preview');
  await apply(); await page.getByText(/Transferred to the Resume draft/).waitFor();
  assert.equal(await page.locator('#resume-config-yaml').inputValue(), 'bge:\n  enabled: true\n');
  // New user edits after preview must survive.
  await preview(); await page.getByText(/Config valid/).waitFor();
  await page.locator('#resume-config-yaml').fill('user edited draft');
  await apply(); await page.getByText(/Preview expired or draft\/source changed/).waitFor();
  assert.equal(await page.locator('#resume-config-yaml').inputValue(), 'user edited draft');
  await page.locator('#resume-config-yaml').fill('');
  await preview(); await page.getByText(/Config valid/).waitFor(); expired = true;
  await apply(); await page.getByText(/Preview expired/).waitFor(); assert.equal(await page.locator('#resume-config-yaml').inputValue(), '');
  expired = false; valid = false;
  await preview(); await page.getByText(/Config invalid/).waitFor(); assert.equal(await page.getByRole('button', { name: 'Transfer to Resume draft' }).isDisabled(), true);
  valid = true;
  await preview(); await page.getByText(/Config valid/).waitFor(); source = 'bge:\n  enabled: true\n';
  await apply(); await page.getByText(/Preview expired or draft\/source changed/).waitFor();
  assert.equal(await page.locator('#resume-config-yaml').inputValue(), '', 'changed run file blocks handoff');
  source = 'bge:\n  enabled: false\n';
  delayPreview = true; await preview();
  await page.getByRole('checkbox', { name: 'bge.enabled: true', exact: true }).uncheck();
  await page.waitForTimeout(240); delayPreview = false;
  assert.equal(await page.getByRole('button', { name: 'Transfer to Resume draft' }).isDisabled(), true, 'late preview cannot restore old selection');
  await select(); wrongUid = true;
  await preview(); await page.getByText(/Run context or target editor changed/).waitFor(); wrongUid = false;
  await page.evaluate(() => window.mount(false, true)); await select(); await preview(); await page.getByText(/Config valid/).waitFor();
  await page.getByText(/requires a full run/).waitFor();
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `card fits ${width}`);
  }
  await page.evaluate(() => window.mount(true)); await select(); assert.equal(await page.getByRole('button', { name: 'Validate selection' }).isDisabled(), true);
  assert.ok(requests.every(r => r.method === 'GET' || r.p === '/api/pi/action-plans/preview'), 'no save, apply, run start or Decision writes');
  assert.deepEqual(errors, []);
  console.log('Action-card fixtures passed: explicit selection, run YAML, real draft handoff, stale editor, expiry, invalid config, UID conflict, PI/Jev shared cards, full-run warning, missing artifacts, desktop/mobile, no saved mutation.');
} finally { await browser.close(); }
