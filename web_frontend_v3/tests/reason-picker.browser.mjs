import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE_PATH || "playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  const errors = []; page.on("pageerror", error => errors.push(error.message));
  let reasonRequests = 0;
  await page.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/pi/reason-codes") {
      reasonRequests += 1;
      return route.fulfill({ contentType: "application/json", body: JSON.stringify({
        schema_version: "pi.user-reason-codes.v1", catalog_version: "1",
        domains: { config: [{ code: "artifacts", label_key: "pi.reason.artifacts" }] },
      }) });
    }
    if (url.pathname === "/fixture") return route.fulfill({ contentType: "text/html", body: `<!doctype html><body><div id="root"></div><script type="module">import {loadLocale} from '/ui/js/i18n/i18n.js'; import {createReasonPicker} from '/ui/js/assistant/reason-picker.js'; await loadLocale('en'); window.values=[]; window.picker=createReasonPicker({domain:'config',onChange:value=>window.values.push(value)}); document.querySelector('#root').append(window.picker.element);</script>` });
    const file = path.resolve(root, url.pathname.replace(/^\/ui\//, "").replace(/^\/i18n\//, "i18n/"));
    assert.ok(file.startsWith(root + path.sep));
    return route.fulfill({ contentType: file.endsWith(".js") ? "text/javascript" : file.endsWith(".json") ? "application/json" : "text/css", body: await fs.readFile(file) });
  });
  await page.goto("https://reason-picker.test/fixture");
  const chip = page.getByRole("checkbox", { name: "Artifacts", exact: true });
  await chip.waitFor();
  await chip.check();
  assert.equal(await page.getByRole("checkbox", { name: "Do not provide a reason", exact: true }).isDisabled(), true);
  await page.getByRole("textbox", { name: "Your reason", exact: true }).fill("visible artifact");
  assert.deepEqual(await page.evaluate(() => window.picker.getValue()), {
    reason_codes: ["artifacts"], catalog_version: "1", text: "visible artifact", no_reason_given: false,
  });
  await chip.uncheck();
  await page.getByRole("textbox", { name: "Your reason", exact: true }).fill("");
  await page.getByRole("checkbox", { name: "Do not provide a reason", exact: true }).check();
  assert.deepEqual(await page.evaluate(() => window.picker.getValue()), {
    reason_codes: [], catalog_version: "1", text: "", no_reason_given: true,
  });
  assert.equal(reasonRequests, 1, "catalog is fetched once");
  assert.deepEqual(errors, []);
  console.log("Reason-picker fixture passed: catalog, localized chips, optional text, and no-reason XOR.");
} finally { await browser.close(); }
