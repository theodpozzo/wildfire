// Real-browser check of docs/index.html (Playwright + Chromium). Run: node scripts/browser_check.js OUTDIR
"use strict";
const { chromium } = require("playwright");
const path = require("path");
(async () => {
  const out = process.argv[2] || "repro_out";
  const url = "file://" + path.resolve(__dirname, "..", "docs", "index.html");
  const browser = await chromium.launch();
  const report = {};
  for (const [name, vp, rm] of [["desktop", { width: 1200, height: 900 }, "no-preference"],
                                 ["phone360", { width: 360, height: 780 }, "no-preference"],
                                 ["reduced", { width: 800, height: 900 }, "reduce"]]) {
    const ctx = await browser.newContext({ viewport: vp, reducedMotion: rm, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    const errors = [], requests = [];
    page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(m.text()); });
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("request", (r) => requests.push(r.url()));
    await page.goto(url);
    await page.waitForTimeout(300);
    const before = await page.evaluate(() => document.getElementById("st-sim").textContent);
    const scrollW = await page.evaluate(() => document.documentElement.scrollWidth);
    // keyboard: tab to the run button and press it
    await page.focus("#run");
    await page.keyboard.press("Enter");
    await page.waitForTimeout(rm === "reduce" ? 500 : 2500);
    const after = await page.evaluate(() => [document.getElementById("st-sim").textContent,
      document.getElementById("st-mod").textContent, document.getElementById("readout").textContent,
      document.getElementById("answer").textContent.slice(0, 80)]);
    await page.screenshot({ path: path.join(out, `demo_${name}.png`), fullPage: true });
    report[name] = { before, after, scrollW, viewport: vp.width, errors,
      nonFileRequests: requests.filter((u) => !u.startsWith("file://")) };
    await ctx.close();
  }
  await browser.close();
  console.log(JSON.stringify(report, null, 1));
})();
