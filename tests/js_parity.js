// Run by tests/test_js_parity.py: prints JS predictions for each fixture case.
"use strict";
const fs = require("fs");
const path = require("path");
const M = require(path.join(__dirname, "..", "docs", "model.js"));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", "js_parity.json"), "utf8"));
const out = [];
for (const c of fx.cases) {
  const g = { L: c.L, s: c.s, fuel: c.fuel, moisture: c.moisture, elevation: c.elevation,
    windX: c.wind[0], windY: c.wind[1], state: c.state };
  const pred = [];
  for (let r = 0; r < c.L; r++) for (let k = 0; k < c.L; k++) pred.push(M.predict(c.kind, c.weights, g, r, k));
  out.push(pred);
}
process.stdout.write(JSON.stringify(out));
