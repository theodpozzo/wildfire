/* Demo wiring. No libraries, no network requests, no innerHTML. */
"use strict";

(function () {
  const D = window.WILDFIRE_DATA;
  const Sim = window.WildfireSim;
  const Model = window.WildfireModel;
  const L = 64;
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = function (id) { return document.getElementById(id); };

  const st = {
    fam: "A", kind: "mpnn1", s: 0.57, seed: 1, land: null,
    sim: null, mod: null, running: false, timer: null, t: 0, cursor: [32, 32], painting: false,
  };

  function famParams() { return D.families[st.fam].params; }

  // ------------------------------------------------------------- landscape
  function newLandscape() {
    st.seed = (st.seed * 1103515245 + 12345) >>> 0;
    st.land = Sim.makeLandscape(famParams(), L, st.seed);
    resetFire();
  }

  function resetFire() {
    stop();
    st.t = 0;
    st.sim = { state: new Uint8Array(L * L), rng: Sim.rng(st.seed ^ 0x9e3779b9), lit: false, crossed: false, alive: false };
    st.mod = { state: new Uint8Array(L * L), rng: Sim.rng(st.seed ^ 0x85ebca6b), lit: false, crossed: false, alive: false };
    $("run").textContent = "Light top row";
    status();
    draw();
  }

  function light() {
    for (const f of [st.sim, st.mod]) {
      f.state = Sim.initialState(st.land);
      f.lit = true; f.alive = true; f.crossed = false;
    }
  }

  // ------------------------------------------------------------- stepping
  function grid(state) {
    return { L: L, s: st.s, fuel: st.land.fuel, moisture: st.land.moisture, elevation: st.land.elevation,
      windX: st.land.windX, windY: st.land.windY, state: state };
  }

  function stepOne() {
    if (!st.sim.lit) light();
    const fam = famParams();
    const w = D.families[st.fam].weights[st.kind];
    if (st.sim.alive) {
      const pf = Sim.trueProb(fam, st.land, st.s, st.sim.state);
      st.sim.alive = Sim.step(st.land, st.sim.state, pf, st.sim.rng) > 0;
    }
    if (st.mod.alive) {
      const g = grid(st.mod.state);
      const snapshot = new Uint8Array(st.mod.state);
      g.state = snapshot;
      const pf = function (i) { return Model.predict(st.kind, w, g, Math.floor(i / L), i % L); };
      st.mod.alive = Sim.step(st.land, st.mod.state, pf, st.mod.rng) > 0;
    }
    st.t++;
    for (const f of [st.sim, st.mod]) {
      for (let c = 0; c < L; c++) if (f.state[(L - 1) * L + c] !== 0 && st.land.fuel[(L - 1) * L + c]) f.crossed = true;
    }
    return st.sim.alive || st.mod.alive;
  }

  function loop() {
    if (!st.running) return;
    const more = stepOne();
    draw(); status();
    if (more) st.timer = window.setTimeout(loop, 45);
    else stop();
  }

  function start() {
    if (st.running) { stop(); return; }
    if (!st.sim.lit) light();
    if (reduced) {
      // no animation: run to the end and show the final maps
      let guard = 0;
      while (stepOne() && guard++ < 20000) { /* run */ }
      draw(); status();
      return;
    }
    st.running = true;
    $("run").textContent = "Pause";
    loop();
  }

  function stop() {
    st.running = false;
    if (st.timer) window.clearTimeout(st.timer);
    st.timer = null;
    if (st.sim && st.sim.lit) $("run").textContent = "Continue";
  }

  function status() {
    for (const [f, id] of [[st.sim, "st-sim"], [st.mod, "st-mod"]]) {
      let msg = "not lit";
      if (f.lit) {
        msg = "step " + st.t + (f.crossed ? " · CROSSED the map" : f.alive ? " · burning" : " · died out");
      }
      $(id).textContent = msg;
    }
  }

  // ------------------------------------------------------------- drawing
  const ctx = { sim: $("cv-sim").getContext("2d"), mod: $("cv-mod").getContext("2d") };
  function cellColour(i, state) {
    const s = state[i];
    if (s === 1) return [245, 158, 11];
    if (s === 2) return [46, 42, 39];
    if (!st.land.fuel[i]) return [230, 220, 198];
    const m = st.land.moisture[i];
    // hillshade from elevation gradient (light from the north-west)
    const r = Math.floor(i / L), c = i % L;
    const zN = r > 0 ? st.land.elevation[i - L] : st.land.elevation[i];
    const zW = c > 0 ? st.land.elevation[i - 1] : st.land.elevation[i];
    const shade = Math.max(-0.25, Math.min(0.25, 0.35 * ((st.land.elevation[i] - zN) + (st.land.elevation[i] - zW))));
    const base = [125 - 62 * m, 138 - 31 * m, 78 + 28 * m];
    return base.map(function (v) { return Math.max(0, Math.min(255, v * (1 + shade))); });
  }

  function drawOne(c2d, state) {
    const img = c2d.createImageData(L, L);
    for (let i = 0; i < L * L; i++) {
      const col = cellColour(i, state);
      img.data[4 * i] = col[0]; img.data[4 * i + 1] = col[1]; img.data[4 * i + 2] = col[2]; img.data[4 * i + 3] = 255;
    }
    c2d.putImageData(img, 0, 0);
    // contour-like tick at the cursor
    c2d.strokeStyle = "rgba(29,78,216,0.9)";
    c2d.lineWidth = 0.4;
    c2d.strokeRect(st.cursor[1] + 0.2, st.cursor[0] + 0.2, 0.6, 0.6);
  }

  function draw() {
    drawOne(ctx.sim, st.sim.state);
    drawOne(ctx.mod, st.mod.state);
  }

  // ------------------------------------------------------------- painting
  function brush() { return document.querySelector("input[name=brush]:checked").value; }
  function paintAt(r, c) {
    if (r < 0 || c < 0 || r >= L || c >= L) return;
    const b = brush();
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      const rr = r + dr, cc = c + dc;
      if (rr < 0 || cc < 0 || rr >= L || cc >= L) continue;
      const i = rr * L + cc;
      if (b === "fuel") st.land.fuel[i] = 1;
      if (b === "clear") st.land.fuel[i] = 0;
      if (b === "wet") { st.land.moisture[i] = 1; st.land.fuel[i] = 1; }
      if (b === "dry") { st.land.moisture[i] = 0; st.land.fuel[i] = 1; }
    }
    draw();
  }
  function cellFromEvent(cv, ev) {
    const rect = cv.getBoundingClientRect();
    return [Math.floor((ev.clientY - rect.top) / rect.height * L), Math.floor((ev.clientX - rect.left) / rect.width * L)];
  }
  for (const cv of [$("cv-sim"), $("cv-mod")]) {
    cv.addEventListener("pointerdown", function (ev) {
      st.painting = true; cv.setPointerCapture(ev.pointerId);
      const rc = cellFromEvent(cv, ev); st.cursor = rc; paintAt(rc[0], rc[1]);
    });
    cv.addEventListener("pointermove", function (ev) {
      if (!st.painting) return;
      const rc = cellFromEvent(cv, ev); st.cursor = rc; paintAt(rc[0], rc[1]);
    });
    cv.addEventListener("pointerup", function () { st.painting = false; });
    cv.addEventListener("keydown", function (ev) {
      const mv = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[ev.key];
      if (mv) {
        st.cursor = [Math.max(0, Math.min(L - 1, st.cursor[0] + mv[0])), Math.max(0, Math.min(L - 1, st.cursor[1] + mv[1]))];
        draw(); ev.preventDefault();
      } else if (ev.key === " " || ev.key === "Enter") {
        paintAt(st.cursor[0], st.cursor[1]); ev.preventDefault();
      }
    });
  }

  // ------------------------------------------------------------- chart (SVG, built with DOM calls)
  const NS = "http://www.w3.org/2000/svg";
  function el(name, attrs, parent) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function logistic(x, xc, w) { return 1 / (1 + Math.exp(-(x - xc) / w)); }
  const CH = { W: 640, H: 260, l: 44, r: 14, t: 12, b: 34, x0: 0.3, x1: 1.0 };
  function sx(x) { return CH.l + (x - CH.x0) / (CH.x1 - CH.x0) * (CH.W - CH.l - CH.r); }
  function sy(y) { return CH.t + (1 - y) * (CH.H - CH.t - CH.b); }

  function curvePath(xc, w) {
    let d = "";
    for (let i = 0; i <= 200; i++) {
      const x = CH.x0 + (CH.x1 - CH.x0) * i / 200;
      d += (i ? "L" : "M") + sx(x).toFixed(1) + "," + sy(logistic(x, xc, w)).toFixed(1);
    }
    return d;
  }

  function drawChart() {
    const host = $("chart");
    while (host.firstChild) host.removeChild(host.firstChild);
    const F = D.families[st.fam];
    CH.W = Math.max(320, Math.min(640, host.clientWidth || 640));
    CH.H = CH.W < 480 ? 240 : 260;
    CH.x0 = Math.max(0.3, Math.round((F.truth.xc - 0.2) * 20) / 20);
    CH.x1 = Math.min(1.0, CH.x0 + 0.4);
    const svg = el("svg", { viewBox: "0 0 " + CH.W + " " + CH.H, role: "img",
      "aria-label": "Crossing probability against s for family " + st.fam }, host);
    const ink = getComputedStyle(document.documentElement).getPropertyValue("--ink-2").trim() || "#554c43";
    for (let y = 0; y <= 1.0001; y += 0.25) {
      el("line", { x1: CH.l, x2: CH.W - CH.r, y1: sy(y), y2: sy(y), stroke: ink, "stroke-opacity": 0.18 }, svg);
      const t = el("text", { x: CH.l - 6, y: sy(y) + 4, "text-anchor": "end" }, svg); t.textContent = Math.round(y * 100) + "%";
    }
    const tick = CH.W < 480 ? 0.1 : 0.05;
    for (let x = CH.x0; x <= CH.x1 + 1e-9; x += tick) {
      const t = el("text", { x: sx(x), y: CH.H - 14, "text-anchor": "middle" }, svg); t.textContent = x.toFixed(2);
    }
    const tl = el("text", { x: CH.W - CH.r, y: CH.H - 1, "text-anchor": "end" }, svg); tl.textContent = "base spread s";
    const simCol = getComputedStyle(document.documentElement).getPropertyValue("--sim").trim();
    const modCol = getComputedStyle(document.documentElement).getPropertyValue(st.kind === "mpnn1" ? "--mpnn1" : "--structured").trim();
    for (const sd of F.models[st.kind].seeds) {
      el("path", { d: curvePath(sd.xc, sd.w), fill: "none", stroke: modCol, "stroke-width": 1.5, "stroke-opacity": 0.8 }, svg);
    }
    el("path", { d: curvePath(F.truth.xc, F.truth.w), fill: "none", stroke: simCol, "stroke-width": 2.5 }, svg);
    for (const p of F.truth.points) {
      if (p[0] < CH.x0 || p[0] > CH.x1) continue;
      el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 2.5, fill: simCol }, svg);
    }
    // direct labels
    const lx = el("text", { x: sx(F.truth.xc) - 6, y: sy(0.88), "text-anchor": "end" }, svg);
    lx.textContent = "simulator";
    const mx = el("text", { x: sx(F.models[st.kind].mean_xc) + 6, y: sy(0.12) }, svg);
    mx.textContent = st.kind === "mpnn1" ? "1-hop MPNN" : "structured";
    // marker
    const mxp = sx(Math.max(CH.x0, Math.min(CH.x1, st.s)));
    const m = el("line", { x1: mxp, x2: mxp, y1: CH.t, y2: CH.H - CH.b, stroke: "#c2410c", "stroke-width": 2 }, svg);
    m.setAttribute("stroke-dasharray", "4 3");
    const pSim = logistic(st.s, F.truth.xc, F.truth.w);
    const pMod = F.models[st.kind].seeds.reduce(function (a, sd) { return a + logistic(st.s, sd.xc, sd.w); }, 0) / F.models[st.kind].seeds.length;
    $("readout").textContent = "At s = " + st.s.toFixed(3) + ": simulator fires cross " + Math.round(100 * pSim) +
      "% of the time; the model's fires cross " + Math.round(100 * pMod) + "% (mean of 5 seeds). " +
      "Simulator tipping point " + F.truth.xc.toFixed(3) + ", model " + F.models[st.kind].mean_xc.toFixed(3) + ".";
  }

  function fillTable() {
    const tb = document.querySelector("#curve-table tbody");
    for (const f of Object.keys(D.families)) {
      const F = D.families[f];
      const tr = document.createElement("tr");
      for (const v of [f, F.truth.xc.toFixed(3), F.models.mpnn1.mean_xc.toFixed(3), F.models.structured.mean_xc.toFixed(3)]) {
        const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
      }
      tb.appendChild(tr);
    }
  }

  // ------------------------------------------------------------- wiring
  function setS(v) {
    st.s = v; $("s").value = String(v); $("s-out").textContent = v.toFixed(3); drawChart();
  }
  $("s").addEventListener("input", function () { setS(parseFloat($("s").value)); });
  $("model").addEventListener("change", function () { st.kind = $("model").value; resetFire(); drawChart(); });
  for (const r of document.querySelectorAll("input[name=family]")) {
    r.addEventListener("change", function () {
      st.fam = r.value; newLandscape(); setS(Math.round(D.families[st.fam].truth.xc * 200) / 200);
    });
  }
  $("gen").addEventListener("click", newLandscape);
  window.addEventListener("resize", function () { drawChart(); });
  $("run").addEventListener("click", start);
  $("stepb").addEventListener("click", function () { stop(); stepOne(); draw(); status(); });
  $("reset").addEventListener("click", resetFire);

  $("answer").textContent = D.headline;
  fillTable();
  newLandscape();
  setS(Math.round(D.families.A.truth.xc * 200) / 200);
})();
