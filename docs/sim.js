/* The fire simulator and landscape generator, in plain JS for the demo.
   Same rules as wildfire/sim.py; landscapes are generated the same way in
   distribution (smoothed Gaussian fields), not bit-for-bit. */
"use strict";

(function (root) {
  const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]]; // spread south, north, east, west

  // small seeded PRNG (mulberry32)
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(r) {
    let u = 0;
    while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }

  // standard normal CDF (Abramowitz-Stegun 7.1.26 erf)
  function ncdf(x) {
    const z = Math.abs(x) / Math.SQRT2;
    const t = 1 / (1 + 0.3275911 * z);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
    return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  }

  // smoothed, standardised Gaussian random field (separable Gaussian blur)
  function field(L, sigma, r) {
    const pad = Math.ceil(4 * sigma);
    const n = L + 2 * pad;
    let a = new Float64Array(n * n);
    for (let i = 0; i < a.length; i++) a[i] = gauss(r);
    if (sigma > 0) {
      const R = Math.ceil(3 * sigma);
      const k = [];
      for (let d = -R; d <= R; d++) k.push(Math.exp(-(d * d) / (2 * sigma * sigma)));
      const tmp = new Float64Array(n * n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        let s = 0;
        for (let d = -R; d <= R; d++) { const xx = x + d; if (xx >= 0 && xx < n) s += k[d + R] * a[y * n + xx]; }
        tmp[y * n + x] = s;
      }
      const out = new Float64Array(n * n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        let s = 0;
        for (let d = -R; d <= R; d++) { const yy = y + d; if (yy >= 0 && yy < n) s += k[d + R] * tmp[yy * n + x]; }
        out[y * n + x] = s;
      }
      a = out;
    }
    const f = new Float64Array(L * L);
    let mean = 0;
    for (let y = 0; y < L; y++) for (let x = 0; x < L; x++) { const v = a[(y + pad) * n + x + pad]; f[y * L + x] = v; mean += v; }
    mean /= f.length;
    let v2 = 0;
    for (let i = 0; i < f.length; i++) { f[i] -= mean; v2 += f[i] * f[i]; }
    const sd = Math.sqrt(v2 / f.length) || 1;
    for (let i = 0; i < f.length; i++) f[i] /= sd;
    return f;
  }

  /* fam: parameters exported from wildfire/families.py */
  function makeLandscape(fam, L, seed) {
    const r = rng(seed);
    const N = L * L;
    const fuel = new Uint8Array(N);
    if (fam.fuel_corr > 0) {
      const g = field(L, fam.fuel_corr, r);
      for (let i = 0; i < N; i++) fuel[i] = ncdf(g[i]) < fam.fuel_density ? 1 : 0;
    } else {
      for (let i = 0; i < N; i++) fuel[i] = r() < fam.fuel_density ? 1 : 0;
    }
    const moisture = new Float64Array(N);
    if (fam.moisture_amp > 0) {
      const g = field(L, fam.moisture_corr, r);
      for (let i = 0; i < N; i++) moisture[i] = ncdf(g[i]);
    }
    const elevation = new Float64Array(N);
    if (fam.elev_amp > 0) {
      const g = field(L, fam.elev_corr, r);
      for (let i = 0; i < N; i++) elevation[i] = fam.elev_amp * g[i];
    }
    let windX = 0;
    if (fam.wind_w > 0) windX = (fam.wind_random_sign && r() < 0.5 ? -1 : 1) * fam.wind_w;
    return { L: L, fuel: fuel, moisture: moisture, elevation: elevation, windX: windX, windY: 0 };
  }

  // true spread probability from source (sr, sc) into target (r, c) in direction k
  function q(fam, land, s, k, sr, sc, r, c) {
    const L = land.L;
    const i = r * L + c;
    let v = s * Math.exp(-fam.moisture_amp * land.moisture[i]);
    if (fam.wind_w > 0) v *= DIRS[k][1] !== 0 ? 1 + DIRS[k][1] * land.windX : fam.wind_ns;
    if (fam.slope_beta !== 0) v *= Math.exp(fam.slope_beta * (land.elevation[i] - land.elevation[sr * L + sc]));
    return Math.min(1, Math.max(0, v));
  }

  function initialState(land) {
    const st = new Uint8Array(land.L * land.L);
    for (let c = 0; c < land.L; c++) st[c] = land.fuel[c] ? 1 : 0;
    return st;
  }

  /* active cells: unburnt fuel with a burning 4-neighbour */
  function activeCells(land, st) {
    const L = land.L;
    const out = [];
    for (let r = 0; r < L; r++) for (let c = 0; c < L; c++) {
      const i = r * L + c;
      if (st[i] !== 0 || !land.fuel[i]) continue;
      for (let k = 0; k < 4; k++) {
        const sr = r - DIRS[k][0], sc = c - DIRS[k][1];
        if (sr >= 0 && sc >= 0 && sr < L && sc < L && st[sr * L + sc] === 1) { out.push(i); break; }
      }
    }
    return out;
  }

  /* one step: probFn(i) -> ignition probability of active cell i */
  function step(land, st, probFn, r) {
    const act = activeCells(land, st);
    const ign = [];
    for (const i of act) if (r() < probFn(i)) ign.push(i);
    let burning = 0;
    for (let i = 0; i < st.length; i++) if (st[i] === 1) st[i] = 2;
    for (const i of ign) { st[i] = 1; burning++; }
    return burning;
  }

  function trueProb(fam, land, s, st) {
    const L = land.L;
    return function (i) {
      const r = Math.floor(i / L), c = i % L;
      let notp = 1;
      for (let k = 0; k < 4; k++) {
        const sr = r - DIRS[k][0], sc = c - DIRS[k][1];
        if (sr >= 0 && sc >= 0 && sr < L && sc < L && st[sr * L + sc] === 1) notp *= 1 - q(fam, land, s, k, sr, sc, r, c);
      }
      return 1 - notp;
    };
  }

  root.WildfireSim = { rng: rng, makeLandscape: makeLandscape, initialState: initialState, step: step,
    trueProb: trueProb, activeCells: activeCells, DIRS: DIRS };
  if (typeof module !== "undefined") module.exports = root.WildfireSim;
})(typeof window !== "undefined" ? window : globalThis);
