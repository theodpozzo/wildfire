/* Hand-written inference for the trained models. Mirrors wildfire/models.py
   (forward_np) exactly; tests/test_js_parity.py checks agreement within 1e-5. */
"use strict";

(function (root) {
  // spread directions (drow, dcol): south, north, east, west
  const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  const F = 9;

  function relu(x) { return x > 0 ? x : 0; }
  function sigmoid(x) { return 0.5 * (1 + Math.tanh(0.5 * x)); }

  // dense layer: x (array len i), W stored row-major as nested [i][o]
  function dense(x, W, b, act) {
    const o = b.length;
    const out = new Float64Array(o);
    for (let k = 0; k < o; k++) out[k] = b[k];
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = W[i];
      for (let k = 0; k < o; k++) out[k] += xi * row[k];
    }
    if (act) for (let k = 0; k < o; k++) out[k] = relu(out[k]);
    return out;
  }

  function mlp(p, prefix, x, n) {
    for (let i = 0; i < n; i++) {
      x = dense(x, p[prefix + i + ".w"], p[prefix + i + ".b"], i < n - 1);
    }
    return x;
  }

  function concat(a, b, c) {
    const out = new Float64Array(a.length + b.length + (c ? c.length : 0));
    out.set(a, 0); out.set(b, a.length);
    if (c) out.set(c, a.length + b.length);
    return out;
  }

  /* Grid accessor. g = {L, s, fuel, moisture, elevation, windX, windY, state}
     with flat typed arrays of length L*L; state: 0 unburnt, 1 burning, 2 burnt. */
  function nodeFeatures(g, r, c) {
    const x = new Float64Array(F);
    if (r < 0 || c < 0 || r >= g.L || c >= g.L) return { x: x, valid: 0 };
    const i = r * g.L + c;
    const st = g.state[i];
    x[0] = g.s; x[1] = g.fuel[i]; x[2] = g.moisture[i]; x[3] = g.elevation[i];
    x[4] = g.windX; x[5] = g.windY;
    x[6] = st === 0 ? 1 : 0; x[7] = st === 1 ? 1 : 0; x[8] = st === 2 ? 1 : 0;
    // values are float32 in Python: round the same way
    for (let k = 0; k < F; k++) x[k] = Math.fround(x[k]);
    return { x: x, valid: 1 };
  }

  function edgeVec(k) { return new Float64Array(DIRS[k]); }

  // one message-passing layer evaluated at node (r, c); h(r, c) gives {x, valid}
  function layer(p, prefix, h, r, c, final) {
    const hi = h(r, c).x;
    let agg = null;
    for (let k = 0; k < 4; k++) {
      const nj = h(r - DIRS[k][0], c - DIRS[k][1]);
      const m = mlp(p, prefix + "msg", concat(hi, nj.x, edgeVec(k)), 2);
      if (!agg) agg = new Float64Array(m.length);
      if (nj.valid) for (let u = 0; u < m.length; u++) agg[u] += relu(m[u]);
    }
    const out = mlp(p, prefix + "upd", concat(hi, agg), 2);
    if (!final) for (let u = 0; u < out.length; u++) out[u] = relu(out[u]);
    return out;
  }

  /* Probability that cell (r, c) is burning at the next step. */
  function predict(kind, p, g, r, c) {
    const nf = function (rr, cc) { return nodeFeatures(g, rr, cc); };
    if (kind === "never") return 0;
    if (kind === "mpnn1") return sigmoid(layer(p, "l1.", nf, r, c, true)[0]);
    if (kind === "mpnn2") {
      const cache = new Map();
      const h1 = function (rr, cc) {
        const key = rr * 100003 + cc;
        if (!cache.has(key)) {
          const v = nf(rr, cc).valid;
          cache.set(key, { x: layer(p, "l1.", nf, rr, cc, false), valid: v });
        }
        return cache.get(key);
      };
      return sigmoid(layer(p, "l2.", h1, r, c, true)[0]);
    }
    if (kind === "structured") {
      const xi = nf(r, c).x;
      let notp = 1;
      for (let k = 0; k < 4; k++) {
        const nj = nf(r - DIRS[k][0], c - DIRS[k][1]);
        const z = mlp(p, "edge", concat(xi, nj.x, edgeVec(k)), 3)[0];
        notp *= 1 - nj.valid * nj.x[7] * sigmoid(z);
      }
      return xi[6] * xi[1] * (1 - notp);
    }
    throw new Error("unknown model kind " + kind);
  }

  root.WildfireModel = { predict: predict, DIRS: DIRS };
  if (typeof module !== "undefined") module.exports = root.WildfireModel;
})(typeof window !== "undefined" ? window : globalThis);
