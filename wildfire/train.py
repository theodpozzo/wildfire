"""Training of the learned models with NumPy and hand-written reverse-mode
gradients (wildfire/nn.py). The graph mirrors wildfire.models.forward_np exactly;
evaluation always uses forward_np on the exported weights."""

from __future__ import annotations

import time

import numpy as np

from wildfire import nn
from wildfire.init import shapes
from wildfire.models import EDGE, NBR, logreg_features

N_LOGREG = 14


def init_params(kind: str, H: int, rng: np.random.Generator) -> dict:
    if kind == "logreg":
        return {"w": np.zeros(N_LOGREG, np.float32), "b": np.array(-6.0, np.float32)}
    p = {}
    for name, (i, o) in shapes(kind, H).items():
        bound = 1 / np.sqrt(i)
        p[name + ".w"] = rng.uniform(-bound, bound, (i, o)).astype(np.float32)
        p[name + ".b"] = rng.uniform(-bound, bound, o).astype(np.float32)
    return p


def _mlp(P, prefix, x, n):
    for i in range(n):
        x = nn.add(nn.matmul(x, P[f"{prefix}{i}.w"]), P[f"{prefix}{i}.b"])
        if i < n - 1:
            x = nn.relu(x)
    return x


def _layer(P, prefix, X, V, center, final):
    """X: T of shape (n, m, d); V: array (n, m)."""
    hi = nn.index(X, (slice(None), center))
    n = X.v.shape[0]
    rows = []
    for k in range(4):
        hj = nn.index(X, (slice(None), int(NBR[center, k])))
        e = nn.const(np.broadcast_to(EDGE[k], (n, 2)).astype(np.float32))
        rows.append(nn.concat([hi, hj, e], axis=1))
    E = nn.stack(rows, 0)  # (4, n, 2d+2)
    Vk = np.stack([V[:, NBR[center, k]] for k in range(4)])[..., None].astype(np.float32)
    msg = nn.relu(_mlp(P, prefix + "msg", E, 2))
    agg = nn.sum_(nn.mul(msg, nn.const(Vk)), 0)
    out = _mlp(P, prefix + "upd", nn.concat([hi, agg], axis=1), 2)
    return out if final else nn.relu(out)


def forward_graph(kind, P, X, V, feats=None):
    """Returns (logit T or None, prob T)."""
    if kind == "logreg":
        z = nn.add(nn.matmul(nn.const(feats), P["w"]), P["b"])
        return z, nn.sigmoid(z)
    Xt = nn.const(X)
    if kind in ("mpnn1", "mpnn1_windblind"):
        z = nn.index(_layer(P, "l1.", Xt, V, 0, True), (slice(None), 0))
        return z, nn.sigmoid(z)
    if kind == "mpnn2":
        Hs = nn.stack([_layer(P, "l1.", Xt, V, a, False) for a in range(5)], 1)
        z = nn.index(_layer(P, "l2.", Hs, V[:, :5], 0, True), (slice(None), 0))
        return z, nn.sigmoid(z)
    if kind == "structured":
        n = X.shape[0]
        xi = nn.index(Xt, (slice(None), 0))
        notp = None
        for k in range(4):
            j = int(NBR[0, k])
            xj = nn.index(Xt, (slice(None), j))
            e = nn.const(np.broadcast_to(EDGE[k], (n, 2)).astype(np.float32))
            zz = nn.index(_mlp(P, "edge", nn.concat([xi, xj, e], axis=1), 3), (slice(None), 0))
            q = nn.sigmoid(zz)
            gate = (V[:, j] * X[:, j, 7]).astype(np.float32)
            term = nn.one_minus(nn.mul(q, nn.const(gate)))
            notp = term if notp is None else nn.mul(notp, term)
        elig = (X[:, 0, 6] * X[:, 0, 1]).astype(np.float32)
        p = nn.mul(nn.one_minus(notp), nn.const(elig))
        return None, p
    raise ValueError(kind)


def loss_graph(kind, P, X, V, y, act, feats=None):
    z, p = forward_graph(kind, P, X, V, feats)
    bce = nn.bce_with_logits(z, y) if z is not None else nn.bce_prob(p, y)
    a = act.astype(np.float32)
    return nn.add(nn.weighted_mean(bce, a), nn.weighted_mean(bce, 1 - a))


def loss_and_grads(kind, params, X, V, y, act, feats=None):
    P = {k: nn.T(v) for k, v in params.items()}
    loss = loss_graph(kind, P, X, V, y, act, feats)
    nn.backward(loss)
    return float(loss.v), {k: (t.g if t.g is not None else np.zeros_like(t.v)) for k, t in P.items()}


def active_brier(kind, params, S) -> float:
    from wildfire.models import forward_np

    m = S["active"]
    p = forward_np(kind, params, S["X"][m], S["V"][m])
    return float(((p - S["y"][m]) ** 2).mean())


def train(kind: str, S: dict, Sval: dict, H: int, seed: int, steps: int, lr: float, batch: int = 2048,
          log=print) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    params = init_params(kind, H, rng)
    m = 13 if kind == "mpnn2" else 5
    X, V, y, act = S["X"], S["V"], S["y"], S["active"]
    feats = logreg_features(X, V) if kind == "logreg" else None
    opt = nn.Adam(params, lr=lr)
    n = X.shape[0]
    t0 = time.time()
    hist = []
    for step in range(steps):
        idx = rng.integers(0, n, batch)
        f = feats[idx] if feats is not None else None
        loss, grads = loss_and_grads(kind, params, X[idx, :m], V[idx, :m], y[idx], act[idx], f)
        lr_t = 0.5 * lr * (1 + np.cos(np.pi * step / steps))  # cosine decay
        opt.step(grads, lr_t)
        if (step + 1) % max(steps // 5, 1) == 0:
            vb = active_brier(kind, params, Sval)
            hist.append({"step": step + 1, "loss": loss, "val_active_brier": vb})
            log(f"{kind} H={H} seed={seed} step {step+1} loss {loss:.4f} val {vb:.5f} {time.time()-t0:.0f}s")
    return {k: np.asarray(v, np.float32) for k, v in params.items()}, {
        "history": hist, "train_seconds": time.time() - t0}
