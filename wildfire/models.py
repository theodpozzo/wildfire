"""Model definitions shared by training (PyTorch), rollouts (NumPy) and the demo (JS).

All learned models see only raw per-cell fields:
    s, fuel, moisture, elevation, wind_x, wind_y, unburnt, burning, burnt
They are never given the simulator's factors or q.

Every model predicts, for a target cell, P(cell is burning at the next step) from
a "diamond" of cells around it: the cell itself, its 4 neighbours and (for 2-hop
models) their neighbours. Out-of-grid cells are marked invalid.

Diamond layout (index -> (drow, dcol) offset from the target):
    0: (0, 0)
    1..4: the source neighbour of edge k, at -DIRS[k]
    5..12: second ring, see DIAMOND below
For a neighbour at diamond index a, NBR[a][k] gives the diamond index of its own
k-th source neighbour (or -1 if outside the diamond; only needed for a = 0..4).
"""

from __future__ import annotations

import numpy as np

from wildfire.sim import BURNING, BURNT, DIRS, UNBURNT

F = 9  # node features
FEATURES = ("s", "fuel", "moisture", "elevation", "wind_x", "wind_y", "unburnt", "burning", "burnt")
# Edge feature for edge k (source -> target): spread direction (drow, dcol)
EDGE = np.array(DIRS, dtype=np.float32)  # (4, 2)

_offsets = [(0, 0)] + [(-dr, -dc) for dr, dc in DIRS]
for a in range(1, 5):
    for dr, dc in DIRS:
        o = (_offsets[a][0] - dr, _offsets[a][1] - dc)
        if o not in _offsets:
            _offsets.append(o)
DIAMOND = np.array(_offsets, dtype=np.int64)  # (13, 2)
NBR = np.full((5, 4), -1, dtype=np.int64)
for a in range(5):
    for k, (dr, dc) in enumerate(DIRS):
        o = (DIAMOND[a][0] - dr, DIAMOND[a][1] - dc)
        NBR[a, k] = _offsets.index(o)
assert DIAMOND.shape == (13, 2) and (NBR[0] == np.arange(1, 5)).all()

PAD = 2


def relu(x):
    return np.maximum(x, 0.0)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# ---------------------------------------------------------------- features
def static_grid(fuel, moisture, elevation, wind, s, wind_blind=False) -> np.ndarray:
    """Padded static feature grid (B, L+4, L+4, 6) and validity (B, L+4, L+4)."""
    B, L, _ = fuel.shape
    g = np.zeros((B, L + 2 * PAD, L + 2 * PAD, 6), np.float32)
    sl = (slice(None), slice(PAD, PAD + L), slice(PAD, PAD + L))
    g[sl + (0,)] = np.asarray(s, np.float32)[:, None, None]
    g[sl + (1,)] = fuel
    g[sl + (2,)] = moisture
    g[sl + (3,)] = elevation
    if not wind_blind:
        g[sl + (4,)] = wind[:, 0][:, None, None]
        g[sl + (5,)] = wind[:, 1][:, None, None]
    return g


def gather_diamond(static_g: np.ndarray, state_p: np.ndarray, b, r, c, radius: int = 2):
    """Build diamond inputs for target cells (b, r, c) in unpadded coordinates.

    static_g: (B, L+4, L+4, 6); state_p: (B, L+4, L+4) int8 padded with -1.
    Returns X (n, m, F) float32 and valid (n, m) float32 with m = 5 or 13."""
    m = 5 if radius == 1 else 13
    off = DIAMOND[:m]
    rr = r[:, None] + PAD + off[None, :, 0]
    cc = c[:, None] + PAD + off[None, :, 1]
    bb = np.broadcast_to(b[:, None], rr.shape)
    st = state_p[bb, rr, cc]
    valid = (st >= 0).astype(np.float32)
    X = np.empty(rr.shape + (F,), np.float32)
    X[..., :6] = static_g[bb, rr, cc]
    X[..., 6] = st == UNBURNT
    X[..., 7] = st == BURNING
    X[..., 8] = st == BURNT
    X *= valid[..., None]
    return X, valid


def pad_state(state: np.ndarray) -> np.ndarray:
    B, L, _ = state.shape
    p = np.full((B, L + 2 * PAD, L + 2 * PAD), -1, np.int8)
    p[:, PAD : PAD + L, PAD : PAD + L] = state
    return p


# ---------------------------------------------------------------- NumPy forward passes
def _mlp(params, prefix, x, n_layers):
    for i in range(n_layers):
        x = x @ params[f"{prefix}{i}.w"] + params[f"{prefix}{i}.b"]
        if i < n_layers - 1:
            x = relu(x)
    return x


def _edge_inputs(hi, hj_list, valid_list):
    """Stack [h_i, h_j, e_k] for the 4 edges: returns (4, n, 2d+2) and (4, n) masks."""
    n = hi.shape[0]
    rows = []
    for k in range(4):
        e = np.broadcast_to(EDGE[k], (n, 2))
        rows.append(np.concatenate([hi, hj_list[k], e], axis=1))
    return np.stack(rows), np.stack(valid_list)


def mpnn_layer(params, prefix, X, valid, center, final):
    """One message-passing step evaluated at diamond index `center`.

    msg = MLP_msg([x_i, x_j, e]) (2 layers, relu after each)
    out = MLP_upd([x_i, sum_k valid_k * msg_k]) (2 layers; relu on hidden; relu on
    output unless final, where the output is a single logit)."""
    hi = X[:, center]
    hj = [X[:, NBR[center, k]] for k in range(4)]
    vj = [valid[:, NBR[center, k]] for k in range(4)]
    E, V = _edge_inputs(hi, hj, vj)
    msg = relu(_mlp(params, prefix + "msg", E, 2))  # (4, n, H)
    agg = (msg * V[..., None]).sum(0)
    out = _mlp(params, prefix + "upd", np.concatenate([hi, agg], axis=1), 2)
    return out if final else relu(out)


def forward_np(kind: str, params: dict, X: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Predicted P(burning next step) for each target, from diamond inputs."""
    n = X.shape[0]
    if kind == "never":
        return np.zeros(n)
    if kind == "persistence":
        return X[:, 0, 7].astype(np.float64)
    if kind == "logreg":
        f = logreg_features(X, valid)
        return sigmoid(f @ params["w"] + params["b"]).astype(np.float64)
    if kind in ("mpnn1", "mpnn1_windblind"):
        return sigmoid(mpnn_layer(params, "l1.", X, valid, 0, True)[:, 0]).astype(np.float64)
    if kind == "mpnn2":
        H = [mpnn_layer(params, "l1.", X, valid, a, False) for a in range(5)]
        Hs = np.stack(H, axis=1)  # (n, 5, H)
        V5 = valid[:, :5]
        out = mpnn_layer(params, "l2.", Hs, V5, 0, True)
        return sigmoid(out[:, 0]).astype(np.float64)
    if kind == "structured":
        xi = X[:, 0]
        notp = np.ones(n)
        for k in range(4):
            j = NBR[0, k]
            e = np.broadcast_to(EDGE[k], (n, 2))
            z = _mlp(params, "edge", np.concatenate([xi, X[:, j], e], axis=1), 3)[:, 0]
            q = sigmoid(z)
            notp *= 1.0 - valid[:, j] * X[:, j, 7] * q
        elig = xi[:, 6] * xi[:, 1]
        return (elig * (1.0 - notp)).astype(np.float64)
    raise ValueError(kind)


def logreg_features(X, valid) -> np.ndarray:
    """Hand-made local features for the logistic-regression baseline."""
    xi = X[:, 0]
    s, m = xi[:, 0], xi[:, 2]
    elig = xi[:, 6] * xi[:, 1]
    nb = np.zeros_like(s)
    ns = np.zeros_like(s)
    dz = np.zeros_like(s)
    wd = np.zeros_like(s)
    for k, (dr, dc) in enumerate(DIRS):
        j = NBR[0, k]
        bj = X[:, j, 7] * valid[:, j]
        nb += bj
        ns += bj * abs(dr)
        dz += bj * (xi[:, 3] - X[:, j, 3])
        wd += bj * (dc * X[:, j, 4] + dr * X[:, j, 5])
    act = elig * (nb > 0)
    cols = [act, act * s, act * nb, act * s * nb, act * m, act * s * m, act * m * nb,
            act * dz, act * s * dz, act * wd, act * s * wd, act * ns, act * s * ns,
            act * s * s * nb]
    return np.stack(cols, axis=1).astype(np.float32)


RADIUS = {"never": 1, "persistence": 1, "logreg": 1, "mpnn1": 1, "mpnn1_windblind": 1,
          "mpnn2": 2, "structured": 1}


def predict_grid(kind, params, ls_fields, state, wind_blind=False, chunk=200000):
    """Predictions on every cell of full grids (B, L, L) -> (B, L, L)."""
    fuel, moisture, elevation, wind, s = ls_fields
    g = static_grid(fuel, moisture, elevation, wind, s, wind_blind)
    sp = pad_state(state)
    B, L, _ = state.shape
    b, r, c = [a.ravel() for a in np.indices((B, L, L))]
    out = np.empty(b.size)
    for i in range(0, b.size, chunk):
        X, V = gather_diamond(g, sp, b[i : i + chunk], r[i : i + chunk], c[i : i + chunk], RADIUS[kind])
        out[i : i + chunk] = forward_np(kind, params, X, V)
    return out.reshape(B, L, L)
