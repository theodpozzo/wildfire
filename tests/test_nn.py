import numpy as np
import pytest

from wildfire import nn
from wildfire.models import forward_np
from wildfire.train import forward_graph, init_params, loss_and_grads, loss_graph

KINDS = ["mpnn1", "mpnn2", "structured", "logreg"]


def batch(seed=0, n=12):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 13, 9)).astype(np.float64)
    X[..., 6:] = 0
    st = rng.integers(0, 3, (n, 13))
    X[np.arange(n)[:, None], np.arange(13)[None], 6 + st] = 1
    X[..., 1] = (rng.random((n, 13)) < 0.8)
    V = (rng.random((n, 13)) < 0.9).astype(np.float64)
    V[:, 0] = 1
    X *= V[..., None]
    y = (rng.random(n) < 0.5).astype(np.float64)
    act = rng.random(n) < 0.5
    return X, V, y, act


@pytest.mark.parametrize("kind", KINDS)
def test_graph_matches_numpy_forward(kind):
    X, V, y, act = batch()
    p = init_params(kind, 6, np.random.default_rng(1))
    if kind == "logreg":
        p["w"] = np.random.default_rng(2).standard_normal(14).astype(np.float32)
    from wildfire.models import logreg_features
    m = 13 if kind == "mpnn2" else 5
    _, pg = forward_graph(kind, {k: nn.T(v) for k, v in p.items()}, X[:, :m], V[:, :m],
                          logreg_features(X, V))
    assert np.allclose(pg.v, forward_np(kind, p, X, V), atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_match_finite_differences(kind):
    X, V, y, act = batch(3)
    rng = np.random.default_rng(4)
    p = {k: v.astype(np.float64) for k, v in init_params(kind, 5, rng).items()}
    if kind == "logreg":
        p["w"] = rng.standard_normal(14) * 0.3
        p["b"] = np.array(0.1)
    from wildfire.models import logreg_features
    m = 13 if kind == "mpnn2" else 5
    f = logreg_features(X, V).astype(np.float64)
    _, grads = loss_and_grads(kind, p, X[:, :m], V[:, :m], y, act, f)

    def L(pp):
        return float(loss_graph(kind, {k: nn.T(v) for k, v in pp.items()}, X[:, :m], V[:, :m], y, act, f).v)

    eps = 1e-6
    for k, v in p.items():
        flat = v.reshape(-1)
        for i in rng.choice(flat.size, size=min(5, flat.size), replace=False):
            pp = {kk: vv.copy() for kk, vv in p.items()}
            pp[k].reshape(-1)[i] += eps
            lp = L(pp)
            pp[k].reshape(-1)[i] -= 2 * eps
            lm = L(pp)
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[k].reshape(-1)[i]) < 1e-5 + 1e-4 * abs(num), (k, i)
