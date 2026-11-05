import numpy as np

from wildfire.families import FAMILIES
from wildfire.init import shapes
from wildfire.models import DIAMOND, NBR, forward_np, gather_diamond, pad_state, predict_grid, static_grid
from wildfire.rollout import model_crossing, model_rollout
from wildfire.sim import BURNING, active_mask, initial_state, make_landscapes


def rand_params(kind, H=8, seed=0):
    rng = np.random.default_rng(seed)
    p = {}
    for k, (i, o) in shapes(kind, H).items():
        p[k + ".w"] = (rng.standard_normal((i, o)) / np.sqrt(i)).astype(np.float32)
        p[k + ".b"] = (0.1 * rng.standard_normal(o)).astype(np.float32)
    return p


def test_diamond_geometry():
    # neighbour lists are consistent with offsets
    from wildfire.sim import DIRS
    for a in range(5):
        for k, (dr, dc) in enumerate(DIRS):
            assert tuple(DIAMOND[NBR[a, k]]) == (DIAMOND[a][0] - dr, DIAMOND[a][1] - dc)
    assert len({tuple(o) for o in DIAMOND}) == 13


def test_gather_matches_grid():
    fam = FAMILIES["C"]
    rng = np.random.default_rng(0)
    ls = make_landscapes(fam, np.full(2, 0.7), 12, rng)
    st = initial_state(ls.fuel)
    g = static_grid(ls.fuel, ls.moisture, ls.elevation, ls.wind, ls.s)
    X, V = gather_diamond(g, pad_state(st), np.array([1]), np.array([0]), np.array([5]))
    assert V[0, 0] == 1 and V[0, 1] == 0  # north of row 0 is outside
    assert np.isclose(X[0, 0, 2], ls.moisture[1, 0, 5])
    assert X[0, 0, 7] == (st[1, 0, 5] == BURNING)
    # source of eastward spread (edge k=2) is the west neighbour
    assert np.isclose(X[0, NBR[0, 2], 3], ls.elevation[1, 0, 4])


def test_structured_zero_off_active():
    fam = FAMILIES["B"]
    rng = np.random.default_rng(1)
    ls = make_landscapes(fam, np.full(2, 0.7), 10, rng)
    st = initial_state(ls.fuel)
    p = predict_grid("structured", rand_params("structured"), (ls.fuel, ls.moisture, ls.elevation, ls.wind, ls.s), st)
    act = active_mask(st, ls.fuel)
    assert (p[~act] == 0).all() and (p[act] > 0).all()


def test_forward_shapes_all_kinds():
    fam = FAMILIES["A"]
    rng = np.random.default_rng(2)
    ls = make_landscapes(fam, np.full(1, 0.6), 8, rng)
    st = initial_state(ls.fuel)
    fields = (ls.fuel, ls.moisture, ls.elevation, ls.wind, ls.s)
    for kind in ("mpnn1", "mpnn2", "structured"):
        p = predict_grid(kind, rand_params(kind), fields, st)
        assert p.shape == (1, 8, 8) and (p >= 0).all() and (p <= 1).all()
    assert (predict_grid("never", {}, fields, st) == 0).all()
    assert (predict_grid("persistence", {}, fields, st) == (st == BURNING)).all()
    lr = {"w": np.zeros(14, np.float32), "b": np.float32(0.0)}
    assert np.allclose(forward_np("logreg", lr, *gather_diamond(
        static_grid(*fields), pad_state(st), np.array([0]), np.array([1]), np.array([1]), 1)), 0.5)


def test_baselines_never_cross():
    r = model_crossing("never", {}, FAMILIES["A"], [1.0], 8, 16, 0)
    assert r["k"] == [0]
    r = model_crossing("persistence", {}, FAMILIES["A"], [1.0], 8, 16, 0)
    assert r["k"] == [0]


def test_unmasked_rollout_counts_spurious():
    fam = FAMILIES["A"]
    rng = np.random.default_rng(3)
    ls = make_landscapes(fam, np.full(4, 0.6), 16, rng)
    p = rand_params("mpnn1")
    p["l1.upd1.b"][:] = 3.0  # high everywhere -> spurious ignitions
    r = model_rollout("mpnn1", p, ls, rng, masked=False, max_steps=5)
    assert r["spurious"] > 0 and r["nonactive_cell_steps"] > 0
