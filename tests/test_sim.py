import numpy as np

from wildfire.fit import fit_logistic
from wildfire.sim import (
    BURNING,
    DIRS,
    UNBURNT,
    Family,
    active_mask,
    ignition_prob,
    initial_state,
    make_landscapes,
    raw_q,
    simulate,
    spread_q,
)


def test_s1_full_fuel_always_crosses_in_L_steps():
    fam = Family("t")
    rng = np.random.default_rng(0)
    ls = make_landscapes(fam, np.ones(4), 16, rng)
    r = simulate(spread_q(fam, ls), ls.fuel, rng, stop_on_cross=False)
    assert r.crossed.all()
    assert (r.burnt_frac == 1).all()
    assert (r.steps == 16).all()  # front moves one row per step, then dies


def test_s0_never_crosses():
    fam = Family("t")
    rng = np.random.default_rng(0)
    ls = make_landscapes(fam, np.zeros(8), 16, rng)
    r = simulate(spread_q(fam, ls), ls.fuel, rng)
    assert not r.crossed.any()
    assert (r.steps == 1).all()


def test_seeded_reproducible():
    fam = Family("t", fuel_density=0.8)
    out = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        ls = make_landscapes(fam, np.full(16, 0.6), 32, rng)
        out.append(simulate(spread_q(fam, ls), ls.fuel, rng).crossed)
    assert (out[0] == out[1]).all()


def test_empty_cells_never_burn():
    fam = Family("t", fuel_density=0.5)
    rng = np.random.default_rng(1)
    ls = make_landscapes(fam, np.ones(4), 24, rng)
    r = simulate(spread_q(fam, ls), ls.fuel, rng, record=True, stop_on_cross=False)
    for fr in r.frames:
        assert (fr[~ls.fuel] == UNBURNT).all()


def test_burning_lasts_one_step_and_spreads_only_to_neighbours():
    fam = Family("t", fuel_density=0.9)
    rng = np.random.default_rng(3)
    ls = make_landscapes(fam, np.full(4, 0.7), 20, rng)
    r = simulate(spread_q(fam, ls), ls.fuel, rng, record=True, stop_on_cross=False)
    for a, b in zip(r.frames[:-1], r.frames[1:], strict=False):
        assert ((a == BURNING) <= (b == 2)).all()  # burning -> burnt
        newly = (b == BURNING)
        assert (newly <= active_mask(a, ls.fuel)).all()


def test_single_edge_ignition_rate_matches_q():
    # 2-row grid: top row burns, bottom cell ignites with prob q from north only
    fam = Family("t")
    rng = np.random.default_rng(5)
    B, L = 4000, 2
    ls = make_landscapes(fam, np.full(B, 0.3), L, rng)
    q = spread_q(fam, ls)
    r = simulate(q, ls.fuel, rng)
    # each bottom cell has one burning neighbour (north) -> P(cross)=1-(1-.3)^2
    assert abs(r.crossed.mean() - (1 - 0.7**2)) < 0.03


def test_raw_q_factors():
    fam = Family("t", wind_w=0.5, wind_ns=0.6, slope_beta=1.0, elev_amp=1.0, moisture_amp=0.5)
    rng = np.random.default_rng(2)
    ls = make_landscapes(fam, np.full(2, 0.4), 8, rng)
    rq = raw_q(fam, ls)
    b, r, c = 0, 4, 4
    for d, (dr, dc) in enumerate(DIRS):
        src = (r - dr, c - dc)
        wf = fam.wind_ns if dc == 0 else 1 + dc * ls.wind[b, 0]
        expect = 0.4 * np.exp(-0.5 * ls.moisture[b, r, c]) * wf * np.exp(
            ls.elevation[b, r, c] - ls.elevation[b, src[0], src[1]])
        assert np.isclose(rq[d, b, r, c], expect, rtol=1e-5)
    # top row has no source from the north
    assert (rq[0, :, 0, :] == 0).all()


def test_ignition_prob_matches_monte_carlo():
    fam = Family("t", wind_w=0.5, wind_ns=0.6)
    rng = np.random.default_rng(9)
    ls = make_landscapes(fam, np.full(1, 0.5), 6, rng)
    q = spread_q(fam, ls)
    st = initial_state(ls.fuel)
    st[0, 1, 2] = BURNING  # add a second burning neighbour for cell (1,1)? no: (1,2) itself
    p = ignition_prob(q, st, ls.fuel)
    # cell (1,1): burning neighbours north (0,1) and east (1,2) -> spreads south and west
    expect = 1 - (1 - q[0, 0, 1, 1]) * (1 - q[3, 0, 1, 1])
    assert np.isclose(p[0, 1, 1], expect)
    assert p[0, 0, 0] == 0  # burning cell


def test_logistic_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    x = np.linspace(0.3, 0.7, 11)
    n = np.full(11, 4000)
    k = rng.binomial(n, 1 / (1 + np.exp(-(x - 0.52) / 0.03)))
    xc, w = fit_logistic(x, k, n)
    assert abs(xc - 0.52) < 0.003 and abs(w - 0.03) < 0.004
