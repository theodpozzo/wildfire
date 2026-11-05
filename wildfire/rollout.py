"""Fire rollouts driven by a model's predicted ignition probabilities."""

from __future__ import annotations

import numpy as np

from wildfire.models import RADIUS, forward_np, gather_diamond, pad_state, static_grid
from wildfire.sim import BURNING, BURNT, DIRS, UNBURNT, Family, initial_state, make_landscapes  # noqa: F401


def _active_cells(state_flat, fuel_flat, burning, B, L):
    N = L * L
    rows = (burning % N) // L
    cols = burning % L
    cand = []
    for dr, dc in DIRS:
        ok = (rows + dr >= 0) & (rows + dr < L) & (cols + dc >= 0) & (cols + dc < L)
        t = burning[ok] + dr * L + dc
        cand.append(t[(state_flat[t] == UNBURNT) & fuel_flat[t]])
    if not cand:
        return np.empty(0, np.int64)
    return np.unique(np.concatenate(cand))


def _oracle_p(qtrue, sf, cand, L, N):
    rows = (cand % N) // L
    cols = cand % L
    notp = np.ones(cand.size)
    for d, (dr, dc) in enumerate(DIRS):
        ok = (rows - dr >= 0) & (rows - dr < L) & (cols - dc >= 0) & (cols - dc < L)
        src = np.where(ok, cand - dr * L - dc, cand)
        burning = ok & (sf[src] == BURNING)
        notp *= 1 - qtrue[d][cand] * burning
    return 1 - notp


def model_rollout(kind, params, ls, rng, masked=True, wind_blind=False, max_steps=None):
    """Roll out B fires using the model. Returns dict with crossed (B,) and, for
    unmasked rollouts, counts of ignitions with no burning neighbour.

    masked: only active cells (unburnt fuel with a burning neighbour) may ignite.
    unmasked: every unburnt fuel cell may ignite with the model's probability."""
    B, L = ls.B, ls.L
    N = L * L
    g = static_grid(ls.fuel, ls.moisture, ls.elevation, ls.wind, ls.s, wind_blind)
    state = initial_state(ls.fuel)
    fuel_flat = ls.fuel.reshape(-1)
    burning = np.flatnonzero(state.reshape(-1) == BURNING)
    crossed = np.zeros(B, bool)
    spurious = 0
    cell_steps = 0
    t = 0
    max_steps = max_steps or 20 * N
    rad = RADIUS.get(kind, 1)
    if kind == "oracle":
        # control: the simulator's true q, driven through this rollout code path
        from wildfire.sim import spread_q

        qtrue = spread_q(params["family"], ls).reshape(4, -1)
    while burning.size and t < max_steps:
        t += 1
        sf = state.reshape(-1)
        act = _active_cells(sf, fuel_flat, burning, B, L)
        if masked:
            cand = act
        else:
            live = np.unique(burning // N)
            m = np.zeros(B, bool)
            m[live] = True
            cand = np.flatnonzero((sf == UNBURNT) & fuel_flat & np.repeat(m, N))
        new = np.empty(0, np.int64)
        if cand.size:
            if kind == "oracle":
                p = _oracle_p(qtrue, sf, cand, L, N)
            else:
                sp = pad_state(state)
                p = forward_np(kind, params, *gather_diamond(g, sp, cand // N, (cand % N) // L, cand % L, rad))
            hit = rng.random(cand.size) < p
            new = cand[hit]
            if not masked:
                isact = np.isin(new, act, assume_unique=True)
                spurious += int((~isact).sum())
                cell_steps += int(cand.size - act.size)
                # spurious ignitions are kept: the rollout is the model's own dynamics
        sf[burning] = BURNT
        if new.size:
            sf[new] = BURNING
            bottom = new[(new % N) // L == L - 1]
            if bottom.size:
                crossed[np.unique(bottom // N)] = True
            new = new[~crossed[new // N]]
        burning = new
    return {"crossed": crossed, "spurious": spurious, "nonactive_cell_steps": cell_steps, "steps": t}


def model_crossing(kind, params, fam: Family, s_values, n_per, L, seed, batch=128, wind_blind=False):
    s_values = np.asarray(s_values, float)
    rng = np.random.default_rng(seed)
    s_all = np.repeat(s_values, n_per)
    crossed = np.zeros(s_all.size, bool)
    for i in range(0, s_all.size, batch):
        ls = make_landscapes(fam, s_all[i : i + batch], L, rng)
        crossed[i : i + batch] = model_rollout(kind, params, ls, rng, wind_blind=wind_blind)["crossed"]
    k = crossed.reshape(len(s_values), n_per).sum(1)
    return {"s": s_values.tolist(), "n": [n_per] * len(s_values), "k": k.astype(int).tolist()}


