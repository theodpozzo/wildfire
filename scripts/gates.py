"""Hard gates G1 (bond percolation) and G2 (site percolation) at L=128."""

import numpy as np

from wildfire.fit import fit_with_ci
from wildfire.io import save_result, smoke
from wildfire.sim import Family, make_landscapes, simulate, spread_q

L = 128
N_PER = 60 if smoke() else 400
SEED = 20261105


def g1():
    fam = Family("G1")
    s_vals = np.round(np.linspace(0.44, 0.56, 13), 4)
    rng = np.random.default_rng(SEED)
    k = []
    for s in s_vals:
        c = 0
        for i in range(0, N_PER, 200):
            b = min(200, N_PER - i)
            ls = make_landscapes(fam, np.full(b, s), L, rng)
            c += int(simulate(spread_q(fam, ls), ls.fuel, rng).crossed.sum())
        k.append(c)
    fit = fit_with_ci(s_vals, k, [N_PER] * len(s_vals), seed=SEED)
    target = 0.5
    fit["target"] = target
    fit["pass"] = bool(abs(fit["xc"] - target) <= 0.010)
    save_result("gate_G1", fit, SEED, {"L": L, "n_per": N_PER, "family": fam.to_dict()})
    print("G1", fit["xc"], fit["ci"], fit["pass"])


def g2():
    fam = Family("G2")
    p_vals = np.round(np.linspace(0.56, 0.625, 14), 4)
    rng = np.random.default_rng(SEED + 1)
    k = []
    for p in p_vals:
        c = 0
        for i in range(0, N_PER, 200):
            b = min(200, N_PER - i)
            ls = make_landscapes(fam, np.ones(b), L, rng, fuel_p=np.full(b, p))
            c += int(simulate(spread_q(fam, ls), ls.fuel, rng).crossed.sum())
        k.append(c)
    fit = fit_with_ci(p_vals, k, [N_PER] * len(p_vals), seed=SEED)
    target = 0.592746
    fit["target"] = target
    fit["pass"] = bool(abs(fit["xc"] - target) <= 0.010)
    save_result("gate_G2", fit, SEED + 1, {"L": L, "n_per": N_PER, "s": 1.0, "family": fam.to_dict()})
    print("G2", fit["xc"], fit["ci"], fit["pass"])


if __name__ == "__main__":
    g1()
    g2()
