"""Ground-truth critical s for each family (L=128 and L=64), adaptive bracketing.

Rule: stage 1, 9 values of s on a coarse range, N1 trials each; logistic fit.
Stage 2: 9 values evenly spaced over xc +/- 1.2 * (10-90 width), N2 trials each.
Then add N2 more trials at the stage-2 values until the 95% bootstrap CI of xc
is no wider than 0.015 (at most 6 extra rounds). All trials are pooled for the fit.
"""

import sys

import numpy as np

from wildfire.families import FAMILIES
from wildfire.fit import fit_with_ci, merge_counts
from wildfire.io import save_result, smoke
from wildfire.sim import crossing_trials, make_landscapes, raw_q

COARSE = {"A": (0.45, 0.75), "B": (0.6, 1.0), "C": (0.55, 0.9)}
N1 = 40 if smoke() else 150
N2 = 60 if smoke() else 300
CI_MAX = 0.015


def corr_length(field: np.ndarray) -> float:
    """1/e distance (cells) of the mean horizontal autocorrelation."""
    f = field - field.mean(axis=(1, 2), keepdims=True)
    var = (f * f).mean()
    for r in range(1, field.shape[1] // 2):
        c = (f[:, :, r:] * f[:, :, :-r]).mean() / var
        if c < np.exp(-1):
            return float(r)
    return float("inf")


def measure(name: str, L: int, seed: int) -> dict:
    fam = FAMILIES[name]
    sweeps = [crossing_trials(fam, np.linspace(*COARSE[name], 9), N1, L, seed)]
    fit = fit_with_ci(*merge_counts(*sweeps), seed=seed)
    span = 1.2 * max(fit["width_10_90"], 0.02)
    s2 = np.round(np.linspace(fit["xc"] - span, fit["xc"] + span, 9), 4)
    rounds = 0
    while True:
        sweeps.append(crossing_trials(fam, s2, N2, L, seed + 1000 + rounds))
        rounds += 1
        fit = fit_with_ci(*merge_counts(*sweeps), seed=seed)
        print(name, L, rounds, round(fit["xc"], 4), round(fit["ci_width"], 4), flush=True)
        if fit["ci_width"] <= CI_MAX or rounds >= 7 or smoke():
            break
    fit["rounds"] = rounds
    fit["stage2_s"] = s2.tolist()
    # diagnostics at the critical point
    rng = np.random.default_rng(seed + 7)
    ls = make_landscapes(fam, np.full(64, fit["xc"]), L, rng)
    rq = raw_q(fam, ls)
    nz = rq[(rq > 0) & ls.fuel[None]]
    fit["clip_frac_at_xc"] = float((nz > 1).mean())
    fit["q_quantiles_at_xc"] = np.quantile(nz, [0.01, 0.5, 0.99]).tolist()
    if fam.fuel_corr > 0 or fam.moisture_amp > 0:
        from wildfire.sim import gaussian_field
        fit["corr_length"] = {
            "fuel": corr_length(ls.fuel.astype(float)),
            "moisture": corr_length(ls.moisture),
            "elevation": corr_length(ls.elevation),
            "L_over_10": L / 10,
        }
        del gaussian_field
    fit["fuel_fraction"] = float(ls.fuel.mean())
    return fit


if __name__ == "__main__":
    names = sys.argv[1:] or list(FAMILIES)
    for name in names:
        i = list(FAMILIES).index(name)
        out = {}
        for L in (128, 64):
            out[f"L{L}"] = measure(name, L, 100 * (i + 1) + L)
        save_result(f"ground_truth_{name}", out, 100 * (i + 1),
                    {"family": FAMILIES[name].to_dict(), "N1": N1, "N2": N2, "ci_max": CI_MAX,
                     "coarse": COARSE[name]})
