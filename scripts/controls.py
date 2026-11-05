"""Controls: oracle rollouts (true q through the model rollout code) and
unmasked rollouts (spurious ignition rate) for seed 0 of each learned model."""

import numpy as np

from scripts.run_models import L_ROLL, SEEDS, kinds_for, sc_true
from wildfire.families import FAMILIES
from wildfire.fit import fit_with_ci
from wildfire.io import results_dir, save_result, smoke
from wildfire.rollout import model_crossing, model_rollout
from wildfire.sim import make_landscapes

N_OR = 40 if smoke() else 200
N_UNMASKED = 4 if smoke() else 16


def oracle(fam):
    sc = sc_true(fam)
    s = np.round(np.linspace(sc - 0.04, sc + 0.04, 9), 4)
    r = model_crossing("oracle", {"family": FAMILIES[fam]}, FAMILIES[fam], s, N_OR, L_ROLL, 4242)
    fit = fit_with_ci(r["s"], r["k"], r["n"], seed=4242)
    return fit


def unmasked(fam, kind):
    w = np.load(results_dir() / "weights" / f"{fam}_{kind}_s0.npz")
    params = {k: w[k] for k in w.files}
    rng = np.random.default_rng(999)
    ls = make_landscapes(FAMILIES[fam], np.full(N_UNMASKED, sc_true(fam)), L_ROLL, rng)
    bk = kind.replace("_band", "")
    r = model_rollout(bk, params, ls, rng, masked=False, wind_blind=kind == "mpnn1_windblind", max_steps=300)
    rate = r["spurious"] / max(r["nonactive_cell_steps"], 1)
    return {"spurious_ignitions": r["spurious"], "nonactive_cell_steps": r["nonactive_cell_steps"],
            "rate": rate, "crossed_frac": float(r["crossed"].mean()), "steps": r["steps"]}


if __name__ == "__main__":
    out = {}
    for fam in FAMILIES:
        out[fam] = {"oracle": oracle(fam), "unmasked": {}}
        for kind in kinds_for(fam):
            out[fam]["unmasked"][kind] = unmasked(fam, kind)
        print(fam, out[fam]["oracle"]["xc"], flush=True)
    save_result("controls", out, 4242, {"n_oracle": N_OR, "n_unmasked": N_UNMASKED, "L": L_ROLL,
                                       "seeds": SEEDS, "max_steps_unmasked": 300})
