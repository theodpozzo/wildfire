"""Exploratory follow-ups ("embers"), written in EMBERS.md before running.

E1: signed bias, not squared error, predicts the tipping-point error.
E2: the same models at L=256: does the error stay while the transition narrows?
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats

from scripts.ground_truth import measure
from scripts.run_models import DATA, sc_true
from wildfire.data import BAND
from wildfire.evaluate import crossing_rule
from wildfire.families import FAMILIES
from wildfire.io import load_result, results_dir, save_result, smoke
from wildfire.models import predict_grid
from wildfire.sim import active_mask

KINDS = ["mpnn1", "mpnn2", "structured", "mpnn1_band", "structured_band"]


def weights(fam, kind, seed):
    w = np.load(results_dir() / "weights" / f"{fam}_{kind}_s{seed}.npz")
    return {k: w[k] for k in w.files}


def inband_bias(fam, kind, seed):
    """Relative signed bias of predicted ignition probability on in-band test active cells."""
    fr = dict(np.load(DATA / f"{fam}_test_frames.npz"))
    m = np.abs(fr["s"] - sc_true(fam)) < BAND
    fields = tuple(fr[k][m] for k in ("fuel", "moisture", "elevation", "wind", "s"))
    pred = predict_grid(kind.replace("_band", ""), weights(fam, kind, seed), fields, fr["state"][m])
    act = active_mask(fr["state"][m], fr["fuel"][m])
    p = fr["p"][m][act].astype(float)
    ph = pred[act]
    return float((ph.mean() - p.mean()) / p.mean())


def e1():
    S = load_result("summary")["result"]
    rows = []
    for fam, F in S["families"].items():
        sc = F["truth"]["L128"]["xc"]
        for kind in KINDS:
            for d in F["models"][kind]["seeds"]:
                if d["err"] is None:
                    continue
                b = inband_bias(fam, kind, d["seed"])
                rows.append({"family": fam, "kind": kind, "seed": d["seed"], "bias": b, "pred_err": -sc * b,
                             "err": d["err"], "excess": d["test"]["offband"]["excess"]})
    pe = np.array([r["pred_err"] for r in rows])
    er = np.array([r["err"] for r in rows])
    ex = np.array([r["excess"] for r in rows])
    r_bias = stats.pearsonr(pe, er)
    rho_exc = stats.spearmanr(ex, np.abs(er))
    slope = float(np.polyfit(pe, er, 1)[0])
    out = {"rows": rows, "pearson_pred_vs_err": float(r_bias.statistic), "pearson_p": float(r_bias.pvalue),
           "slope": slope, "spearman_excess_vs_abs_err": float(rho_exc.statistic),
           "spearman_p": float(rho_exc.pvalue), "n": len(rows)}
    out["supported"] = bool(out["pearson_pred_vs_err"] > 0.5 and 0.5 <= slope <= 2.0
                            and abs(out["spearman_excess_vs_abs_err"]) < out["pearson_pred_vs_err"])
    return out


def e2():
    S = load_result("summary")["result"]
    L = 128 if smoke() else 256
    out = {"L": L, "families": {}}
    for fam, F in S["families"].items():
        # rule: the default-data GNN with the largest |error| at L=128 among seed-0 runs
        cand = [(abs(F["models"][k]["seeds"][0]["err"]), k) for k in ("mpnn1", "mpnn2", "structured")
                if F["models"][k]["seeds"][0]["err"] is not None]
        _, kind = max(cand)
        gt = measure(fam, L, 9000 + list(FAMILIES).index(fam))
        cr = crossing_rule(kind, weights(fam, kind, 0), FAMILIES[fam], gt["xc"], L, seed=91000)
        e128 = F["models"][kind]["seeds"][0]["err"]
        out["families"][fam] = {
            "kind": kind, "truth_xc": gt["xc"], "truth_ci": gt["ci"], "truth_width": gt["width_10_90"],
            "truth_width_128": F["truth"]["L128"]["width_10_90"], "model_xc": cr.get("xc"), "model_ci": cr.get("ci"),
            "err": None if cr.get("xc") is None else cr["xc"] - gt["xc"], "err_128": e128,
        }
        print(fam, out["families"][fam], flush=True)
    ok = []
    for r in out["families"].values():
        persists = r["err"] is not None and abs(r["err"]) >= 0.5 * abs(r["err_128"])
        narrows = r["truth_width"] <= 0.75 * r["truth_width_128"]
        r["error_persists"], r["width_narrows"] = bool(persists), bool(narrows)
        ok.append(persists and narrows)
    out["supported"] = bool(all(ok))
    out["verdict"] = ("Pre-stated criterion (each error at least half its L=128 size, each simulator width at "
                      "most three quarters of its L=128 size): " + ("supported." if all(ok) else "not supported."))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2"]
    res = {}
    if "e1" in which:
        res["e1"] = e1()
        save_result("ember_e1", res["e1"], None, {"kinds": KINDS})
    if "e2" in which:
        res["e2"] = e2()
        save_result("ember_e2", res["e2"], 91000, {})
