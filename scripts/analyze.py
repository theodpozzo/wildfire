"""Aggregate model results and evaluate H1-H5 exactly as pre-registered.
Writes results/summary.json (or $WILDFIRE_OUT/summary.json)."""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from scripts.run_models import SEEDS, kinds_for
from wildfire.families import FAMILIES
from wildfire.fit import t_interval
from wildfire.io import load_result, save_result

MARGIN_EXCELLENT = 0.002
MARGIN_STRICT = 0.0005
H4_MARGIN = 0.02
FLOOR = 0.01
PREREG_COMMIT = "1c697d0338f8384833044f184ef5d878f1c96c83"
H2_KINDS = ["mpnn1", "mpnn2", "structured", "mpnn1_band", "structured_band"]
GNN_DEFAULT = ["mpnn1", "mpnn2", "structured"]
BASELINES = ["never", "persistence", "logreg"]


def welch(a, b):
    """95% CI of mean(a) - mean(b), Welch."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    d = a.mean() - b.mean()
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1)) if se > 0 else 1e9
    t = stats.t.ppf(0.975, df)
    return {"diff": float(d), "lo": float(d - t * se), "hi": float(d + t * se), "df": float(df)}


def overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def main():
    summary = {"prereg_commit": PREREG_COMMIT, "margins": {
        "excellent": MARGIN_EXCELLENT, "strict": MARGIN_STRICT, "h4": H4_MARGIN, "floor": FLOOR}}
    gates = {g: load_result(f"gate_{g}")["result"] for g in ("G1", "G2")}
    summary["gates"] = {g: {k: gates[g][k] for k in ("xc", "ci", "target", "pass", "width_10_90")}
                        | {"n_values": len(gates[g]["x"]), "n_per": gates[g]["n"][0]} for g in gates}
    try:
        _c = load_result("controls")
        controls = _c["result"]
        n_unmasked = _c["config"]["n_unmasked"]
    except FileNotFoundError:
        controls = None
    fams = {}
    for fam in FAMILIES:
        gt = load_result(f"ground_truth_{fam}")["result"]
        truth = gt["L128"]
        thr = max(FLOOR, truth["width_10_90"] / 2)
        F = {"truth": {L: {k: gt[L][k] for k in ("xc", "ci", "ci_width", "width_10_90", "gof_p",
                                                  "clip_frac_at_xc")} | {"n_trials": int(sum(gt[L]["n"]))}
                       for L in ("L128", "L64")},
             "h2_threshold": thr, "models": {}}
        if "corr_length" in truth:
            F["truth"]["L128"]["corr_length"] = truth["corr_length"]
        for kind in BASELINES[:2]:
            r = load_result(f"models/{fam}_{kind}_s0")["result"]
            F["models"][kind] = {"seeds": [{"seed": 0, "xc": None, "status": r["crossing"]["status"],
                                            "test": _slim(r["test"])}],
                                 "accuracy_by_s": r["test"]["accuracy_by_s"]}
        for kind in kinds_for(fam):
            seeds = []
            for s in SEEDS:
                try:
                    r = load_result(f"models/{fam}_{kind}_s{s}")["result"]
                except FileNotFoundError:
                    continue
                cr = r["crossing"]
                xc = cr.get("xc")
                exc = r["test"]["offband"]["excess"]
                seeds.append({
                    "seed": s, "H": r["H"], "xc": xc, "ci": cr.get("ci"), "status": cr.get("status"),
                    "width_10_90": cr.get("width_10_90"), "err": None if xc is None else xc - truth["xc"],
                    "excellent": exc <= MARGIN_EXCELLENT, "strict": exc <= MARGIN_STRICT,
                    "test": _slim(r["test"]),
                    "curve": {"x": cr.get("x"), "k": cr.get("k"), "n": cr.get("n"), "w": cr.get("w")},
                })
            M = {"seeds": seeds}
            xs = [d["xc"] for d in seeds if d["xc"] is not None]
            if len(xs) == len(seeds) and xs:
                M["xc"] = t_interval(xs)
                M["err"] = t_interval([x - truth["xc"] for x in xs])
                M["abs_err_of_mean"] = abs(M["xc"]["mean"] - truth["xc"])
            for part in ("offband", "inband"):
                for met in ("active_brier", "excess", "all_cell_accuracy", "ece"):
                    M[f"{part}_{met}"] = t_interval([d["test"][part][met] for d in seeds])
            F["models"][kind] = M
        fams[fam] = F
    summary["families"] = fams
    if controls:
        summary["controls"] = {f: {"oracle_xc": controls[f]["oracle"]["xc"], "oracle_ci": controls[f]["oracle"]["ci"],
                                   "unmasked": controls[f]["unmasked"]} for f in controls}
        summary["n_unmasked"] = n_unmasked
    summary["n_seeds"] = len(SEEDS)
    summary["H"] = hypotheses(fams)
    save_result("summary", summary, None, {"seeds": SEEDS})
    print(json.dumps(summary["H"], indent=1, default=lambda o: o.item())[:3000])


def _slim(test):
    out = {}
    for part in ("all", "offband", "inband"):
        t = test[part]
        out[part] = {k: t[k] for k in ("all_cell_accuracy", "active_brier", "bayes_floor", "excess", "ece",
                                       "n_active", "mean_pred_active", "mean_true_active")}
        out[part]["calibration"] = t["calibration"]
    return out


def hypotheses(fams):
    H = {}
    # H1
    h1 = {}
    for fam, F in fams.items():
        M = F["models"]
        best = min(GNN_DEFAULT, key=lambda k: M[k]["offband_active_brier"]["mean"])
        b_hi = M[best]["offband_active_brier"]["hi"]
        lr_lo = M["logreg"]["offband_active_brier"]["lo"]
        others = {k: M[k]["seeds"][0]["test"]["offband"]["active_brier"] for k in ("never", "persistence")}
        ok = b_hi < lr_lo and all(b_hi < v for v in others.values())
        h1[fam] = {"best": best, "best_hi": b_hi, "best_mean": M[best]["offband_active_brier"]["mean"],
                   "logreg_lo": lr_lo, "logreg_mean": M["logreg"]["offband_active_brier"]["mean"],
                   "never": others["never"], "persistence": others["persistence"], "supported": ok}
    H["H1"] = {"per_family": h1, "verdict": "supported" if all(v["supported"] for v in h1.values())
               else "not supported"}
    # H2 per seed
    passes, tried, rows = 0, 0, []
    n_excellent = 0
    for fam, F in fams.items():
        tci = F["truth"]["L128"]["ci"]
        thr = F["h2_threshold"]
        for kind in H2_KINDS:
            for d in F["models"][kind]["seeds"]:
                tried += 1
                exc = d["excellent"]
                n_excellent += exc
                if d["xc"] is None:
                    big, sep = True, True
                else:
                    big = abs(d["err"]) > thr
                    sep = not overlap(d["ci"], tci)
                ok = bool(exc and big and sep)
                passes += ok
                rows.append({"family": fam, "kind": kind, "seed": d["seed"], "excellent": exc, "err": d["err"],
                             "big": bool(big), "separated": bool(sep), "pass": ok,
                             "excess": d["test"]["offband"]["excess"]})
    sep_excellent = sum(1 for r in rows if r["excellent"] and r["separated"])
    H["H2"] = {"passes": passes, "tried": tried, "n_excellent": n_excellent, "rows": rows,
               "excellent_and_ci_separated": sep_excellent,
               "verdict": "supported" if passes > 0 else "not supported"}
    # secondary per-model view
    per_model = []
    for fam, F in fams.items():
        tci = F["truth"]["L128"]["ci"]
        for kind in H2_KINDS:
            M = F["models"][kind]
            if "xc" not in M:
                continue
            allexc = all(d["excellent"] for d in M["seeds"])
            per_model.append({"family": fam, "kind": kind, "mean_err": M["err"]["mean"],
                              "t_lo": M["xc"]["lo"], "t_hi": M["xc"]["hi"], "all_excellent": allexc,
                              "pass": bool(allexc and abs(M["err"]["mean"]) > F["h2_threshold"]
                                           and not overlap([M["xc"]["lo"], M["xc"]["hi"]], tci))})
    H["H2"]["per_model"] = per_model
    H["H2"]["per_model_passes"] = sum(r["pass"] for r in per_model)
    # H3
    cells = {}
    for fam, F in fams.items():
        t = F["truth"]["L128"]["xc"]
        for base in ("mpnn1", "structured"):
            Md, Mb = F["models"][base], F["models"][base + "_band"]
            ed, eb = Md.get("abs_err_of_mean"), Mb.get("abs_err_of_mean")
            if ed is None or eb is None:
                cells[f"{fam}/{base}"] = {"verdict": "inconclusive", "reason": "censored"}
                continue
            contains = Md["xc"]["lo"] <= t <= Md["xc"]["hi"]
            if ed < FLOOR or contains:
                v = "inconclusive"
            else:
                v = "supported" if eb <= 0.5 * ed else "not supported"
            cells[f"{fam}/{base}"] = {"err_default": ed, "err_band": eb, "ratio": eb / ed if ed > 0 else None,
                                      "default_ci_contains_truth": contains, "verdict": v}
    conc = [c for c in cells.values() if c["verdict"] != "inconclusive"]
    H["H3"] = {"cells": cells, "verdict": "inconclusive" if not conc else (
        "supported" if all(c["verdict"] == "supported" for c in conc) else "not supported"),
        "n_conclusive": len(conc), "n_supported": sum(c["verdict"] == "supported" for c in conc)}
    # H4
    h4 = {}
    for fam, F in fams.items():
        a = [d["xc"] for d in F["models"]["mpnn2"]["seeds"]]
        b = [d["xc"] for d in F["models"]["mpnn1"]["seeds"]]
        if None in a or None in b:
            h4[fam] = {"verdict": "inconclusive", "reason": "censored"}
            continue
        w = welch(a, b)
        if -H4_MARGIN < w["lo"] and w["hi"] < H4_MARGIN:
            v = "supported"
        elif w["lo"] >= H4_MARGIN or w["hi"] <= -H4_MARGIN:
            v = "not supported"
        else:
            v = "inconclusive"
        h4[fam] = w | {"verdict": v}
    H["H4"] = {"per_family": h4, "verdict": _overall(h4)}
    # H5
    h5 = {}
    for fam, F in fams.items():
        g = [abs(d["err"]) for d in F["models"]["mpnn1"]["seeds"] if d["err"] is not None]
        s = [abs(d["err"]) for d in F["models"]["structured"]["seeds"] if d["err"] is not None]
        w = welch(g, s)
        if np.mean(g) < FLOOR and np.mean(s) < FLOOR:
            v = "inconclusive"
        else:
            v = "supported" if w["lo"] > 0 else "not supported"
        h5[fam] = w | {"mean_abs_err_mpnn1": float(np.mean(g)), "mean_abs_err_structured": float(np.mean(s)),
                       "verdict": v}
    H["H5"] = {"per_family": h5, "verdict": _overall(h5)}
    return H


def _overall(d):
    vs = [v["verdict"] for v in d.values()]
    if all(v == "supported" for v in vs):
        return "supported"
    if any(v == "not supported" for v in vs):
        return "not supported"
    return "inconclusive"


if __name__ == "__main__":
    main()
