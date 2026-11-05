"""Next-step metrics on the fixed test set and the adaptive crossing-curve rule."""

from __future__ import annotations

import numpy as np

from wildfire.data import BAND
from wildfire.fit import fit_with_ci, merge_counts
from wildfire.models import predict_grid
from wildfire.rollout import model_crossing
from wildfire.sim import active_mask

S_MAX_MODEL = 1.3  # models may be swept above s=1 (s is only an input to them)


def test_metrics(kind, params, fr: dict, sc: float, wind_blind=False, n_bins=10) -> dict:
    """All-cell accuracy, active-cell Brier, Bayes floor, excess, calibration.

    Reported for the whole test set and split into off-band (|s - sc| >= 0.08, the
    training distribution) and in-band (|s - sc| < 0.08)."""
    fields = (fr["fuel"], fr["moisture"], fr["elevation"], fr["wind"], fr["s"])
    pred = np.empty(fr["state"].shape)
    for i in range(0, pred.shape[0], 64):
        sl = slice(i, i + 64)
        pred[sl] = predict_grid(kind, params, tuple(f[sl] for f in fields), fr["state"][sl], wind_blind)
    y = fr["y"].astype(float)
    p_true = fr["p"].astype(float)
    act = active_mask(fr["state"], fr["fuel"])
    inband = np.abs(fr["s"] - sc) < BAND
    out = {}
    for name, fm in (("all", np.ones_like(inband)), ("offband", ~inband), ("inband", inband)):
        a = act[fm]
        pa, ya, ta = pred[fm][a], y[fm][a], p_true[fm][a]
        acc = float(((pred[fm] >= 0.5) == (y[fm] > 0.5)).mean())
        brier = float(((pa - ya) ** 2).mean())
        floor_real = float(((ta - ya) ** 2).mean())
        floor = float((ta * (1 - ta)).mean())
        excess = float(((pa - ta) ** 2).mean())
        bins = np.minimum((pa * n_bins).astype(int), n_bins - 1)
        cal = []
        ece = 0.0
        for k in range(n_bins):
            m = bins == k
            if m.any():
                cal.append({"bin": k, "n": int(m.sum()), "pred": float(pa[m].mean()), "obs": float(ya[m].mean()),
                            "true_p": float(ta[m].mean())})
                ece += m.mean() * abs(pa[m].mean() - ya[m].mean())
        out[name] = {
            "all_cell_accuracy": acc, "active_brier": brier, "bayes_floor": floor,
            "bayes_floor_realised": floor_real, "excess": excess, "excess_brier_realised": brier - floor_real,
            "ece": float(ece), "calibration": cal, "n_active": int(a.sum()), "n_cells": int(fm.sum() * act[0].size),
            "mean_pred_active": float(pa.mean()), "mean_true_active": float(ta.mean()),
        }
    # all-cell accuracy by s for the misleading-accuracy control
    edges = np.linspace(fr["s"].min(), fr["s"].max() + 1e-9, 9)
    by_s = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (fr["s"] >= lo) & (fr["s"] < hi)
        if m.any():
            by_s.append({"s_lo": float(lo), "s_hi": float(hi),
                         "acc": float(((pred[m] >= 0.5) == (y[m] > 0.5)).mean())})
    out["accuracy_by_s"] = by_s
    return out


def crossing_rule(kind, params, fam, sc_true, L, seed, n1=64, n2=128, ci_target=0.02, wind_blind=False,
                  s_lo=0.02):
    """Pre-registered adaptive rule for a model's crossing curve (see PREREGISTRATION.md)."""
    lo, hi = max(sc_true - 0.2, s_lo), min(sc_true + 0.2, S_MAX_MODEL)
    sweeps = [model_crossing(kind, params, fam, np.round(np.linspace(lo, hi, 9), 4), n1, L, seed,
                             wind_blind=wind_blind)]
    x, k, n = merge_counts(*sweeps)
    f = k / n
    status = "ok"
    if f.max() < 0.5:
        # does it cross at the largest allowed s?
        extra = model_crossing(kind, params, fam, [S_MAX_MODEL], n1, L, seed + 1, wind_blind=wind_blind)
        sweeps.append(extra)
        x, k, n = merge_counts(*sweeps)
        if (k / n).max() < 0.5:
            return {"status": "never_crosses_below_smax", "sweeps": sweeps, "xc": None}
    if f.min() > 0.5 and lo <= s_lo:
        return {"status": "always_crosses", "sweeps": sweeps, "xc": None}
    fit = fit_with_ci(x, k, n, seed=seed)
    for r in range(2):
        span = 1.2 * max(fit["width_10_90"], 0.02)
        s2 = np.round(np.linspace(max(fit["xc"] - span, s_lo), min(fit["xc"] + span, S_MAX_MODEL), 9), 4)
        sweeps.append(model_crossing(kind, params, fam, s2, n2, L, seed + 10 + r, wind_blind=wind_blind))
        fit = fit_with_ci(*merge_counts(*sweeps), seed=seed)
        if fit["ci_width"] <= ci_target:
            break
    fit["status"] = status
    fit["sweeps"] = sweeps
    return fit
