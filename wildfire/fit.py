"""Logistic fits of crossing probability against a control parameter."""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

LN9 = float(np.log(9.0))


def _nll(params, x, k, n):
    xc, logw = params
    z = (x - xc) / np.exp(logw)
    # log p = -log(1+e^-z), log(1-p) = -log(1+e^z)
    lp = -np.logaddexp(0, -z)
    lq = -np.logaddexp(0, z)
    return -(k * lp + (n - k) * lq).sum()


def fit_logistic(x, k, n) -> tuple[float, float]:
    """MLE of P(cross) = 1 / (1 + exp(-(x - xc) / w)). Returns (xc, w)."""
    x = np.asarray(x, float)
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    f = k / n
    # initial guess: interpolate 0.5 crossing
    order = np.argsort(x)
    xs, fs = x[order], f[order]
    if (fs > 0.5).any() and (fs < 0.5).any():
        xc0 = float(np.interp(0.5, np.maximum.accumulate(fs), xs))
    else:
        xc0 = float(np.mean(x))
    w0 = max((x.max() - x.min()) / 10, 1e-3)
    best = None
    for w_init in (w0, w0 / 4, w0 * 2):
        r = optimize.minimize(
            _nll, x0=[xc0, np.log(w_init)], args=(x, k, n), method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000},
        )
        if best is None or r.fun < best.fun:
            best = r
    xc, logw = best.x
    return float(xc), float(np.exp(logw))


def fit_with_ci(x, k, n, n_boot: int = 400, seed: int = 0) -> dict:
    """Logistic fit with nonparametric bootstrap CI (trials resampled within each
    control value) and a Pearson goodness-of-fit test."""
    x = np.asarray(x, float)
    k = np.asarray(k, int)
    n = np.asarray(n, int)
    xc, w = fit_logistic(x, k, n)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        kb = rng.binomial(n, k / n)
        try:
            boots.append(fit_logistic(x, kb, n)[0])
        except Exception:  # pragma: no cover
            continue
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 1 / (1 + np.exp(-(x - xc) / w))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    chi2 = float((((k - n * p) ** 2) / (n * p * (1 - p))).sum())
    dof = max(len(x) - 2, 1)
    return {
        "xc": xc,
        "w": w,
        "width_10_90": 2 * LN9 * w,
        "ci": [float(lo), float(hi)],
        "ci_width": float(hi - lo),
        "boot_sd": float(boots.std()),
        "chi2": chi2,
        "dof": dof,
        "gof_p": float(stats.chi2.sf(chi2, dof)),
        "x": x.tolist(),
        "k": k.tolist(),
        "n": n.tolist(),
    }


def merge_counts(*dicts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool counts from several sweeps (same x values are combined)."""
    acc: dict[float, list[int]] = {}
    for d in dicts:
        for xv, kv, nv in zip(d["s"], d["k"], d["n"], strict=False):
            key = round(float(xv), 6)
            a = acc.setdefault(key, [0, 0])
            a[0] += int(kv)
            a[1] += int(nv)
    xs = np.array(sorted(acc))
    k = np.array([acc[x][0] for x in xs])
    n = np.array([acc[x][1] for x in xs])
    return xs, k, n


def t_interval(values, conf: float = 0.95) -> dict:
    v = np.asarray(values, float)
    m = float(v.mean())
    if v.size < 2:
        return {"mean": m, "lo": m, "hi": m, "n": int(v.size), "min": m, "max": m}
    se = v.std(ddof=1) / np.sqrt(v.size)
    t = stats.t.ppf(0.5 + conf / 2, v.size - 1)
    return {
        "mean": m, "lo": m - t * se, "hi": m + t * se, "n": int(v.size),
        "min": float(v.min()), "max": float(v.max()), "sd": float(v.std(ddof=1)),
    }
