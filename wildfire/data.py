"""Training / validation / test data from L=64 simulator rollouts."""

from __future__ import annotations

import numpy as np

from wildfire.models import gather_diamond, pad_state, static_grid
from wildfire.sim import BURNING, Family, active_mask, ignition_prob, make_landscapes, simulate, spread_q

L_TRAIN = 64
S_HALF_RANGE = 0.30
BAND = 0.08


def s_sampler(sc: float, which: str, rng, n: int) -> np.ndarray:
    """Draw s for trials. 'default': uniform on [sc-0.3, sc+0.3] minus the band
    |s - sc| < 0.08 (clipped to [0.02, 1]); 'band': uniform in the band;
    'full': uniform on the whole range including the band."""
    lo, hi = max(sc - S_HALF_RANGE, 0.02), min(sc + S_HALF_RANGE, 1.0)
    if which == "band":
        return rng.uniform(sc - BAND, min(sc + BAND, 1.0), n)
    if which == "full":
        return rng.uniform(lo, hi, n)
    a = (lo, sc - BAND)
    b = (sc + BAND, hi)
    la, lb = max(a[1] - a[0], 0), max(b[1] - b[0], 0)
    u = rng.uniform(0, la + lb, n)
    return np.where(u < la, a[0] + u, b[0] + (u - la))


def make_frames(fam: Family, s: np.ndarray, rng, frames_per_trial: int, L: int = L_TRAIN, batch: int = 128):
    """Run full simulator rollouts and keep a random subset of frames per trial.

    Returns dict of arrays: statics per frame and (state_t, burning_{t+1}, p_true)."""
    out = {k: [] for k in ("fuel", "moisture", "elevation", "wind", "s", "state", "y", "p")}
    for i in range(0, s.size, batch):
        sb = s[i : i + batch]
        ls = make_landscapes(fam, sb, L, rng)
        q = spread_q(fam, ls)
        res = simulate(q, ls.fuel, rng, stop_on_cross=False, record=True)
        fr = np.stack(res.frames)  # (T+1, B, L, L)
        T = fr.shape[0] - 1
        for b in range(sb.size):
            alive = [t for t in range(T) if (fr[t, b] == BURNING).any()]
            pick = rng.choice(alive, size=min(frames_per_trial, len(alive)), replace=False)
            for t in pick:
                st = fr[t, b][None]
                out["fuel"].append(ls.fuel[b])
                out["moisture"].append(ls.moisture[b])
                out["elevation"].append(ls.elevation[b])
                out["wind"].append(ls.wind[b])
                out["s"].append(sb[b])
                out["state"].append(fr[t, b])
                out["y"].append(fr[t + 1, b] == BURNING)
                out["p"].append(ignition_prob(q[:, b : b + 1], st, ls.fuel[b : b + 1])[0].astype(np.float32))
    return {k: np.stack(v) for k, v in out.items()}


def frames_to_samples(fr: dict, rng, inactive_ratio: float = 1.0, wind_blind: bool = False, radius: int = 2):
    """Diamond samples: every active cell plus a random set of other cells."""
    act = active_mask(fr["state"], fr["fuel"])
    B, L, _ = act.shape
    ab, ar, ac = np.nonzero(act)
    n_in = int(inactive_ratio * ab.size)
    ib = rng.integers(0, B, n_in)
    ir = rng.integers(0, L, n_in)
    ic = rng.integers(0, L, n_in)
    keep = ~act[ib, ir, ic]
    b = np.concatenate([ab, ib[keep]])
    r = np.concatenate([ar, ir[keep]])
    c = np.concatenate([ac, ic[keep]])
    g = static_grid(fr["fuel"], fr["moisture"], fr["elevation"], fr["wind"], fr["s"], wind_blind)
    X, V = gather_diamond(g, pad_state(fr["state"]), b, r, c, radius)
    return {
        "X": X, "V": V, "y": fr["y"][b, r, c].astype(np.float32), "p": fr["p"][b, r, c],
        "active": act[b, r, c], "s": fr["s"][b],
    }
