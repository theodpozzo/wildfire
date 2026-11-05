"""Stochastic grid fire simulator.

Cells sit on an L x L grid with 4-neighbour connectivity. Each cell is fuel or
empty, and unburnt, burning or burnt. A burning cell burns for exactly one step.
Each burning neighbour j independently ignites an unburnt fuel cell i with
probability

    q = clip(s * moisture_factor(i) * wind_factor(dir) * slope_factor(j -> i), 0, 1)

where dir is the direction of spread from j to i. The whole top row's fuel
ignites at step 0. A run "crosses" if fire ever reaches the bottom row.

Many independent trials run together as a batch; only cells adjacent to the
burning front are touched each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UNBURNT, BURNING, BURNT = 0, 1, 2

# Direction of spread (from source j to target i), as (drow, dcol).
# Index order is used everywhere: 0 = spreads south, 1 = spreads north,
# 2 = spreads east, 3 = spreads west.
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR_NAMES = ("south", "north", "east", "west")


@dataclass
class Family:
    """Parameters of a scenario family (the ground-truth distribution)."""

    name: str
    fuel_density: float = 1.0
    fuel_corr: float = 0.0  # Gaussian smoothing sigma (cells) of fuel field; 0 = iid
    moisture_amp: float = 0.0  # moisture_factor = exp(-moisture_amp * m), m in [0, 1]
    moisture_corr: float = 3.0
    slope_beta: float = 0.0  # slope_factor = exp(slope_beta * (z_target - z_source))
    elev_amp: float = 0.0  # elevation field std (arbitrary units)
    elev_corr: float = 3.0
    wind_w: float = 0.0  # downwind factor 1 + w, upwind 1 - w
    wind_ns: float = 1.0  # factor for spread perpendicular to the wind (north/south)
    wind_random_sign: bool = False  # wind blows east or west with equal probability
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Landscapes:
    """A batch of B landscapes of size L x L (raw fields only)."""

    fuel: np.ndarray  # (B, L, L) bool
    moisture: np.ndarray  # (B, L, L) float32 in [0, 1]
    elevation: np.ndarray  # (B, L, L) float32
    wind: np.ndarray  # (B, 2) float32: (east component, south component)
    s: np.ndarray  # (B,) float64

    @property
    def B(self) -> int:
        return self.fuel.shape[0]

    @property
    def L(self) -> int:
        return self.fuel.shape[1]


def gaussian_field(rng: np.random.Generator, B: int, L: int, sigma: float) -> np.ndarray:
    """Zero-mean, unit-variance Gaussian random field smoothed with a Gaussian
    kernel of width sigma (cells). Generated on a padded periodic grid and cropped
    so that opposite edges are not correlated."""
    if sigma <= 0:
        return rng.standard_normal((B, L, L)).astype(np.float32)
    pad = int(np.ceil(4 * sigma))
    n = L + 2 * pad
    white = rng.standard_normal((B, n, n))
    k = np.fft.fftfreq(n)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    filt = np.exp(-2 * (np.pi**2) * (sigma**2) * k2)
    f = np.fft.ifft2(np.fft.fft2(white) * filt).real[:, pad : pad + L, pad : pad + L]
    f = f - f.mean(axis=(1, 2), keepdims=True)
    f = f / (f.std(axis=(1, 2), keepdims=True) + 1e-12)
    return f.astype(np.float32)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(x)


def make_landscapes(
    fam: Family, s: np.ndarray, L: int, rng: np.random.Generator, fuel_p: np.ndarray | None = None
) -> Landscapes:
    """Draw fresh landscapes. fuel_p optionally overrides fuel density per trial."""
    s = np.asarray(s, dtype=np.float64)
    B = s.shape[0]
    dens = np.full(B, fam.fuel_density) if fuel_p is None else np.asarray(fuel_p, dtype=np.float64)
    if fam.fuel_corr > 0:
        g = gaussian_field(rng, B, L, fam.fuel_corr)
        u = _norm_cdf(g)  # marginally uniform
        fuel = u < dens[:, None, None]
    else:
        fuel = rng.random((B, L, L)) < dens[:, None, None]
    if fam.moisture_amp > 0:
        moisture = _norm_cdf(gaussian_field(rng, B, L, fam.moisture_corr)).astype(np.float32)
    else:
        moisture = np.zeros((B, L, L), np.float32)
    if fam.elev_amp > 0:
        elevation = (fam.elev_amp * gaussian_field(rng, B, L, fam.elev_corr)).astype(np.float32)
    else:
        elevation = np.zeros((B, L, L), np.float32)
    wind = np.zeros((B, 2), np.float32)
    if fam.wind_w > 0:
        sign = np.where(rng.random(B) < 0.5, -1.0, 1.0) if fam.wind_random_sign else np.ones(B)
        wind[:, 0] = sign * fam.wind_w
    return Landscapes(fuel=fuel, moisture=moisture, elevation=elevation, wind=wind, s=s)


def raw_q(fam: Family, ls: Landscapes) -> np.ndarray:
    """Unclipped spread probability into every cell from each of the 4 directions.

    Returns (4, B, L, L): entry [d, b, r, c] is the raw q for spread in direction
    DIRS[d] from the source neighbour (r - dr, c - dc) into target (r, c).
    Entries whose source lies outside the grid are set to 0.
    """
    B, L = ls.B, ls.L
    mf = np.exp(-fam.moisture_amp * ls.moisture.astype(np.float64))
    base = ls.s[:, None, None] * mf
    z = ls.elevation.astype(np.float64)
    out = np.zeros((4, B, L, L))
    for d, (dr, dc) in enumerate(DIRS):
        # wind factor for this spread direction
        if fam.wind_w > 0:
            # wind[:,0] = +w (east) or -w (west). Spread east has dc = +1.
            if dc != 0:
                wf = 1.0 + dc * ls.wind[:, 0].astype(np.float64)
            else:
                wf = np.full(B, fam.wind_ns)
        else:
            wf = np.ones(B)
        # slope: source elevation shifted onto target
        zsrc = np.zeros_like(z)
        valid = np.zeros((L, L), bool)
        rs = slice(max(dr, 0), L + min(dr, 0))
        cs = slice(max(dc, 0), L + min(dc, 0))
        rsrc = slice(max(-dr, 0), L + min(-dr, 0))
        csrc = slice(max(-dc, 0), L + min(-dc, 0))
        zsrc[:, rs, cs] = z[:, rsrc, csrc]
        valid[rs, cs] = True
        sf = np.exp(fam.slope_beta * (z - zsrc)) if fam.slope_beta != 0 else 1.0
        out[d] = base * wf[:, None, None] * sf * valid[None]
    return out


def spread_q(fam: Family, ls: Landscapes) -> np.ndarray:
    """Clipped spread probabilities, float32, shape (4, B, L, L)."""
    return np.clip(raw_q(fam, ls), 0.0, 1.0).astype(np.float32)


def ignition_prob(q: np.ndarray, state: np.ndarray, fuel: np.ndarray) -> np.ndarray:
    """True probability that each cell is burning at the next step.

    q: (4, B, L, L); state: (B, L, L) int8; fuel: (B, L, L) bool.
    Returns (B, L, L) float64: 1 - prod_d (1 - q_d * burning(source_d)) on unburnt
    fuel, 0 elsewhere."""
    burning = state == BURNING
    L = state.shape[1]
    notp = np.ones(state.shape, np.float64)
    for d, (dr, dc) in enumerate(DIRS):
        src = np.zeros_like(burning)
        rs = slice(max(dr, 0), L + min(dr, 0))
        cs = slice(max(dc, 0), L + min(dc, 0))
        rsrc = slice(max(-dr, 0), L + min(-dr, 0))
        csrc = slice(max(-dc, 0), L + min(-dc, 0))
        src[:, rs, cs] = burning[:, rsrc, csrc]
        notp *= 1.0 - q[d].astype(np.float64) * src
    p = 1.0 - notp
    return p * ((state == UNBURNT) & fuel)


def active_mask(state: np.ndarray, fuel: np.ndarray) -> np.ndarray:
    """Unburnt fuel cells with at least one burning 4-neighbour."""
    b = state == BURNING
    nb = np.zeros_like(b)
    nb[:, 1:, :] |= b[:, :-1, :]
    nb[:, :-1, :] |= b[:, 1:, :]
    nb[:, :, 1:] |= b[:, :, :-1]
    nb[:, :, :-1] |= b[:, :, 1:]
    return nb & (state == UNBURNT) & fuel


def initial_state(fuel: np.ndarray) -> np.ndarray:
    state = np.zeros(fuel.shape, np.int8)
    state[:, 0, :] = np.where(fuel[:, 0, :], BURNING, UNBURNT)
    return state


@dataclass
class RunResult:
    crossed: np.ndarray  # (B,) bool
    steps: np.ndarray  # (B,) number of steps until extinction (or crossing if stopped)
    burnt_frac: np.ndarray  # (B,) fraction of fuel cells that burned
    frames: list | None = None  # list of (B, L, L) int8 states, if recorded


def simulate(
    q: np.ndarray,
    fuel: np.ndarray,
    rng: np.random.Generator,
    stop_on_cross: bool = True,
    record: bool = False,
    max_steps: int | None = None,
) -> RunResult:
    """Run B independent fires. q: (4, B, L, L) clipped spread probabilities.

    Only the burning front is processed each step (sparse frontier update)."""
    B, L, _ = fuel.shape
    N = L * L
    state = initial_state(fuel).reshape(-1)
    fuel_flat = fuel.reshape(-1)
    qf = [q[d].reshape(-1) for d in range(4)]
    burning = np.flatnonzero(state == BURNING)
    crossed = np.zeros(B, bool)
    steps = np.zeros(B, np.int64)
    frames = [state.reshape(B, L, L).copy()] if record else None
    offsets = [dr * L + dc for dr, dc in DIRS]
    t = 0
    max_steps = max_steps or 50 * N
    while burning.size and t < max_steps:
        t += 1
        rows = (burning % N) // L
        cols = burning % L
        new = []
        for d, (dr, dc) in enumerate(DIRS):
            ok = np.ones(burning.size, bool)
            if dr == 1:
                ok &= rows < L - 1
            elif dr == -1:
                ok &= rows > 0
            if dc == 1:
                ok &= cols < L - 1
            elif dc == -1:
                ok &= cols > 0
            tgt = burning[ok] + offsets[d]
            tgt = tgt[(state[tgt] == UNBURNT) & fuel_flat[tgt]]
            if tgt.size:
                hit = rng.random(tgt.size) < qf[d][tgt]
                new.append(tgt[hit])
        state[burning] = BURNT
        steps[np.unique(burning // N)] = t
        if new:
            nw = np.unique(np.concatenate(new))
            state[nw] = BURNING
            bottom = nw[(nw % N) // L == L - 1]
            if bottom.size:
                crossed[np.unique(bottom // N)] = True
            if stop_on_cross and crossed.any():
                nw = nw[~crossed[nw // N]]
            burning = nw
        else:
            burning = np.empty(0, np.int64)
        if record:
            frames.append(state.reshape(B, L, L).copy())
    st = state.reshape(B, L, L)
    nf = fuel.reshape(B, -1).sum(1)
    burnt = (st != UNBURNT).reshape(B, -1).sum(1)
    return RunResult(
        crossed=crossed,
        steps=steps,
        burnt_frac=burnt / np.maximum(nf, 1),
        frames=frames,
    )


def crossing_trials(
    fam: Family, s_values, n_per: int, L: int, seed: int, batch: int = 256
) -> dict:
    """Crossing counts for each s value (fresh landscapes per trial)."""
    s_values = np.asarray(s_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s_all = np.repeat(s_values, n_per)
    crossed = np.zeros(s_all.size, bool)
    clipfrac = []
    for i in range(0, s_all.size, batch):
        sb = s_all[i : i + batch]
        ls = make_landscapes(fam, sb, L, rng)
        rq = raw_q(fam, ls)
        q = np.clip(rq, 0, 1).astype(np.float32)
        res = simulate(q, ls.fuel, rng)
        crossed[i : i + batch] = res.crossed
        nz = rq[rq > 0]
        clipfrac.append((float((nz > 1).sum()), float(nz.size)))
    k = crossed.reshape(len(s_values), n_per).sum(1)
    return {
        "s": s_values.tolist(),
        "n": [n_per] * len(s_values),
        "k": k.astype(int).tolist(),
        "clip_frac": sum(a for a, _ in clipfrac) / max(sum(b for _, b in clipfrac), 1),
    }
