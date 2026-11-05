"""Build and cache the L=64 train / band / validation / test sets for a family.

The test set is fixed (seeded) and built before any model is trained; its
SHA-256 is saved in results/data_<family>.json."""

import hashlib
import sys
from pathlib import Path

import numpy as np

from wildfire.data import BAND, S_HALF_RANGE, frames_to_samples, make_frames, s_sampler
from wildfire.families import FAMILIES
from wildfire.io import ROOT, load_result, results_dir, save_result, smoke

DATA = Path(ROOT / "data") if not smoke() else results_dir() / "data"
N_TRAIN = 120 if smoke() else 1500
N_VAL = 40 if smoke() else 300
N_TEST = 60 if smoke() else 400
FRAMES = 6
TEST_FRAMES = 4


def critical_s(fam: str) -> float:
    return load_result(f"ground_truth_{fam}", results_dir())["result"]["L128"]["xc"]


def build(fam_name: str):
    fam = FAMILIES[fam_name]
    sc = critical_s(fam_name)
    base = 1000 * (list(FAMILIES).index(fam_name) + 1)
    lo, hi = max(sc - S_HALF_RANGE, 0.02), min(sc + S_HALF_RANGE, 1.0)
    default_width = (sc - BAND - lo) + max(hi - (sc + BAND), 0)
    band_width = min(sc + BAND, 1.0) - (sc - BAND)
    n_band = int(round(N_TRAIN * band_width / default_width))
    DATA.mkdir(parents=True, exist_ok=True)
    info = {"critical_s_L128": sc, "range": [lo, hi], "band": [sc - BAND, min(sc + BAND, 1.0)],
            "n_train": N_TRAIN, "n_band": n_band, "n_val": N_VAL, "n_test": N_TEST}
    sets = {"train": ("default", N_TRAIN, FRAMES, 1), "band": ("band", n_band, FRAMES, 2),
            "val": ("default", N_VAL, FRAMES, 3), "test": ("full", N_TEST, TEST_FRAMES, 4)}
    for split, (which, n, nf, off) in sets.items():
        rng = np.random.default_rng(base + off)
        s = s_sampler(sc, which, rng, n)
        fr = make_frames(fam, s, rng, nf)
        if split == "test":
            path = DATA / f"{fam_name}_test_frames.npz"
            np.savez_compressed(path, **fr)
            info["test_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            info["test_frames"] = int(fr["state"].shape[0])
        for wb in (False, True):
            if wb and fam.wind_w == 0:
                continue
            S = frames_to_samples(fr, np.random.default_rng(base + off + 10), wind_blind=wb)
            tag = "_windblind" if wb else ""
            np.savez(DATA / f"{fam_name}_{split}{tag}.npz", **S)
            info[f"{split}{tag}_samples"] = int(S["X"].shape[0])
            info[f"{split}{tag}_active"] = int(S["active"].sum())
        print(fam_name, split, info.get(f"{split}_samples"), flush=True)
    save_result(f"data_{fam_name}", info, base, {"family": fam.to_dict(), "frames_per_trial": FRAMES,
                                                 "test_frames_per_trial": TEST_FRAMES})


if __name__ == "__main__":
    for f in sys.argv[1:] or list(FAMILIES):
        build(f)
