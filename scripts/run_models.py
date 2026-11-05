"""Train, evaluate and roll out every model for every family.

Usage:
  python scripts/run_models.py hp        # hyperparameter selection (validation Brier only)
  python scripts/run_models.py seeds     # all seeds of all models, with rollouts
  python scripts/run_models.py baselines # never-ignite and persistence
Jobs run in a process pool; each job writes its own JSON, so a restart skips
finished jobs (checkpointing)."""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from wildfire.families import FAMILIES
from wildfire.io import ROOT, load_result, results_dir, save_result, smoke

DATA = ROOT / "data" if not smoke() else results_dir() / "data"
SEEDS = [0, 1] if smoke() else [0, 1, 2, 3, 4]
H_GRID = [16, 32]
STEPS = 300 if smoke() else int(os.environ.get("WILDFIRE_STEPS", "4000"))
LR = 3e-3
BATCH = 2048
L_ROLL = 64 if smoke() else 128
N1, N2 = (24, 48) if smoke() else (64, 128)


def kinds_for(fam: str) -> list[str]:
    k = ["logreg", "mpnn1", "mpnn2", "structured", "mpnn1_band", "structured_band"]
    if FAMILIES[fam].wind_w > 0:
        k.append("mpnn1_windblind")
    return k


def base_kind(kind: str) -> str:
    return kind.replace("_band", "")


def load_samples(fam, split, wind_blind=False):
    tag = "_windblind" if wind_blind else ""
    d = np.load(DATA / f"{fam}_{split}{tag}.npz")
    return {k: d[k] for k in d.files}


def train_set(fam, kind):
    wb = kind == "mpnn1_windblind"
    S = load_samples(fam, "train", wb)
    if kind.endswith("_band"):
        B = load_samples(fam, "band", wb)
        S = {k: np.concatenate([S[k], B[k]]) for k in S}
    return S


def sc_true(fam):
    return load_result(f"ground_truth_{fam}", results_dir())["result"]["L128"]["xc"]


def hp_job(args):
    fam, kind, H = args
    name = f"hp/{fam}_{kind}_H{H}"
    if (results_dir() / f"{name}.json").exists():
        return name
    from wildfire.train import active_brier, train

    S = train_set(fam, kind)
    Sval = load_samples(fam, "val", kind == "mpnn1_windblind")
    params, info = train(base_kind(kind), S, Sval, H, seed=0, steps=STEPS, lr=LR, batch=BATCH)
    info["val_active_brier"] = active_brier(base_kind(kind), params, Sval)
    save_result(name, info, 0, {"family": fam, "kind": kind, "H": H, "steps": STEPS, "lr": LR, "batch": BATCH})
    return name


def chosen_H(fam, kind):
    """H chosen by validation active-cell Brier of the default-data model (seed 0).
    Band and wind-blind variants use the H chosen for their base model."""
    k0 = kind.replace("_band", "").replace("_windblind", "")
    if k0 == "logreg":
        return 0
    best = None
    for H in H_GRID:
        r = load_result(f"hp/{fam}_{k0}_H{H}")["result"]["val_active_brier"]
        if best is None or r < best[1]:
            best = (H, r)
    return best[0]


def seed_job(args):
    fam, kind, seed = args
    name = f"models/{fam}_{kind}_s{seed}"
    if (results_dir() / f"{name}.json").exists():
        return name
    from wildfire.evaluate import crossing_rule, test_metrics
    from wildfire.train import train

    t0 = time.time()
    H = chosen_H(fam, kind)
    wb = kind == "mpnn1_windblind"
    S = train_set(fam, kind)
    Sval = load_samples(fam, "val", wb)
    bk = base_kind(kind)
    params, info = train(bk, S, Sval, H, seed=seed, steps=STEPS, lr=LR, batch=BATCH, log=lambda *a: None)
    wdir = results_dir() / "weights"
    wdir.mkdir(exist_ok=True)
    np.savez(wdir / f"{fam}_{kind}_s{seed}.npz", **params)
    fr = dict(np.load(DATA / f"{fam}_test_frames.npz"))
    sc = sc_true(fam)
    met = test_metrics(bk, params, fr, sc, wind_blind=wb)
    t1 = time.time()
    cr = crossing_rule(bk, params, FAMILIES[fam], sc, L_ROLL, seed=50000 + 97 * seed, n1=N1, n2=N2,
                       wind_blind=wb)
    out = {"H": H, "train": info, "test": met, "crossing": cr, "rollout_seconds": time.time() - t1,
           "total_seconds": time.time() - t0}
    save_result(name, out, seed, {"family": fam, "kind": kind, "H": H, "steps": STEPS, "lr": LR,
                                  "batch": BATCH, "L_rollout": L_ROLL, "n1": N1, "n2": N2})
    return name


def baseline_job(args):
    fam, kind = args
    name = f"models/{fam}_{kind}_s0"
    if (results_dir() / f"{name}.json").exists():
        return name
    from wildfire.evaluate import S_MAX_MODEL, test_metrics
    from wildfire.rollout import model_crossing

    fr = dict(np.load(DATA / f"{fam}_test_frames.npz"))
    sc = sc_true(fam)
    met = test_metrics(kind, {}, fr, sc)
    cr = model_crossing(kind, {}, FAMILIES[fam], [sc, 1.0, S_MAX_MODEL], N1, L_ROLL, 777)
    out = {"test": met, "crossing": {"status": "never_crosses_below_smax" if sum(cr["k"]) == 0 else "crosses",
                                     "sweeps": [cr], "xc": None}}
    save_result(name, out, 0, {"family": fam, "kind": kind, "L_rollout": L_ROLL})
    return name


def run(jobs, fn, procs):
    with Pool(procs) as pool:
        for name in pool.imap_unordered(fn, jobs):
            print(time.strftime("%H:%M:%S"), "done", name, flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    fams = sys.argv[2:] or list(FAMILIES)
    procs = int(os.environ.get("WILDFIRE_PROCS", "4"))
    if what == "hp":
        jobs = [(f, k, H) for f in fams for k in ("mpnn1", "mpnn2", "structured") for H in H_GRID]
        run(jobs, hp_job, procs)
    elif what == "seeds":
        jobs = [(f, k, s) for s in SEEDS for f in fams for k in kinds_for(f)]
        # expensive 2-hop jobs first for better packing
        jobs.sort(key=lambda j: (j[2], j[1] != "mpnn2"))
        run(jobs, seed_job, procs)
    elif what == "baselines":
        run([(f, k) for f in fams for k in ("never", "persistence")], baseline_job, procs)
    else:
        raise SystemExit(json.dumps({"unknown": what}))
