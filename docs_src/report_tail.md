## Limitations

- **This is not fire.** Real fire spread depends on fuel type and load, live and dead fuel moisture, wind
  that varies in time and with terrain, spotting (embers carried far ahead of the front), radiant and
  convective heating, suppression, and weather. Here a cell burns for exactly one step, spread is
  nearest-neighbour only, the wind is constant, and every factor is a made-up multiplier. The "tipping
  point" is a finite-size percolation-style transition on a 128 x 128 grid. Nothing here transfers to
  real landscapes, and it must never inform real fire or evacuation decisions.
- **Finite size.** Critical s is a finite-size, finite-grid quantity, measured at L=128 and reported at
  L=64. It is not an infinite-lattice threshold.
- **The held-out band is centred on the known answer.** A real modeller would not know where the critical
  region is. Centring the gap on it is a best case for "the model never saw the transition".
- **Small models, one architecture family.** Hidden sizes 16 or 32, fixed training budget, a
  hyperparameter grid of two values. Larger or differently trained models could behave differently.
- **Multiple comparisons.** H2 is judged per seed across many seeds and families; a single passing seed
  is a weaker claim than a per-model pass. Both views are reported.
- **The excellence margin is a choice.** 0.002 excess Brier was fixed in advance. A stricter tier (0.0005)
  is reported in the tables (excess column) so a reader can apply their own.
- **Family B clips.** A quarter of fuel-to-fuel edges at B's critical point have raw q above 1 and are
  clipped (the downwind edges). This was left as specified rather than tuned away.
- **Cited work.** Kesten (1980) proved the square-lattice bond percolation threshold is 1/2; Newman and
  Ziff (2000) estimated the site threshold 0.592746. I am confident both exist; I did not re-check them
  online, because network use was limited to package downloads.

## What I got wrong along the way

- I first tried to install PyTorch. The CPU wheel index was blocked and the CUDA build was too slow to
  download, so training is NumPy with hand-written gradients. This cost time and is recorded in
  DEVIATIONS.md.
- A timing pilot trained two models before the pre-registration was committed, so I had seen two results
  before fixing the plan. The margins had been drafted before the pilot and were not changed; this is
  disclosed in the pre-registration itself.
- My first ground-truth script derived each family's random seed from its position in the command line,
  so running one family alone would have changed its seed. Fixed before any committed result depended on it.
- See DEVIATIONS.md for every change of plan after the pre-registration.

## Reproduce

```
uv venv .venv && uv pip install -p .venv/bin/python --require-hashes -r requirements.txt
make test        # pytest + ruff, includes the JS/Python parity test (needs node)
make reproduce   # small version of every figure into repro_out/, under 10 minutes
make full        # everything, several hours on 4 vCPUs
```

`make reproduce` writes only to `repro_out/` and never touches `results/` or `figures/`. Every result
JSON in `results/` carries its seed, config, git hash and date. REPORT.md and README.md are rendered by
`scripts/render.py`; `tests/test_report.py` fails if any number in them does not match the JSON.
