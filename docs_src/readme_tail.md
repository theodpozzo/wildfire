## The demo

`docs/index.html` is a single static page (vanilla JS, no libraries, no build step, no network
requests, no cookies). Open it straight from disk, or serve `docs/` with GitHub Pages. Paint or
generate a landscape, set s with the slider, light the top row and watch the simulator and a trained
model burn the same landscape side by side, with the measured crossing curves underneath. The model
weights ship as JSON inside `docs/data.js`, and the JS inference is hand-written; a test checks it
matches Python within 1e-5.

## Layout

- `wildfire/` simulator, fits, models, NumPy autodiff, training, rollouts, evaluation
- `scripts/` gates, ground truth, data, training runs, analysis, figures, rendering, demo export
- `results/` every result as JSON (seed, config, git hash, date); `figures/` the plots
- `PREREGISTRATION.md` (fixed before training), `DEVIATIONS.md`, `EMBERS.md`, `REPORT.md`
- `tests/` simulator, gradients, models, JS parity, report-number checks

## Reproduce

```
uv venv .venv && uv pip install -p .venv/bin/python --require-hashes -r requirements.txt
make test
make reproduce
```

MIT licence, copyright Theo Dal Pozzo.
