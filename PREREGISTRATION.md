# Pre-registration

Written and committed before any model was trained. After this commit the file
does not change; later changes of plan go in `DEVIATIONS.md` with a reason.

## Question

Can a learned fire-spread model that is excellent at predicting the next step
still put the tipping point (the critical base spread control `s` at which half
of the fires cross an L=128 map) in the wrong place?

## Ground truth (already measured, committed in `results/`)

The simulator (`wildfire/sim.py`) passed both hard gates at L=128 with 400 trials
at each of 13 / 14 control values: G1 bond percolation critical s = 0.4995
(target 0.5, tolerance 0.010); G2 site percolation critical p = 0.5925 (target
0.592746, tolerance 0.010).

"Critical s" is the 50% point of a binomial maximum-likelihood logistic fit of
crossing against s, at L=128. It is a finite-size value, not an infinite-lattice
threshold. CIs are 95% percentile bootstrap intervals (400 resamples, trials
resampled within each s value).

Families (`wildfire/families.py`), fresh landscapes per trial:

| family | definition | critical s, L=128 (95% CI) | L=64 | 10-90% width (L=128) |
|---|---|---|---|---|
| A uniform | iid fuel density 0.9, nothing else | 0.5719 (0.5711, 0.5729) | 0.5709 | 0.0324 |
| B wind | A + wind east or west (random sign per landscape), along-wind factors 1 ± 0.5, north/south factor 0.6 | 0.8112 (0.8083, 0.8139) | 0.8124 | 0.1066 |
| C patchy | smoothed fuel field (sigma 2, density 0.9), moisture factor exp(-0.6 m) with smooth m in [0,1] (sigma 3), slope factor exp(0.4 (z_target - z_source)) with smooth z (sigma 3, sd 1) | 0.7180 (0.7163, 0.7199) | 0.7206 | 0.0728 |

B differs from A by 0.24 (requirement: at least 0.05) at the prescribed starting
point w=0.5, N/S factor 0.6, so no adjustment was made. At B's critical s, 25% of
fuel-to-fuel edges have raw q > 1 and are clipped (the downwind edges, 1.5 x 0.81).
This is disclosed, not tuned away. C: critical s in [0.3, 0.9]; clipped edges at
critical s 0.001%; 1/e correlation lengths 3 (fuel), 6 (moisture), 6 (elevation)
cells, at most L/10 at both L=64 and L=128; logistic goodness of fit p = 0.14.
The wind sign is random per landscape so that a wind-blind control is
informative; the prompt's "blowing east" is kept as one of the two signs.

## Data

L=64 simulator rollouts run to extinction; up to 6 random frames per trial with
fire present. Each sample is a target cell with its 2-hop neighbourhood of raw
fields (s, fuel, moisture, elevation, wind vector, one-hot state); the label is
"burning at the next step". Samples: every active cell in the frame plus an equal
number of random cells. Training s is uniform on
[sc - 0.30, sc + 0.30] clipped to [0.02, 1.0], **excluding the band
|s - sc| < 0.08, where sc is the measured L=128 critical s of the family**. The
gap is centred on the known answer; a real modeller would not know where to
centre it. Band data: extra trials uniform in the band, at the same density per
unit s as the default set. Sizes: 1500 train trials, band trials in proportion,
300 validation trials (default distribution), 400 test trials (whole range
including the band, 4 frames each). The test set is fixed, seeded, and was built
before training; its SHA-256 is in `results/data_<family>.json`.

## Models

All learned models see only raw fields; never the factors or q.

- never-ignite: P = 0 everywhere.
- persistence: next state = current state (P = 1 on burning cells, else 0).
- logreg: logistic regression on 14 hand-made local features
  (`wildfire/models.py::logreg_features`).
- mpnn1: generic 1-hop MPNN. Message MLP on [x_i, x_j, spread direction]
  (2 layers, H units, ReLU), summed over valid neighbours, update MLP on
  [x_i, sum] to one logit.
- mpnn2: two such message-passing rounds (2-hop receptive field).
- mpnn1_windblind (B only): mpnn1 with the wind vector input set to zero.
- structured: an edge MLP (3 layers) predicts q_ij for each directed edge; the
  cell's ignition probability is (unburnt fuel) x (1 - prod_j (1 - burning_j q_ij)).
  **This model has the true functional form by construction**; H5 measures the
  value of that knowledge, not a discovery.
- mpnn1_band, structured_band: as mpnn1 / structured, trained on default + band data.

Training: Adam, lr 3e-3 with cosine decay, batch 2048, a fixed number of steps
(STEPS, below), loss = mean BCE on active samples + mean BCE on other samples.
Hyperparameter selection: hidden size H in {16, 32}, chosen per family and model
by validation active-cell Brier of seed 0 trained on default data, never on
critical s. Band and wind-blind variants use their base model's H. Seeds 0-4 for
every learned model (logreg included) and family. Weights are evaluated with the
NumPy implementation in `wildfire/models.py`.

## Metrics

On the fixed test set (L=64):

- all-cell accuracy (threshold 0.5, every cell of every test frame);
- active-cell Brier, where an active cell is unburnt fuel with a burning
  4-neighbour;
- Bayes floor = mean over active cells of p(1-p), p = 1 - prod(1 - q) from the
  simulator's true q;
- excess = mean over active cells of (p_hat - p)^2 (the expected Brier minus the
  floor, without label noise);
- calibration: 10 equal-width bins on active cells, ECE.

Each is reported on the off-band part of the test set (|s - sc| >= 0.08, the
training distribution), the in-band part, and overall.

**Locally excellent**: excess on the off-band test set at most **0.002**
(root-mean-square probability error about 0.045, about 1.2% of the floor, which is
0.15-0.20 in these families). A stricter tier, excess at most 0.0005, is also
reported but not used for the hypotheses.

Critical s of a model: masked rollouts at L=128 on fresh simulator landscapes:
each step only active cells may ignite, with the model's probability. Secondary
unmasked rollouts (16 trials at the true critical s, at most 300 steps, seed 0 of
each model) report the rate of ignitions of cells with no burning neighbour.

## Timing and the trial-count rule

Measured before this file (4 vCPUs): one simulator batch of 128 fires at L=128 near
critical takes about 1.2 s; one model batch of 128 rollouts takes about 1.5-4 s
for 1-hop models with random weights, more for 2-hop models.
PyTorch could not be installed (the CPU wheel index is blocked by this
environment's network policy; the PyPI CUDA build was still downloading after
15 minutes), so training uses NumPy with hand-written reverse-mode gradients
(`wildfire/nn.py`, finite-difference tested). Measured training cost per step
(batch 2048, H=32, one core): mpnn1 10 ms, structured 5 ms, mpnn2 80 ms, logreg
2 ms. A pilot (family C, mpnn1 and structured, 3000 steps, H=32, seed 0, data
and test set as below) took 17-22 s to train and 32-34 s for the crossing rule
below. The pilot's numbers are not used in any result. **Disclosure:** strictly,
the pilot means two models were trained before this file was committed, and I
saw their results: off-band excess 0.00047 and 0.00010, critical s 0.730 and
0.734 against truth 0.718. The hypotheses, the 0.002 excellence margin and the
H2 thresholds were drafted before the pilot ran and were not changed after it. Chosen: **STEPS = 4000**
for every learned model; 4 worker processes. Estimated total: about 1 hour.

Rollout-machinery control: the same masked rollout code driven by the
simulator's true q ("oracle") gave critical s 0.7157 (C) and 0.5705 (A) with 200
trials at 9 values, versus ground truth 0.7180 and 0.5719. The oracle is re-run
and reported for every family as a control.

Rule for each model and seed (`wildfire/evaluate.py::crossing_rule`):
1. 9 values of s evenly spaced over [sc - 0.2, sc + 0.2] (clipped to
   [0.02, 1.3]; s above 1 is allowed for models since s is only an input),
   64 trials each.
2. If no value reaches 50% crossing, run 64 trials at s = 1.3; if still below,
   record "never crosses below 1.3" (critical s censored).
3. Fit; then 9 values evenly spaced over xc ± 1.2 x max(10-90% width, 0.02),
   128 trials each. Refit on pooled counts. If the CI is wider than 0.02, repeat
   step 3 once more around the new fit.

## Statistics across seeds

For each model and family: range and two-sided 95% t-interval of the 5 seeds'
critical s and of their errors (model - truth).

## Hypotheses

**H1** (per family): the best GNN (mpnn1, mpnn2 or structured, default data,
lowest mean off-band active-cell Brier) beats the baselines: the upper end of
its seed t-interval of off-band active-cell Brier is below the lower end of
logreg's interval and below never-ignite and persistence. Supported overall if
it holds in all three families; not supported if it fails in any.

**H2** (main question), **judged per seed**: a seed passes if (a) it is locally
excellent, (b) |critical s error| > max(0.01, half the family's 10-90% width
at L=128) = A 0.0162, B 0.0533, C 0.0364, and (c) its 95% bootstrap CI does not
overlap the ground-truth CI. Eligible: all seeds of mpnn1, mpnn2, structured,
mpnn1_band, structured_band in all families (75 seeds). Baselines (never,
persistence, logreg) and the wind-blind control do not count. Supported if at
least one seed passes; report passes / tried and which. Also reported
per model (secondary): the mean critical s across seeds with its t-interval vs
the truth CI.

**H3** (per family, for mpnn1 and for structured): error = |mean critical s over
seeds - truth|. Supported in a cell if error(band) <= 0.5 x error(default).
Inconclusive in a cell where error(default) < 0.01 (the measurement floor) or
the default model's seed t-interval contains the truth. Overall: supported if
every conclusive cell is supported, not supported if any conclusive cell is not,
inconclusive if no cell is conclusive.

**H4** (per family): mpnn2 is equivalent to mpnn1 on critical s within an
equivalence margin of **0.02**: the Welch 95% CI of the difference of seed means
lies inside (-0.02, 0.02) -> supported; lies entirely outside -> not supported;
otherwise (CI crosses a margin) inconclusive. Overall: supported if all families
supported, not supported if any family is not supported, otherwise inconclusive.

**H5** (per family): structured has lower critical-s error than mpnn1 (default
data): the Welch 95% CI of mean |error|(mpnn1) - mean |error|(structured) is
entirely above 0 -> supported; otherwise not supported; inconclusive if both mean
|errors| are below 0.01. Overall as H4.

## Controls reported prominently

- never-ignite all-cell accuracy, overall and by s, next to its critical s
  (it never crosses).
- wind-blind MPNN in family B.

## Things that are not hypotheses

Up to 3 exploratory follow-ups ("embers") may run after H1-H5, each written in
`EMBERS.md` and committed before it runs, and labelled exploratory.
