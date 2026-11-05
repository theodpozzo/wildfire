# Embers: exploratory follow-ups

Written and committed before running. Chosen after seeing H1-H5, so they are
exploratory and separate from the confirmatory results. Code: `scripts/embers.py`.

## What surprised me

(Updated after H1-H5 finished, still before either ember ran.) All 75 eligible GNN
seeds were locally excellent, none passed H2, but 46 of the 75 had a critical-s
interval that did not overlap the simulator's.

Most GNN seeds were locally excellent by a wide margin (excess Brier often 10-50x
below the 0.002 margin), and their critical-s errors were small next to the
transition width, yet many of those errors were clearly non-zero: the model's
95% interval did not overlap the simulator's. Excess Brier is a squared error,
so it is blind to the sign of a model's bias; the tipping point should respond
to the signed, systematic bias. And the H2 threshold (half the 10-90% width)
shrinks as the grid grows, while a model's systematic bias need not.

## E1: signed bias, not squared error, tracks the tipping-point error

For every GNN seed (mpnn1, mpnn2, structured and the band variants, all
families), compute the relative signed bias b = (mean p_hat - mean p_true) /
mean p_true over active cells of the in-band test frames (|s - sc| < 0.08, L=64,
already on disk), and the predicted error -sc x b (a uniform relative bias in
spread probability is roughly a rescaling of s).

Hypothesis: across seeds, Pearson r between predicted and measured critical-s
error is above 0.5, the least-squares slope of measured on predicted is in
[0.5, 2], and |Spearman rho| between off-band excess Brier and |error| is
smaller than that Pearson r. No new rollouts.

## E2: the same models at L=256

For each family, take the seed-0 default-data GNN (mpnn1, mpnn2 or structured)
with the largest |error| at L=128. Measure the simulator's critical s at L=256
with the ground-truth rule, and the model's with the pre-registered crossing
rule at L=256.

Hypothesis: in every family, the model's |error| at L=256 is at least half its
L=128 value, and the simulator's 10-90% width at L=256 is at most 0.75 of its
L=128 value. If so, an error that sits below the H2 threshold at L=128 is a
larger fraction of the transition at L=256.
