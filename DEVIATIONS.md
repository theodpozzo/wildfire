# Deviations from the brief and from the pre-registration

Each entry: what changed, and why. Entries before the pre-registration commit
describe choices made while setting it up.

## Before pre-registration

1. **NumPy instead of PyTorch.** `download.pytorch.org` is blocked by the
   environment's network policy (HTTP 403 at CONNECT). The PyPI build of torch
   pulls several GB of CUDA libraries and was still downloading after about 15
   minutes, so I stopped it. Training uses NumPy with a small hand-written
   reverse-mode autodiff (`wildfire/nn.py`), checked against finite differences
   in `tests/test_nn.py`. `requirements-torch.*` was removed.
2. **Wind sign randomised in family B.** Each landscape's wind blows east or west
   with equal probability (factors 1 ± 0.5 along the wind, 0.6 north/south). With
   a single fixed direction the wind input would be a constant, and the wind-blind
   control could not differ from the normal model in what it is told. Crossing is
   north-to-south, so the sign does not change the ground truth.
3. **Masked rollouts use the active-cell mask** (unburnt fuel with a burning
   neighbour) rather than "any cell with a burning neighbour": burnt or empty cells
   cannot burn in the simulator's state machine, so letting a model ignite them
   would test something other than spread.
4. **A timing pilot trained two models before the pre-registration commit** (see
   PREREGISTRATION.md, "Timing"). Their results were seen. Nothing in the
   hypotheses or margins was changed after seeing them.
