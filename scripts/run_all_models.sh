#!/bin/sh
# Full model pipeline (checkpointed: finished jobs are skipped on restart).
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
PY=${PY:-.venv/bin/python}
$PY scripts/run_models.py hp
$PY scripts/run_models.py baselines
$PY scripts/run_models.py seeds
$PY scripts/controls.py
