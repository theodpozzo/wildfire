PY ?= .venv/bin/python
export PYTHONPATH := .
export TZ := UTC
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1

.PHONY: test lint reproduce full render demo

test: lint
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

# Small version of every figure, written to repro_out/ only (never results/ or figures/).
reproduce:
	rm -rf repro_out
	WILDFIRE_OUT=repro_out WILDFIRE_SMOKE=1 $(PY) scripts/gates.py
	WILDFIRE_OUT=repro_out WILDFIRE_SMOKE=1 $(PY) scripts/ground_truth.py
	WILDFIRE_OUT=repro_out WILDFIRE_SMOKE=1 $(PY) scripts/make_data.py
	WILDFIRE_OUT=repro_out WILDFIRE_SMOKE=1 $(PY) scripts/run_models.py hp
	WILDFIRE_OUT=repro_out WILDFIRE_SMOKE=1 $(PY) scripts/run_models.py baselines
	WILDFIRE_OUT=repro_out WILDFIRE_SMOKE=1 $(PY) scripts/run_models.py seeds
	WILDFIRE_OUT=repro_out WILDFIRE_SMOKE=1 $(PY) scripts/controls.py
	WILDFIRE_OUT=repro_out WILDFIRE_SMOKE=1 $(PY) scripts/analyze.py > /dev/null
	WILDFIRE_OUT=repro_out WILDFIRE_SMOKE=1 $(PY) scripts/figures.py
	@echo "small figures in repro_out/figures"

# Everything (several hours on 4 vCPUs). Overwrites results/ and figures/.
full:
	$(PY) scripts/gates.py
	$(PY) scripts/ground_truth.py
	$(PY) scripts/make_data.py
	./scripts/run_all_models.sh
	$(PY) scripts/embers.py
	$(MAKE) render

render:
	$(PY) scripts/analyze.py > /dev/null
	$(PY) scripts/figures.py
	$(PY) scripts/export_demo.py
	$(PY) scripts/render.py
