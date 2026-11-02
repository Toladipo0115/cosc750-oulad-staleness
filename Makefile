# One command reproduces the baseline table:  make baselines
PY ?= python

.PHONY: all data test baselines preliminary uncertainty verify benchmark diagnostics tables breakdown clean

all: data
	$(PY) scripts/run.py --condition fresh --models B0 B1 B2 B3
	$(PY) scripts/run.py --condition augmented --models B1 B3
	$(PY) scripts/uncertainty.py
	$(PY) scripts/tables.py
	$(PY) scripts/feedback_breakdown.py
	$(PY) scripts/presentation_bootstrap.py
	$(PY) scripts/reproduction_check.py
	$(PY) scripts/diagnostics.py
	$(PY) scripts/verify_predictions.py

data:
	$(PY) scripts/get_data.py

test: data
	$(PY) -m pytest -q tests

baselines: data
	$(PY) scripts/run.py --condition fresh --models B0 B1 B2 B3
	$(PY) scripts/uncertainty.py
	$(PY) scripts/tables.py
	$(PY) scripts/feedback_breakdown.py --seed-only

preliminary: data
	$(PY) scripts/run.py --condition augmented --models B1 B3
	$(PY) scripts/uncertainty.py
	$(PY) scripts/tables.py
	$(PY) scripts/feedback_breakdown.py

uncertainty:
	$(PY) scripts/uncertainty.py

verify:
	$(PY) scripts/verify_predictions.py

breakdown: data
	$(PY) scripts/feedback_breakdown.py
	$(PY) scripts/presentation_bootstrap.py

diagnostics: data
	$(PY) scripts/diagnostics.py

benchmark: data
	$(PY) scripts/reproduction_check.py



tables:
	$(PY) scripts/tables.py

clean:
	rm -rf results/cache results/predictions results/runs_*.csv results/*.tex results/*.pdf results/*.csv results/*.csv.gz
