# One command reproduces the baseline table:  make baselines
PY ?= python

.PHONY: all data test baselines preliminary uncertainty verify benchmark diagnostics tables breakdown stage4 rq3 stage4-tables clean

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

# Stage 4: RQ3 (five pairs fixed in stage4_plan.json), ablations and sensitivity analyses
stage4: data
	$(PY) scripts/stage4.py rq3 norecency randomage wide size chrono fullcohort
	$(PY) scripts/rq3_bootstrap.py
	$(PY) scripts/ablation_bootstrap.py
	$(PY) scripts/stage4_tables.py

rq3: data
	$(PY) scripts/stage4.py rq3
	$(PY) scripts/rq3_bootstrap.py
	$(PY) scripts/stage4_tables.py

stage4-tables:
	$(PY) scripts/stage4_tables.py



tables:
	$(PY) scripts/tables.py

clean:
	rm -rf results/cache results/predictions results/runs_*.csv results/*.tex results/*.pdf results/*.csv results/*.csv.gz
