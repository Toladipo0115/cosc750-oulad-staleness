# Source-specific feature staleness in early withdrawal recognition (OULAD)

Code for the COSC 750 (Fall 2026, Bowie State University) project
*Classification Under Source-Specific Feature Staleness: A Freshness-Aware Study of
Early Student-Withdrawal Recognition* (Tolulope Oladipo).

The question: at day 28 of a course, how much does withdrawal recognition lose when
VLE activity and assessment submissions arrive late, each by 0, 1, 3, 7 or 14 days,
and does telling the model how old each source is help beyond training on delayed data?

Repository: https://github.com/Toladipo0115/cosc750-oulad-staleness

## Reproduce

Use `make all` after installing requirements to regenerate both training conditions, all bootstrap summaries, paper tables, the benchmark comparison, diagnostics and the feedback breakdown, then verify all saved metrics. This is a long-running command. `make baselines` alone reproduces the baseline table; `make verify` requires both `make baselines` and `make preliminary`.


```bash
pip install -r requirements.txt      # tested with Python 3.13
make data                            # downloads OULAD (or uses data/oulad/) and checks SHA-256 checksums
make baselines                       # B0 to B3 on fresh data, bootstrap intervals, results/baseline_table.*
make preliminary                     # augmented vs freshness-aware runs (B1, B3)
make verify                          # run after both conditions; checks six metrics from saved predictions
make test                            # 13 checks: checksums, cohort, day-28 cutoff, delays, no shared students, SVM calibration
make diagnostics                     # B3 confusion matrices, permutation importance, model comparison chart
make benchmark                       # B3 on the published study's cohort definition (benchmark comparison)
```

Both training commands are long-running. The original bootstrap estimate was about 30 minutes; total runtime depends on hardware, data loading and whether the feature cache exists. The sequential `make all` computes the full bootstrap once.
Partition/model seeds are 0, 1, 2; delay-copy seed is 1000 * partition_seed + outer_fold; bootstrap seed is 20261015. Diagnostic permutation seed is 10000 + 100 * partition_seed + outer_fold. If the UCI download is blocked, unpack the
dataset into `data/oulad/` yourself; `make data` still verifies the checksums in
`data_checksums.json`.

## Layout

| Path | What it does |
|---|---|
| `staleness/cohort.py` | Primary cohort (27,428 records enrolled by day 28, 5,013 later withdrawals) with the author's original assertions; `day28_eligible=False` gives the 27,444-record Stage 2 cohort; registration-date audit |
| `staleness/features.py` | 12 day-28 features recomputed for every pair of VLE and assessment delays (25 cells); 2 source-age indicators |
| `staleness/models.py` | B0 majority class, B1 logistic regression (+ age interactions), B2 RBF-SVM with Platt scaling fitted on the grouped inner folds (scikit-learn's internal probability step is off), B3 calibrated histogram gradient boosting; fixed grids |
| `staleness/experiment.py` | Nested evaluation: 5 outer folds grouped by module presentation x 3 seeds; inner grouped folds for tuning, calibration and threshold; students shared with a validation fold are removed from the matching training fold (outer and inner) |
| `staleness/inference.py` | Weighted average precision and the student-cluster bootstrap |
| `scripts/run.py` | Runs a condition; writes `results/runs_<condition>.csv`, `results/splits.csv` and the test predictions in `results/predictions/` |
| `scripts/uncertainty.py` | Seed-level estimates and 95% student-cluster bootstrap intervals (2,000 draws) for every reported comparison |
| `scripts/verify_predictions.py` | Recomputes all metrics from the saved predictions and checks them against the runs files |
| `scripts/reproduction_check.py` | Benchmark comparison: B3 on all 32,593 enrollments (the cohort of Wu, Duan and Luo 2026) |
| `scripts/diagnostics.py` | B3 confusion matrices, permutation feature importance, model comparison chart |
| `scripts/tables.py` | Baseline table, staleness surface, preliminary comparison, heat map |
| `tests/test_pipeline.py` | Checksums, cohort counts, day-28 cutoff boundary, no extra information at larger delays, independent delays, no presentation or student overlap in outer or inner folds, weighted AP |

## Protocol in brief

* Cohort: 32,593 enrollments, minus 5,047 withdrawals by day 28, 93 withdrawals without a date,
  9 inconsistent unregistrations, and 16 registrations after day 28 (not enrolled at prediction
  time) = 27,428 records. The 7 records without a registration date have no VLE or submission
  record at all; the date is imputed with the training-fold median.
* Prediction time t = 28. A source with delay d contributes records dated on or before
  t - 1 - d. Registration features are always current. Scores and banked assessments are never used.
* Outer test folds share no module presentation and no student with training; inner
  validation folds share no student with their inner training folds.
* Grid search (average precision), calibration and the macro-F1 threshold use inner
  grouped folds only. Each test fold is scored once, at all 25 delay cells.
* The 15 folds (3 partitions of the same 22 presentations) are not independent. Fold mean
  +/- SD is descriptive; uncertainty comes from the three seed-level means and a
  student-cluster bootstrap of the saved test predictions (95% intervals; Bonferroni over
  the 24 pairs of a surface), with the matching two-sided bootstrap test (p_boot, p_bonf). No fold-level
  significance test is used.

## Results

Fresh training and test data, mean ± SD over 15 test folds (`results/baseline_table.csv`):

| Method | PR-AUC | ROC-AUC | Macro-F1 | Recall | Brier | Train time (s) | PR-AUC minus B3, 95% interval (bootstrap p ≤ 0.001 for each) |
|---|---|---|---|---|---|---|---|
| B0 majority class | 0.181 ± 0.038 | 0.500 ± 0.000 | 0.450 ± 0.012 | 0.000 ± 0.000 | 0.149 ± 0.025 | 0 | -0.135 to -0.115 |
| B1 logistic regression | 0.287 ± 0.055 | 0.627 ± 0.033 | 0.580 ± 0.019 | 0.307 ± 0.066 | 0.143 ± 0.023 | 1 | -0.023 to -0.013 |
| B2 RBF-SVM | 0.247 ± 0.045 | 0.549 ± 0.017 | 0.551 ± 0.013 | 0.222 ± 0.064 | 0.148 ± 0.025 | 8 | -0.064 to -0.051 |
| B3 calibrated gradient boosting | 0.305 ± 0.058 | 0.651 ± 0.028 | 0.581 ± 0.018 | 0.378 ± 0.151 | 0.140 ± 0.021 | 7 | ref. |

Freshness-aware comparison, mean PR-AUC over all 25 delay pairs (`results/preliminary.csv`):

| Model | Fresh-trained | Augmented | Freshness-aware | Aware minus augmented (folds) | 95% interval |
|---|---|---|---|---|---|
| B1 logistic regression | 0.271 | 0.270 | 0.274 | +0.0040 ± 0.0028 (15/15) | 0.0025 to 0.0056 |
| B3 calibrated gradient boosting | 0.280 | 0.283 | 0.287 | +0.0045 ± 0.0029 (14/15) | 0.0026 to 0.0064 |

Benchmark comparison (`make benchmark`), not a reproduction: B3 on the cohort definition of Wu,
Duan and Luo (2026), all 32,593 enrollments: PR-AUC 0.721 ± 0.025, ROC-AUC 0.801 ± 0.013 (published: 0.664 and 0.794).

Diagnostics (`make diagnostics`): B3 at its inner-fold threshold, pooled over the 15 test folds:

| Delay pair | Flagged | Recall | Precision |
|---|---|---|---|
| (0, 0) | 23.2% | 0.408 | 0.322 |
| (0, 14) | 71.2% | 0.831 | 0.213 |
| (14, 14) | 72.8% | 0.838 | 0.211 |

Most important feature on fresh data (permutation importance, drop in PR-AUC; predictive
reliance, not a cause of withdrawal): assessments past their deadline with no observed submission (0.070), then studied credits (0.028) and mean days submitted early (0.028).

## Data

Kuzilek, Hlosta and Zdrahal (2017), *Open University Learning Analytics Dataset*,
Scientific Data 4:170171, CC BY 4.0. The data are not redistributed here;
`data_checksums.json` holds the SHA-256 of the seven CSV files used.

## Authorship of the code

The cohort logic and its assertions are the author's. The rest of the pipeline was written
with Claude (Anthropic) and reviewed and run by the author. A methodological review of the
Chatgpt reviewed the code, provided guideance- the inner-fold student
purge, the bootstrap intervals, the day-28 eligibility rule and the planned Stage 4 robustness
checks; see the AI-use disclosure in the paper.

## Revision for instructor feedback

`make breakdown` reads saved predictions and writes paired gain by source, delay and original presentation, plus complete-partition seed means. Presentation AP is recomputed within each presentation and averaged over cells and seeds; these descriptive values need not equal fold-mean AP. Student-cluster intervals still condition on fitted models and do not estimate new-presentation or retraining variability.

The original published baseline grids and saved model predictions are unchanged. Table 1 in the manuscript reports mean and SD over the three complete seed means (`results/baseline_seed_means.csv` and `results/paper_seed_table.tex`), with training time from `results/baseline_table.csv`, which also holds the fold SDs. `make breakdown` also writes `results/adjacent_contrasts_B3.csv`: B3 adjacent contrasts and per-day finite differences on the fresh-trained surface (Table A5). ChatGPT added the feedback reporting script, six-metric verification and missing-file message, vector heatmap output, this documentation, and the serial `make all` workflow. `make breakdown` also runs `scripts/presentation_bootstrap.py` (about 7 minutes on 2 CPU cores): a presentation-cluster bootstrap of the age gain that resamples the 22 presentations (2,000 draws, fitted models held fixed) and writes `results/presentation_bootstrap.csv` (Appendix G). Claude added the adjacent-contrast export, the presentation bootstrap and checked the revised scripts against the saved predictions.

`stage4_plan.json` records exploratory exclusion pairs selected after inspecting Stage 3 results. It is a plan, not an implemented or completed RQ3 experiment. Do not call these previously unseen results.

`results/paper_seed_table.tex` is the numeric table used by the revised LaTeX source. After regenerating results, copy it to the source package at `generated/paper_seed_table.tex`; copy regenerated figure PDFs to `figures/`. The original fold-level tables remain available for descriptive comparison.
