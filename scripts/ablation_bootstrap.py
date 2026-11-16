"""Student-cluster bootstrap for the Stage 4 ablations, from their saved test predictions.

  norecency  freshness-aware minus augmented, both trained without the three recency features
             (results/norecency/predictions/)
  randomage  shuffled ages minus augmented, and freshness-aware minus shuffled ages, against
             the Stage 3 augmented run (results/randomage/predictions/, results/predictions/)

Same statistic, resampling unit (students with all their enrollments, 2,000 draws, fitted
models fixed) and p value as scripts/uncertainty.py and scripts/rq3_bootstrap.py, over all 25
pairs. Writes results/ablation_bootstrap.csv.   Run:  python scripts/ablation_bootstrap.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.rq3_bootstrap import N_BOOT, boot_p, engine, stat                     # noqa: E402
from staleness.experiment import CELLS                                            # noqa: E402
from staleness.inference import interval, load_pooled, student_weights            # noqa: E402


def main(y=None, students=None):
    if y is None:
        from scripts.run import load
        cohort, _, _ = load("data/oulad", "results/cache/features.pkl")
        y, students = cohort["y"].to_numpy(), cohort["id_student"].to_numpy()
    n = len(y)
    main_, _, folds = load_pooled("augmented", n, with_folds=True)
    nr, _, f_nr = load_pooled("norecency", n, results="results/norecency", with_folds=True)
    ra, _, f_ra = load_pooled("randomage", n, results="results/randomage", with_folds=True)
    for f in (f_nr, f_ra):
        assert all((a == b).all() for a, b in zip(folds, f)), "ablations must use the Stage 3 folds"
    S = len(folds)
    comps = []
    for m in ["B1", "B3"]:
        comps.append((m, "norecency", "aware_minus_augmented", nr[(m, "freshness_aware")], nr[(m, "augmented")]))
        comps.append((m, "randomage", "random_minus_augmented", ra[(m, "random_age")], main_[(m, "augmented")]))
        comps.append((m, "randomage", "aware_minus_random", main_[(m, "freshness_aware")], ra[(m, "random_age")]))
    weights = list(student_weights(students, N_BOOT))
    ones = np.ones(n)
    rows = []
    for m, exp, lab, A, B in comps:
        ea, eb = engine(A, y, folds, CELLS), engine(B, y, folds, CELLS)
        seeds = [np.mean([np.mean([a.ap(ones[idx]) for idx, a in ea[(s, c)]]) -
                          np.mean([a.ap(ones[idx]) for idx, a in eb[(s, c)]]) for c in CELLS])
                 for s in range(S)]
        draws = np.array([stat(ea, w, CELLS, S) - stat(eb, w, CELLS, S) for w in weights])
        lo, hi = interval(draws)
        rows.append(dict(model=m, experiment=exp, comparison=lab, pairs="all_25",
                         estimate=float(np.mean(seeds)), seed_sd=float(np.std(seeds, ddof=1)),
                         ci_low=lo, ci_high=hi, p_boot=boot_p(draws), n_boot=N_BOOT))
        print(m, exp, lab, round(rows[-1]["estimate"], 4), round(lo, 4), round(hi, 4), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/ablation_bootstrap.csv", index=False)
    print(out.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
