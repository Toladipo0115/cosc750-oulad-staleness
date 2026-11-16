"""Student-cluster bootstrap for RQ3, from the saved test predictions of the RQ3 run.

The RQ3 run (python scripts/stage4.py rq3) trains both arms without the five delay pairs fixed
in stage4_plan.json and saves its test predictions in results/rq3/predictions/. This script
resamples students with all their enrollments (2,000 draws, fitted models held fixed) and
recomputes the same fold-mean average precision as scripts/uncertainty.py for

  aware_minus_augmented       freshness-aware minus augmented, both trained without the pairs
  heldout_minus_full_<arm>    the same arm trained without the pairs minus trained with them
                              (the Stage 3 augmented run, results/predictions/augmented_*)

on the five held-out pairs and on all 25 pairs, for B1 and B3.
Writes results/rq3_bootstrap.csv.   Run:  python scripts/rq3_bootstrap.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from staleness.experiment import CELLS                                            # noqa: E402
from staleness.inference import APIndex, interval, load_pooled, student_weights   # noqa: E402

N_BOOT = int(os.environ.get("N_BOOT", 2000))
HOLDOUT = [tuple(p) for p in json.load(open(os.path.join(ROOT, "stage4_plan.json")))
           ["excluded_training_and_inner_validation_pairs"]]
J = {c: i for i, c in enumerate(CELLS)}


def boot_p(d):
    d = d[~np.isnan(d)]
    k = min((d <= 0).sum(), (d >= 0).sum())
    return float(min(1.0, 2 * (k + 1) / (len(d) + 1)))


def engine(P, y, folds, cells):
    """APIndex per (seed, cell, outer fold) for one prediction set."""
    return {(s, c): [(idx, APIndex(P[s][idx, J[c]].astype(float), y[idx]))
                     for idx in [np.flatnonzero(folds[s] == k) for k in np.unique(folds[s])]]
            for s in range(len(P)) for c in cells}


def stat(e, w, cells, n_seeds):
    """Fold-mean AP averaged over cells and seeds."""
    return float(np.mean([np.mean([a.ap(w[idx]) for idx, a in e[(s, c)]])
                          for s in range(n_seeds) for c in cells]))


def main(y=None, students=None):
    if y is None:
        from scripts.run import load
        cohort, _, _ = load("data/oulad", "results/cache/features.pkl")
        y, students = cohort["y"].to_numpy(), cohort["id_student"].to_numpy()
    n = len(y)
    rq3, _, folds = load_pooled("rq3", n, results="results/rq3", with_folds=True)
    full, _, folds_full = load_pooled("augmented", n, with_folds=True)
    assert all((a == b).all() for a, b in zip(folds, folds_full)), "RQ3 run must use the Stage 3 folds"
    S = len(folds)
    groups = {"held_out_5": HOLDOUT, "all_25": CELLS}
    comps = []
    for m in ["B1", "B3"]:
        comps.append((m, "aware_minus_augmented", rq3[(m, "freshness_aware")], rq3[(m, "augmented")]))
        for arm in ["augmented", "freshness_aware"]:
            comps.append((m, f"heldout_minus_full_{arm}", rq3[(m, arm)], full[(m, arm)]))
    rows = []
    for gname, cells in groups.items():
        for m, lab, A, B in comps:
            ea, eb = engine(A, y, folds, cells), engine(B, y, folds, cells)
            ones = np.ones(n)
            seeds = [np.mean([np.mean([a.ap(ones[idx]) for idx, a in ea[(s, c)]]) -
                              np.mean([a.ap(ones[idx]) for idx, a in eb[(s, c)]]) for c in cells])
                     for s in range(S)]
            draws = np.array([stat(ea, w, cells, S) - stat(eb, w, cells, S)
                              for w in student_weights(students, N_BOOT)])
            lo, hi = interval(draws)
            rows.append(dict(model=m, comparison=lab, pairs=gname, estimate=float(np.mean(seeds)),
                             seed_sd=float(np.std(seeds, ddof=1)), seed0=seeds[0], seed1=seeds[1],
                             seed2=seeds[2], ci_low=lo, ci_high=hi, p_boot=boot_p(draws), n_boot=N_BOOT))
            print(gname, m, lab, round(rows[-1]["estimate"], 4), round(lo, 4), round(hi, 4), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/rq3_bootstrap.csv", index=False)
    print(out.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
