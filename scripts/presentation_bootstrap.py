"""Presentation-cluster bootstrap of the paired age benefit, from saved predictions.

Complements the student-cluster bootstrap in scripts/uncertainty.py. Here the 22 module
presentations are resampled with replacement (each drawn presentation brings all of its
records), and the RQ2 statistic is recomputed on each draw: average precision per outer
test fold, averaged over the folds of a seed, the 25 delay pairs and the three seeds, for
the freshness-aware arm minus the augmented arm. No model is refitted, so the interval
reflects which presentations are evaluated, not retraining. A fold whose presentations are
all absent from a draw is skipped in that draw. Students enrolled in more than one
presentation are not tied together across presentations by this resampling.

Writes results/presentation_bootstrap.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from staleness.cohort import build_cohort
from staleness.experiment import CELLS
from staleness.inference import APIndex, interval, load_pooled

N_BOOT = 2000
SEED = 20261101


def boot_p(d):
    d = d[~np.isnan(d)]
    k = min((d <= 0).sum(), (d >= 0).sum())
    return float(min(1.0, 2 * (k + 1) / (len(d) + 1)))


def main():
    cohort, _ = build_cohort('data/oulad', verbose=False)
    y = cohort.y.to_numpy()
    pres = cohort.presentation.to_numpy()
    n = len(cohort)
    _, _, folds = load_pooled('fresh', n, with_folds=True)
    aug, _ = load_pooled('augmented', n)
    uniq, inv = np.unique(pres, return_inverse=True)
    S = len(folds)

    # fold parts per seed, with a precomputed AP index per (model, arm, seed, cell, fold)
    parts = {s: [np.flatnonzero(folds[s] == k) for k in np.unique(folds[s])] for s in range(S)}
    idx = {}
    for m in ['B1', 'B3']:
        for arm in ['augmented', 'freshness_aware']:
            for s in range(S):
                P = aug[(m, arm)][s]
                for j, _ in enumerate(CELLS):
                    idx[(m, arm, s, j)] = [APIndex(P[r, j].astype(float), y[r]) for r in parts[s]]

    def stat(w, m):
        vals = []
        for s in range(S):
            for j in range(len(CELLS)):
                a = [ix.ap(w[r]) if w[r].sum() > 0 else np.nan
                     for ix, r in zip(idx[(m, 'freshness_aware', s, j)], parts[s])]
                b = [ix.ap(w[r]) if w[r].sum() > 0 else np.nan
                     for ix, r in zip(idx[(m, 'augmented', s, j)], parts[s])]
                vals.append(np.nanmean(a) - np.nanmean(b))
        return float(np.mean(vals))

    rng = np.random.default_rng(SEED)
    weights = []
    for _ in range(N_BOOT):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        weights.append(counts[inv].astype(float))

    rows = []
    for m in ['B1', 'B3']:
        est = stat(np.ones(n), m)
        draws = np.array([stat(w, m) for w in weights])
        lo, hi = interval(draws)
        rows.append(dict(model=m, comparison='aware_minus_augmented', pairs='all 25',
                         resampled_unit='presentation', n_units=len(uniq), n_boot=N_BOOT,
                         estimate=est, ci_low=lo, ci_high=hi, p_boot=boot_p(draws),
                         share_draws_positive=float((draws > 0).mean())))
        print(m, 'done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv('results/presentation_bootstrap.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
