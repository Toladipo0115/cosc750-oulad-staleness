"""Uncertainty that respects the dependence in the data.

The 15 outer test folds (5 folds x 3 seeds) are not independent: the three seeds
partition the same 22 presentations, training sets overlap, and some students have
several enrollments. So no test treats the folds as independent observations.
Instead, results report:

  * fold-level mean and SD and the number of folds with a given sign (descriptive);
  * seed-level estimates: each seed is one complete partition in which every record
    is tested exactly once, so the three seed means show partition-to-partition spread;
  * a student-cluster bootstrap: students are resampled with replacement (all of a
    student's enrollments together) and the reported statistic (average precision per
    outer test fold, averaged over the 15 folds) is recomputed on each draw. Intervals
    are conditional on the fitted models: they reflect which students are evaluated,
    not retraining variability.

Average precision is computed with record weights (a record's weight is the number of
times its student is drawn), which equals scikit-learn's average_precision_score on the
expanded sample, including the treatment of tied scores.
"""
import glob
import os

import numpy as np

from .experiment import CELLS


class APIndex:
    """Precomputed ordering of one score vector for fast weighted average precision."""

    def __init__(self, p, y):
        order = np.argsort(-p, kind="mergesort")
        ps = p[order]
        self.order = order
        self.ends = np.flatnonzero(np.r_[np.diff(ps) != 0, True])   # last index of each tie group
        self.y = y[order].astype(float)

    def ap(self, w):
        """w: weights per record in the original order (counts from the bootstrap)."""
        ws = w[self.order]
        tp = np.cumsum(ws * self.y)[self.ends]
        cw = np.cumsum(ws)[self.ends]
        if tp[-1] == 0:
            return np.nan
        dtp = np.diff(tp, prepend=0.0)
        prec = np.divide(tp, cw, out=np.zeros_like(tp), where=cw > 0)
        return float(np.sum(dtp * prec) / tp[-1])


def load_pooled(condition, n, results="results", with_folds=False):
    """Pool the saved test predictions of each seed: {(model, arm): [P_seed (n x 25), ...]},
    the per-record threshold of the fold in which the record was tested and, if requested,
    the outer fold of each record per seed."""
    preds, thr, folds = {}, {}, None
    files = sorted(glob.glob(os.path.join(results, "predictions", f"{condition}_seed*_fold*.npz")))
    seeds = sorted({int(os.path.basename(f).split("_seed")[1].split("_")[0]) for f in files})
    for f in files:
        s = seeds.index(int(os.path.basename(f).split("_seed")[1].split("_")[0]))
        z = np.load(f)
        rows = z["rows"]
        if folds is None:
            folds = [np.full(n, -1) for _ in seeds]
        folds[s][rows] = int(os.path.basename(f).split("_fold")[1][:-4])
        for key in [k[3:] for k in z.files if k.startswith("p__")]:
            mk = tuple(key.split("__"))
            if mk not in preds:
                preds[mk] = [np.full((n, len(CELLS)), np.nan, dtype=np.float32) for _ in seeds]
                thr[mk] = [np.full(n, np.nan) for _ in seeds]
            preds[mk][s][rows] = z[f"p__{key}"]
            thr[mk][s][rows] = float(z[f"thr__{key}"][0])
    for mk in preds:
        assert all(not np.isnan(P).any() for P in preds[mk]), f"missing predictions for {mk}"
    return (preds, thr, folds) if with_folds else (preds, thr)


def student_weights(students, n_boot, seed=20261015):
    """Yield record weights for each bootstrap draw (students resampled with replacement)."""
    uniq, inv = np.unique(students, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        yield counts[inv].astype(float)


def interval(draws, level=0.95):
    a = (1 - level) / 2
    return float(np.nanpercentile(draws, 100 * a)), float(np.nanpercentile(draws, 100 * (1 - a)))
