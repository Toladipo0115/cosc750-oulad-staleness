"""Grouped, nested evaluation of the baselines and the freshness-aware model.

Protocol (fixed before any test fold was scored):
  * outer: 5 folds grouped by module presentation (the 22 presentations are
    shuffled per seed and dealt into 5 folds), repeated for seeds 0, 1, 2 ->
    15 outer test folds per method;
  * training records of any student who also appears in the test fold are
    dropped, so no student is in both partitions;
  * inner: 3 folds grouped by presentation, with the same student purge
    (training records of students who appear in the inner validation fold are
    dropped), used for the grid search (average precision), for Platt-style
    calibration and for the decision threshold (maximum inner macro-F1).
    The test fold is used once;
  * every fitted model is scored on the test fold at all 25 delay pairs.
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict

from .features import AGE_COLS, BASE_COLS, DELAYS
from .models import CALIBRATE, SVM_MAX_TRAIN, SVM_MAX_TRAIN_LARGE, make_model

CELLS = [(dv, da) for dv in DELAYS for da in DELAYS]
SEEDS = (0, 1, 2)
N_OUTER, N_INNER, COPIES = 5, 3, 5


def outer_folds(cohort, seed):
    """Yield (fold, train_idx, test_idx) with student overlap removed from train."""
    # The 22 presentations are shuffled with the seed and dealt round-robin into 5
    # folds (4 or 5 presentations each), so each seed gives a different partition.
    # (StratifiedGroupKFold returned the same partition for every seed on these data.)
    pres = np.array(sorted(cohort["presentation"].unique()))
    order = np.random.default_rng(seed).permutation(pres)
    fold_of = {p: i % N_OUTER for i, p in enumerate(order)}
    fold = cohort["presentation"].map(fold_of).to_numpy()
    for k in range(N_OUTER):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        test_students = set(cohort["id_student"].to_numpy()[te])
        keep = ~cohort["id_student"].iloc[tr].isin(test_students).to_numpy()
        yield k, tr[keep], te


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    top = np.argsort(-p)[: max(1, int(round(0.10 * len(p))))]
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p),
            "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "precision_top10": y[top].mean()}


def best_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    f = [f1_score(y, (p >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(f))])


class Fitted:
    """A tuned model plus its calibration map and decision threshold."""

    def __init__(self, est, cal, thr, params, seconds):
        self.est, self.cal, self.thr, self.params, self.seconds = est, cal, thr, params, seconds

    def predict(self, X):
        p = self.est.predict_proba(X)[:, 1]
        if self.cal is not None:
            p = self.cal.predict_proba(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
        return p


def inner_splits(y, groups, students):
    """Inner folds grouped by presentation; training records of students who also
    appear in the validation fold are dropped (the same rule as the outer folds)."""
    out = []
    for tr, va in GroupKFold(n_splits=N_INNER).split(np.zeros(len(y)), y, groups):
        shared = np.isin(students[tr], np.unique(students[va]))
        out.append((tr[~shared], va))
    return out


def fit_tuned(name, X, y, groups, seed, with_ages=False, n_vle=6, n_asm=3, students=None):
    """Tune on inner folds, calibrate, set the threshold, refit on all rows.
    `students` (id per row) enables the inner student purge; every caller passes it."""
    t0 = time.time()
    if students is None:
        raise ValueError("fit_tuned needs student ids for the inner student purge")
    cap = {"B2": SVM_MAX_TRAIN, "B2L": SVM_MAX_TRAIN_LARGE}.get(name)
    if cap and len(y) > cap:
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c),
                                         int(round(cap * np.mean(y == c))), replace=False)
                              for c in (0, 1)])
        X, y, groups, students = X[idx], y[idx], groups[idx], students[idx]
    est, grid = make_model(name, with_ages, seed, n_vle, n_asm)
    inner = inner_splits(y, groups, students)
    if grid:
        gs = GridSearchCV(est, grid, scoring="average_precision", cv=inner, n_jobs=2, refit=True)
        gs.fit(X, y)
        best, params = gs.best_estimator_, gs.best_params_
    else:
        best, params = clone(est).fit(X, y), {}
    oof = cross_val_predict(clone(best), X, y, cv=inner, method="predict_proba", n_jobs=2)[:, 1]
    cal = None
    if CALIBRATE[name]:
        cal = LogisticRegression().fit(logit(np.clip(oof, 1e-6, 1 - 1e-6)).reshape(-1, 1), y)
        oof = cal.predict_proba(logit(np.clip(oof, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
    thr = best_threshold(y, oof) if name != "B0" else 0.5
    return Fitted(best, cal, thr, params, time.time() - t0)


def score_all_cells(fitted, grid, te, y_te, with_ages, cols=None, return_preds=False):
    """Metrics at every delay pair; optionally also the test predictions (len(te) x 25)."""
    rows, P = [], np.empty((len(te), len(CELLS)), dtype=np.float32)
    for j, (dv, da) in enumerate(CELLS):
        X = (grid[(dv, da)] if cols is None else grid[(dv, da)][cols]).to_numpy()[te]
        if with_ages:
            X = np.hstack([X, np.tile([dv, da], (len(te), 1))])
        # metrics use the same float32 values that are saved, so they can be recomputed exactly
        p = fitted.predict(X).astype(np.float32)
        P[:, j] = p
        p = p.astype(float)
        rows.append({"d_vle": dv, "d_asm": da, **metrics(y_te, p, fitted.thr)})
    return (rows, P) if return_preds else rows


def save_predictions(path, te, preds, thresholds):
    """One outer fold: test row indices, predictions per (model, arm) and thresholds.
    Metrics in the runs files can be recomputed from these with scripts/verify_predictions.py."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {"rows": te.astype(np.int32)}
    for key, P in preds.items():
        arrays[f"p__{key}"] = P
        arrays[f"thr__{key}"] = np.array([thresholds[key]])
    np.savez_compressed(path, **arrays)


def augmented_training(grid, tr, rng, cells=None, cols=None):
    """COPIES delay pairs per training record, sampled without replacement."""
    cells = CELLS if cells is None else cells
    picks = np.stack([rng.choice(len(cells), COPIES, replace=False) for _ in tr])
    Xs, ages, rows = [], [], []
    for j in range(COPIES):
        for c in np.unique(picks[:, j]):
            sel = tr[picks[:, j] == c]
            g = grid[cells[c]] if cols is None else grid[cells[c]][cols]
            Xs.append(g.to_numpy()[sel])
            ages.append(np.tile(cells[c], (len(sel), 1)))
            rows.append(sel)
    return np.vstack(Xs), np.vstack(ages), np.concatenate(rows)


def run(cohort, grid, models, condition, seeds=SEEDS, log=print, checkpoint=None):
    """condition: 'fresh' (train at (0,0)) or 'augmented' (both arms: no ages / ages).

    If `checkpoint` is a CSV path, finished (seed, fold) blocks are appended to it
    and skipped on a rerun, so an interrupted run resumes where it stopped."""
    y_all, g_all = cohort["y"].to_numpy(), cohort["presentation"].to_numpy()
    s_all = cohort["id_student"].to_numpy()
    out, done = [], set()
    if checkpoint and os.path.exists(checkpoint):
        prev = pd.read_csv(checkpoint)
        out = prev.to_dict("records")
        done = set(zip(prev["seed"], prev["fold"]))
    for seed in seeds:
        for k, tr, te in outer_folds(cohort, seed):
            if (seed, k) in done:
                continue
            start = len(out)
            y_te = y_all[te]
            preds, thrs = {}, {}
            if condition == "fresh":
                arms = [("fresh", grid[(0, 0)].to_numpy()[tr], y_all[tr], g_all[tr], s_all[tr], False)]
            else:
                rng = np.random.default_rng(1000 * seed + k)
                X, ages, rows = augmented_training(grid, tr, rng)
                arms = [("augmented", X, y_all[rows], g_all[rows], s_all[rows], False),
                        ("freshness_aware", np.hstack([X, ages]), y_all[rows], g_all[rows], s_all[rows], True)]
            for name in models:
                for arm, X, y, g, s, with_ages in arms:
                    f = fit_tuned(name, X, y, g, seed, with_ages, students=s)
                    cell_rows, P = score_all_cells(f, grid, te, y_te, with_ages, return_preds=True)
                    preds[f"{name}__{arm}"], thrs[f"{name}__{arm}"] = P, f.thr
                    for r in cell_rows:
                        out.append({"seed": seed, "fold": k, "model": name, "condition": arm,
                                    "n_train": len(y), "n_test": len(te),
                                    "train_seconds": f.seconds, "threshold": f.thr,
                                    "params": str(f.params), **r})
                    fresh = [r for r in out[-len(CELLS):] if r["d_vle"] == 0 and r["d_asm"] == 0][0]
                    log(f"seed {seed} fold {k} {name:>2} {arm:<16} PR-AUC(0,0)={fresh['pr_auc']:.3f} "
                        f"t={f.seconds:.0f}s {f.params}")
            save_predictions(f"results/predictions/{condition}_seed{seed}_fold{k}.npz", te, preds, thrs)
            if checkpoint:
                pd.DataFrame(out[start:]).to_csv(checkpoint, mode="a", index=False,
                                                 header=not os.path.exists(checkpoint))
    return pd.DataFrame(out)


FEATURES = BASE_COLS
FEATURES_WITH_AGES = BASE_COLS + AGE_COLS
