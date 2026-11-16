"""Stage 4 experiments (same cohort, features, folds, seeds and tuning as Stage 3).

  rq3         augmented vs freshness-aware (B1, B3) trained WITHOUT the five unequal delay pairs
              fixed in stage4_plan.json at Stage 3, (VLE, assessment) = (0, 1), (1, 3), (3, 7),
              (7, 14), (14, 0); each single delay value still appears in training. Scored on all
              25 pairs, so the held-out pairs test transfer to unseen combinations (RQ3).
              An earlier exploratory run that held out (7, 1), (1, 7), (14, 3), (3, 14) is kept
              in results/pilot_rq3_four_pairs/ and is not the RQ3 test.
  norecency   augmented vs freshness-aware (B1, B3) without the three recency features
              (days since last VLE activity, VLE clicks in the last 7 and 14 days) (RQ2).
  randomage   control for model capacity: the freshness-aware arm (B1, B3) trained on the same
              delayed copies but with the two age inputs shuffled across training copies, and
              scored with random ages; ages then carry no information about staleness (RQ2).
  wide        sensitivity of the baselines: B3W (wider B3 grid) and B2L (RBF-SVM on 12,000
              records instead of 6,000), fresh training, scored on all 25 pairs.
  size        training-set size: B3 fresh, augmented and freshness-aware trained on 50% and 75%
              of the training students of each outer fold (100% is the main run).
  chrono      chronological transfer: train on the 2013 presentations, test on the 2014 ones
              (shared students removed from training); B1 and B3, all three regimes, seeds 0-2.
  fullcohort  cohort sensitivity: the original 27,444-record Stage 2 cohort (including the 16
              registrations after day 28); B3 fresh, augmented and freshness-aware.

Per-module, record-level and alert-volume analyses need no refits: they are computed from
the saved test predictions of the main runs (results/predictions/) in stage4_tables.py.

Every fit purges students across inner folds as well as across the outer split.
Each experiment writes results/stage4_<name>.csv and resumes from a .partial.csv if stopped.
Run:  python scripts/stage4.py rq3 norecency randomage wide size chrono fullcohort
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run import load                                                      # noqa: E402
from staleness.experiment import (CELLS, SEEDS, augmented_training, fit_tuned,  # noqa: E402
                                  metrics, outer_folds, save_predictions, score_all_cells)
from staleness.features import BASE_COLS                                          # noqa: E402

import json                                                                         # noqa: E402
_PLAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage4_plan.json")
HOLDOUT = [tuple(p) for p in json.load(open(_PLAN))["excluded_training_and_inner_validation_pairs"]]
RECENCY = ["vle_recency", "vle_clicks_7d", "vle_clicks_14d"]
SIZES = (0.5, 0.75)
R = "results"
ALL_EXPERIMENTS = ["rq3", "norecency", "randomage", "wide", "size", "chrono", "fullcohort"]


def _arrays(cohort):
    return cohort["y"].to_numpy(), cohort["presentation"].to_numpy(), cohort["id_student"].to_numpy()


def arms_for(exp, grid, tr, y_all, g_all, s_all, seed, k):
    """Training arms for one outer fold: (name, X, y, groups, students, with_ages)."""
    if exp in ("rq3", "norecency"):
        cells = [c for c in CELLS if c not in HOLDOUT] if exp == "rq3" else None
        cols = [c for c in BASE_COLS if c not in RECENCY] if exp == "norecency" else None
        rng = np.random.default_rng(1000 * seed + k)
        X, ages, rows = augmented_training(grid, tr, rng, cells=cells, cols=cols)
        return cols, [("augmented", X, y_all[rows], g_all[rows], s_all[rows], False),
                      ("freshness_aware", np.hstack([X, ages]), y_all[rows], g_all[rows], s_all[rows], True)]
    return None, [("fresh", grid[(0, 0)].to_numpy()[tr], y_all[tr], g_all[tr], s_all[tr], False)]


def _checkpointed(out_path):
    ckpt = out_path.replace(".csv", ".partial.csv")
    done = set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        done = set(zip(prev["seed"], prev["fold"]))
    return ckpt, done


def _finish(out_path, ckpt):
    pd.read_csv(ckpt).to_csv(out_path, index=False)
    os.remove(ckpt)
    print("wrote", out_path, flush=True)


def run_experiment(exp, models, cohort, grid):
    out_path = f"{R}/stage4_{exp}.csv"
    ckpt, done = _checkpointed(out_path)
    y_all, g_all, s_all = _arrays(cohort)
    n_vle = 3 if exp == "norecency" else 6
    for seed in SEEDS:
        for k, tr, te in outer_folds(cohort, seed):
            if (seed, k) in done:
                continue
            cols, arms = arms_for(exp, grid, tr, y_all, g_all, s_all, seed, k)
            rows, preds, thrs = [], {}, {}
            for name in models:
                for arm, X, y, g, s, with_ages in arms:
                    f = fit_tuned(name, X, y, g, seed, with_ages, n_vle=n_vle, students=s)
                    cell_rows, P = score_all_cells(f, grid, te, y_all[te], with_ages, cols=cols,
                                                   return_preds=True)
                    preds[f"{name}__{arm}"], thrs[f"{name}__{arm}"] = P, f.thr
                    for r in cell_rows:
                        rows.append({"experiment": exp, "seed": seed, "fold": k, "model": name,
                                     "condition": arm, "n_train": len(y), "train_seconds": f.seconds,
                                     "threshold": f.thr, "params": str(f.params), **r})
                    r00 = [r for r in rows[-len(CELLS):] if r["d_vle"] == 0 and r["d_asm"] == 0][0]
                    print(f"{exp} seed {seed} fold {k} {name} {arm} PR-AUC(0,0)={r00['pr_auc']:.3f} "
                          f"t={f.seconds:.0f}s {f.params}", flush=True)
            if exp in ("rq3", "norecency"):
                # saved test predictions give a student-cluster bootstrap
                # (scripts/rq3_bootstrap.py, scripts/ablation_bootstrap.py)
                save_predictions(f"{R}/{exp}/predictions/{exp}_seed{seed}_fold{k}.npz", te, preds, thrs)
            pd.DataFrame(rows).to_csv(ckpt, mode="a", index=False, header=not os.path.exists(ckpt))
    _finish(out_path, ckpt)


def run_randomage(cohort, grid):
    """Same delayed copies as the Stage 3 augmented run; ages shuffled across training copies
    (training) and drawn at random from the 25 pairs (test), so they carry no information."""
    out_path = f"{R}/stage4_randomage.csv"
    ckpt, done = _checkpointed(out_path)
    y_all, g_all, s_all = _arrays(cohort)
    for seed in SEEDS:
        for k, tr, te in outer_folds(cohort, seed):
            if (seed, k) in done:
                continue
            X, ages, rows_ = augmented_training(grid, tr, np.random.default_rng(1000 * seed + k))
            shuf = np.random.default_rng(5000 + 1000 * seed + k)
            Xr = np.hstack([X, ages[shuf.permutation(len(ages))]])
            out, preds, thrs = [], {}, {}
            for name in ["B1", "B3"]:
                f = fit_tuned(name, Xr, y_all[rows_], g_all[rows_], seed, True, students=s_all[rows_])
                P = np.zeros((len(te), len(CELLS)), dtype=np.float32)
                for j, (dv, da) in enumerate(CELLS):
                    rnd = np.random.default_rng(9000 + 1000 * seed + 100 * k + 10 * dv + da)
                    fake = np.array(CELLS)[rnd.integers(0, len(CELLS), len(te))]
                    p = f.predict(np.hstack([grid[(dv, da)].to_numpy()[te], fake]))
                    P[:, j] = p
                    out.append({"experiment": "randomage", "seed": seed, "fold": k, "model": name,
                                "condition": "random_age", "d_vle": dv, "d_asm": da, **metrics(y_all[te], p, f.thr)})
                preds[f"{name}__random_age"], thrs[f"{name}__random_age"] = P, f.thr
                print(f"randomage seed {seed} fold {k} {name} t={f.seconds:.0f}s", flush=True)
            save_predictions(f"{R}/randomage/predictions/randomage_seed{seed}_fold{k}.npz", te, preds, thrs)
            pd.DataFrame(out).to_csv(ckpt, mode="a", index=False, header=not os.path.exists(ckpt))
    _finish(out_path, ckpt)


def run_size(cohort, grid):
    """B3 fresh, augmented and freshness-aware on a random 50% or 75% of the training students."""
    out_path = f"{R}/stage4_size.csv"
    ckpt, done = _checkpointed(out_path)
    y_all, g_all, s_all = _arrays(cohort)
    for seed in SEEDS:
        for k, tr, te in outer_folds(cohort, seed):
            if (seed, k) in done:
                continue
            out = []
            for frac in SIZES:
                rng = np.random.default_rng(7000 + 1000 * seed + 10 * k + int(100 * frac))
                studs = np.unique(s_all[tr])
                keep = set(rng.choice(studs, int(round(frac * len(studs))), replace=False))
                sub = tr[np.isin(s_all[tr], list(keep))]
                X, ages, rows_ = augmented_training(grid, sub, np.random.default_rng(1000 * seed + k))
                arms = [("fresh", grid[(0, 0)].to_numpy()[sub], sub, False),
                        ("augmented", X, rows_, False), ("freshness_aware", np.hstack([X, ages]), rows_, True)]
                for arm, Xa, idx, wa in arms:
                    f = fit_tuned("B3", Xa, y_all[idx], g_all[idx], seed, wa, students=s_all[idx])
                    for r in score_all_cells(f, grid, te, y_all[te], wa):
                        out.append({"experiment": "size", "seed": seed, "fold": k, "model": "B3",
                                    "fraction": frac, "condition": arm, "n_train_records": len(sub), **r})
                print(f"size seed {seed} fold {k} fraction {frac}", flush=True)
            pd.DataFrame(out).to_csv(ckpt, mode="a", index=False, header=not os.path.exists(ckpt))
    _finish(out_path, ckpt)


def run_chrono(cohort, grid):
    """Train on 2013 presentations, test on 2014 presentations; shared students removed from training."""
    out_path = f"{R}/stage4_chrono.csv"
    y_all, g_all, s_all = _arrays(cohort)
    year = cohort["code_presentation"].str[:4].to_numpy()
    tr0, te = np.flatnonzero(year == "2013"), np.flatnonzero(year == "2014")
    tr = tr0[~np.isin(s_all[tr0], np.unique(s_all[te]))]
    out = []
    for seed in SEEDS:
        X, ages, rows_ = augmented_training(grid, tr, np.random.default_rng(1000 * seed + 99))
        arms = [("fresh", grid[(0, 0)].to_numpy()[tr], tr, False),
                ("augmented", X, rows_, False), ("freshness_aware", np.hstack([X, ages]), rows_, True)]
        for name in ["B1", "B3"]:
            for arm, Xa, idx, wa in arms:
                f = fit_tuned(name, Xa, y_all[idx], g_all[idx], seed, wa, students=s_all[idx])
                for r in score_all_cells(f, grid, te, y_all[te], wa):
                    out.append({"experiment": "chrono", "seed": seed, "model": name, "condition": arm,
                                "n_train": len(tr), "n_removed_shared": len(tr0) - len(tr), "n_test": len(te), **r})
                print(f"chrono seed {seed} {name} {arm} t={f.seconds:.0f}s", flush=True)
    pd.DataFrame(out).to_csv(out_path, index=False)
    print("wrote", out_path, flush=True)


def run_fullcohort():
    """B3 on the original 27,444-record cohort (16 registrations after day 28 kept)."""
    from staleness.cohort import build_cohort
    from staleness.features import build_grid, load_sources
    sub, _ = build_cohort("data/oulad", verbose=False, day28_eligible=False)
    sgrid = build_grid(load_sources("data/oulad", sub), sub)
    out_path = f"{R}/stage4_fullcohort.csv"
    ckpt, done = _checkpointed(out_path)
    y_all, g_all, s_all = _arrays(sub)
    for seed in SEEDS:
        for k, tr, te in outer_folds(sub, seed):
            if (seed, k) in done:
                continue
            X, ages, rows_ = augmented_training(sgrid, tr, np.random.default_rng(1000 * seed + k))
            arms = [("fresh", sgrid[(0, 0)].to_numpy()[tr], tr, False),
                    ("augmented", X, rows_, False), ("freshness_aware", np.hstack([X, ages]), rows_, True)]
            out = []
            for arm, Xa, idx, wa in arms:
                f = fit_tuned("B3", Xa, y_all[idx], g_all[idx], seed, wa, students=s_all[idx])
                for r in score_all_cells(f, sgrid, te, y_all[te], wa):
                    out.append({"experiment": "fullcohort", "seed": seed, "fold": k, "model": "B3",
                                "condition": arm, "n_cohort": len(sub), **r})
            print(f"fullcohort seed {seed} fold {k}", flush=True)
            pd.DataFrame(out).to_csv(ckpt, mode="a", index=False, header=not os.path.exists(ckpt))
    _finish(out_path, ckpt)


if __name__ == "__main__":
    cohort, _, grid = load("data/oulad", "results/cache/features.pkl")
    for exp in sys.argv[1:] or ALL_EXPERIMENTS:
        if exp == "randomage":
            run_randomage(cohort, grid)
        elif exp == "size":
            run_size(cohort, grid)
        elif exp == "chrono":
            run_chrono(cohort, grid)
        elif exp == "fullcohort":
            run_fullcohort()
        elif exp == "wide":
            run_experiment(exp, ["B3W", "B2L"], cohort, grid)
        else:
            run_experiment(exp, ["B1", "B3"], cohort, grid)
