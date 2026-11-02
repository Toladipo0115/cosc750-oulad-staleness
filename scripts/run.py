"""Run one experiment and write per-fold, per-delay-cell results to CSV.

  python scripts/run.py --condition fresh --models B0 B1 B2 B3
  python scripts/run.py --condition augmented --models B1 B3
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from staleness.cohort import build_cohort          # noqa: E402
from staleness.experiment import SEEDS, run         # noqa: E402
from staleness.features import build_grid, load_sources  # noqa: E402


def write_splits(cohort, path="results/splits.csv"):
    """Record every outer split: per seed and fold, the test presentations and the counts of
    test records, training records, and training records removed because the student is in the test fold."""
    from staleness.experiment import outer_folds
    rows, n = [], len(cohort)
    for seed in SEEDS:
        for k, tr, te in outer_folds(cohort, seed):
            pres = sorted(cohort["presentation"].iloc[te].unique())
            rows.append({"seed": seed, "fold": k, "test_presentations": " ".join(pres),
                         "n_test": len(te), "n_train": len(tr), "n_train_removed_shared_student": n - len(te) - len(tr),
                         "test_prevalence": round(float(cohort["y"].iloc[te].mean()), 4)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


def load(data_dir, cache):
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    cohort, audit = build_cohort(data_dir)
    grid = build_grid(load_sources(data_dir, cohort), cohort)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump((cohort, audit, grid), fh)
    return cohort, audit, grid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/oulad")
    ap.add_argument("--condition", choices=["fresh", "augmented"], required=True)
    ap.add_argument("--models", nargs="+", default=["B0", "B1", "B2", "B3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cohort, audit, grid = load(a.data, "results/cache/features.pkl")
    write_splits(cohort)
    out = a.out or f"results/runs_{a.condition}.csv"
    ckpt = out.replace(".csv", ".partial.csv")
    df = run(cohort, grid, a.models, a.condition, seeds=a.seeds, checkpoint=ckpt)
    df.to_csv(out, index=False)
    os.remove(ckpt)
    print(f"wrote {out} ({len(df)} rows)")
