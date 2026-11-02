"""Reproduction check for B3 against Wu, Duan and Luo (2026).

They report PR-AUC 0.664 for calibrated gradient boosting on withdrawal at
28 days, keeping all 32,593 enrollments (students already withdrawn by day 28
included) and using richer features than ours (assessment scores and
demographics). We cannot run their code here, so we re-run our B3 under the
same protocol on their cohort definition with our 12 features, to separate
the effect of the cohort from the effect of the features.

Writes results/reproduction_check.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from staleness.cohort import JOIN_KEYS                            # noqa: E402
from staleness.experiment import SEEDS, fit_tuned, metrics, outer_folds  # noqa: E402
from staleness.features import (BASE_COLS, asm_features, load_sources,  # noqa: E402
                                reg_features, vle_features)

DATA = "data/oulad"

if __name__ == "__main__":
    reg = pd.read_csv(f"{DATA}/studentRegistration.csv", na_values="?")
    info = pd.read_csv(f"{DATA}/studentInfo.csv")
    full = reg.merge(info, on=JOIN_KEYS, validate="one_to_one").reset_index(drop=True)
    full["y"] = full["final_result"].eq("Withdrawn").astype(int)
    full["presentation"] = full["code_module"] + "_" + full["code_presentation"]
    src = load_sources(DATA, full)
    X = pd.concat([vle_features(src, 0), asm_features(src, full, 0), reg_features(full)], axis=1)[BASE_COLS]
    X, y, g = X.to_numpy(), full["y"].to_numpy(), full["presentation"].to_numpy()
    print(f"all enrollments: n={len(y):,}, withdrawals={y.sum():,} ({y.mean():.3f})")
    rows = []
    for seed in SEEDS:
        for k, tr, te in outer_folds(full, seed):
            f = fit_tuned("B3", X[tr], y[tr], g[tr], seed, students=full["id_student"].to_numpy()[tr])
            rows.append({"seed": seed, "fold": k, **metrics(y[te], f.predict(X[te]), f.thr)})
            print(seed, k, round(rows[-1]["pr_auc"], 3), round(rows[-1]["roc_auc"], 3))
    d = pd.DataFrame(rows)
    d.to_csv("results/reproduction_check.csv", index=False)
    print(d[["pr_auc", "roc_auc"]].agg(["mean", "std"]).round(3))
