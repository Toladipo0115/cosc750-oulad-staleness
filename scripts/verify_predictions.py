"""Recompute the reported metrics from the saved test predictions and check that they match
results/runs_<condition>.csv exactly. Run after `make baselines` and `make preliminary`.

  python scripts/verify_predictions.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run import load                                # noqa: E402
from staleness.experiment import CELLS, metrics             # noqa: E402

if __name__ == "__main__":
    missing = [f"results/runs_{c}.csv" for c in ["fresh", "augmented"]
               if not os.path.exists(f"results/runs_{c}.csv")]
    if missing:
        sys.exit("Missing " + ", ".join(missing) +
                 ". Run make baselines and make preliminary before make verify.")
    cohort, _, _ = load("data/oulad", "results/cache/features.pkl")
    y = cohort["y"].to_numpy()
    checked = 0
    for cond in ["fresh", "augmented"]:
        runs = pd.read_csv(f"results/runs_{cond}.csv").set_index(["seed", "fold", "model", "condition", "d_vle", "d_asm"])
        for path in sorted(glob.glob(f"results/predictions/{cond}_seed*_fold*.npz")):
            name = os.path.basename(path)[:-4]
            seed, fold = int(name.split("_seed")[1].split("_")[0]), int(name.split("_fold")[1])
            z = np.load(path)
            te = z["rows"]
            for key in [k[3:] for k in z.files if k.startswith("p__")]:
                model, arm = key.split("__")
                P, thr = z[f"p__{key}"], float(z[f"thr__{key}"][0])
                for j, (dv, da) in enumerate(CELLS):
                    m = metrics(y[te], P[:, j].astype(float), thr)
                    ref = runs.loc[(seed, fold, model, arm, dv, da)]
                    for col in ["pr_auc", "roc_auc", "macro_f1", "recall", "brier", "precision_top10"]:
                        assert abs(m[col] - ref[col]) < 1e-12, (path, key, (dv, da), col, m[col], ref[col])
                    checked += 1
    print(f"all {checked:,} metric cells recomputed from saved predictions match the runs files")
