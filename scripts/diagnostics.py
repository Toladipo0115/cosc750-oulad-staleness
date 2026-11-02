"""Diagnostics for the Stage 3 baselines (same folds, seeds and tuning as scripts/run.py).

Writes to results/:
  confusion_B3.csv / confusion_B3.pdf   B3 confusion matrix at its inner-fold threshold,
                                        pooled over the 15 test folds, at (0,0), (0,14), (14,14)
  importance_B3.csv / importance_B3.pdf permutation importance of B3 (drop in test PR-AUC when a
                                        feature is shuffled), fresh data vs both sources 14 days late
  model_comparison.pdf                  PR-AUC of B0 to B3 (mean +/- SD over 15 folds)

Run:  python scripts/diagnostics.py      (about 3 minutes on a 2-core CPU)
      python scripts/diagnostics.py --figures-only   (redraw from the saved CSVs)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run import load                                   # noqa: E402
from staleness.experiment import SEEDS, fit_tuned, outer_folds  # noqa: E402
from staleness.features import BASE_COLS                       # noqa: E402

R = "results"
CELLS = [(0, 0), (0, 14), (14, 14)]
N_REPEATS = 5
LABELS = {"vle_clicks": "VLE total clicks", "vle_active_days": "VLE active days",
          "vle_resources": "VLE distinct resources", "vle_recency": "Days since last VLE activity",
          "vle_clicks_7d": "VLE clicks, last 7 days", "vle_clicks_14d": "VLE clicks, last 14 days",
          "asm_submitted": "Submissions observed", "asm_missed": "Past deadline, not submitted",
          "asm_days_early": "Mean days submitted early", "prev_attempts": "Previous attempts",
          "studied_credits": "Studied credits", "registration_day": "Registration day"}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def permutation_drop(fitted, X, y, rng):
    base = average_precision_score(y, fitted.predict(X))
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(N_REPEATS):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - average_precision_score(y, fitted.predict(Xp)))
        out[j] = np.mean(drops)
    return out


def main():
    cohort, _, grid = load("data/oulad", "results/cache/features.pkl")
    y_all, g_all = cohort["y"].to_numpy(), cohort["presentation"].to_numpy()
    conf, imp, check = [], [], []
    for seed in SEEDS:
        for k, tr, te in outer_folds(cohort, seed):
            f = fit_tuned("B3", grid[(0, 0)].to_numpy()[tr], y_all[tr], g_all[tr], seed,
                          students=cohort["id_student"].to_numpy()[tr])
            rng = np.random.default_rng(10_000 + 100 * seed + k)
            for dv, da in CELLS:
                X = grid[(dv, da)].to_numpy()[te]
                p = f.predict(X)
                tn, fp, fn, tp = confusion_matrix(y_all[te], (p >= f.thr).astype(int), labels=[0, 1]).ravel()
                conf.append({"seed": seed, "fold": k, "d_vle": dv, "d_asm": da,
                             "tn": tn, "fp": fp, "fn": fn, "tp": tp})
                if (dv, da) == (0, 0):
                    check.append(average_precision_score(y_all[te], p))
                if (dv, da) in [(0, 0), (14, 14)]:
                    for name, v in zip(BASE_COLS, permutation_drop(f, X, y_all[te], rng)):
                        imp.append({"seed": seed, "fold": k, "d_vle": dv, "d_asm": da,
                                    "feature": name, "pr_auc_drop": v})
            print(f"seed {seed} fold {k} done; PR-AUC(0,0)={check[-1]:.3f}")

    # the refitted models must reproduce the baseline run exactly
    ref = pd.read_csv(f"{R}/runs_fresh.csv")
    ref = ref[(ref.model == "B3") & (ref.d_vle == 0) & (ref.d_asm == 0)].sort_values(["seed", "fold"]).pr_auc.to_numpy()
    assert np.allclose(ref, check), "diagnostic refit differs from runs_fresh.csv"

    c = pd.DataFrame(conf)
    pooled = c.groupby(["d_vle", "d_asm"])[["tn", "fp", "fn", "tp"]].sum().reset_index()
    pooled["recall"] = pooled.tp / (pooled.tp + pooled.fn)
    pooled["precision"] = pooled.tp / (pooled.tp + pooled.fp)
    pooled["flag_rate"] = (pooled.tp + pooled.fp) / pooled[["tn", "fp", "fn", "tp"]].sum(axis=1)
    pooled["note"] = "pooled over 15 test folds (each record appears 3 times, once per seed)"
    pooled.to_csv(f"{R}/confusion_B3.csv", index=False)
    c.to_csv(f"{R}/confusion_B3_per_fold.csv", index=False)

    i = pd.DataFrame(imp)
    i.to_csv(f"{R}/importance_B3_per_fold.csv", index=False)
    s = i.groupby(["feature", "d_vle", "d_asm"]).pr_auc_drop.agg(["mean", "std"]).reset_index()
    wide = s.pivot_table(index="feature", columns=["d_vle"], values=["mean", "std"])
    wide.columns = [f"{a}_{'fresh' if b == 0 else 'stale14'}" for a, b in wide.columns]
    wide = wide.sort_values("mean_fresh", ascending=False).reset_index()
    wide.to_csv(f"{R}/importance_B3.csv", index=False)
    print(pooled.round(3).to_string(index=False))
    print(wide.round(4).to_string(index=False))
    figures(pooled, wide)


def figures(pooled, wide):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})

    # 1. confusion matrices (counts and row percentages), one panel per delay cell
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    titles = {(0, 0): "Fresh (0, 0)", (0, 14): "Assessments 14 days late (0, 14)",
              (14, 14): "Both 14 days late (14, 14)"}
    vmax = pooled[["tn", "fp", "fn", "tp"]].to_numpy().max()
    for ax, (_, r) in zip(axes, pooled.iterrows()):
        m = np.array([[r.tp, r.fn], [r.fp, r.tn]])
        ax.pcolormesh(np.arange(3)-0.5, np.arange(3)-0.5, m, cmap="Blues",
                      vmin=0, vmax=vmax, shading="flat", rasterized=False)
        ax.set_ylim(1.5, -0.5)
        rows = m.sum(axis=1, keepdims=True)
        for a in range(2):
            for b in range(2):
                v = m[a, b]
                ax.text(b, a, f"{v:,}\n({100 * v / rows[a, 0]:.0f}%)", ha="center", va="center",
                        color="white" if v > 0.55 * vmax else INK, fontsize=7.5)
        ax.set_xticks([0, 1], ["Withdrawal", "No\nwithdrawal"], fontsize=7)
        ax.set_yticks([0, 1], ["Withdrawal", "No\nwithdrawal"], fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_title(titles[(r.d_vle, r.d_asm)], fontsize=8, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(f"{R}/confusion_B3.pdf")
    plt.close(fig)

    # 2. model comparison
    b = pd.read_csv(f"{R}/baseline_table.csv").set_index("model").loc[["B0", "B1", "B2", "B3"]]
    names = ["B0 majority\nclass", "B1 logistic\nregression", "B2 RBF-SVM", "B3 calibrated\nboosting"]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    x = np.arange(4)
    ax.bar(x, b.pr_auc_mean, width=0.5, color=BLUE, yerr=b.pr_auc_sd, capsize=3,
           error_kw={"ecolor": MUTED, "elinewidth": 1})
    for xi, (mval, sd) in enumerate(zip(b.pr_auc_mean, b.pr_auc_sd)):
        ax.text(xi, mval + sd + 0.008, f"{mval:.3f}", ha="center", color=INK, fontsize=7.5)
    ax.set_xticks(x, names)
    ax.set_ylabel("Test PR-AUC (mean ± SD, 15 folds)")
    ax.set_ylim(0, 0.42)
    ax.yaxis.grid(True, color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{R}/model_comparison.pdf")
    plt.close(fig)

    # 3. permutation importance, fresh vs both sources 14 days late
    w = wide.iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    yv = np.arange(len(w))
    h = 0.36
    ax.barh(yv + h / 2 + 0.02, w.mean_fresh, height=h, color=BLUE, label="Fresh data (0, 0)")
    ax.barh(yv - h / 2 - 0.02, w.mean_stale14, height=h, color=ORANGE, label="Both sources 14 days late")
    ax.set_yticks(yv, [LABELS[f] for f in w.feature])
    ax.axvline(0, color=MUTED, linewidth=0.6)
    ax.set_xlabel("Drop in test PR-AUC when the feature is shuffled")
    ax.xaxis.grid(True, color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{R}/importance_B3.pdf")
    plt.close(fig)


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        figures(pd.read_csv(f"{R}/confusion_B3.csv"), pd.read_csv(f"{R}/importance_B3.csv"))
    else:
        main()
