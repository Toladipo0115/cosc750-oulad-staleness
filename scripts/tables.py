"""Build the Stage 3 tables and figure from results/runs_*.csv and results/uncertainty_*.csv.

Fold-level values are descriptive (mean +/- SD over the 15 outer folds, 3 seeds x 5 folds);
the folds are not independent, so uncertainty comes from scripts/uncertainty.py
(seed-level estimates and student-cluster bootstrap intervals), not from fold-level tests.

Outputs (results/):
  baseline_table.csv / .tex   fresh training, fresh test, mean +/- SD over the 15 folds; the CSV also
                              holds the 95% bootstrap interval of AP(model) - AP(B3) (table caption)
  key_cells_fresh.csv         change from (0,0) at selected pairs, folds lower, bootstrap interval
  staleness_surface.csv       mean change in PR-AUC from the (0,0) cell, per model
  surface_B3.pdf              heat map of that change for B3
  preliminary.csv / .tex      augmented vs freshness-aware (if runs_augmented.csv exists)
"""
import os

import numpy as np
import pandas as pd

R = "results"
NAMES = {"B0": "B0 majority class", "B1": "B1 logistic regression",
         "B2": "B2 RBF-SVM", "B3": "B3 calibrated boosting"}
METRICS = ["pr_auc", "roc_auc", "macro_f1", "recall", "precision_top10", "brier", "train_seconds"]


def fmt_ci(lo, hi, d=3):
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]".replace("+", "$+$").replace("-", "$-$")


def unc(name):
    return pd.read_csv(f"{R}/uncertainty_{name}.csv")


def ms(x, d=3):
    return f"{np.mean(x):.{d}f} $\\pm$ {np.std(x, ddof=1):.{d}f}"


def baseline_table(df):
    f = df[(df.d_vle == 0) & (df.d_asm == 0)]
    u = unc("baselines").set_index("model")
    rows = []
    for m in sorted(f.model.unique()):
        g = f[f.model == m].sort_values(["seed", "fold"])
        row = {"model": m, "n_runs": len(g)}
        for c in METRICS:
            row[c + "_mean"], row[c + "_sd"] = g[c].mean(), g[c].std(ddof=1)
        if m != "B3":
            row["diff_vs_B3_estimate"] = u.loc[m, "estimate"]
            row["diff_vs_B3_ci_low"], row["diff_vs_B3_ci_high"] = u.loc[m, "ci_low"], u.loc[m, "ci_high"]
            row["diff_vs_B3_p_boot"] = u.loc[m, "p_boot"]
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{R}/baseline_table.csv", index=False)
    lines = []
    for _, r in t.iterrows():
        g = f[f.model == r.model]
        lines.append(f"{NAMES[r.model]} & {ms(g.pr_auc)} & {ms(g.roc_auc)} & {ms(g.macro_f1)} & "
                     f"{ms(g.recall)} & {ms(g.brier)} & {g.train_seconds.mean():.0f} \\\\")
    open(f"{R}/baseline_table.tex", "w").write("\n".join(lines) + "\n")
    return t


def surface(df):
    base = df[(df.d_vle == 0) & (df.d_asm == 0)][["seed", "fold", "model", "condition", "pr_auc"]]
    d = df.merge(base, on=["seed", "fold", "model", "condition"], suffixes=("", "_00"))
    d["delta"] = d.pr_auc - d.pr_auc_00
    s = d.groupby(["condition", "model", "d_vle", "d_asm"]).agg(
        pr_auc=("pr_auc", "mean"), delta=("delta", "mean"), delta_sd=("delta", "std")).reset_index()
    return s


def key_cells(df, condition="fresh"):
    """Change from (0,0) at selected pairs and the compounding term
    PR(dv,da) - PR(dv,0) - PR(0,da) + PR(0,0): fold mean, SD and number of folds below zero
    (descriptive), with the seed-level estimates and 95% bootstrap interval."""
    u = unc("surface")
    rows = []
    for m in sorted(df.model.unique()):
        if m == "B0":
            continue
        g = df[(df.model == m) & (df.condition == condition)].pivot_table(
            index=["seed", "fold"], columns=["d_vle", "d_asm"], values="pr_auc")
        base = g[(0, 0)]
        cand = [("change", c, g[c] - base) for c in [(7, 0), (14, 0), (0, 7), (0, 14), (7, 14), (14, 14)]]
        cand += [("interaction", c, g[c] - g[(c[0], 0)] - g[(0, c[1])] + base) for c in [(7, 14), (14, 14)]]
        for kind, c, dd in cand:
            b_ = u[(u.model == m) & (u.kind == kind) & (u.d_vle == c[0]) & (u.d_asm == c[1])].iloc[0]
            rows.append({"model": m, "cell": str(c), "kind": kind, "mean": dd.mean(), "sd": dd.std(ddof=1),
                         "folds_negative": int((dd < 0).sum()), "pooled_estimate": b_.estimate,
                         "seed_min": min(b_.seed0, b_.seed1, b_.seed2), "seed_max": max(b_.seed0, b_.seed1, b_.seed2),
                         "ci_low": b_.ci_low, "ci_high": b_.ci_high, "p_boot": b_.p_boot,
                         "p_bonf": b_.get("p_bonf", np.nan)})
    t = pd.DataFrame(rows)
    t.to_csv(f"{R}/key_cells_{condition}.csv", index=False)
    return t


def heatmap(s, model, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = s[(s.model == model) & (s.condition == "fresh")].pivot(index="d_asm", columns="d_vle", values="delta")
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    lim = np.abs(g.to_numpy()).max()
    im = ax.pcolormesh(np.arange(6)-0.5, np.arange(6)-0.5, g.to_numpy(),
                       cmap="RdBu", vmin=-lim, vmax=lim, shading="flat", rasterized=False)
    ax.set_xticks(range(5), g.columns); ax.set_yticks(range(5), g.index)
    ax.set_xlabel("VLE delay $\\delta_V$ (days)"); ax.set_ylabel("assessment delay $\\delta_A$ (days)")
    for i in range(5):
        for j in range(5):
            v = g.to_numpy()[i, j]
            r = round(v, 3)
            label = "0.000" if r == 0 else ("+" if r > 0 else "\u2212") + f"{abs(r):.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * lim else "black")
    ax.set_title(title, fontsize=8)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    bar.ax.tick_params(labelsize=6)
    bar.solids.set_rasterized(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def preliminary(df, fresh):
    """Compare fresh-trained, augmented and freshness-aware models (B1, B3), averaged over
    all 25 delay cells and at the worst cell (14, 14); paired over the 15 test folds."""
    rows, lines = [], []
    u = unc("rq2")
    both = pd.concat([df, fresh[fresh.model.isin(df.model.unique())]])
    for m in sorted(df.model.unique()):
        g = both[both.model == m]
        cell = g.groupby(["condition", "seed", "fold"]).pr_auc.mean().unstack(0)
        worst = g[(g.d_vle == 14) & (g.d_asm == 14)].groupby(["condition", "seed", "fold"]).pr_auc.mean().unstack(0)
        diff = cell["freshness_aware"] - cell["augmented"]
        aug_gain = cell["augmented"] - cell["fresh"]
        rows.append({"model": m,
                     "fresh_trained_mean": cell["fresh"].mean(), "fresh_trained_sd": cell["fresh"].std(ddof=1),
                     "augmented_mean": cell["augmented"].mean(), "augmented_sd": cell["augmented"].std(ddof=1),
                     "aware_mean": cell["freshness_aware"].mean(), "aware_sd": cell["freshness_aware"].std(ddof=1),
                     "aware_minus_augmented_mean": diff.mean(), "aware_minus_augmented_sd": diff.std(ddof=1),
                     "folds_aware_better": int((diff > 0).sum()), "n_folds": len(diff),
                     **{f"aware_vs_aug_{k}": u[(u.model == m) & (u.comparison == "aware_minus_augmented") & (u.pairs == "all 25")][k].iloc[0]
                        for k in ["estimate", "seed0", "seed1", "seed2", "ci_low", "ci_high", "p_boot"]},
                     "augmented_minus_fresh_mean": aug_gain.mean(), "augmented_minus_fresh_sd": aug_gain.std(ddof=1),
                     "folds_augmented_better": int((aug_gain > 0).sum()),
                     "worst_cell_fresh": worst["fresh"].mean(), "worst_cell_augmented": worst["augmented"].mean(),
                     "worst_cell_aware": worst["freshness_aware"].mean(),
                     "worst_cell_aug_minus_fresh_sd": (worst["augmented"] - worst["fresh"]).std(ddof=1),
                     "worst_cell_folds_aug_better": int(((worst["augmented"] - worst["fresh"]) > 0).sum()),
                     "cell00_fresh": g[(g.condition == "fresh") & (g.d_vle == 0) & (g.d_asm == 0)].pr_auc.mean(),
                     "cell00_augmented": g[(g.condition == "augmented") & (g.d_vle == 0) & (g.d_asm == 0)].pr_auc.mean(),
                     "cell00_aware": g[(g.condition == "freshness_aware") & (g.d_vle == 0) & (g.d_asm == 0)].pr_auc.mean()})
        r = rows[-1]
        lines.append(f"{NAMES[m]} & {ms(cell['fresh'])} & {ms(cell['augmented'])} & {ms(cell['freshness_aware'])} & "
                     f"{diff.mean():+.4f} $\\pm$ {diff.std(ddof=1):.4f} & {r['folds_aware_better']}/{r['n_folds']} & "
                     f"{fmt_ci(r['aware_vs_aug_ci_low'], r['aware_vs_aug_ci_high'])} \\\\")
    pd.DataFrame(rows).to_csv(f"{R}/preliminary.csv", index=False)
    open(f"{R}/preliminary.tex", "w").write("\n".join(lines) + "\n")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    fresh = pd.read_csv(f"{R}/runs_fresh.csv")
    print(baseline_table(fresh).round(4).T)
    print(key_cells(fresh).round(4))
    frames = [fresh]
    if os.path.exists(f"{R}/runs_augmented.csv"):
        aug = pd.read_csv(f"{R}/runs_augmented.csv")
        print(preliminary(aug, fresh).round(4).T)
        frames.append(aug)
    s = surface(pd.concat(frames))
    s.to_csv(f"{R}/staleness_surface.csv", index=False)
    heatmap(s, "B3", f"{R}/surface_B3.pdf", "B3: change in PR-AUC vs. (0, 0)")
    print(s[s.condition == "fresh"].pivot_table(index=["model", "d_vle"], columns="d_asm", values="delta").round(3))
