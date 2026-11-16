"""Stage 4 tables and figures.

Fold-level numbers are descriptive (mean +/- SD over 15 outer folds, folds with a given sign).
The folds are not independent, so uncertainty comes from (a) seed-level estimates, one per
complete partition, and (b) student-cluster bootstrap intervals from scripts/uncertainty.py.
No fold-level significance test is used.

Reads results/runs_*.csv, results/stage4_*.csv, results/uncertainty_*.csv and the saved
predictions in results/predictions/. Writes results/:
  stage4_surface.csv          change from (0,0) for every pair and model, fold summary, seed range,
                              95% and Bonferroni (24 pairs) bootstrap intervals          (RQ1)
  stage4_surfaces.pdf         the three surfaces; bold where the Bonferroni interval excludes 0
  stage4_main.csv             every method and regime at (0,0), (14,14) and over all 25 pairs,
                              with the bootstrap interval of freshness-aware B3 minus each method
  stage4_rq2_rq3.csv          freshness-aware minus augmented by experiment and pair group, random-age
                              control, cost of holding pairs out of training              (RQ2, RQ3)
  stage4_rq2_cells.csv        regime comparisons at single pairs and by assessment freshness
  stage4_wide_summary.csv     wider B3 grid and larger SVM sample vs the main baselines
  stage4_size_summary.csv     training-set size: 50%, 75%, 100% of training students
  stage4_size.pdf             the same as a figure
  stage4_prevalence.csv       evaluation at prevalence 0.10, 0.183, 0.30 (class imbalance)
  stage4_chrono_summary.csv   train 2013 -> test 2014
  stage4_fullcohort_summary.csv  original 27,444-record cohort vs the primary cohort
  stage4_modules.csv / .pdf   B3 per module, fresh vs assessments 14 days late
  stage4_false_alarms.csv     record-level check of the false alarms added by assessment delay
  stage4_alerts_summary.csv   share flagged, recall, precision by regime, with intervals
  stage4_alerts.pdf           the same as a figure
  stage4_alerts_share.pdf     its share-flagged panel alone, for the slides
  stage4_alerts_budget.csv    precision among the top 10% of scores (matched alert budget) by regime
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R = "results"
import json                                                                         # noqa: E402
_PLAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage4_plan.json")
HOLDOUT = {tuple(p) for p in json.load(open(_PLAN))["excluded_training_and_inner_validation_pairs"]}
L = [0, 1, 3, 7, 14]
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    return plt


def fold_summary(diff):
    """Descriptive summary of a per-fold difference Series indexed by (seed, fold)."""
    seeds = diff.groupby(level=0).mean()
    return {"mean": diff.mean(), "sd": diff.std(ddof=1), "folds_positive": int((diff > 0).sum()),
            "n_folds": len(diff), "seed_min": seeds.min(), "seed_max": seeds.max()}


def unc(name):
    return pd.read_csv(f"{R}/uncertainty_{name}.csv")


# ----------------------------------------------------------------------------- RQ1
def surface():
    f = pd.read_csv(f"{R}/runs_fresh.csv")
    u = unc("surface")
    rows = []
    for m in ["B1", "B2", "B3"]:
        g = f[f.model == m].pivot_table(index=["seed", "fold"], columns=["d_vle", "d_asm"], values="pr_auc")
        for _, b in u[(u.model == m) & (u.kind == "change")].iterrows():
            dd = g[(b.d_vle, b.d_asm)] - g[(0, 0)]
            s = fold_summary(dd)
            rows.append({"model": m, "d_vle": b.d_vle, "d_asm": b.d_asm, "fold_mean_change": s["mean"],
                         "fold_sd": s["sd"], "folds_lower": s["n_folds"] - s["folds_positive"],
                         "pooled_estimate": b.estimate, "seed_min": min(b.seed0, b.seed1, b.seed2),
                         "seed_max": max(b.seed0, b.seed1, b.seed2), "ci_low": b.ci_low, "ci_high": b.ci_high,
                         "bonf_low": b.bonf_low, "bonf_high": b.bonf_high, "p_boot": b.p_boot, "p_bonf": b.p_bonf,
                         "bonf_excludes_zero": bool(b.bonf_high < 0 or b.bonf_low > 0)})
    t = pd.DataFrame(rows)
    t.to_csv(f"{R}/stage4_surface.csv", index=False)
    return t


def surface_figure(t):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("loss", ["#f7f6f3", "#fbe1d2", "#f6b993", "#ef9a6a"])
    vmax = -t.fold_mean_change.min()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.32))
    names = {"B1": "B1 logistic regression", "B2": "B2 RBF-SVM", "B3": "B3 calibrated boosting"}
    for ax, m in zip(axes, ["B1", "B2", "B3"]):
        g = t[t.model == m]
        M = np.zeros((5, 5)); sig = np.zeros((5, 5), bool)
        for _, r in g.iterrows():
            M[L.index(r.d_vle), L.index(r.d_asm)] = -r.fold_mean_change
            sig[L.index(r.d_vle), L.index(r.d_asm)] = r.bonf_excludes_zero
        # pcolormesh keeps the heatmap as vector cells (imshow would embed a raster image)
        edges = np.arange(6) - 0.5
        im = ax.pcolormesh(edges, edges, np.clip(M, 0, None), cmap=cmap, vmin=0, vmax=vmax,
                           edgecolors="face", linewidth=0)
        ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5)
        for i in range(5):
            for j in range(5):
                if i == 0 and j == 0:
                    ax.text(j, i, "ref.", ha="center", va="center", fontsize=6.5, color=MUTED)
                    continue
                v = 0.0 if abs(M[i, j]) < 0.0005 else M[i, j]
                txt = f"{v:.3f}" if v >= 0 else f"\u2212{-v:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                        color=INK if sig[i, j] else "#5f5e5a",
                        fontweight="bold" if sig[i, j] else "normal")
        ax.set_xticks(range(5), L); ax.set_yticks(range(5), L)
        ax.set_xlabel("Assessment delay (days)", fontsize=7, labelpad=1); ax.set_title(names[m], fontsize=7.5, pad=3)
        ax.tick_params(labelsize=6.5, length=2, pad=1)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("VLE delay (days)", fontsize=7, labelpad=1)
    fig.subplots_adjust(wspace=0.12)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("PR-AUC lost vs (0, 0)", fontsize=7); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5, length=2)
    # draw the colour bar as vector strips (matplotlib rasterises long colour bars by default)
    cb.solids.set_rasterized(False); cb.solids.set_edgecolor("face")
    fig.savefig(f"{R}/stage4_surfaces.pdf", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- main comparison
def main_comparison():
    f = pd.read_csv(f"{R}/runs_fresh.csv"); a = pd.read_csv(f"{R}/runs_augmented.csv")
    d = pd.concat([f, a]); d["cond"] = d.condition.replace({"freshness_aware": "aware"})
    u = unc("rq2").set_index("comparison")
    key = ["seed", "fold"]
    rows = []
    for m, c in [("B0", "fresh"), ("B1", "fresh"), ("B2", "fresh"), ("B3", "fresh"),
                 ("B1", "augmented"), ("B1", "aware"), ("B3", "augmented"), ("B3", "aware")]:
        s = d[(d.model == m) & (d.cond == c)]
        allm = s.groupby(key).pr_auc.mean()
        c00 = s[(s.d_vle == 0) & (s.d_asm == 0)].set_index(key).pr_auc
        c14 = s[(s.d_vle == 14) & (s.d_asm == 14)].set_index(key).pr_auc
        r = {"model": m, "training": c, "pr_auc_00": c00.mean(), "pr_auc_00_sd": c00.std(ddof=1),
             "pr_auc_1414": c14.mean(), "pr_auc_1414_sd": c14.std(ddof=1),
             "pr_auc_all25": allm.mean(), "pr_auc_all25_sd": allm.std(ddof=1),
             "roc_auc_all25": s.groupby(key).roc_auc.mean().mean()}
        if (m, c) != ("B3", "aware"):
            b = u.loc[f"aware_B3_minus_{m}_{'freshness_aware' if c == 'aware' else c}"]
            r.update({"aware_B3_minus_this": b.estimate, "ci_low": b.ci_low, "ci_high": b.ci_high, "p_boot": b.p_boot,
                      "seed_min": min(b.seed0, b.seed1, b.seed2), "seed_max": max(b.seed0, b.seed1, b.seed2)})
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(f"{R}/stage4_main.csv", index=False)
    return t


# ----------------------------------------------------------------------------- RQ2 / RQ3
def _group(dv, da):
    if (dv, da) in HOLDOUT:
        return "held_out_unequal"
    return "equal" if dv == da else "seen_unequal"


def _pairs(d, a, b, grp):
    s = d if grp == "all" else d[[_group(x, y) == grp for x, y in zip(d.d_vle, d.d_asm)]]
    c = s.groupby(["condition", "seed", "fold"]).pr_auc.mean().unstack(0)
    return c[a] - c[b], c


def rq2_rq3():
    aug = pd.read_csv(f"{R}/runs_augmented.csv")
    frames = {"main": aug}
    for e in ["rq3", "norecency"]:
        if os.path.exists(f"{R}/stage4_{e}.csv"):
            frames[e] = pd.read_csv(f"{R}/stage4_{e}.csv")
    rows = []
    for name, d in frames.items():
        for m in sorted(d.model.unique()):
            for grp in ["all", "held_out_unequal", "seen_unequal", "equal"]:
                diff, c = _pairs(d[d.model == m], "freshness_aware", "augmented", grp)
                rows.append({"experiment": name, "model": m, "pairs": grp, "comparison": "aware_minus_augmented",
                             "augmented": c["augmented"].mean(), "aware": c["freshness_aware"].mean(),
                             **fold_summary(diff)})
    if os.path.exists(f"{R}/stage4_randomage.csv"):
        ra = pd.read_csv(f"{R}/stage4_randomage.csv")
        d = pd.concat([aug, ra])
        for m in ["B1", "B3"]:
            for grp in ["all", "held_out_unequal"]:
                for lab, a, b in [("random_minus_augmented", "random_age", "augmented"),
                                  ("aware_minus_random", "freshness_aware", "random_age")]:
                    diff, c = _pairs(d[d.model == m], a, b, grp)
                    rows.append({"experiment": "randomage", "model": m, "pairs": grp, "comparison": lab,
                                 "augmented": c["augmented"].mean(), "aware": c["freshness_aware"].mean(),
                                 "random_age": c["random_age"].mean(), **fold_summary(diff)})
    if "rq3" in frames:
        for m in ["B1", "B3"]:
            for arm in ["augmented", "freshness_aware"]:
                sel = lambda x: x[(x.model == m) & (x.condition == arm) &                        # noqa: E731
                                  np.array([(i, j) in HOLDOUT for i, j in zip(x.d_vle, x.d_asm)])]
                ca = sel(frames["rq3"]).groupby(["seed", "fold"]).pr_auc.mean()
                cb = sel(aug).groupby(["seed", "fold"]).pr_auc.mean()
                rows.append({"experiment": "rq3_minus_full_training", "model": m, "pairs": "held_out_unequal",
                             "comparison": f"{arm}: trained without the {len(HOLDOUT)} pairs minus trained with them",
                             "augmented": cb.mean(), "aware": ca.mean(), **fold_summary(ca - cb)})
    t = pd.DataFrame(rows); t.to_csv(f"{R}/stage4_rq2_rq3.csv", index=False)
    return t


def rq2_cells():
    f = pd.read_csv(f"{R}/runs_fresh.csv"); a = pd.read_csv(f"{R}/runs_augmented.csv")
    d = pd.concat([f, a]).pivot_table(index=["model", "seed", "fold", "d_vle", "d_asm"],
                                       columns="condition", values="pr_auc").reset_index()
    u = unc("rq2")
    rows = []
    for _, b in u[~u.comparison.str.startswith("aware_B3_minus")].iterrows():
        s = d[d.model == b.model]
        if b.pairs == "all 25":
            s = s
        elif b.pairs.startswith("("):
            dv, da = map(int, b.pairs.strip("()").split(","))
            s = s[(s.d_vle == dv) & (s.d_asm == da)]
        elif b.pairs == "assessment delay <= 1":
            s = s[s.d_asm <= 1]
        else:
            s = s[s.d_asm == 14]
        x, y = {"aware_minus_augmented": ("freshness_aware", "augmented"),
                "augmented_minus_fresh": ("augmented", "fresh"),
                "aware_minus_fresh": ("freshness_aware", "fresh")}[b.comparison]
        diff = (s[x] - s[y]).groupby([s.seed, s.fold]).mean()
        rows.append({"model": b.model, "comparison": b.comparison, "pairs": b.pairs, **fold_summary(diff),
                     "pooled_estimate": b.estimate, "ci_low": b.ci_low, "ci_high": b.ci_high, "p_boot": b.p_boot})
    t = pd.DataFrame(rows); t.to_csv(f"{R}/stage4_rq2_cells.csv", index=False)
    return t


# ----------------------------------------------------------------------------- sensitivity
def wide():
    if not os.path.exists(f"{R}/stage4_wide.csv"):
        return None
    w = pd.read_csv(f"{R}/stage4_wide.csv"); f = pd.read_csv(f"{R}/runs_fresh.csv")
    rows = []
    for new, old in [("B3W", "B3"), ("B2L", "B2")]:
        for cell in [(0, 0), (0, 14), (14, 14)]:
            a = w[(w.model == new) & (w.d_vle == cell[0]) & (w.d_asm == cell[1])].set_index(["seed", "fold"]).pr_auc
            b = f[(f.model == old) & (f.d_vle == cell[0]) & (f.d_asm == cell[1])].set_index(["seed", "fold"]).pr_auc
            rows.append({"comparison": f"{new} vs {old}", "d_vle": cell[0], "d_asm": cell[1],
                         "old_pr_auc": b.mean(), "new_pr_auc": a.mean(), "new_sd": a.std(ddof=1),
                         **fold_summary(a - b),
                         "new_train_seconds": w[w.model == new].groupby(["seed", "fold"]).train_seconds.first().mean(),
                         "old_train_seconds": f[f.model == old].groupby(["seed", "fold"]).train_seconds.first().mean()})
    sel = w[(w.d_vle == 0) & (w.d_asm == 0)].groupby("model").params.agg(lambda s: s.value_counts().index[0])
    t = pd.DataFrame(rows); t["most_selected_params_new"] = t.comparison.str.split().str[0].map(sel)
    t.to_csv(f"{R}/stage4_wide_summary.csv", index=False)
    return t


def size():
    if not os.path.exists(f"{R}/stage4_size.csv"):
        return None
    s = pd.read_csv(f"{R}/stage4_size.csv")
    f = pd.read_csv(f"{R}/runs_fresh.csv"); a = pd.read_csv(f"{R}/runs_augmented.csv")
    full = pd.concat([f[f.model == "B3"], a[a.model == "B3"]]).assign(fraction=1.0)
    d = pd.concat([s, full])
    rows = []
    for (frac, cond), g in d.groupby(["fraction", "condition"]):
        per = g.groupby(["seed", "fold"])
        rows.append({"fraction": frac, "condition": cond,
                     "pr_auc_00": g[(g.d_vle == 0) & (g.d_asm == 0)].pr_auc.mean(),
                     "pr_auc_1414": g[(g.d_vle == 14) & (g.d_asm == 14)].pr_auc.mean(),
                     "pr_auc_all25": per.pr_auc.mean().mean(), "pr_auc_all25_sd": per.pr_auc.mean().std(ddof=1)})
    t = pd.DataFrame(rows)
    gain = []
    for frac, g in d.groupby("fraction"):
        diff, _ = _pairs(g, "freshness_aware", "augmented", "all")
        gain.append({"fraction": frac, "condition": "aware_minus_augmented", **fold_summary(diff)})
    t = pd.concat([t, pd.DataFrame(gain)], ignore_index=True)
    t.to_csv(f"{R}/stage4_size.csv".replace(".csv", "_summary.csv"), index=False)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for cond, lab, col in [("fresh", "Fresh-trained", BLUE), ("augmented", "Augmented", ORANGE),
                           ("freshness_aware", "Freshness-aware", AQUA)]:
        g = t[t.condition == cond].sort_values("fraction")
        ax.plot(g.fraction * 100, g.pr_auc_all25, marker="o", color=col, label=lab, linewidth=2, markersize=5)
    ax.set_xticks([50, 75, 100]); ax.set_xlabel("Training students used (%)")
    ax.set_ylabel("B3 PR-AUC, mean of 25 pairs"); ax.yaxis.grid(True, color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{R}/stage4_size.pdf"); plt.close(fig)
    return t


def prevalence():
    """Class-imbalance sensitivity: reweight the pooled test predictions so the evaluated
    prevalence is 0.10, 0.183 (actual) or 0.30; the trained models are unchanged."""
    from scripts.run import load
    from staleness.experiment import CELLS
    from staleness.inference import APIndex, load_pooled
    cohort, _, _ = load("data/oulad", "results/cache/features.pkl")
    y = cohort["y"].to_numpy(); n = len(y); base = y.mean()
    fr, _ = load_pooled("fresh", n); au, _ = load_pooled("augmented", n)
    P = {**fr, **au}
    rows = []
    for target in [0.10, base, 0.30]:
        w = np.where(y == 1, target / base, (1 - target) / (1 - base))
        for key in [("B3", "fresh"), ("B3", "augmented"), ("B3", "freshness_aware"), ("B1", "fresh")]:
            vals = [[APIndex(P[key][s][:, j].astype(float), y).ap(w) for j in range(len(CELLS))] for s in range(3)]
            v = np.array(vals)
            rows.append({"prevalence": round(target, 3), "model": key[0], "training": key[1],
                         "pr_auc_00": v[:, CELLS.index((0, 0))].mean(), "pr_auc_1414": v[:, CELLS.index((14, 14))].mean(),
                         "pr_auc_all25": v.mean(), "loss_0_14": (v[:, CELLS.index((0, 14))] - v[:, CELLS.index((0, 0))]).mean()})
    t = pd.DataFrame(rows); t.to_csv(f"{R}/stage4_prevalence.csv", index=False)
    return t


def chrono():
    if not os.path.exists(f"{R}/stage4_chrono.csv"):
        return None
    c = pd.read_csv(f"{R}/stage4_chrono.csv")
    rows = []
    for (m, cond), g in c.groupby(["model", "condition"]):
        per_seed = g.groupby("seed").pr_auc.mean()
        rows.append({"model": m, "condition": cond, "n_train": g.n_train.iloc[0], "n_test": g.n_test.iloc[0],
                     "n_removed_shared": g.n_removed_shared.iloc[0],
                     "pr_auc_00": g[(g.d_vle == 0) & (g.d_asm == 0)].pr_auc.mean(),
                     "pr_auc_0_14": g[(g.d_vle == 0) & (g.d_asm == 14)].pr_auc.mean(),
                     "pr_auc_1414": g[(g.d_vle == 14) & (g.d_asm == 14)].pr_auc.mean(),
                     "pr_auc_all25": per_seed.mean(), "seed_min": per_seed.min(), "seed_max": per_seed.max()})
    t = pd.DataFrame(rows)
    t.to_csv(f"{R}/stage4_chrono_summary.csv", index=False)
    return t


def fullcohort():
    if not os.path.exists(f"{R}/stage4_fullcohort.csv"):
        return None
    fc = pd.read_csv(f"{R}/stage4_fullcohort.csv")
    f = pd.read_csv(f"{R}/runs_fresh.csv"); a = pd.read_csv(f"{R}/runs_augmented.csv")
    prim = pd.concat([f[f.model == "B3"], a[a.model == "B3"]])
    rows = []
    for name, d in [("primary_27428", prim), ("original_27444", fc)]:
        g = d.pivot_table(index=["seed", "fold", "d_vle", "d_asm"], columns="condition", values="pr_auc").reset_index()
        fr = g[(g.d_vle == 0) & (g.d_asm == 0)]
        loss = g[(g.d_vle == 0) & (g.d_asm == 14)].set_index(["seed", "fold"]).fresh - fr.set_index(["seed", "fold"]).fresh
        loss2 = g[(g.d_vle == 14) & (g.d_asm == 14)].set_index(["seed", "fold"]).fresh - fr.set_index(["seed", "fold"]).fresh
        gain = (g.freshness_aware - g.augmented).groupby([g.seed, g.fold]).mean()
        all25 = g.groupby(["seed", "fold"])[["fresh", "augmented", "freshness_aware"]].mean().mean()
        rows.append({"cohort": name, "b3_fresh_00": fr.fresh.mean(),
                     "loss_0_14": loss.mean(), "loss_14_14": loss2.mean(),
                     "fresh_all25": all25.fresh, "augmented_all25": all25.augmented, "aware_all25": all25.freshness_aware,
                     "aware_minus_augmented_all25": gain.mean(), "folds_aware_better": int((gain > 0).sum())})
    t = pd.DataFrame(rows); t.to_csv(f"{R}/stage4_fullcohort_summary.csv", index=False)
    return t


# ----------------------------------------------------------------------------- errors and mechanism
def modules_and_records():
    from sklearn.metrics import average_precision_score
    from scripts.run import load
    from staleness.experiment import CELLS
    from staleness.inference import load_pooled
    cohort, _, grid = load("data/oulad", "results/cache/features.pkl")
    y = cohort["y"].to_numpy(); n = len(y)
    P, T = load_pooled("fresh", n)
    P, T = P[("B3", "fresh")], T[("B3", "fresh")]
    j00, j014 = CELLS.index((0, 0)), CELLS.index((0, 14))
    a = pd.read_csv("data/oulad/assessments.csv"); a["date"] = pd.to_numeric(a.date, errors="coerce")
    early_due = a[(a.assessment_type != "Exam") & (a.date <= 27)].groupby("code_module").size()
    submitted_early = (grid[(0, 0)]["asm_submitted"].to_numpy() > 0)
    rows = []
    mod = cohort["code_module"].to_numpy()
    for mname in sorted(set(mod)):
        idx = np.flatnonzero(mod == mname)
        for s in range(3):
            for cell, j in [((0, 0), j00), ((0, 14), j014)]:
                p = P[s][idx, j]; flag = p >= T[s][idx]
                rows.append({"module": mname, "seed": s, "d_vle": cell[0], "d_asm": cell[1], "n": len(idx),
                             "prevalence": y[idx].mean(), "pr_auc": average_precision_score(y[idx], p),
                             "flagged": flag.mean(), "recall": flag[y[idx] == 1].mean(),
                             "precision": y[idx][flag].mean() if flag.any() else np.nan})
    s_ = pd.DataFrame(rows).groupby(["module", "d_vle", "d_asm"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    s_["assessments_due_by_day27"] = s_.module.map(early_due).fillna(0).astype(int)
    s_["share_with_submission_by_day27"] = s_.module.map(
        pd.Series(submitted_early).groupby(mod).mean())
    s_.to_csv(f"{R}/stage4_modules.csv", index=False)
    # record-level check of the false alarms added by assessment delay
    missed0, missed14 = grid[(0, 0)]["asm_missed"].to_numpy(), grid[(0, 14)]["asm_missed"].to_numpy()
    sub0, sub14 = grid[(0, 0)]["asm_submitted"].to_numpy(), grid[(0, 14)]["asm_submitted"].to_numpy()
    rec = []
    for s in range(3):
        f0, f14 = P[s][:, j00] >= T[s], P[s][:, j014] >= T[s]
        new_fa = np.flatnonzero(~f0 & f14 & (y == 0))
        all_fa = np.flatnonzero(f14 & (y == 0))
        rec.append({"seed": s, "new_flags": int((~f0 & f14).sum()), "new_false_alarms": len(new_fa),
                    "share_new_flags_false": len(new_fa) / max(1, int((~f0 & f14).sum())),
                    "new_false_alarms_missed_count_raised": float((missed14[new_fa] > missed0[new_fa]).mean()),
                    "new_false_alarms_fewer_submissions_seen": float((sub14[new_fa] < sub0[new_fa]).mean()),
                    "all_false_alarms_0_14": len(all_fa),
                    "all_false_alarms_missed_count_raised": float((missed14[all_fa] > missed0[all_fa]).mean())})
    pd.DataFrame(rec).to_csv(f"{R}/stage4_false_alarms.csv", index=False)
    plt = _plt()
    A = s_[(s_.d_vle == 0) & (s_.d_asm == 0)].set_index("module")
    B = s_[(s_.d_vle == 0) & (s_.d_asm == 14)].set_index("module")
    mods = A.index.tolist(); x = np.arange(len(mods)); h = 0.36
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.bar(x - h / 2 - 0.02, A.pr_auc, width=h, color=BLUE, label="Fresh data (0, 0)")
    ax.bar(x + h / 2 + 0.02, B.loc[mods].pr_auc, width=h, color=ORANGE, label="Assessments 14 days late (0, 14)")
    ax.plot(x, A.prevalence, "_", color=INK, markersize=14, label="Prevalence (floor)")
    ax.set_xticks(x, mods); ax.set_xlabel("Module"); ax.set_ylabel("B3 test PR-AUC (mean of 3 seeds)")
    ax.yaxis.grid(True, color="#e5e4df", linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout(); fig.savefig(f"{R}/stage4_modules.pdf"); plt.close(fig)
    return s_, pd.DataFrame(rec)


def alerts_budget():
    """Matched alert budget: precision among the top 10% of scores, fold mean, by regime and pair."""
    d = pd.concat([pd.read_csv(f"{R}/runs_fresh.csv"), pd.read_csv(f"{R}/runs_augmented.csv")])
    d = d[d.model.isin(["B1", "B3"])]
    t = (d.groupby(["model", "condition", "d_vle", "d_asm"]).precision_top10.agg(["mean", "std"])
         .reset_index().rename(columns={"mean": "precision_top10", "std": "sd"}))
    t.to_csv(f"{R}/stage4_alerts_budget.csv", index=False)
    return t


def alerts():
    u = unc("alerts")
    u.to_csv(f"{R}/stage4_alerts_summary.csv", index=False)
    plt = _plt()
    b = u[u.model == "B3"]
    cells = ["(0,0)", "(0,14)", "(14,14)"]
    arms = [("fresh", "Fresh-trained", BLUE), ("augmented", "Augmented", ORANGE),
            ("freshness_aware", "Freshness-aware", AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    x = np.arange(len(cells)); w = 0.26
    for ax, q, lab in [(axes[0], "flagged", "Share of test records flagged"), (axes[1], "recall", "Recall"),
                       (axes[2], "precision", "Precision")]:
        for k, (r, name, col) in enumerate(arms):
            v = [b[(b.pair == c) & (b.regime == r) & (b.quantity == q)].estimate.iloc[0] for c in cells]
            xs = x + (k - 1) * (w + 0.02)
            ax.bar(xs, v, width=w, color=col, label=name)
            for xi, vi in zip(xs, v):
                ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", fontsize=5.6, color=INK)
        ax.set_xticks(x, [c.replace(",", ", ") for c in cells]); ax.set_xlabel("Delay pair (VLE, assessment)")
        ax.set_ylabel(lab); ax.yaxis.grid(True, color="#e5e4df", linewidth=0.6); ax.set_axisbelow(True)
        ax.set_ylim(0, 1.0 if q != "precision" else 0.45)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{R}/stage4_alerts.pdf"); plt.close(fig)

    # single panel (share of test records flagged) for the slides
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k, (r, name, col) in enumerate(arms):
        v = [b[(b.pair == c) & (b.regime == r) & (b.quantity == "flagged")].estimate.iloc[0] for c in cells]
        xs = x + (k - 1) * (w + 0.02)
        ax.bar(xs, v, width=w, color=col, label=name)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, [c.replace(",", ", ") for c in cells]); ax.set_xlabel("Delay pair (VLE, assessment)")
    ax.set_ylabel("Share of test records flagged"); ax.set_ylim(0, 0.85)
    ax.yaxis.grid(True, color="#e5e4df", linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{R}/stage4_alerts_share.pdf"); plt.close(fig)
    return u


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    t = surface(); surface_figure(t)
    print("RQ1 pairs whose Bonferroni interval excludes 0:", t[t.bonf_excludes_zero].groupby("model").size().to_dict())
    print(main_comparison().round(4).to_string(index=False))
    print(rq2_rq3().round(4).to_string(index=False))
    print(rq2_cells().round(4).to_string(index=False))
    for fn in (wide, size, prevalence, chrono, fullcohort):
        r = fn()
        if r is not None:
            print(fn.__name__); print(r.round(4).to_string(index=False))
    m, rec = modules_and_records(); print(m.round(3).to_string(index=False)); print(rec.round(4).to_string(index=False))
    alerts(); alerts_budget()
