"""Seed-level estimates and student-cluster bootstrap intervals for every comparison reported
in the papers (see staleness/inference.py for why no fold-level significance test is used).

Reads results/predictions/*.npz (saved by scripts/run.py). Writes results/:
  uncertainty_baselines.csv   AP(model) - AP(B3) at (0,0), fresh training
  uncertainty_surface.csv     AP(cell) - AP(0,0) per model and delay pair, fresh training, with
                              per-cell 95% intervals and Bonferroni intervals over the 24 pairs;
                              plus the compounding term AP(dv,da) - AP(dv,0) - AP(0,da) + AP(0,0)
  uncertainty_rq2.csv         augmented / freshness-aware / fresh comparisons, averaged over the
                              25 pairs, at single pairs, and by assessment freshness
  uncertainty_alerts.csv      share flagged, recall and precision at the inner-fold threshold,
                              and differences between training regimes

Columns: estimate (mean over the 15 outer folds, as in the fold-level tables), seed0..seed2
(mean over the 5 folds of each seed, one complete partition each),
ci_low/ci_high (95% student-cluster bootstrap), p_boot (two-sided bootstrap test of no difference;
the smallest attainable value with 2,000 draws is about 0.001), and for the surface bonf_low/bonf_high
and p_bonf (Bonferroni over the 24 pairs).

Run:  python scripts/uncertainty.py            (about 10 minutes on 2 CPU cores)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run import load                                          # noqa: E402
from staleness.experiment import CELLS                                # noqa: E402
from staleness.inference import APIndex, interval, load_pooled, student_weights  # noqa: E402

R = "results"
N_BOOT = int(os.environ.get("N_BOOT", 2000))
J = {c: i for i, c in enumerate(CELLS)}


class FoldAP:
    """Mean over the 5 outer test folds of one seed of the (weighted) average precision,
    i.e. the same statistic as the fold-level tables, for one (model/arm, seed, cell)."""

    def __init__(self, p, y, fold):
        self.parts = [(np.flatnonzero(fold == k), None) for k in np.unique(fold)]
        self.parts = [(idx, APIndex(p[idx], y[idx])) for idx, _ in self.parts]

    def ap(self, w):
        return float(np.mean([a.ap(w[idx]) for idx, a in self.parts]))


def ap_engine(preds, y, keys, folds):
    """FoldAP for every (key, seed, cell)."""
    return {(k, s, c): FoldAP(preds[k][s][:, J[c]].astype(float), y, folds[s])
            for k in keys for s in range(len(preds[k])) for c in CELLS}


def run_bootstrap(stat, students, n_seeds):
    """stat(w, s) -> vector for seed s. Returns point estimates per seed and bootstrap draws
    of the seed-averaged statistic."""
    ones = np.ones(len(students))
    per_seed = np.array([stat(ones, s) for s in range(n_seeds)])
    draws = np.array([np.mean([stat(w, s) for s in range(n_seeds)], axis=0)
                      for w in student_weights(students, N_BOOT)])
    return per_seed, draws


def boot_p(d):
    """Two-sided bootstrap p-value for 'no difference': twice the smaller share of draws on
    either side of zero, with the (k + 1) / (B + 1) correction so p is never exactly 0."""
    d = d[~np.isnan(d)]
    k = min((d <= 0).sum(), (d >= 0).sum())
    return float(min(1.0, 2 * (k + 1) / (len(d) + 1)))


def rows_from(names, per_seed, draws, extra=None, bonf=None):
    out = []
    for i, name in enumerate(names):
        r = dict(name) if isinstance(name, dict) else {"quantity": name}
        r["estimate"] = per_seed[:, i].mean()
        for s in range(per_seed.shape[0]):
            r[f"seed{s}"] = per_seed[s, i]
        r["ci_low"], r["ci_high"] = interval(draws[:, i])
        r["p_boot"] = boot_p(draws[:, i])
        if bonf:
            r["bonf_low"], r["bonf_high"] = interval(draws[:, i], 1 - 0.05 / bonf)
            r["p_bonf"] = min(1.0, bonf * r["p_boot"])
        if extra:
            r.update(extra)
        out.append(r)
    return out


def main():
    cohort, _, _ = load("data/oulad", "results/cache/features.pkl")
    y, stud, n = cohort["y"].to_numpy(), cohort["id_student"].to_numpy(), len(cohort)
    fresh, fthr, folds = load_pooled("fresh", n, with_folds=True)
    aug, athr = load_pooled("augmented", n)          # empty until `make preliminary` has run
    S = len(fresh[("B3", "fresh")])
    allp = {**fresh, **aug}
    eng = ap_engine(allp, y, list(allp), folds)
    memo = {"w": None, "v": {}}

    def ap(k, s, c, w):
        """Average precision for (model/arm, seed, cell) under weights w, memoised per draw."""
        if memo["w"] is not w:
            memo["w"], memo["v"] = w, {}
        key = (k, s, c)
        if key not in memo["v"]:
            memo["v"][key] = eng[key].ap(w)
        return memo["v"][key]

    # 1. baselines vs B3 at (0,0)
    names = ["B0", "B1", "B2"]
    ps, dr = run_bootstrap(lambda w, s: np.array([ap((m, "fresh"), s, (0, 0), w) - ap(("B3", "fresh"), s, (0, 0), w)
                                                  for m in names]), stud, S)
    pd.DataFrame(rows_from([{"model": m, "comparison": "AP(model) - AP(B3), fresh (0,0)"} for m in names], ps, dr)) \
        .to_csv(f"{R}/uncertainty_baselines.csv", index=False)
    print("baselines done", flush=True)

    # 2. staleness surface per model, fresh training
    rows = []
    for m in ["B1", "B2", "B3"]:
        k = (m, "fresh")
        others = [c for c in CELLS if c != (0, 0)]
        inter = [(7, 14), (14, 14)]

        def stat(w, s):
            base = ap(k, s, (0, 0), w)
            v = [ap(k, s, c, w) - base for c in others]
            v += [ap(k, s, c, w) - ap(k, s, (c[0], 0), w) - ap(k, s, (0, c[1]), w) + base for c in inter]
            return np.array(v)
        ps, dr = run_bootstrap(stat, stud, S)
        nm = [{"model": m, "kind": "change", "d_vle": c[0], "d_asm": c[1]} for c in others] + \
             [{"model": m, "kind": "interaction", "d_vle": c[0], "d_asm": c[1]} for c in inter]
        rows += rows_from(nm, ps, dr, bonf=24)
        print("surface", m, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{R}/uncertainty_surface.csv", index=False)

    if not aug:
        print("no augmented predictions yet: skipping RQ2 and alert intervals")
        return

    # 3. RQ2 comparisons
    rows = []
    fresh_cells = [c for c in CELLS if c[1] <= 1]
    stale_cells = [c for c in CELLS if c[1] == 14]
    for m in ["B1", "B3"]:
        F, A, W = (m, "fresh"), (m, "augmented"), (m, "freshness_aware")
        comps = [("aware_minus_augmented", W, A), ("augmented_minus_fresh", A, F), ("aware_minus_fresh", W, F)]
        names, funcs = [], []
        for lab, a, b in comps:
            names.append({"model": m, "comparison": lab, "pairs": "all 25"})
            funcs.append(lambda w, s, a=a, b=b: np.mean([ap(a, s, c, w) - ap(b, s, c, w) for c in CELLS]))
            for c in [(0, 0), (0, 14), (14, 0), (14, 14)]:
                names.append({"model": m, "comparison": lab, "pairs": f"({c[0]},{c[1]})"})
                funcs.append(lambda w, s, a=a, b=b, c=c: ap(a, s, c, w) - ap(b, s, c, w))
        names += [{"model": m, "comparison": "aware_minus_augmented", "pairs": "assessment delay <= 1"},
                  {"model": m, "comparison": "aware_minus_augmented", "pairs": "assessment delay 14"}]
        funcs += [lambda w, s, W=W, A=A: np.mean([ap(W, s, c, w) - ap(A, s, c, w) for c in fresh_cells]),
                  lambda w, s, W=W, A=A: np.mean([ap(W, s, c, w) - ap(A, s, c, w) for c in stale_cells])]
        ps, dr = run_bootstrap(lambda w, s: np.array([f(w, s) for f in funcs]), stud, S)
        rows += rows_from(names, ps, dr)
        print("rq2", m, "done", flush=True)
    # main comparison: every method's 25-pair mean against freshness-aware B3
    ref = ("B3", "freshness_aware")
    others = [("B0", "fresh"), ("B1", "fresh"), ("B2", "fresh"), ("B3", "fresh"), ("B1", "augmented"),
              ("B1", "freshness_aware"), ("B3", "augmented")]
    ps, dr = run_bootstrap(lambda w, s: np.array([np.mean([ap(ref, s, c, w) - ap(o, s, c, w) for c in CELLS])
                                                  for o in others]), stud, S)
    rows += rows_from([{"model": "B3", "comparison": f"aware_B3_minus_{o[0]}_{o[1]}", "pairs": "all 25"}
                       for o in others], ps, dr)
    pd.DataFrame(rows).to_csv(f"{R}/uncertainty_rq2.csv", index=False)
    print("main comparison done", flush=True)

    # 4. alerts: share flagged, recall, precision (weighted), and regime differences
    def alert_stats(P, T, w, c):
        flag = (P[:, J[c]] >= T).astype(float)
        fw, yw = flag * w, y * w
        return np.array([fw.sum() / w.sum(), (fw * y).sum() / yw.sum(), (fw * y).sum() / max(fw.sum(), 1e-12)])
    rows = []
    cells = [(0, 0), (0, 14), (14, 0), (14, 14)]
    regimes = {"B3": [("fresh", fresh, fthr), ("augmented", aug, athr), ("freshness_aware", aug, athr)],
               "B1": [("fresh", fresh, fthr), ("augmented", aug, athr), ("freshness_aware", aug, athr)]}
    for m, regs in regimes.items():
        def stat(w, s, regs=regs, m=m):
            v = []
            for c in cells:
                st = {r: alert_stats(src[(m, r)][s], th[(m, r)][s], w, c) for r, src, th in regs}
                for r, _, _ in regs:
                    v += list(st[r])
                v += list(st["freshness_aware"] - st["augmented"]) + list(st["freshness_aware"] - st["fresh"])
            return np.array(v)
        ps, dr = run_bootstrap(stat, stud, S)
        names = []
        for c in cells:
            for r in ["fresh", "augmented", "freshness_aware", "aware_minus_augmented", "aware_minus_fresh"]:
                for q in ["flagged", "recall", "precision"]:
                    names.append({"model": m, "pair": f"({c[0]},{c[1]})", "regime": r, "quantity": q})
        rows += rows_from(names, ps, dr)
        print("alerts", m, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{R}/uncertainty_alerts.csv", index=False)


if __name__ == "__main__":
    main()
