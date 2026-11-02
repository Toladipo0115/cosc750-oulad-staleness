"""Day-28 features under independent source delays.

Prediction time is t = 28 (course days 0..27 have happened). A source with
delay d has loaded only the records dated on or before t - 1 - d. VLE
activity (V) and assessment submissions (A) are delayed independently;
registration features (R) are always current.

Twelve inputs (Stage 2 plan):
  V: total clicks, active days, distinct resources, recency, clicks in the
     last 7 and last 14 calendar days before t
  A: submissions observed, assessments past deadline with no observed
     submission, mean days submitted before the deadline
  R: previous attempts, studied credits, registration day
Two source-age indicators (a_V = d_V, a_A = d_A) are added only for the
freshness-aware model, giving d = 14.

Windows are anchored to the calendar (t - 7, t - 14), not to the last loaded
day, because an advisor's dashboard counts "the last week" from today. A
stale source therefore shows fewer recent clicks and a longer recency, which
is exactly the confusion between staleness and disengagement under study.
Scores and banked assessments are never used.
"""
import numpy as np
import pandas as pd

from .cohort import CUTOFF_DAY, JOIN_KEYS

DELAYS = (0, 1, 3, 7, 14)
V_COLS = ["vle_clicks", "vle_active_days", "vle_resources", "vle_recency",
          "vle_clicks_7d", "vle_clicks_14d"]
A_COLS = ["asm_submitted", "asm_missed", "asm_days_early"]
R_COLS = ["prev_attempts", "studied_credits", "registration_day"]
BASE_COLS = V_COLS + A_COLS + R_COLS
AGE_COLS = ["age_vle", "age_asm"]
NO_ACTIVITY_RECENCY = 60  # days; larger than any observable gap (earliest record is day -25)


def load_sources(data_dir, cohort):
    """Load VLE and assessment records dated before t, for cohort members only."""
    keys = cohort[JOIN_KEYS].assign(row=np.arange(len(cohort)))
    vle = pd.read_csv(f"{data_dir}/studentVle.csv",
                      usecols=JOIN_KEYS + ["id_site", "date", "sum_click"],
                      dtype={"id_student": "int64", "id_site": "int64",
                             "date": "int32", "sum_click": "int32"})
    vle = vle[vle["date"] < CUTOFF_DAY].merge(keys, on=JOIN_KEYS)
    daily = vle.groupby(["row", "date"], as_index=False)["sum_click"].sum()
    first_site = vle.groupby(["row", "id_site"], as_index=False)["date"].min()

    asm = pd.read_csv(f"{data_dir}/assessments.csv", na_values="?")
    sub = pd.read_csv(f"{data_dir}/studentAssessment.csv", na_values="?")
    sub = sub[sub["is_banked"] == 0].merge(
        asm[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment").rename(columns={"date": "deadline"})
    sub = sub[sub["date_submitted"] < CUTOFF_DAY].merge(keys, on=JOIN_KEYS)
    # Deadlines are on the published schedule, so they are known without delay.
    due = asm[(asm["assessment_type"] != "Exam") & (asm["date"] < CUTOFF_DAY)]
    return {"daily": daily, "first_site": first_site, "sub": sub, "due": due,
            "n": len(cohort)}


def _sum_by_row(frame, col, n):
    out = np.zeros(n)
    s = frame.groupby("row")[col].sum()
    out[s.index.to_numpy()] = s.to_numpy()
    return out


def vle_features(src, delay):
    n, last = src["n"], CUTOFF_DAY - 1 - delay
    d = src["daily"][src["daily"]["date"] <= last]
    f = pd.DataFrame(index=range(n))
    f["vle_clicks"] = np.log1p(_sum_by_row(d, "sum_click", n))
    f["vle_active_days"] = _sum_by_row(d.assign(one=1), "one", n)
    fs = src["first_site"]
    f["vle_resources"] = _sum_by_row(fs[fs["date"] <= last].assign(one=1), "one", n)
    rec = np.full(n, float(NO_ACTIVITY_RECENCY))
    lastday = d.groupby("row")["date"].max()
    rec[lastday.index.to_numpy()] = CUTOFF_DAY - lastday.to_numpy()
    f["vle_recency"] = rec
    f["vle_clicks_7d"] = np.log1p(_sum_by_row(d[d["date"] >= CUTOFF_DAY - 7], "sum_click", n))
    f["vle_clicks_14d"] = np.log1p(_sum_by_row(d[d["date"] >= CUTOFF_DAY - 14], "sum_click", n))
    return f


def asm_features(src, cohort, delay):
    n, last = src["n"], CUTOFF_DAY - 1 - delay
    s = src["sub"][src["sub"]["date_submitted"] <= last]
    f = pd.DataFrame(index=range(n))
    f["asm_submitted"] = _sum_by_row(s.assign(one=1), "one", n)
    # Assessments whose deadline has passed by t, per presentation.
    due = src["due"].groupby(["code_module", "code_presentation"])["id_assessment"].apply(set)
    seen = s.groupby("row")["id_assessment"].apply(set)
    missed = np.zeros(n)
    for i, (m, p) in enumerate(zip(cohort["code_module"], cohort["code_presentation"])):
        ids = due.get((m, p), set())
        if ids:
            missed[i] = len(ids - seen.get(i, set()))
    f["asm_missed"] = missed
    early = (s["deadline"] - s["date_submitted"]).groupby(s["row"]).mean()
    v = np.zeros(n)
    v[early.index.to_numpy()] = early.to_numpy()
    f["asm_days_early"] = v
    return f


def reg_features(cohort):
    return pd.DataFrame({
        "prev_attempts": cohort["num_of_prev_attempts"].to_numpy(float),
        "studied_credits": cohort["studied_credits"].to_numpy(float),
        "registration_day": cohort["date_registration"].to_numpy(float),  # 7 missing; imputed in-fold
    })


def build_grid(src, cohort):
    """Return {(d_V, d_A): DataFrame of the 12 base features} for all 25 delay pairs."""
    V = {d: vle_features(src, d) for d in DELAYS}
    A = {d: asm_features(src, cohort, d) for d in DELAYS}
    R = reg_features(cohort)
    return {(dv, da): pd.concat([V[dv], A[da], R], axis=1)[BASE_COLS]
            for dv in DELAYS for da in DELAYS}
