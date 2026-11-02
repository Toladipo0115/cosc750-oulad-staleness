"""Primary cohort for day-28 withdrawal recognition.

The category logic and the assertions are the author's original cohort
validation code (Stage 1 and 2 notebook), wrapped in a function so the
pipeline can import it. The category logic is unchanged; the label column,
the presentation key, the registration-date audit and the day-28
eligibility step are additions.

Day-28 eligibility (Stage 3 onward): the 27,444 outcome-valid records of
Stages 1 and 2 include 16 students who registered after day 28, who are not
in the institution's risk set on day 28. The primary cohort removes them
(27,428 records, 5,013 later withdrawals). The 7 records with no registration
date are kept; their registration day is imputed with the training-fold
median. `day28_eligible=False` returns the original 27,444 records, used as
a sensitivity analysis.
"""
import pandas as pd

JOIN_KEYS = ["code_module", "code_presentation", "id_student"]
CUTOFF_DAY = 28


def build_cohort(data_dir, verbose=True, day28_eligible=True):
    """Return the primary cohort (27,428 records) with a binary label `y`.

    y = 1 for a withdrawal recorded after day 28, y = 0 for a Pass, Fail or
    Distinction with no unregistration date. Raises AssertionError if the
    counts differ from the verified Stage 1 cohort or the day-28 eligible cohort.
    """
    reg = pd.read_csv(f"{data_dir}/studentRegistration.csv")
    info = pd.read_csv(f"{data_dir}/studentInfo.csv")

    # Convert valid dates to numbers and treat nonnumeric values as missing.
    reg["date_unregistration"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")
    reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")

    df = reg.merge(info, on=JOIN_KEYS, how="inner", validate="one_to_one")
    total = len(df)

    has_unreg = df["date_unregistration"].notna()
    withdrawn_label = df["final_result"].eq("Withdrawn")
    valid_nonwithdrawal_result = df["final_result"].isin(["Pass", "Fail", "Distinction"])

    # Mutually exclusive categories
    valid_early_withdrawal = withdrawn_label & has_unreg & (df["date_unregistration"] <= CUTOFF_DAY)
    valid_later_withdrawal = withdrawn_label & has_unreg & (df["date_unregistration"] > CUTOFF_DAY)
    valid_nonwithdrawal = (~has_unreg) & valid_nonwithdrawal_result
    edge_case_a = has_unreg & (~withdrawn_label)
    edge_case_b = withdrawn_label & (~has_unreg)

    # Confirm that every record belongs to exactly one category.
    category_count = (valid_early_withdrawal.astype(int) + valid_later_withdrawal.astype(int)
                      + valid_nonwithdrawal.astype(int) + edge_case_a.astype(int)
                      + edge_case_b.astype(int))
    unaccounted = int((category_count == 0).sum())
    multiply_accounted = int((category_count > 1).sum())

    primary_mask = valid_later_withdrawal | valid_nonwithdrawal
    cohort = df.loc[primary_mask].copy()
    cohort["y"] = valid_later_withdrawal[primary_mask].astype(int).to_numpy()
    cohort["presentation"] = cohort["code_module"] + "_" + cohort["code_presentation"]

    # Assertions stop execution if the cohort differs from the verified result.
    assert total == 32_593
    assert valid_early_withdrawal.sum() == 5_047
    assert valid_later_withdrawal.sum() == 5_016
    assert valid_nonwithdrawal.sum() == 22_428
    assert edge_case_a.sum() == 9
    assert edge_case_b.sum() == 93
    assert unaccounted == 0
    assert multiply_accounted == 0
    assert len(cohort) == 27_444 and int(cohort["y"].sum()) == 5_016

    # Day-28 eligibility: drop registrations after the prediction day (missing dates kept).
    late = cohort["date_registration"] > CUTOFF_DAY
    if day28_eligible:
        cohort = cohort.loc[~late].copy()
        assert len(cohort) == 27_428 and int(cohort["y"].sum()) == 5_013

    audit = {
        "total": total,
        "early_withdrawals_excluded": int(valid_early_withdrawal.sum()),
        "missing_unregistration_date_excluded": int(edge_case_b.sum()),
        "inconsistent_unregistration_excluded": int(edge_case_a.sum()),
        "outcome_valid_stage2": 27_444,
        "registered_after_day_28_excluded": int(late.sum()) if day28_eligible else 0,
        "eligible": len(cohort),
        "positive": int(cohort["y"].sum()),
        "negative": int((cohort["y"] == 0).sum()),
        # Registration-date audit promised in the Stage 2 paper.
        "registration_date_missing": int(cohort["date_registration"].isna().sum()),
        "registered_after_day_28_remaining": int((cohort["date_registration"] > CUTOFF_DAY).sum()),
    }
    if verbose:
        for k, v in audit.items():
            print(f"{k}: {v:,}")
    return cohort.reset_index(drop=True), audit
