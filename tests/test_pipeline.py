"""Checks on the cohort, the delayed features and the splits.

Run with:  python -m pytest -q tests
Needs the OULAD CSVs in data/oulad (make data).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "oulad")

from staleness.cohort import build_cohort                       # noqa: E402
from staleness.experiment import outer_folds                    # noqa: E402
from staleness.features import (A_COLS, DELAYS, R_COLS, V_COLS,  # noqa: E402
                                build_grid, load_sources)
from staleness.models import AgeInteractions                    # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(DATA), reason="run `make data` first")


@pytest.fixture(scope="module")
def built():
    cohort, audit = build_cohort(DATA, verbose=False)
    return cohort, audit, build_grid(load_sources(DATA, cohort), cohort)


def test_cohort_counts_and_day28_eligibility(built):
    cohort, audit, _ = built
    assert len(cohort) == 27_428 and cohort["y"].sum() == 5_013
    assert audit["early_withdrawals_excluded"] + audit["missing_unregistration_date_excluded"] \
        + audit["inconsistent_unregistration_excluded"] + audit["outcome_valid_stage2"] == 32_593
    assert audit["outcome_valid_stage2"] - audit["registered_after_day_28_excluded"] == audit["eligible"]
    assert not (cohort["date_registration"] > 28).any()           # nobody registers after day 28
    assert cohort["date_registration"].isna().sum() == 7           # kept; imputed in-fold
    full, _ = build_cohort(DATA, verbose=False, day28_eligible=False)
    assert len(full) == 27_444 and full["y"].sum() == 5_016        # Stage 2 cohort (sensitivity)


def test_data_checksums():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from get_data import verify
    assert verify(DATA) == []


@pytest.mark.parametrize("delay", [0, 3, 14])
def test_cutoff_boundary_vle_and_assessments(built, delay):
    """Features at delay d use exactly the records dated on or before day 27 - d:
    removing later records changes nothing, removing the boundary day changes something."""
    from staleness.features import asm_features, vle_features
    cohort, _, _ = built
    src = load_sources(DATA, cohort)
    last = 27 - delay
    cut = dict(src, daily=src["daily"][src["daily"]["date"] <= last],
               first_site=src["first_site"][src["first_site"]["date"] <= last],
               sub=src["sub"][src["sub"]["date_submitted"] <= last])
    assert vle_features(cut, delay).equals(vle_features(src, delay))
    assert asm_features(cut, cohort, delay).equals(asm_features(src, cohort, delay))
    edge = dict(src, daily=src["daily"][src["daily"]["date"] != last],
                sub=src["sub"][src["sub"]["date_submitted"] != last])
    assert not vle_features(edge, delay).equals(vle_features(src, delay))
    assert not asm_features(edge, cohort, delay).equals(asm_features(src, cohort, delay))


def test_grid_has_25_cells_with_12_features(built):
    _, _, grid = built
    assert len(grid) == 25
    assert all(g.shape == (27_428, 12) for g in grid.values())


def test_more_delay_never_reveals_more_activity(built):
    _, _, grid = built
    for small, large in zip(DELAYS, DELAYS[1:]):
        a, b = grid[(small, 0)], grid[(large, 0)]
        for col in ["vle_clicks", "vle_active_days", "vle_resources", "vle_clicks_7d", "vle_clicks_14d"]:
            assert (b[col] <= a[col] + 1e-12).all(), col
        assert (b["vle_recency"] >= a["vle_recency"]).all()
        c, d = grid[(0, small)], grid[(0, large)]
        assert (d["asm_submitted"] <= c["asm_submitted"]).all()
        assert (d["asm_missed"] >= c["asm_missed"]).all()


def test_sources_are_delayed_independently(built):
    _, _, grid = built
    assert grid[(14, 0)][A_COLS + R_COLS].equals(grid[(0, 0)][A_COLS + R_COLS])
    assert grid[(0, 14)][V_COLS + R_COLS].equals(grid[(0, 0)][V_COLS + R_COLS])


def test_outer_folds_share_no_presentation_or_student(built):
    cohort, _, _ = built
    for _, tr, te in outer_folds(cohort, seed=0):
        assert not set(cohort["presentation"].iloc[tr]) & set(cohort["presentation"].iloc[te])
        assert not set(cohort["id_student"].iloc[tr]) & set(cohort["id_student"].iloc[te])


def test_age_interactions():
    X = np.arange(2 * 14, dtype=float).reshape(2, 14)
    Z = AgeInteractions().fit_transform(X)
    assert Z.shape == (2, 14 + 6 + 3)
    assert np.allclose(Z[:, 14], X[:, 0] * X[:, 12]) and np.allclose(Z[:, 20], X[:, 6] * X[:, 13])


def test_inner_folds_share_no_student(built):
    cohort, _, _ = built
    from staleness.experiment import inner_splits
    _, tr, _ = next(iter(outer_folds(cohort, seed=0)))
    y = cohort["y"].to_numpy()[tr]
    g = cohort["presentation"].to_numpy()[tr]
    s = cohort["id_student"].to_numpy()[tr]
    splits = inner_splits(y, g, s)
    assert len(splits) == 3
    covered = np.sort(np.concatenate([va for _, va in splits]))
    assert np.array_equal(covered, np.arange(len(y)))          # validation folds partition the rows
    for itr, va in splits:
        assert not set(g[itr]) & set(g[va])                      # no shared presentation
        assert not set(s[itr]) & set(s[va])                      # no shared student


def test_weighted_average_precision_matches_sklearn():
    from sklearn.metrics import average_precision_score
    from staleness.inference import APIndex
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    p = np.round(rng.random(2000), 2)                     # many tied scores
    w = rng.integers(0, 3, 2000).astype(float)
    rep = np.repeat(np.arange(2000), w.astype(int))
    idx = APIndex(p, y)
    assert abs(idx.ap(np.ones(2000)) - average_precision_score(y, p)) < 1e-12
    assert abs(idx.ap(w) - average_precision_score(y[rep], p[rep])) < 1e-12


def test_svm_uses_external_calibration():
    """B2 has no internal Platt step (whose folds ignore grouping); it is calibrated externally."""
    from staleness.models import CALIBRATE, SVCScore, make_model
    est, _ = make_model("B2")
    assert isinstance(est.named_steps["clf"], SVCScore) and est.named_steps["clf"].probability is not True
    assert CALIBRATE["B2"] and CALIBRATE["B2L"]
