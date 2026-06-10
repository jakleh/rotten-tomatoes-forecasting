"""Tests for gates/pfresh_lib.py — the p_fresh regression program's pure helpers.

No network/DB. The load-bearing checks: the parser's F14 family set, shipped-prior
parity with estimate_p_fresh, the tied-timestamp visible_state rule, per-movie weight
normalization, the temporal-fit asserts, and GLM sign recovery on toy data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gates import pfresh_lib as pl
from rotten_tomatoes_forecasting import estimate_p_fresh


# ---- parser -----------------------------------------------------------------------

@pytest.mark.parametrize("raw, expect", [
    ("3/5", ("frac_5", 3.0)),
    ("3.5/4", ("frac_4", 3.5)),
    ("3.5/4.0", ("frac_4", 3.5)),          # decimal denominator normalizes
    (".5/4", ("frac_4", 0.5)),             # leading dot
    ("7.5/10", ("frac_10", 7.5)),
    ("17/20", ("frac_20", 17.0)),
    ("85/100", ("frac_100", 85.0)),
    ("70%", ("frac_100", 70.0)),
    ("B+", ("letter", 9.0)),
    ("a-minus", ("letter", 10.0)),
    ("b-plus", ("letter", 9.0)),
    ("F+", ("letter", 0.0)),
    ("4 stars", ("stars", 4.0)),
    ("2.5 stars", ("stars", 2.5)),
    ("3 1/2 stars", ("stars", 3.5)),
    ("5/5 stars", ("frac_5", 5.0)),
    ("3 out of 4", ("frac_4", 3.0)),
    ("7/9", None),                          # unknown denominator
    ("3 1/2", None),                        # mixed number without the star word
    ("85", None),                           # bare numeral -> ambiguous
    ("recommended", None),
    ("", None),
    (None, None),
    (float("nan"), None),
])
def test_parse_subjective(raw, expect):
    assert pl.parse_subjective(raw) == expect


def test_dominant_family_assignment():
    df = pd.DataFrame({
        "reviewer_name": ["j"] * 3 + ["k"] * 2,
        "subjective_score": ["3/5", "4/5", "B+", "A", "B"],
        "tomatometer_sentiment": ["positive"] * 5,
        "estimated_timestamp": pd.date_range("2026-01-01", periods=5, tz="UTC"),
    })
    out = pl.add_parsed_scores(df)
    assert (out.loc[out["reviewer_name"] == "j", "dominant_family"] == "frac_5").all()
    # j's letter row is parsed but non-dominant -> excluded from anchored rows
    assert out["anchored_scored"].tolist() == [True, True, False, True, True]


# ---- curves + anchors ---------------------------------------------------------------

def _scored_frame(n_per_level):
    rows = []
    for lvl, (n, p) in n_per_level.items():
        for i in range(n):
            rows.append({"reviewer_name": f"c{i % 7}", "score_family": "frac_5",
                         "score_level": lvl, "anchored_scored": True,
                         "tomatometer_sentiment": "positive" if i < p * n else "negative"})
    return pd.DataFrame(rows)


def test_global_curves_support_and_interpolation():
    df = _scored_frame({2.0: (30, 0.2), 4.0: (30, 0.9), 3.0: (5, 1.0)})
    curves = pl.global_curves(df)
    assert curves[("frac_5", 2.0)] == pytest.approx(0.2, abs=0.04)
    assert curves[("frac_5", 4.0)] == pytest.approx(0.9, abs=0.04)
    thin = curves[("frac_5", 3.0)]           # interpolated, NOT its own 1.0
    assert 0.2 < thin < 0.9


def test_anchor_shrinks_toward_global():
    rows = pd.DataFrame({
        "reviewer_name": ["j"] * 4, "score_family": ["frac_5"] * 4,
        "score_level": [2.0, 2.0, 4.0, 4.0], "anchored_scored": [True] * 4,
        "tomatometer_sentiment": ["negative", "negative", "positive", "positive"],
    })
    a = pl.fit_anchor(rows)
    assert a.n == 4 and a.w == pytest.approx(4 / 14)
    curves = {("frac_5", 3.0): 0.5}
    p = a.p(3.0, curves)
    assert 0.0 <= p <= 1.0
    # with w small, the blend sits near the global value
    assert abs(p - 0.5) < 0.25


def test_intensity_excess_no_scored_rows_is_zero():
    df = pd.DataFrame({"reviewer_name": ["j"], "score_family": [None],
                       "score_level": [np.nan], "anchored_scored": [False],
                       "tomatometer_sentiment": ["positive"]})
    excess, n = pl.intensity_excess(df, {})
    assert excess == 0.0 and n == 0


# ---- priors --------------------------------------------------------------------------

def _pool_reviews():
    rows = []
    for slug in [f"m{i}" for i in range(6)]:
        for critic, sent in [("a", "positive"), ("b", "negative"), ("c", "positive")]:
            rows.append({"movie_slug": slug, "reviewer_name": critic,
                         "tomatometer_sentiment": sent})
    return pd.DataFrame(rows)


def test_prior_remaining_matches_shipped_estimator_at_total_zero():
    """estimate_p_fresh(total=0) returns its prior — pl.prior_remaining (raw) must
    reproduce it exactly (single source of semantics)."""
    rv = _pool_reviews()
    slugs = [f"m{i}" for i in range(6)]
    for observed in [set(), {"a"}, {"a", "b"}]:
        ours = pl.prior_remaining(rv, slugs, observed)
        shipped = estimate_p_fresh(rv, slugs, observed, 0, 0)
        assert ours == pytest.approx(shipped, abs=1e-12)


def test_prior_remaining_shrunk_moves_toward_global():
    rv = _pool_reviews()
    slugs = [f"m{i}" for i in range(6)]
    raw = pl.prior_remaining(rv, slugs, set())
    shrunk = pl.prior_remaining(rv, slugs, set(), shrink_k=10.0)
    g = 2 / 3                                  # pool global fresh rate
    assert abs(shrunk - g) < abs(raw - g) or raw == pytest.approx(shrunk)


def test_prior_actual():
    rem = pd.DataFrame({"reviewer_name": ["a", "b", "zz"]})
    rates = {"a": 1.0, "b": 0.0}
    assert pl.prior_actual(rem, rates, 0.5) == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert pl.prior_actual(rem, rates, 0.5, raw_default=0.25) == pytest.approx(
        (1.0 + 0.0 + 0.25) / 3)


# ---- visible_state -------------------------------------------------------------------

def test_visible_state_ties_excluded():
    ts = pd.to_datetime(["2026-01-01 12:00", "2026-01-02 12:00", "2026-01-02 12:00",
                         "2026-01-03 12:00"], utc=True)
    df = pd.DataFrame({"estimated_timestamp": ts,
                       "tomatometer_sentiment": ["positive", "positive", "negative",
                                                 "negative"]})
    vs = pl.visible_state(df)
    # rows 1 and 2 are tied: both see only row 0's state (1 fresh / 1 total)
    assert vs["visible_total"].tolist() == [0, 1, 1, 3]
    assert vs["visible_fresh"].tolist() == [0, 1, 1, 2]


# ---- weights / GLM -------------------------------------------------------------------

def _toy_rows():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        slug = f"m{i}"
        x = rng.normal()
        p = 1 / (1 + np.exp(-(0.5 + 2.0 * x)))
        for snap in (1, 2):
            n = int(rng.integers(5, 40))
            rows.append({"slug": slug, "snap_days": snap, "x": x, "n_rem": n,
                         "fresh_rem": int(rng.binomial(n, p)),
                         "close": f"2026-01-{(i % 27) + 1:02d}T14:00:00Z"})
    return pd.DataFrame(rows)


def test_row_weights_sum_to_one_per_movie():
    rows = _toy_rows()
    w = pl.row_weights(rows) * rows["n_rem"]
    sums = w.groupby(rows["slug"]).sum()
    assert np.allclose(sums, 1.0)


def test_glm_recovers_sign_and_normalizes():
    rows = _toy_rows()
    model, best_c, oos = pl.fit_binomial_glm(rows, ["x"])
    assert model.coef_[0][0] > 0.5              # true slope 2.0, heavy shrink OK
    assert best_c in pl.C_GRID and np.isfinite(oos)


def test_deviance_of_predictor_prefers_truth():
    rows = _toy_rows()
    rows["p_true"] = 1 / (1 + np.exp(-(0.5 + 2.0 * rows["x"])))
    rows["p_flat"] = 0.5
    assert pl.deviance_of_predictor(rows, "p_true") < pl.deviance_of_predictor(
        rows, "p_flat")


# ---- temporal_rows -------------------------------------------------------------------

def test_temporal_rows_asserts():
    rows = _toy_rows()
    target_close = pd.Timestamp("2026-01-20T14:00:00Z")
    # toy closes are daily at 14:00; a 12h window has no training close inside
    min_snap = target_close - pd.Timedelta(hours=12)
    with pytest.raises(ValueError):             # floor breach at 60 movies
        pl.temporal_rows(rows, "mX", target_close, min_snap, floor=60)
    sub = pl.temporal_rows(rows, "mX", target_close, min_snap, floor=5)
    closes = pd.to_datetime(sub["close"], utc=True)
    assert (closes < target_close).all()
    # M5 collision: a training movie closing inside (snap, close] must raise
    bad = rows.copy()
    bad.loc[0, "close"] = "2026-01-20T06:00:00Z"
    with pytest.raises(AssertionError):
        pl.temporal_rows(bad, "mX", target_close, min_snap, floor=5)


def test_emp_logit_and_clip():
    assert pl.emp_logit(0, 0) == 0.0
    assert pl.emp_logit(10, 10) == pytest.approx(np.log((10.5 / 11) / (0.5 / 11)))
    assert pl.logit_clip(1.0) == pytest.approx(np.log(0.99 / 0.01))
