"""Tests for features.py — ET-midnight feature extraction, noon-shift, skip rules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rotten_tomatoes_forecasting.features import (
    FEATURE_NAMES,
    VALID_SNAP_DAYS,
    apply_noon_shift,
    extract_lambda_features,
    midnight_et_of_close,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COHORT_REVIEWS = REPO_ROOT / "reviews.csv"
COHORT_MOVIES = REPO_ROOT / "movies_index.csv"


# -- Fixtures ----------------------------------------------------------------


def _make_cohort(n_movies: int = 30, n_critics: int = 40, seed: int = 7):
    rng = np.random.default_rng(seed)
    base_close = pd.Timestamp("2026-03-01 14:00:00", tz="UTC")
    close_date_map: dict[str, pd.Timestamp] = {}
    rows = []
    for i in range(n_movies):
        slug = f"movie_{i:02d}"
        close_ts = base_close + pd.Timedelta(days=i * 3)
        close_date_map[slug] = close_ts
        first_rev = close_ts - pd.Timedelta(days=rng.uniform(5, 12))
        for c in range(n_critics):
            if rng.random() < 0.35:
                continue
            offset = pd.Timedelta(hours=rng.uniform(0, (close_ts - first_rev).total_seconds() / 3600))
            rows.append(
                {
                    "movie_slug": slug,
                    "reviewer_name": f"critic_{c:02d}",
                    "tomatometer_sentiment": rng.choice(["positive", "negative"], p=[0.7, 0.3]),
                    "estimated_timestamp": first_rev + offset,
                    "top_critic": rng.choice(["t", "f"], p=[0.3, 0.7]),
                    "publication_name": f"pub_{rng.integers(0, 20)}",
                    "timestamp_confidence": rng.choice(["m", "h", "d"], p=[0.05, 0.15, 0.8]),
                }
            )
    return pd.DataFrame(rows), close_date_map


@pytest.fixture
def cohort():
    return _make_cohort()


# -- midnight_et_of_close ----------------------------------------------------


class TestMidnightET:
    def test_midnight_et_on_edt(self):
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        mid = midnight_et_of_close(close)
        assert (close - mid).total_seconds() / 3600 == pytest.approx(10.0)

    def test_midnight_et_on_est(self):
        close = pd.Timestamp("2026-01-15 15:00:00", tz="UTC")
        mid = midnight_et_of_close(close)
        assert (close - mid).total_seconds() / 3600 == pytest.approx(10.0)

    def test_rejects_naive_timestamp(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            midnight_et_of_close(pd.Timestamp("2026-04-06 14:00:00"))


# -- noon-shift --------------------------------------------------------------


class TestApplyNoonShift:
    def test_shifts_day_level_only(self):
        df = pd.DataFrame(
            {
                "movie_slug": ["a", "a", "a"],
                "estimated_timestamp": pd.to_datetime(
                    [
                        "2026-04-01 00:00:00",
                        "2026-04-01 08:30:00",
                        "2026-04-01 14:15:23",
                    ],
                    utc=True,
                ),
                "timestamp_confidence": ["d", "h", "m"],
            }
        )
        out = apply_noon_shift(df)
        assert out.loc[0, "estimated_timestamp"] == pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
        assert out.loc[1, "estimated_timestamp"] == pd.Timestamp("2026-04-01 08:30:00", tz="UTC")
        assert out.loc[2, "estimated_timestamp"] == pd.Timestamp("2026-04-01 14:15:23", tz="UTC")

    def test_returns_copy(self):
        df = pd.DataFrame(
            {
                "movie_slug": ["a"],
                "estimated_timestamp": pd.to_datetime(["2026-04-01 00:00:00"], utc=True),
                "timestamp_confidence": ["d"],
            }
        )
        out = apply_noon_shift(df)
        assert df.loc[0, "estimated_timestamp"] == pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
        assert out.loc[0, "estimated_timestamp"] == pd.Timestamp("2026-04-01 12:00:00", tz="UTC")


# -- extract_lambda_features -------------------------------------------------


class TestExtractFeatures:
    def test_all_17_features_present(self, cohort):
        reviews, close_date_map = cohort
        slug = "movie_29"
        close_ts = close_date_map[slug]
        feats = extract_lambda_features(
            slug, snap_days=3, close_ts=close_ts, reviews_df=reviews, close_date_map=close_date_map
        )
        assert feats is not None
        assert set(feats.keys()) == set(FEATURE_NAMES)
        assert len(feats) == 17

    def test_observed_count_positive(self, cohort):
        reviews, close_date_map = cohort
        feats = extract_lambda_features(
            "movie_29",
            snap_days=3,
            close_ts=close_date_map["movie_29"],
            reviews_df=reviews,
            close_date_map=close_date_map,
        )
        assert feats is not None
        assert feats["observed_count"] > 0

    def test_rejects_out_of_range_snap_days(self, cohort):
        reviews, close_date_map = cohort
        with pytest.raises(ValueError):
            extract_lambda_features(
                "movie_29",
                snap_days=6,
                close_ts=close_date_map["movie_29"],
                reviews_df=reviews,
                close_date_map=close_date_map,
            )
        with pytest.raises(ValueError):
            extract_lambda_features(
                "movie_29",
                snap_days=0,
                close_ts=close_date_map["movie_29"],
                reviews_df=reviews,
                close_date_map=close_date_map,
            )

    def test_snap_days_is_keyword_only(self, cohort):
        """Plan §13 decision #3: snap_days must be keyword-only to match estimate_lambda."""
        reviews, close_date_map = cohort
        with pytest.raises(TypeError):
            # Positional snap_days must raise — guards against caller misuse.
            extract_lambda_features(  # type: ignore[misc]
                "movie_29", 3, close_date_map["movie_29"], reviews, close_date_map
            )

    def test_none_when_target_not_in_close_map(self, cohort):
        reviews, close_date_map = cohort
        feats = extract_lambda_features(
            "nonexistent",
            snap_days=3,
            close_ts=pd.Timestamp("2026-04-01 14:00:00", tz="UTC"),
            reviews_df=reviews,
            close_date_map=close_date_map,
        )
        assert feats is None

    def test_none_when_too_few_critics(self, cohort):
        reviews, close_date_map = cohort
        target = "movie_29"
        mask = ~(reviews["movie_slug"] == target)
        first_two = reviews[reviews["movie_slug"] == target].head(2)
        trimmed = pd.concat([reviews[mask], first_two], ignore_index=True)
        feats = extract_lambda_features(
            target,
            snap_days=3,
            close_ts=close_date_map[target],
            reviews_df=trimmed,
            close_date_map=close_date_map,
        )
        assert feats is None

    def test_finite_pool_features_in_range(self, cohort):
        reviews, close_date_map = cohort
        feats = extract_lambda_features(
            "movie_29",
            snap_days=3,
            close_ts=close_date_map["movie_29"],
            reviews_df=reviews,
            close_date_map=close_date_map,
        )
        assert feats is not None
        assert 0 <= feats["pool_mass_consumed"] <= 1
        assert 0 <= feats["observed_top_tier_frac"] <= 1
        assert feats["remaining_base_rate_sum"] >= 0

    def test_valid_snap_days_covers_all_supported(self):
        assert VALID_SNAP_DAYS == (1, 2, 3, 4, 5)

    def test_rate_delta_matches_components(self, cohort):
        reviews, close_date_map = cohort
        feats = extract_lambda_features(
            "movie_29",
            snap_days=3,
            close_ts=close_date_map["movie_29"],
            reviews_df=reviews,
            close_date_map=close_date_map,
        )
        assert feats is not None
        assert feats["rate_delta"] == pytest.approx(
            feats["rate_last_day"] - feats["rate_first_day"]
        )

    def test_log_transforms_consistent(self, cohort):
        reviews, close_date_map = cohort
        feats = extract_lambda_features(
            "movie_29",
            snap_days=3,
            close_ts=close_date_map["movie_29"],
            reviews_df=reviews,
            close_date_map=close_date_map,
        )
        assert feats is not None
        assert feats["log_observed_count"] == pytest.approx(np.log1p(feats["observed_count"]))
        assert feats["log_rate_last_day"] == pytest.approx(np.log1p(feats["rate_last_day"]))
        assert feats["sqrt_rate_last_day"] == pytest.approx(
            np.sqrt(max(feats["rate_last_day"], 0))
        )


# -- Feature parity against notebooks/proposed_ship_stack_test.ipynb ----------
# Per plan §8.1: hardcoded feature vectors from the ET-midnight ship-stack notebook.
# Expected values are from the cell output of `21dc3161` in proposed_ship_stack_test.ipynb
# (noon-shifted reviews, ET-midnight convention, A1 pool n=20, top-tier n=30).
# If these exact-value assertions break on a cohort refresh, investigate before editing
# expected values — drift would indicate a feature arithmetic regression.

PARITY_CASES: list[dict] = [
    {
        "slug": "28_years_later",
        "snap_days": 3,
        "expected": {
            "observed_count": 131.0,
            "first_review_dbc": 5.041667,
            "target_gap": 5.041667,
            "pub_diversity": 130.0,
            "pub_entropy": 4.864615,
            "low_activity_frac": 0.022901,
            "log_observed_count": 4.882802,
            "log_rate_last_day": 3.806662,
            "sqrt_rate_last_day": 6.633250,
            "rate_delta": -43.0,
            "remaining_base_rate_sum": 121.60,
            "pool_mass_consumed": 0.277481,
            "observed_top_tier_frac": 0.666667,
        },
    },
    {
        "slug": "28_years_later",
        "snap_days": 2,
        "expected": {
            "observed_count": 190.0,
            "first_review_dbc": 5.041667,
            "target_gap": 5.041667,
            "pub_diversity": 189.0,
            "pub_entropy": 5.239728,
            "low_activity_frac": 0.042105,
            "log_observed_count": 5.252273,
            "log_rate_last_day": 4.094345,
            "sqrt_rate_last_day": 7.681146,
            "rate_delta": -28.0,
            "remaining_base_rate_sum": 101.60,
            "pool_mass_consumed": 0.396316,
            "observed_top_tier_frac": 0.866667,
        },
    },
    {
        "slug": "28_years_later",
        "snap_days": 1,
        "expected": {
            "observed_count": 203.0,
            "first_review_dbc": 5.041667,
            "target_gap": 5.041667,
            "pub_diversity": 200.0,
            "pub_entropy": 5.292719,
            "low_activity_frac": 0.044335,
            "log_observed_count": 5.318120,
            "log_rate_last_day": 2.639057,
            "sqrt_rate_last_day": 3.605551,
            "rate_delta": -74.0,
            "remaining_base_rate_sum": 98.75,
            "pool_mass_consumed": 0.413250,
            "observed_top_tier_frac": 0.866667,
        },
    },
    {
        "slug": "28_years_later_the_bone_temple",
        "snap_days": 4,
        "expected": {
            "observed_count": 131.0,
            "first_review_dbc": 6.125000,
            "target_gap": 6.125000,
            "pub_diversity": 130.0,
            "pub_entropy": 4.864615,
            "low_activity_frac": 0.061069,
            "log_observed_count": 4.882802,
            "log_rate_last_day": 3.784190,
            "sqrt_rate_last_day": 6.557439,
            "rate_delta": -45.0,
            "remaining_base_rate_sum": 99.20,
            "pool_mass_consumed": 0.259425,
            "observed_top_tier_frac": 0.433333,
        },
    },
    {
        "slug": "28_years_later_the_bone_temple",
        "snap_days": 3,
        "expected": {
            "observed_count": 172.0,
            "first_review_dbc": 6.125000,
            "target_gap": 6.125000,
            "pub_diversity": 170.0,
            "pub_entropy": 5.131375,
            "low_activity_frac": 0.052326,
            "log_observed_count": 5.153292,
            "log_rate_last_day": 3.737670,
            "sqrt_rate_last_day": 6.403124,
            "rate_delta": -47.0,
            "remaining_base_rate_sum": 87.65,
            "pool_mass_consumed": 0.345651,
            "observed_top_tier_frac": 0.566667,
        },
    },
]


@pytest.fixture(scope="module")
def ship_cohort():
    if not (COHORT_REVIEWS.exists() and COHORT_MOVIES.exists()):
        pytest.skip("reviews.csv / movies_index.csv not present (gitignored)")
    reviews = pd.read_csv(COHORT_REVIEWS)
    reviews["estimated_timestamp"] = pd.to_datetime(
        reviews["estimated_timestamp"], format="ISO8601", utc=True
    )
    reviews = apply_noon_shift(reviews)
    movies = pd.read_csv(COHORT_MOVIES)
    movies["Bet Close Date"] = pd.to_datetime(
        movies["Bet Close Date"], utc=True, errors="coerce"
    )
    now = pd.Timestamp.now(tz="UTC")
    resolved = movies.dropna(subset=["Bet Close Date"])
    resolved = resolved[resolved["Bet Close Date"] < now]
    close_date_map = resolved.set_index("Slug")["Bet Close Date"].to_dict()
    return reviews, close_date_map


@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda c: f"{c['slug']}_T-{c['snap_days']}d")
class TestExtractFeaturesParity:
    def test_matches_notebook_vector(self, ship_cohort, case):
        reviews, close_date_map = ship_cohort
        if case["slug"] not in close_date_map:
            pytest.skip(f"{case['slug']} not in cohort close_date_map")
        feats = extract_lambda_features(
            case["slug"],
            snap_days=case["snap_days"],
            close_ts=close_date_map[case["slug"]],
            reviews_df=reviews,
            close_date_map=close_date_map,
        )
        assert feats is not None, f"skip rules unexpectedly failed for {case['slug']}"
        for name, expected in case["expected"].items():
            actual = feats[name]
            tol = 1e-3 if isinstance(expected, float) else 0
            assert actual == pytest.approx(expected, abs=max(tol, abs(expected) * 1e-4)), (
                f"{case['slug']} T-{case['snap_days']}d: {name} = {actual}, expected {expected}"
            )
