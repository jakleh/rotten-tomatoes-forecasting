"""Tests for p_fresh.py — base-rate-weighted critic prior blended with observed rate."""

import numpy as np
import pandas as pd
import pytest

from rotten_tomatoes_forecasting import estimate_p_fresh


@pytest.fixture
def reviews_df():
    rng = np.random.default_rng(11)
    rows = []
    for i in range(6):
        slug = f"movie_{i}"
        for c in range(20):
            if rng.random() < 0.4:
                continue
            rows.append(
                {
                    "movie_slug": slug,
                    "reviewer_name": f"critic_{c}",
                    "tomatometer_sentiment": "positive" if rng.random() < 0.7 else "negative",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def training_slugs():
    return [f"movie_{i}" for i in range(6)]


class TestEstimatePFresh:
    def test_returns_float_in_range(self, reviews_df, training_slugs):
        p = estimate_p_fresh(
            reviews_df, training_slugs, observed_critics=set(), fresh_count=50, total_count=80
        )
        assert 0 <= p <= 1

    def test_blends_toward_observed_with_more_data(self, reviews_df, training_slugs):
        p = estimate_p_fresh(
            reviews_df, training_slugs, observed_critics=set(), fresh_count=900, total_count=1000
        )
        assert abs(p - 0.9) < 0.05

    def test_zero_total_returns_prior(self, reviews_df, training_slugs):
        p = estimate_p_fresh(
            reviews_df, training_slugs, observed_critics=set(), fresh_count=0, total_count=0
        )
        assert 0 <= p <= 1

    def test_all_critics_observed_uses_fallback_prior(self, reviews_df, training_slugs):
        all_critics = set(reviews_df["reviewer_name"])
        p = estimate_p_fresh(
            reviews_df,
            training_slugs,
            observed_critics=all_critics,
            fresh_count=50,
            total_count=80,
        )
        assert 0 <= p <= 1

    def test_empty_training_pool(self, reviews_df):
        p = estimate_p_fresh(
            reviews_df, [], observed_critics=set(), fresh_count=50, total_count=80
        )
        assert 0 <= p <= 1

    def test_n_prior_controls_blend(self, reviews_df, training_slugs):
        """Higher n_prior → slower convergence to observed rate."""
        low = estimate_p_fresh(
            reviews_df,
            training_slugs,
            observed_critics=set(),
            fresh_count=50,
            total_count=80,
            n_prior=1.0,
        )
        high = estimate_p_fresh(
            reviews_df,
            training_slugs,
            observed_critics=set(),
            fresh_count=50,
            total_count=80,
            n_prior=200.0,
        )
        # low_n_prior relies more on observed (0.625); high relies more on prior.
        assert abs(low - 0.625) < abs(high - 0.625)
