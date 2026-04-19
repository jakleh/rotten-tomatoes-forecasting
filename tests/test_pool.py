"""Tests for pool.py — A1 pool context + shared base-rate primitive."""

import numpy as np
import pandas as pd
import pytest

from rotten_tomatoes_forecasting.pool import (
    A1Context,
    _most_recent_resolved_slugs,
    build_a1_pool_context,
    compute_critic_base_rates,
)


@pytest.fixture
def close_date_map():
    base = pd.Timestamp("2026-03-01", tz="UTC")
    return {f"movie_{i}": base + pd.Timedelta(days=i) for i in range(10)}


@pytest.fixture
def reviews_df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(10):
        slug = f"movie_{i}"
        for c in range(15):
            if rng.random() < 0.5:
                continue
            rows.append({
                "movie_slug": slug,
                "reviewer_name": f"critic_{c}",
                "tomatometer_sentiment": "positive" if rng.random() < 0.7 else "negative",
                "estimated_timestamp": pd.Timestamp("2026-02-20", tz="UTC")
                + pd.Timedelta(days=rng.uniform(0, 10)),
            })
    return pd.DataFrame(rows)


class TestMostRecentResolvedSlugs:
    def test_respects_before(self, close_date_map):
        cutoff = pd.Timestamp("2026-03-05", tz="UTC")
        slugs = _most_recent_resolved_slugs(close_date_map, before=cutoff, n=20)
        for s in slugs:
            assert close_date_map[s] < cutoff

    def test_exclude_slug(self, close_date_map):
        cutoff = pd.Timestamp("2026-03-10", tz="UTC")
        slugs = _most_recent_resolved_slugs(
            close_date_map, before=cutoff, n=20, exclude_slug="movie_3"
        )
        assert "movie_3" not in slugs

    def test_respects_n(self, close_date_map):
        cutoff = pd.Timestamp("2026-03-10", tz="UTC")
        slugs = _most_recent_resolved_slugs(close_date_map, before=cutoff, n=3)
        assert len(slugs) <= 3


class TestComputeCriticBaseRates:
    def test_base_rate_in_unit_range(self, reviews_df):
        slugs = [f"movie_{i}" for i in range(5)]
        rates = compute_critic_base_rates(reviews_df, slugs)
        for v in rates.values():
            assert 0 <= v <= 1

    def test_empty_training(self, reviews_df):
        assert compute_critic_base_rates(reviews_df, []) == {}

    def test_critic_count_matches_movies_reviewed(self, reviews_df):
        slugs = [f"movie_{i}" for i in range(5)]
        rates = compute_critic_base_rates(reviews_df, slugs)
        critic = next(iter(rates))
        reviewed = (
            reviews_df[
                (reviews_df["movie_slug"].isin(slugs))
                & (reviews_df["reviewer_name"] == critic)
            ]["movie_slug"]
            .nunique()
        )
        assert rates[critic] == pytest.approx(reviewed / len(slugs))


class TestBuildA1PoolContext:
    def test_returns_a1_context(self, close_date_map, reviews_df):
        ctx = build_a1_pool_context("movie_9", close_date_map, reviews_df, n=5, top_tier_n=3)
        assert isinstance(ctx, A1Context)
        assert ctx.training_slugs == ["movie_8", "movie_7", "movie_6", "movie_5", "movie_4"]

    def test_target_excluded_from_training(self, close_date_map, reviews_df):
        ctx = build_a1_pool_context("movie_9", close_date_map, reviews_df, n=5)
        assert "movie_9" not in ctx.training_slugs

    def test_returns_none_on_missing_target(self, close_date_map, reviews_df):
        assert build_a1_pool_context("nonexistent", close_date_map, reviews_df) is None

    def test_returns_none_when_pool_too_small(self, close_date_map, reviews_df):
        assert build_a1_pool_context("movie_0", close_date_map, reviews_df) is None

    def test_top_tier_size(self, close_date_map, reviews_df):
        ctx = build_a1_pool_context("movie_9", close_date_map, reviews_df, n=5, top_tier_n=3)
        assert len(ctx.top_tier) <= 3

    def test_base_rate_and_total_consistent(self, close_date_map, reviews_df):
        ctx = build_a1_pool_context("movie_9", close_date_map, reviews_df, n=5)
        assert ctx.total_sum == pytest.approx(sum(ctx.base_rate.values()))
