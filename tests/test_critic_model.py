"""Tests for rotten_tomatoes_forecasting.critic_model — KDE model building and estimation."""

import numpy as np
import pandas as pd
import pytest

from rotten_tomatoes_forecasting.critic_model import (
    CriticProfiles,
    KDELambdaModel,
    build_critic_profiles,
    build_kde_lambda_model,
    estimate_lambda,
    estimate_p_fresh,
    default_training_slugs,
    _blended_integral,
    _fit_critic_kde,
    _fit_population_prior,
)


# -- Fixtures ------------------------------------------------------------------


def _make_reviews_df(n_movies=5, n_critics=10, reviews_per_critic_per_movie=0.6):
    """Generate a synthetic reviews DataFrame for testing."""
    rng = np.random.default_rng(42)
    rows = []
    slugs = [f"movie_{i}" for i in range(n_movies)]

    for slug in slugs:
        for c in range(n_critics):
            if rng.random() > reviews_per_critic_per_movie:
                continue
            rows.append({
                "movie_slug": slug,
                "reviewer_name": f"critic_{c}",
                "tomatometer_sentiment": rng.choice(["positive", "negative"], p=[0.7, 0.3]),
                "estimated_timestamp": pd.Timestamp("2026-03-01", tz="UTC")
                    + pd.Timedelta(days=rng.uniform(0, 20)),
            })

    return pd.DataFrame(rows)


def _make_movies_df(n_movies=5):
    """Generate a synthetic movies DataFrame for testing."""
    rows = []
    for i in range(n_movies):
        rows.append({
            "Slug": f"movie_{i}",
            "Bet Close Date": pd.Timestamp("2026-03-25", tz="UTC") + pd.Timedelta(days=i),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def reviews_df():
    return _make_reviews_df()


@pytest.fixture
def movies_df():
    return _make_movies_df()


@pytest.fixture
def profiles(reviews_df, movies_df):
    slugs = [f"movie_{i}" for i in range(5)]
    return build_critic_profiles(reviews_df, movies_df, slugs, verbose=False)


@pytest.fixture
def model(profiles):
    return build_kde_lambda_model(profiles, verbose=False)


# -- CriticProfiles tests -----------------------------------------------------


class TestBuildCriticProfiles:
    def test_returns_critic_profiles(self, reviews_df, movies_df):
        slugs = [f"movie_{i}" for i in range(5)]
        profiles = build_critic_profiles(reviews_df, movies_df, slugs, verbose=False)
        assert isinstance(profiles, CriticProfiles)
        assert profiles.training_slug_count == 5

    def test_df_has_required_columns(self, profiles):
        required = {"reviewer_name", "base_rate", "fresh_rate", "timing_data", "n_reviews"}
        assert required.issubset(set(profiles.df.columns))

    def test_base_rate_range(self, profiles):
        assert (profiles.df["base_rate"] >= 0).all()
        assert (profiles.df["base_rate"] <= 1).all()

    def test_fresh_rate_range(self, profiles):
        assert (profiles.df["fresh_rate"] >= 0).all()
        assert (profiles.df["fresh_rate"] <= 1).all()

    def test_verbose_false_suppresses_output(self, reviews_df, movies_df, capsys):
        slugs = [f"movie_{i}" for i in range(5)]
        build_critic_profiles(reviews_df, movies_df, slugs, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true_prints_output(self, reviews_df, movies_df, capsys):
        slugs = [f"movie_{i}" for i in range(5)]
        build_critic_profiles(reviews_df, movies_df, slugs, verbose=True)
        captured = capsys.readouterr()
        assert "[profiles]" in captured.out


# -- KDELambdaModel tests -----------------------------------------------------


class TestBuildKDELambdaModel:
    def test_returns_kde_model(self, profiles):
        model = build_kde_lambda_model(profiles, verbose=False)
        assert isinstance(model, KDELambdaModel)
        assert model.shrinkage_k == 3.0
        assert model.bandwidth_floor == 0.5

    def test_critic_kdes_populated(self, model):
        assert len(model.critic_kdes) > 0
        for name, entry in model.critic_kdes.items():
            assert "empirical" in entry
            assert "n" in entry
            assert "k" in entry

    def test_verbose_false_suppresses_output(self, profiles, capsys):
        build_kde_lambda_model(profiles, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_custom_shrinkage(self, profiles):
        model = build_kde_lambda_model(profiles, shrinkage_k=10.0, verbose=False)
        assert model.shrinkage_k == 10.0


# -- estimate_lambda tests ----------------------------------------------------


class TestEstimateLambda:
    def test_returns_positive_float(self, model):
        lam = estimate_lambda(model, days_before_close=3.0, hours_to_close=72.0,
                              observed_critics=set())
        assert isinstance(lam, float)
        assert lam >= 0

    def test_zero_hours_returns_zero(self, model):
        lam = estimate_lambda(model, days_before_close=0, hours_to_close=0,
                              observed_critics=set())
        assert lam == 0.0

    def test_observed_critics_reduces_lambda(self, model):
        """More observed critics → fewer remaining → lower lambda."""
        all_names = set(model.profiles.df["reviewer_name"])
        half_names = set(list(all_names)[:len(all_names) // 2])

        lam_none = estimate_lambda(model, 3.0, 72.0, observed_critics=set())
        lam_half = estimate_lambda(model, 3.0, 72.0, observed_critics=half_names)
        lam_all = estimate_lambda(model, 3.0, 72.0, observed_critics=all_names)

        assert lam_none >= lam_half
        assert lam_half >= lam_all

    def test_negative_dbc_returns_zero(self, model):
        """Past close → no reviews expected."""
        lam = estimate_lambda(model, days_before_close=-1.0, hours_to_close=-24.0,
                              observed_critics=set())
        assert lam == 0.0


# -- estimate_p_fresh tests ---------------------------------------------------


class TestEstimatePFresh:
    def test_returns_float_in_range(self, profiles):
        p = estimate_p_fresh(profiles, observed_critics=set(),
                             fresh_count=50, total_count=80)
        assert 0 <= p <= 1

    def test_blends_toward_observed_with_more_data(self, profiles):
        """With high total_count, should converge toward observed rate."""
        observed_rate = 0.90
        p = estimate_p_fresh(profiles, observed_critics=set(),
                             fresh_count=900, total_count=1000)
        assert abs(p - observed_rate) < 0.05

    def test_zero_total_returns_prior(self, profiles):
        """With no observations, return is purely the prior."""
        p = estimate_p_fresh(profiles, observed_critics=set(),
                             fresh_count=0, total_count=0)
        assert 0 <= p <= 1

    def test_all_critics_observed_uses_default_prior(self, profiles):
        """If all critics are observed, remaining set is empty → default prior."""
        all_names = set(profiles.df["reviewer_name"])
        p = estimate_p_fresh(profiles, observed_critics=all_names,
                             fresh_count=50, total_count=80)
        assert 0 <= p <= 1


# -- default_training_slugs tests ---------------------------------------------


class TestDefaultTrainingSlugs:
    def test_returns_list_of_strings(self, movies_df):
        slugs = default_training_slugs(movies_df)
        assert isinstance(slugs, list)
        assert all(isinstance(s, str) for s in slugs)

    def test_excludes_slug(self, movies_df):
        slugs = default_training_slugs(movies_df, exclude_slug="movie_0")
        assert "movie_0" not in slugs

    def test_respects_n(self, movies_df):
        slugs = default_training_slugs(movies_df, n=2)
        assert len(slugs) <= 2

    def test_before_date_filters(self, movies_df):
        cutoff = pd.Timestamp("2026-03-27", tz="UTC")
        slugs = default_training_slugs(movies_df, before_date=cutoff)
        # Only movies with close date < cutoff should be included
        for slug in slugs:
            close = movies_df[movies_df["Slug"] == slug]["Bet Close Date"].iloc[0]
            assert close < cutoff


# -- Internal helpers ----------------------------------------------------------


class TestBlendedIntegral:
    def test_fallback_returns_pop_integral(self):
        """When empirical is None, should return population prior integral."""
        pop_prior = _fit_population_prior(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        entry = {"empirical": None, "n": 0, "k": 3.0}
        result = _blended_integral(entry, pop_prior, 0, 5)
        pop_only = pop_prior.integrate_box_1d(0, 5)
        assert abs(result - pop_only) < 1e-10

    def test_blending_weights(self):
        """With n=k, empirical and prior should be weighted 50/50."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        pop_prior = _fit_population_prior(data)
        entry = _fit_critic_kde(data[:4], pop_prior, shrinkage_k=4.0, bandwidth_floor=0.5)
        # n=4, k=4 → 50/50 blend
        assert entry["n"] == 4
        assert entry["k"] == 4.0
