"""Tests for rotten_tomatoes_forecasting package structure and public API surface."""

import rotten_tomatoes_forecasting


class TestPublicAPI:
    """Verify the 8 public API symbols are importable from rotten_tomatoes_forecasting."""

    def test_compute_edge(self):
        assert callable(rotten_tomatoes_forecasting.compute_edge)

    def test_build_critic_profiles(self):
        assert callable(rotten_tomatoes_forecasting.build_critic_profiles)

    def test_build_kde_lambda_model(self):
        assert callable(rotten_tomatoes_forecasting.build_kde_lambda_model)

    def test_estimate_lambda(self):
        assert callable(rotten_tomatoes_forecasting.estimate_lambda)

    def test_estimate_p_fresh(self):
        assert callable(rotten_tomatoes_forecasting.estimate_p_fresh)

    def test_default_training_slugs(self):
        assert callable(rotten_tomatoes_forecasting.default_training_slugs)

    def test_critic_profiles_type(self):
        from dataclasses import fields
        f = {field.name for field in fields(rotten_tomatoes_forecasting.CriticProfiles)}
        assert "df" in f
        assert "training_slug_count" in f

    def test_kde_lambda_model_type(self):
        from dataclasses import fields
        f = {field.name for field in fields(rotten_tomatoes_forecasting.KDELambdaModel)}
        assert "profiles" in f
        assert "population_prior" in f
        assert "critic_kdes" in f

    def test_version(self):
        assert hasattr(rotten_tomatoes_forecasting, "__version__")
        assert isinstance(rotten_tomatoes_forecasting.__version__, str)


class TestSubmoduleAccess:
    """Verify internal symbols are accessible via submodule imports."""

    def test_edge_result_type(self):
        from rotten_tomatoes_forecasting.edge import EdgeResult
        assert EdgeResult is not None

    def test_naive_estimators(self):
        from rotten_tomatoes_forecasting.edge import naive_lambda, naive_p_fresh
        assert callable(naive_lambda)
        assert callable(naive_p_fresh)

    def test_internal_helpers(self):
        from rotten_tomatoes_forecasting.critic_model import _blended_integral, _compute_scaling
        assert callable(_blended_integral)
        assert callable(_compute_scaling)


class TestCrossRepoImport:
    """Simulate how the orchestrator imports rotten_tomatoes_forecasting."""

    def test_orchestrator_import_pattern(self):
        """The exact import pattern from the orchestrator's evaluate.py pseudocode."""
        from rotten_tomatoes_forecasting import (
            compute_edge,
            build_critic_profiles,
            build_kde_lambda_model,
            estimate_lambda,
            estimate_p_fresh,
            default_training_slugs,
        )
        # All should be the actual functions, not None
        for fn in [compute_edge, build_critic_profiles, build_kde_lambda_model,
                    estimate_lambda, estimate_p_fresh, default_training_slugs]:
            assert callable(fn)

    def test_compute_edge_from_external(self):
        """Call compute_edge exactly as the orchestrator would."""
        from rotten_tomatoes_forecasting import compute_edge
        result = compute_edge(
            threshold=75,
            market_price=42,
            fresh_count=60,
            total_count=80,
            hours_to_close=24,
            lambda_rate=1.5,
            p_fresh=0.72,
        )
        assert "edge_cents" in result
        assert "p_yes" in result
        assert "p_no" in result
        assert abs(result["p_yes"] + result["p_no"] - 1.0) < 1e-10
