"""Tests for rotten_tomatoes_forecasting package structure and 0.2.0 public API surface."""

import rotten_tomatoes_forecasting


class TestPublicAPI:
    """Verify the 0.2.0 public API symbols are importable from rotten_tomatoes_forecasting."""

    def test_compute_edge(self):
        assert callable(rotten_tomatoes_forecasting.compute_edge)

    def test_estimate_lambda(self):
        assert callable(rotten_tomatoes_forecasting.estimate_lambda)

    def test_estimate_p_fresh(self):
        assert callable(rotten_tomatoes_forecasting.estimate_p_fresh)

    def test_fit_lambda_regressor(self):
        assert callable(rotten_tomatoes_forecasting.fit_lambda_regressor)

    def test_load_default_regressor(self):
        assert callable(rotten_tomatoes_forecasting.load_default_regressor)

    def test_extract_lambda_features(self):
        assert callable(rotten_tomatoes_forecasting.extract_lambda_features)

    def test_compute_close_day_phase2(self):
        assert callable(rotten_tomatoes_forecasting.compute_close_day_phase2)

    def test_naive_estimators(self):
        assert callable(rotten_tomatoes_forecasting.naive_lambda)
        assert callable(rotten_tomatoes_forecasting.naive_p_fresh)

    def test_lambda_regressor_dataclass(self):
        from dataclasses import fields

        names = {f.name for f in fields(rotten_tomatoes_forecasting.LambdaRegressor)}
        assert {"snap_models", "features", "training_residuals", "metadata"} <= names

    def test_lambda_prediction_dataclass(self):
        from dataclasses import fields

        names = {f.name for f in fields(rotten_tomatoes_forecasting.LambdaPrediction)}
        assert {
            "rate_per_hour",
            "phase1_pred",
            "phase2_pred",
            "total_pred",
            "p90_abs_err_estimate",
        } <= names

    def test_version(self):
        assert rotten_tomatoes_forecasting.__version__ == "0.2.0"

    def test_no_kde_symbols_reexported(self):
        """0.2.0 removed KDE-era symbols; their absence is the breaking-change signal."""
        for removed in [
            "CriticProfiles",
            "KDELambdaModel",
            "build_critic_profiles",
            "build_kde_lambda_model",
            "default_training_slugs",
        ]:
            assert not hasattr(rotten_tomatoes_forecasting, removed), (
                f"removed symbol {removed} is still re-exported"
            )


class TestSubmoduleAccess:
    def test_edge_result_type(self):
        from rotten_tomatoes_forecasting.edge import EdgeResult

        assert EdgeResult is not None

    def test_lambda_model_internals(self):
        from rotten_tomatoes_forecasting.lambda_model import (
            SnapModel,
            LambdaRegressorMetadata,
            save_regressor,
            load_regressor,
        )

        assert SnapModel is not None
        assert LambdaRegressorMetadata is not None
        assert callable(save_regressor)
        assert callable(load_regressor)

    def test_features_constants(self):
        from rotten_tomatoes_forecasting.features import FEATURE_NAMES, VALID_SNAP_DAYS

        assert len(FEATURE_NAMES) == 17
        assert set(VALID_SNAP_DAYS) == {1, 2, 3, 4, 5}

    def test_pool_primitives(self):
        from rotten_tomatoes_forecasting.pool import (
            A1Context,
            build_a1_pool_context,
            compute_critic_base_rates,
        )

        assert A1Context is not None
        assert callable(build_a1_pool_context)
        assert callable(compute_critic_base_rates)


class TestCrossRepoImport:
    def test_orchestrator_import_pattern(self):
        from rotten_tomatoes_forecasting import (
            compute_edge,
            estimate_lambda,
            estimate_p_fresh,
            extract_lambda_features,
            load_default_regressor,
        )

        for fn in [
            compute_edge,
            estimate_lambda,
            estimate_p_fresh,
            extract_lambda_features,
            load_default_regressor,
        ]:
            assert callable(fn)

    def test_compute_edge_from_external(self):
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
        assert abs(result["p_yes"] + result["p_no"] - 1.0) < 1e-10
