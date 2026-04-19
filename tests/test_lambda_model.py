"""Tests for lambda_model.py — fit, predict, phase-2, JSON artifact round-trip, DST."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rotten_tomatoes_forecasting.features import FEATURE_NAMES
from rotten_tomatoes_forecasting.lambda_model import (
    DEFAULT_SNAPS,
    LambdaPrediction,
    LambdaRegressor,
    LambdaRegressorMetadata,
    SnapModel,
    compute_close_day_phase2,
    estimate_lambda,
    fit_lambda_regressor,
    load_default_regressor,
    load_regressor,
    save_regressor,
)


# -- compute_close_day_phase2 ------------------------------------------------


class TestComputeCloseDayPhase2:
    def test_10am_edt_returns_10h(self):
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        count, hours = compute_close_day_phase2(close, C=1.0)
        assert hours == pytest.approx(10.0)
        assert count == pytest.approx(1.0)

    def test_10am_est_returns_10h(self):
        close = pd.Timestamp("2026-01-15 15:00:00", tz="UTC")
        count, hours = compute_close_day_phase2(close, C=1.0)
        assert hours == pytest.approx(10.0)
        assert count == pytest.approx(1.0)

    def test_dst_spring_forward_9h(self):
        # 2026-03-08 is the DST spring-forward Sunday. 10am ET = 14:00 UTC that day,
        # but midnight ET to 10am ET is only 9h because 2→3am is skipped.
        close = pd.Timestamp("2026-03-08 14:00:00", tz="UTC")
        _, hours = compute_close_day_phase2(close)
        assert hours == pytest.approx(9.0)

    def test_dst_fall_back_11h(self):
        # 2026-11-01 is the DST fall-back Sunday; midnight ET to 10am ET is 11h.
        close = pd.Timestamp("2026-11-01 15:00:00", tz="UTC")
        _, hours = compute_close_day_phase2(close)
        assert hours == pytest.approx(11.0)

    def test_non_standard_close_2am_et(self):
        close = pd.Timestamp("2026-04-06 06:00:00", tz="UTC")  # 2am ET EDT
        count, hours = compute_close_day_phase2(close, C=1.0)
        assert hours == pytest.approx(2.0)
        assert count == pytest.approx(0.2)

    def test_c_override_scales_linearly(self):
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        count_c1, _ = compute_close_day_phase2(close, C=1.0)
        count_c3, _ = compute_close_day_phase2(close, C=3.0)
        assert count_c3 == pytest.approx(3.0 * count_c1)

    def test_rejects_naive_timestamp(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            compute_close_day_phase2(pd.Timestamp("2026-04-06 14:00:00"))


# -- estimate_lambda shape / range / skip -----------------------------------


def _make_trivial_regressor(snap_days: int = 3, n_features: int = 17) -> LambdaRegressor:
    """A hand-rolled regressor for shape/arithmetic tests (no real fit)."""
    model = SnapModel(
        alpha=10.0,
        scaler_mean=np.zeros(n_features),
        scaler_scale=np.ones(n_features),
        ridge_coef=np.zeros(n_features),
        ridge_intercept=5.0,
    )
    meta = LambdaRegressorMetadata(
        artifact_version="1.0",
        library_version="0.2.0",
        sklearn_version="1.8.0",
        fit_date="2026-04-19",
        cohort_size=10,
        phase2_C=1.0,
        snap_alphas={snap_days: 10.0},
    )
    return LambdaRegressor(
        snap_models={snap_days: model},
        features=list(FEATURE_NAMES),
        training_residuals={snap_days: np.array([-2.0, 1.0, 3.0, -1.0, 0.5])},
        metadata=meta,
    )


def _zero_features() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


class TestEstimateLambda:
    def test_returns_lambda_prediction(self):
        reg = _make_trivial_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        pred = estimate_lambda(
            reg,
            _zero_features(),
            snap_days=3,
            close_ts=close,
            hours_to_close=72.0 + 10.0,
        )
        assert isinstance(pred, LambdaPrediction)

    def test_composition_arithmetic(self):
        reg = _make_trivial_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        pred = estimate_lambda(
            reg,
            _zero_features(),
            snap_days=3,
            close_ts=close,
            hours_to_close=82.0,
        )
        assert pred.total_pred == pytest.approx(pred.phase1_pred + pred.phase2_pred)
        assert pred.rate_per_hour == pytest.approx(pred.total_pred / 82.0)

    def test_phase1_equals_intercept_at_zero_features(self):
        reg = _make_trivial_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        pred = estimate_lambda(
            reg,
            _zero_features(),
            snap_days=3,
            close_ts=close,
            hours_to_close=82.0,
        )
        # Scaler mean=0, scale=1, coef=0, intercept=5 → predict = 5.
        assert pred.phase1_pred == pytest.approx(5.0)

    def test_p90_matches_residual_quantile(self):
        reg = _make_trivial_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        pred = estimate_lambda(
            reg,
            _zero_features(),
            snap_days=3,
            close_ts=close,
            hours_to_close=82.0,
        )
        expected = float(np.quantile(np.abs(reg.training_residuals[3]), 0.9))
        assert pred.p90_abs_err_estimate == pytest.approx(expected)

    def test_phase2_C_override(self):
        reg = _make_trivial_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        pred_c1 = estimate_lambda(
            reg, _zero_features(), snap_days=3, close_ts=close, hours_to_close=82.0
        )
        pred_c2 = estimate_lambda(
            reg, _zero_features(), snap_days=3, close_ts=close, hours_to_close=82.0, phase2_C=2.0
        )
        assert pred_c2.phase2_pred == pytest.approx(2.0 * pred_c1.phase2_pred)

    def test_rejects_out_of_range_snap_days(self):
        reg = _make_trivial_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        with pytest.raises(ValueError):
            estimate_lambda(
                reg, _zero_features(), snap_days=7, close_ts=close, hours_to_close=82.0
            )

    def test_rejects_missing_snap_model(self):
        reg = _make_trivial_regressor(snap_days=3)
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        with pytest.raises(ValueError, match="no snap_model"):
            estimate_lambda(
                reg, _zero_features(), snap_days=4, close_ts=close, hours_to_close=82.0
            )

    def test_rejects_missing_features(self):
        reg = _make_trivial_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        incomplete = {k: 0.0 for k in FEATURE_NAMES[:5]}
        with pytest.raises(ValueError, match="missing required keys"):
            estimate_lambda(
                reg, incomplete, snap_days=3, close_ts=close, hours_to_close=82.0
            )

    def test_rejects_zero_hours(self):
        reg = _make_trivial_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        with pytest.raises(ValueError, match="hours_to_close"):
            estimate_lambda(
                reg, _zero_features(), snap_days=3, close_ts=close, hours_to_close=0.0
            )


# -- Artifact round-trip -----------------------------------------------------


class TestArtifactRoundTrip:
    def test_save_then_load_matches(self):
        reg = _make_trivial_regressor()
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            save_regressor(reg, path)
            loaded = load_regressor(path)
            assert loaded.features == reg.features
            assert loaded.metadata.cohort_size == reg.metadata.cohort_size
            assert loaded.metadata.phase2_C == reg.metadata.phase2_C
            np.testing.assert_allclose(
                loaded.snap_models[3].ridge_coef, reg.snap_models[3].ridge_coef
            )
            np.testing.assert_allclose(
                loaded.snap_models[3].scaler_scale, reg.snap_models[3].scaler_scale
            )
            np.testing.assert_allclose(
                loaded.training_residuals[3], reg.training_residuals[3]
            )
        finally:
            path.unlink(missing_ok=True)

    def test_predictions_match_pre_save(self):
        reg = _make_trivial_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        pre = estimate_lambda(
            reg, _zero_features(), snap_days=3, close_ts=close, hours_to_close=82.0
        )
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            save_regressor(reg, path)
            loaded = load_regressor(path)
            post = estimate_lambda(
                loaded, _zero_features(), snap_days=3, close_ts=close, hours_to_close=82.0
            )
            assert post.phase1_pred == pytest.approx(pre.phase1_pred)
            assert post.total_pred == pytest.approx(pre.total_pred)
            assert post.rate_per_hour == pytest.approx(pre.rate_per_hour)
        finally:
            path.unlink(missing_ok=True)

    def test_artifact_is_valid_json(self):
        reg = _make_trivial_regressor()
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            save_regressor(reg, path)
            payload = json.loads(path.read_text())
            assert payload["artifact_version"] == "1.0"
            assert "snap_models" in payload
            assert "training_residuals" in payload
        finally:
            path.unlink(missing_ok=True)


# -- Shipped artifact --------------------------------------------------------


class TestShippedArtifact:
    def test_default_regressor_loads(self):
        reg = load_default_regressor()
        assert isinstance(reg, LambdaRegressor)
        assert set(reg.snap_models) == set(DEFAULT_SNAPS)
        assert len(reg.features) == 17

    def test_default_regressor_metadata_fields(self):
        reg = load_default_regressor()
        assert reg.metadata.cohort_size > 0
        assert reg.metadata.phase2_C == pytest.approx(1.0)
        assert reg.metadata.artifact_version == "1.0"

    def test_default_regressor_predicts(self):
        reg = load_default_regressor()
        close = pd.Timestamp("2026-04-06 14:00:00", tz="UTC")
        pred = estimate_lambda(
            reg,
            {name: 0.0 for name in reg.features},
            snap_days=3,
            close_ts=close,
            hours_to_close=82.0,
        )
        assert isinstance(pred, LambdaPrediction)
        assert pred.p90_abs_err_estimate > 0
