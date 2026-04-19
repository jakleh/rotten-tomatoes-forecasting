"""
Ridge lambda model — fit, predict, and JSON artifact serialization.

Architecture:
  - One Ridge regressor per snap horizon (T-1d, T-2d, T-3d, T-4d, T-5d).
  - α chosen per snap by 5-fold CV over a fixed grid.
  - Features standardized via training-set StandardScaler stats (no leakage).
  - Target: phase-1 review count for the (midnight ET close, snap_time] window.
  - Phase 2 (close-day reviews) is added as a constant via `compute_close_day_phase2`.

Artifact: JSON at `_artifacts/default_regressor.json`. See README in the plan (§3.3b,
§5.2) for rationale against pickle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rotten_tomatoes_forecasting.features import (
    FEATURE_NAMES,
    VALID_SNAP_DAYS,
    extract_lambda_features,
    midnight_et_of_close,
)
from rotten_tomatoes_forecasting.pool import build_a1_pool_context

ARTIFACT_VERSION: str = "1.0"
DEFAULT_ARTIFACT_NAME: str = "default_regressor.json"
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_SNAPS: tuple[int, ...] = (5, 4, 3, 2, 1)
DEFAULT_CV_FOLDS: int = 5
DEFAULT_CV_SEED: int = 42
DEFAULT_PHASE2_C: float = 1.0


# -- Dataclasses --------------------------------------------------------------


@dataclass
class SnapModel:
    """Per-snap fitted Ridge coefficients plus scaler stats.

    Stored separately from sklearn objects so the artifact is version-agnostic JSON.
    Prediction applies `y = ((x − mean) / scale) @ coef + intercept`.
    """

    alpha: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    ridge_coef: np.ndarray
    ridge_intercept: float

    def predict(self, features_row: np.ndarray) -> float:
        scaled = (features_row - self.scaler_mean) / self.scaler_scale
        return float(scaled @ self.ridge_coef + self.ridge_intercept)


@dataclass
class LambdaRegressorMetadata:
    """Metadata attached to a fit artifact. Lets consumers assess staleness."""

    artifact_version: str
    library_version: str
    sklearn_version: str
    fit_date: str
    cohort_size: int
    phase2_C: float
    snap_alphas: dict[int, float]
    notes: str = ""


@dataclass
class LambdaRegressor:
    """Fitted Ridge lambda model across all snap horizons.

    Produced by `fit_lambda_regressor` or loaded via `load_default_regressor` /
    `load_regressor`. Consumed by `estimate_lambda`.
    """

    snap_models: dict[int, SnapModel]
    features: list[str]
    training_residuals: dict[int, np.ndarray]
    metadata: LambdaRegressorMetadata


@dataclass
class LambdaPrediction:
    """Lambda estimate plus its decomposition and uncertainty.

    `rate_per_hour` is what `compute_edge` consumes; the other fields expose the
    phase-1 Ridge output, the phase-2 close-day constant, and the p90|err|
    uncertainty derived from training LOO residuals at this snap.
    """

    rate_per_hour: float
    phase1_pred: float
    phase2_pred: float
    total_pred: float
    p90_abs_err_estimate: float


# -- Fit ----------------------------------------------------------------------


def _select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid: tuple[float, ...],
    cv_folds: int,
    cv_seed: int,
) -> float:
    """Pick the alpha with lowest CV MAE."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
    best_alpha: float | None = None
    best_mae = np.inf
    for alpha in alpha_grid:
        errs: list[float] = []
        for train_idx, test_idx in kf.split(X):
            pipe = Pipeline(
                [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
            )
            pipe.fit(X[train_idx], y[train_idx])
            errs.extend(np.abs(pipe.predict(X[test_idx]) - y[test_idx]).tolist())
        mae = float(np.mean(errs))
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    assert best_alpha is not None
    return best_alpha


def _loo_residuals(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """LOO residuals `(pred − actual)` for each sample at the given alpha."""
    preds = np.zeros(len(X))
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        pipe.fit(X[mask], y[mask])
        preds[i] = pipe.predict(X[i : i + 1])[0]
    return preds - y


def _fit_full(X: np.ndarray, y: np.ndarray, alpha: float) -> SnapModel:
    """Fit StandardScaler + Ridge on the full sample and return a SnapModel."""
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pipe.fit(X, y)
    scaler = pipe.named_steps["scaler"]
    ridge = pipe.named_steps["ridge"]
    return SnapModel(
        alpha=float(alpha),
        scaler_mean=scaler.mean_.astype(float),
        scaler_scale=scaler.scale_.astype(float),
        ridge_coef=ridge.coef_.astype(float),
        ridge_intercept=float(ridge.intercept_),
    )


def _compute_actual_phase1(
    target_slug: str,
    snap_days: int,
    close_ts: pd.Timestamp,
    reviews_df: pd.DataFrame,
) -> int:
    """Count reviews in phase 1 window: (midnight ET close − N days, midnight ET close]."""
    midnight_et_close = midnight_et_of_close(close_ts)
    midnight_et_dbc = (close_ts - midnight_et_close).total_seconds() / 86400
    snap_time = midnight_et_close - pd.Timedelta(days=snap_days)
    snap_dbc_eff = (close_ts - snap_time).total_seconds() / 86400

    mr = reviews_df[reviews_df["movie_slug"] == target_slug]
    if mr.empty:
        return 0
    dbc = ((close_ts - mr["estimated_timestamp"]).dt.total_seconds() / 86400).values
    return int(((dbc > midnight_et_dbc) & (dbc <= snap_dbc_eff)).sum())


def fit_lambda_regressor(
    cohort_reviews: pd.DataFrame,
    close_date_map: dict[str, pd.Timestamp],
    snap_days_list: list[int] | tuple[int, ...] = DEFAULT_SNAPS,
    alpha_grid: list[float] | tuple[float, ...] = DEFAULT_ALPHA_GRID,
    *,
    cv_folds: int = DEFAULT_CV_FOLDS,
    cv_seed: int = DEFAULT_CV_SEED,
    phase2_C: float = DEFAULT_PHASE2_C,
    library_version: str = "0.2.0",
    notes: str = "",
) -> LambdaRegressor:
    """Fit per-snap Ridge models across the cohort.

    Per snap:
      1. For each target, LOO-clean A1 pool built from `close_date_map` minus target.
      2. Feature extraction via `extract_lambda_features` (17 features).
      3. Target = phase-1 review count under ET-midnight convention.
      4. CV-select alpha over `alpha_grid`.
      5. Record LOO residuals for the snap (used later for uncertainty).
      6. Fit on the full cohort to get the shipped coefficients.

    Returns a LambdaRegressor ready for `estimate_lambda`.

    Note: the caller is responsible for any preprocessing (e.g., noon-shift). The
    function does not mutate `cohort_reviews`.
    """
    import sklearn

    snap_models: dict[int, SnapModel] = {}
    residuals: dict[int, np.ndarray] = {}
    alphas: dict[int, float] = {}

    for snap in snap_days_list:
        rows: list[dict[str, float]] = []
        targets_y: list[float] = []
        for target_slug, close_ts in close_date_map.items():
            a1_ctx = build_a1_pool_context(
                target_slug, close_date_map, cohort_reviews
            )
            if a1_ctx is None:
                continue
            feats = extract_lambda_features(
                target_slug,
                snap_days=snap,
                close_ts=close_ts,
                reviews_df=cohort_reviews,
                close_date_map=close_date_map,
                a1_context=a1_ctx,
            )
            if feats is None:
                continue
            y = _compute_actual_phase1(target_slug, snap, close_ts, cohort_reviews)
            rows.append(feats)
            targets_y.append(float(y))

        if len(rows) < cv_folds * 2:
            raise ValueError(
                f"snap T-{snap}d has only {len(rows)} valid targets; need at least {cv_folds * 2}"
            )

        X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=float)
        y = np.array(targets_y, dtype=float)
        alpha = _select_alpha(X, y, tuple(alpha_grid), cv_folds, cv_seed)
        residuals[snap] = _loo_residuals(X, y, alpha)
        snap_models[snap] = _fit_full(X, y, alpha)
        alphas[snap] = alpha

    cohort_size = len(close_date_map)
    fit_date = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    metadata = LambdaRegressorMetadata(
        artifact_version=ARTIFACT_VERSION,
        library_version=library_version,
        sklearn_version=sklearn.__version__,
        fit_date=fit_date,
        cohort_size=cohort_size,
        phase2_C=phase2_C,
        snap_alphas=alphas,
        notes=notes,
    )
    return LambdaRegressor(
        snap_models=snap_models,
        features=list(FEATURE_NAMES),
        training_residuals=residuals,
        metadata=metadata,
    )


# -- Phase 2 close-day constant ----------------------------------------------


def compute_close_day_phase2(
    close_ts: pd.Timestamp,
    C: float = DEFAULT_PHASE2_C,
) -> tuple[float, float]:
    """Phase 2 of the piecewise close-day correction.

    Represents expected pre-market critic arrivals in the
    (midnight ET of close day, close_ts] window.

    Returns (phase2_count, phase2_hours):
      - phase2_hours: actual elapsed hours from midnight ET to close_ts. Normally 10h
        for a 10am ET close; 9h on DST spring-forward Sunday, 11h on fall-back Sunday;
        other values for atypical close times (e.g., 2h for 2am ET close).
      - phase2_count: C, scaled linearly by (phase2_hours / 10.0) for non-10h windows.

    C=1.0 is calibrated for the 10h / 10am-ET case — empirical mean across five h/m
    targets with full scraper coverage was {2, 1, 1, 1, 0}. For non-10h windows the
    linear scaling is the defensible default; callers with better information can
    override C.
    """
    if close_ts.tz is None:
        raise ValueError("close_ts must be timezone-aware")
    midnight_et = midnight_et_of_close(close_ts)
    phase2_hours = (close_ts - midnight_et).total_seconds() / 3600
    phase2_count = C * (phase2_hours / 10.0)
    return phase2_count, phase2_hours


# -- Predict ------------------------------------------------------------------


def estimate_lambda(
    regressor: LambdaRegressor,
    features: dict[str, float],
    *,
    snap_days: int,
    close_ts: pd.Timestamp,
    hours_to_close: float,
    phase2_C: float | None = None,
) -> LambdaPrediction:
    """Estimate review arrival rate between snap_time and close_ts.

    Args:
        regressor: Fitted LambdaRegressor (from `fit_lambda_regressor` or
            `load_default_regressor`).
        features: Output of `extract_lambda_features`. Must contain every name in
            `regressor.features` in identical order.
        snap_days: Which snap horizon to predict. Must be one of {1, 2, 3, 4, 5}.
        close_ts: Market close timestamp (UTC, tz-aware). Used to compute phase-2
            hours dynamically (DST-aware).
        hours_to_close: Hours between "now" and close. Used to convert the total
            expected-review count into a rate. Typically `N × 24 + phase2_hours`.
        phase2_C: Override for the phase-2 constant. Defaults to
            `regressor.metadata.phase2_C`.

    Returns:
        LambdaPrediction with `rate_per_hour`, the phase-1 and phase-2 components,
        the total, and a p90|err| estimate from training residuals.

    Raises:
        ValueError when snap_days is out of range or a required feature is missing.
    """
    if snap_days not in VALID_SNAP_DAYS:
        raise ValueError(
            f"snap_days must be one of {VALID_SNAP_DAYS}, got {snap_days}"
        )
    snap_model = regressor.snap_models.get(snap_days)
    if snap_model is None:
        raise ValueError(
            f"Regressor has no snap_model for snap_days={snap_days}; "
            f"available: {sorted(regressor.snap_models)}"
        )
    if hours_to_close <= 0:
        raise ValueError(f"hours_to_close must be > 0, got {hours_to_close}")

    missing = [f for f in regressor.features if f not in features]
    if missing:
        raise ValueError(f"features dict missing required keys: {missing}")

    row = np.array([features[f] for f in regressor.features], dtype=float)
    phase1_pred = snap_model.predict(row)

    C = phase2_C if phase2_C is not None else regressor.metadata.phase2_C
    phase2_pred, _ = compute_close_day_phase2(close_ts, C=C)

    total_pred = phase1_pred + phase2_pred
    rate_per_hour = total_pred / hours_to_close

    resids = regressor.training_residuals.get(snap_days)
    if resids is None or len(resids) == 0:
        p90 = 0.0
    else:
        p90 = float(np.quantile(np.abs(resids), 0.9))

    return LambdaPrediction(
        rate_per_hour=rate_per_hour,
        phase1_pred=phase1_pred,
        phase2_pred=phase2_pred,
        total_pred=total_pred,
        p90_abs_err_estimate=p90,
    )


# -- Artifact I/O -------------------------------------------------------------


def _snap_model_to_jsonable(model: SnapModel) -> dict[str, Any]:
    return {
        "alpha": model.alpha,
        "scaler_mean": model.scaler_mean.tolist(),
        "scaler_scale": model.scaler_scale.tolist(),
        "ridge_coef": model.ridge_coef.tolist(),
        "ridge_intercept": model.ridge_intercept,
    }


def _snap_model_from_jsonable(d: dict[str, Any]) -> SnapModel:
    return SnapModel(
        alpha=float(d["alpha"]),
        scaler_mean=np.array(d["scaler_mean"], dtype=float),
        scaler_scale=np.array(d["scaler_scale"], dtype=float),
        ridge_coef=np.array(d["ridge_coef"], dtype=float),
        ridge_intercept=float(d["ridge_intercept"]),
    )


def save_regressor(regressor: LambdaRegressor, path: str | Path) -> None:
    """Serialize a LambdaRegressor to JSON. Round-trips via `load_regressor`."""
    payload: dict[str, Any] = {
        "artifact_version": regressor.metadata.artifact_version,
        "library_version": regressor.metadata.library_version,
        "sklearn_version": regressor.metadata.sklearn_version,
        "fit_date": regressor.metadata.fit_date,
        "cohort_size": regressor.metadata.cohort_size,
        "phase2_C": regressor.metadata.phase2_C,
        "snap_alphas": {str(k): v for k, v in regressor.metadata.snap_alphas.items()},
        "notes": regressor.metadata.notes,
        "features": list(regressor.features),
        "snap_models": {
            str(k): _snap_model_to_jsonable(v) for k, v in regressor.snap_models.items()
        },
        "training_residuals": {
            str(k): v.tolist() for k, v in regressor.training_residuals.items()
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def _regressor_from_dict(payload: dict[str, Any]) -> LambdaRegressor:
    metadata = LambdaRegressorMetadata(
        artifact_version=str(payload["artifact_version"]),
        library_version=str(payload["library_version"]),
        sklearn_version=str(payload["sklearn_version"]),
        fit_date=str(payload["fit_date"]),
        cohort_size=int(payload["cohort_size"]),
        phase2_C=float(payload["phase2_C"]),
        snap_alphas={int(k): float(v) for k, v in payload["snap_alphas"].items()},
        notes=str(payload.get("notes", "")),
    )
    snap_models = {
        int(k): _snap_model_from_jsonable(v)
        for k, v in payload["snap_models"].items()
    }
    residuals = {
        int(k): np.array(v, dtype=float)
        for k, v in payload["training_residuals"].items()
    }
    return LambdaRegressor(
        snap_models=snap_models,
        features=list(payload["features"]),
        training_residuals=residuals,
        metadata=metadata,
    )


def load_regressor(path: str | Path) -> LambdaRegressor:
    """Load a LambdaRegressor from a JSON artifact at `path`."""
    payload = json.loads(Path(path).read_text())
    return _regressor_from_dict(payload)


def load_default_regressor() -> LambdaRegressor:
    """Load the shipped default regressor from package data.

    Uses `importlib.resources` so it works whether the package is installed or
    run in-place. Fails loudly if the artifact wasn't included in the wheel.
    """
    try:
        artifact = (
            resources.files("rotten_tomatoes_forecasting._artifacts")
            / DEFAULT_ARTIFACT_NAME
        )
    except ModuleNotFoundError as e:
        raise FileNotFoundError(
            f"Default regressor artifact package not found: {e}. "
            f"Ensure `rotten_tomatoes_forecasting/_artifacts/` exists and is included "
            f"in [tool.setuptools.package-data]."
        ) from e
    with resources.as_file(artifact) as path:
        return load_regressor(path)
