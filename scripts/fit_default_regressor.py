"""Fit the default shipped LambdaRegressor on the current cohort and save as JSON.

Run from repo root:
    uv run python scripts/fit_default_regressor.py

Writes:
    rotten_tomatoes_forecasting/_artifacts/default_regressor.json
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rotten_tomatoes_forecasting.features import apply_noon_shift
from rotten_tomatoes_forecasting.lambda_model import (
    DEFAULT_ARTIFACT_NAME,
    fit_lambda_regressor,
    load_regressor,
    save_regressor,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = ROOT / "rotten_tomatoes_forecasting" / "_artifacts" / DEFAULT_ARTIFACT_NAME


def load_cohort() -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    reviews = pd.read_csv(ROOT / "reviews.csv")
    reviews["estimated_timestamp"] = pd.to_datetime(
        reviews["estimated_timestamp"], format="ISO8601", utc=True
    )
    assert reviews["estimated_timestamp"].notna().all(), "NaT timestamps in reviews"

    movies = pd.read_csv(ROOT / "movies_index.csv")
    movies["Bet Close Date"] = pd.to_datetime(
        movies["Bet Close Date"], utc=True, errors="coerce"
    )
    now = pd.Timestamp.now(tz="UTC")
    resolved = movies.dropna(subset=["Bet Close Date"])
    resolved = resolved[resolved["Bet Close Date"] < now]
    close_date_map = resolved.set_index("Slug")["Bet Close Date"].to_dict()
    return reviews, close_date_map


def main() -> None:
    reviews, close_date_map = load_cohort()
    print(f"cohort: {len(close_date_map)} resolved movies")

    reviews_shifted = apply_noon_shift(reviews)
    n_shifted = int((reviews["timestamp_confidence"] == "d").sum())
    print(f"noon-shifted {n_shifted} day-level reviews")

    regressor = fit_lambda_regressor(
        reviews_shifted,
        close_date_map,
        notes="ET-midnight convention, noon-shifted day-level reviews, A1 pool n=20.",
    )
    print(f"fit complete. alphas: {regressor.metadata.snap_alphas}")
    for snap, resids in sorted(regressor.training_residuals.items()):
        mae = float((resids**2).mean() ** 0.5)
        print(f"  T-{snap}d  n={len(resids)}  RMSE={mae:.2f}")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_regressor(regressor, ARTIFACT_PATH)
    print(f"wrote {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size / 1024:.1f} KB)")

    round_tripped = load_regressor(ARTIFACT_PATH)
    assert round_tripped.metadata.cohort_size == regressor.metadata.cohort_size
    assert round_tripped.features == regressor.features
    assert sorted(round_tripped.snap_models) == sorted(regressor.snap_models)
    print("round-trip load OK")


if __name__ == "__main__":
    main()
