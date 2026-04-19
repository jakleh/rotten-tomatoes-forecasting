"""
Feature extraction for the Ridge lambda model.

Produces the 17-feature vector consumed by `fit_lambda_regressor` and `estimate_lambda`:
  - 10 observation-window features (counts, rates, critic mix, publication mix)
  - 4 nonlinear transforms of the dominant rate/count features
  - 3 finite-pool aggregates from the A1 pool context

All timestamp logic is anchored to Eastern-midnight per `CLAUDE.md` "Current Conventions".
The caller owns any noon-shift preprocessing (`apply_noon_shift=False` by default); the
library does not silently mutate timestamps across calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rotten_tomatoes_forecasting.pool import A1Context, build_a1_pool_context

FEATURE_NAMES: list[str] = [
    "observed_count",
    "first_review_dbc",
    "target_gap",
    "observed_rate",
    "rate_last_day",
    "rate_first_day",
    "top_critic_frac",
    "pub_diversity",
    "pub_entropy",
    "low_activity_frac",
    "log_observed_count",
    "log_rate_last_day",
    "sqrt_rate_last_day",
    "rate_delta",
    "remaining_base_rate_sum",
    "pool_mass_consumed",
    "observed_top_tier_frac",
]

VALID_SNAP_DAYS: tuple[int, ...] = (1, 2, 3, 4, 5)

MIN_OBSERVED_CRITICS: int = 3
LOW_ACTIVITY_THRESHOLD: int = 5
TOP_TIER_N: int = 30
A1_POOL_SIZE: int = 20


def midnight_et_of_close(close_ts: pd.Timestamp) -> pd.Timestamp:
    """Return midnight Eastern time on the close date (DST-aware).

    Example: close_ts = 2026-04-06 10:00 ET → midnight_et = 2026-04-06 00:00 ET.
    """
    if close_ts.tz is None:
        raise ValueError("close_ts must be timezone-aware")
    return close_ts.tz_convert("US/Eastern").normalize()


def apply_noon_shift(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Shift day-level reviews from midnight UTC to 12:00 UTC.

    Returns a copy; the input DataFrame is not mutated. Idempotent on rows that
    have already been shifted only when the caller applies it exactly once per
    ingest (the function has no memory of prior application — callers are
    responsible for applying it exactly once).

    Rationale: day-level `estimated_timestamp` values round to midnight UTC,
    which creates KDE-era boundary spikes and shifts arrival time off the
    day they belong to in ET. A 12h shift centers them in-day.
    """
    out = reviews_df.copy()
    mask = out["timestamp_confidence"] == "d"
    out.loc[mask, "estimated_timestamp"] = out.loc[mask, "estimated_timestamp"] + pd.Timedelta(
        hours=12
    )
    return out


def _pub_entropy(pubs: pd.Series) -> float:
    if len(pubs) == 0:
        return 0.0
    counts = pubs.fillna("unknown").value_counts(normalize=True).values
    if len(counts) == 0:
        return 0.0
    return float(-(counts * np.log(counts + 1e-12)).sum())


def _critic_activity_counts(reviews_df: pd.DataFrame) -> dict[str, int]:
    """Distinct movies reviewed per critic, cohort-wide. Used for low_activity_frac."""
    return reviews_df.groupby("reviewer_name")["movie_slug"].nunique().to_dict()


def extract_lambda_features(
    target_slug: str,
    *,
    snap_days: int,
    close_ts: pd.Timestamp,
    reviews_df: pd.DataFrame,
    close_date_map: dict[str, pd.Timestamp],
    a1_context: A1Context | None = None,
    activity_lookup: dict[str, int] | None = None,
    apply_noon_shift: bool = False,
) -> dict[str, float] | None:
    """Compute the 17-feature vector for one (target, snap) pair.

    Args:
        target_slug: Movie slug to extract features for.
        snap_days: Integer in {1, 2, 3, 4, 5}. Controls snap_time via ET-midnight anchor.
        close_ts: Market close timestamp (UTC, tz-aware). Used for snap/phase anchors.
        reviews_df: Reviews for the cohort. Must exclude the target's post-close reviews
            if LOO-cleanliness is required (caller's responsibility for training; at
            prediction time the target's live reviews are expected).
        close_date_map: Resolved close timestamps for cohort movies. Excludes the target's
            slug for LOO-clean A1 pool construction during fits.
        a1_context: Pre-built A1 pool context. If None, the function builds one via
            `build_a1_pool_context`. Pass pre-built for batch efficiency.
        activity_lookup: Pre-computed `{critic: distinct_movies_count}` dict. If None,
            computed from `reviews_df`. Pass pre-built for batch efficiency.
        apply_noon_shift: When True, copies `reviews_df` and shifts day-level reviews
            to noon UTC before computing features. Default False — the caller is
            expected to apply (or skip) the shift once at ingest and reuse.

    Returns:
        Dict keyed by `FEATURE_NAMES`. Returns None when skip rules don't pass:
          - A1 pool has < 5 training movies
          - first_review_dbc < snap_dbc_eff + 1 (first review too close to snap)
          - observed_critics < 3
          - observed_count == 0
          - gap or obs-window computations produce non-positive values
    """
    if snap_days not in VALID_SNAP_DAYS:
        raise ValueError(
            f"snap_days must be one of {VALID_SNAP_DAYS}, got {snap_days}"
        )
    if close_ts.tz is None:
        raise ValueError("close_ts must be timezone-aware")

    if apply_noon_shift:
        reviews_df = _apply_noon_shift_local(reviews_df)

    if a1_context is None:
        a1_context = build_a1_pool_context(
            target_slug, close_date_map, reviews_df, n=A1_POOL_SIZE, top_tier_n=TOP_TIER_N
        )
    if a1_context is None:
        return None

    if activity_lookup is None:
        activity_lookup = _critic_activity_counts(reviews_df)

    midnight_et_close = midnight_et_of_close(close_ts)
    snap_time = midnight_et_close - pd.Timedelta(days=snap_days)
    snap_dbc_eff = (close_ts - snap_time).total_seconds() / 86400

    target_rows = reviews_df[reviews_df["movie_slug"] == target_slug]
    obs = target_rows[
        (target_rows["estimated_timestamp"] < snap_time)
        & (target_rows["estimated_timestamp"] < close_ts)
    ]
    if obs.empty:
        return None

    observed_critics = set(obs["reviewer_name"])
    observed_count = int(len(obs))
    if observed_count == 0 or len(observed_critics) < MIN_OBSERVED_CRITICS:
        return None

    first_review_ts = obs["estimated_timestamp"].min()
    first_review_dbc = float((close_ts - first_review_ts).total_seconds() / 86400)
    if first_review_dbc < snap_dbc_eff + 1.0:
        return None

    obs_window_days = first_review_dbc - snap_dbc_eff
    if obs_window_days <= 0:
        return None

    target_gap = float((close_ts - first_review_ts).total_seconds() / 86400)

    last_day_start = snap_time - pd.Timedelta(days=1)
    rate_last_day = int(
        (
            (obs["estimated_timestamp"] >= last_day_start)
            & (obs["estimated_timestamp"] < snap_time)
        ).sum()
    )
    first_day_end = first_review_ts + pd.Timedelta(days=1)
    rate_first_day = int(
        (
            (obs["estimated_timestamp"] >= first_review_ts)
            & (obs["estimated_timestamp"] < first_day_end)
        ).sum()
    )

    is_top = obs["top_critic"].astype(str).str.lower().isin(["t", "true", "1"])
    top_critic_frac = float(is_top.mean()) if observed_count > 0 else 0.0

    pubs = obs["publication_name"].fillna("unknown")
    pub_diversity = int(pubs.nunique())
    pub_entropy = _pub_entropy(pubs)

    low_activity_count = sum(
        1 for c in observed_critics if activity_lookup.get(c, 0) < LOW_ACTIVITY_THRESHOLD
    )
    low_activity_frac = low_activity_count / len(observed_critics)

    observed_rate = observed_count / obs_window_days

    obs_base_rate_sum = float(
        sum(a1_context.base_rate.get(c, 0.0) for c in observed_critics)
    )
    total_sum = a1_context.total_sum
    remaining_base_rate_sum = max(total_sum - obs_base_rate_sum, 0.0)
    pool_mass_consumed = (obs_base_rate_sum / total_sum) if total_sum > 0 else 0.0
    observed_top_tier_frac = len(observed_critics & a1_context.top_tier) / TOP_TIER_N

    return {
        "observed_count": float(observed_count),
        "first_review_dbc": first_review_dbc,
        "target_gap": target_gap,
        "observed_rate": observed_rate,
        "rate_last_day": float(rate_last_day),
        "rate_first_day": float(rate_first_day),
        "top_critic_frac": top_critic_frac,
        "pub_diversity": float(pub_diversity),
        "pub_entropy": pub_entropy,
        "low_activity_frac": float(low_activity_frac),
        "log_observed_count": float(np.log1p(observed_count)),
        "log_rate_last_day": float(np.log1p(rate_last_day)),
        "sqrt_rate_last_day": float(np.sqrt(max(rate_last_day, 0))),
        "rate_delta": float(rate_last_day - rate_first_day),
        "remaining_base_rate_sum": remaining_base_rate_sum,
        "pool_mass_consumed": pool_mass_consumed,
        "observed_top_tier_frac": observed_top_tier_frac,
    }


def _apply_noon_shift_local(reviews_df: pd.DataFrame) -> pd.DataFrame:
    return apply_noon_shift(reviews_df)
