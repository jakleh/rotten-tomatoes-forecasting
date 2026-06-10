"""Oracle λ / p_fresh inputs for Gate 2.

The oracle feeds *realized* remaining review-rate and fresh-rate (the MLE — the best
inputs that exist) through ``compute_edge``; the Poisson×Binomial dispersion around
them is genuine real-time uncertainty, so this measures the ceiling of a real-time
forecaster, not perfect foresight. See ``plans/plan_gate_1_2_calibration.md``.

Placement rules (locked 2026-06-04, "Mixed-granularity placement"):
- m/h-confidence reviews enter at ``effective_ts = estimated_timestamp + 1 min``
  (look-ahead-safe latest edge for floored relative timestamps).
- d-confidence reviews are **crowd-forwarded** to their date's latest edge — here the
  next UTC midnight (DB day-level rows land at 00:00 UTC) — so a same-day-as-snap
  review falls on the *remaining* side.

Two observation boundaries (2026-06-07 amendment):
- ``mode='pure'``   — publication-time availability (architecture ceiling, headline).
- ``mode='lagged'`` — scrape-time availability (current-pipeline reality; ``scrape_time``
  is first-seen because the table is insert-only with dedup).

Both modes share the same **terminal** state (publication-time through close — the
settlement score is RT-display-based), so ``observed + remaining == terminal`` holds by
construction in 'pure' and by definition (terminal − observed) in 'lagged'; a review
scraped before the snap necessarily has ``estimated_timestamp ≤ scrape_time``, so the
lagged remainder is never negative. The settlement *label* stays Kalshi ``result`` —
never reconstructed here.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LAG = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class OracleInputs:
    """Everything ``compute_edge`` needs, plus the decomposition for invariant checks."""
    total_obs: int
    fresh_obs: int
    n_remaining: int
    fresh_remaining: int
    t_rem_hours: float        # fractional, never rounded
    lambda_rate: float        # n_remaining / t_rem_hours (per hour)
    p_fresh: float            # fresh_remaining / n_remaining; 0.5 sentinel when n_remaining == 0
    terminal_total: int
    terminal_fresh: int
    mode: str


def effective_ts(reviews: pd.DataFrame) -> pd.Series:
    """Publication-time availability per review under the locked placement rules."""
    est = pd.to_datetime(reviews["estimated_timestamp"], utc=True, format="ISO8601") \
        if reviews["estimated_timestamp"].dtype == object else reviews["estimated_timestamp"]
    is_d = reviews["timestamp_confidence"] == "d"
    eod = est.dt.normalize() + pd.Timedelta(days=1)   # crowd-forward: next UTC midnight
    return eod.where(is_d, est + LAG)


def oracle_inputs(reviews: pd.DataFrame, close_ts, snap_ts, *, mode: str = "pure") -> OracleInputs:
    """Oracle decomposition for one (movie, snap).

    ``reviews``: that movie's rows — ``estimated_timestamp``, ``scrape_time``
    (tz-aware), ``timestamp_confidence`` ∈ {m,h,d}, ``tomatometer_sentiment``.
    ``close_ts`` / ``snap_ts``: tz-aware datetimes, ``snap_ts < close_ts``.
    """
    if not snap_ts < close_ts:
        raise ValueError(f"snap_ts must precede close_ts ({snap_ts} >= {close_ts})")
    if mode not in ("pure", "lagged"):
        raise ValueError(f"mode must be 'pure' or 'lagged', got {mode!r}")

    eff = effective_ts(reviews)
    fresh = (reviews["tomatometer_sentiment"] == "positive").to_numpy()

    terminal = (eff <= close_ts).to_numpy()
    if mode == "pure":
        observed = (eff <= snap_ts).to_numpy()
    else:
        scr = pd.to_datetime(reviews["scrape_time"], utc=True, format="ISO8601") \
            if reviews["scrape_time"].dtype == object else reviews["scrape_time"]
        observed = ((scr <= snap_ts) & terminal).to_numpy()

    total_obs = int(observed.sum())
    fresh_obs = int((observed & fresh).sum())
    terminal_total = int(terminal.sum())
    terminal_fresh = int((terminal & fresh).sum())
    n_remaining = terminal_total - total_obs
    fresh_remaining = terminal_fresh - fresh_obs
    assert n_remaining >= 0 and fresh_remaining >= 0, "remaining must be non-negative"

    t_rem_hours = (close_ts - snap_ts).total_seconds() / 3600.0
    lambda_rate = n_remaining / t_rem_hours
    p_fresh = (fresh_remaining / n_remaining) if n_remaining > 0 else 0.5  # inert: λ=0

    return OracleInputs(
        total_obs=total_obs, fresh_obs=fresh_obs,
        n_remaining=n_remaining, fresh_remaining=fresh_remaining,
        t_rem_hours=t_rem_hours, lambda_rate=lambda_rate, p_fresh=p_fresh,
        terminal_total=terminal_total, terminal_fresh=terminal_fresh, mode=mode,
    )
