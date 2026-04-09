# Finding: Bet Close Time Miscalibration

**Date:** 2026-04-09

## Problem

`Bet Close Date` in `movies_index.csv` is stored as a date-only string (e.g., `2026-04-06`). When pandas parses this, it becomes midnight UTC (`2026-04-06T00:00:00Z`). But Kalshi RT markets close at 10:00 AM Eastern on that date, which is:

- **14:00 UTC** during EDT (second Sunday in March through first Sunday in November)
- **15:00 UTC** during EST (first Sunday in November through second Sunday in March)

This creates a **~14-15 hour gap** between the reference point the model uses and when markets actually close.

## What It Affects

### KDE training (critic_model.py)

`days_before_close = (bet_close - estimated_timestamp) / 86400`

With `bet_close` at midnight UTC instead of ~14:00 UTC:
- All `days_before_close` values are shifted ~0.58-0.63 days too low
- Reviews on the close date get `days_before_close = 0` and are **filtered out** by `train[train["days_before_close"] > 0]`
- The KDE has zero mass for the last ~14-15h before actual close

This means any critic who tends to review on close day is invisible to the model in that window. Lambda is systematically underestimated, inflating apparent No edge. See `brainstorm/brainstorm_close_day_lambda_bias.md` for the full analysis and patch approaches.

### Backtest (kde_backtest.ipynb)

The backtest separately infers close time from the last price CSV timestamp, which is a good proxy for actual close (~14:00 UTC). So the backtest's `hours_to_close` calculations are roughly correct, but they're on a different time axis than the KDE training data. The KDE thinks "close" is midnight; the backtest thinks "close" is ~14:00.

### Impact on `days_before_close` at signal time

| Snapshot | Midnight-referenced | Actual (14:00 UTC) | Error |
|---|---|---|---|
| T-1d | 1.00 days | 1.58 days | -37% |
| T-3d | 3.00 days | 3.58 days | -16% |
| T-5d | 5.00 days | 5.58 days | -10% |

The error is largest close to close and shrinks at longer horizons. In the validated action window (T-5d to T-1d), the error ranges from 10-37%.

## Fix

Replace the date-only `Bet Close Date` column in `movies_index.csv` with a full UTC datetime reflecting the actual market close time.

Steps:
1. Confirm Kalshi RT markets always close at 10:00 AM Eastern.
2. For each movie, determine whether the close date falls in EDT or EST.
3. Convert to UTC: `10:00 AM EDT = 14:00 UTC`, `10:00 AM EST = 15:00 UTC`.
4. Store as ISO 8601 with timezone (e.g., `2026-04-06T14:00:00Z`).

No changes needed to:
- Reviews data (already UTC)
- Price history CSVs (already UTC, timestamps end with `Z`)
- Scraper or database config

The only downstream code change: `critic_model.py` and notebooks that parse `Bet Close Date` will automatically get the correct reference point once the column contains full datetimes instead of date-only strings.

## What This Doesn't Fix

Even with correct close times, ~98% of reviews still have day-level timestamps. A review timestamped to `2026-04-05T00:00:00Z` (midnight) could have actually arrived anytime on April 5. The close-time fix resolves the systematic ~14h shift in the reference point, but the ~24h noise in review timestamps remains. See `brainstorm/brainstorm_close_day_lambda_bias.md` for approaches to recover close-day review mass using minute-level data from live-tracked movies.

## Status

**Implemented (2026-04-09).** Confirmed 10:00 AM ET from both the Kalshi rules contract (`rt-rules-contract.pdf`) and empirical price data (all 141 movies end at 14:00Z or 15:00Z, matching EDT/EST). `Bet Close Date` in `movies_index.csv` now contains full UTC datetimes derived from the last timestamp in each movie's hourly price CSV.

Downstream code (`critic_model.py`, backtest notebooks) will automatically pick up the correct close times on next run. No code changes needed — the column name is unchanged, and pandas parses ISO 8601 datetimes natively.

See `findings/kalshi_rt_contract_rules.md` for other contract details (position limits, resolution rules).
