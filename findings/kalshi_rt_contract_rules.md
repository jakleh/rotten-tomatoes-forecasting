# Kalshi RT Contract Rules

**Source:** `rt-rules-contract.pdf`
**Date reviewed:** 2026-04-09

## Key Rules

### Expiration

- **Expiration time:** 10:00 AM ET (Eastern Time, follows DST)
- **Expiration value:** Rotten Tomatoes "All Critics" Tomatometer score at expiration time
- **Revisions after expiration are ignored.** The score is locked at 10:00 AM ET.
- **Fallback:** If no RT data at 10:00 AM ET on the Monday following wide release, Tuesday's 10:00 AM ET value is used. If still unavailable a week later, all markets resolve to No.

### Position Limits

- **$25,000 per member per contract.** At $1/contract, max 25,000 contracts per position.
- This caps per-movie exposure regardless of bankroll. The bankroll simulation's "10% of bankroll per movie" stops being the binding constraint once bankroll exceeds $250K.
- Each threshold (Above 65, Above 70, etc.) is a separate contract, so the $25K limit applies per threshold, not per movie. Total per-movie exposure across all thresholds could be much higher.

### Trading

- **Last Trading Date:** The first Monday following wide release (Last Trading Time = 10:00 AM ET), or one year after `<date>` (Last Trading Time = 11:59 PM ET), whichever is sooner.
- **Minimum tick:** $0.01
- **Settlement:** No later than the day after expiration, unless under review.

### Payout Criterion

- Contracts resolve based on whether the expiration value is **above**, **below**, or **between** specified count levels.
- Count levels range from 0 to 100 in increments of 1.
- "Between" contracts pay out if the value is >= the lower bound and <= the upper bound.

## Implications for the Model

1. **Close time is 10:00 AM ET, not midnight UTC.** See `findings/bet_close_time_calibration.md` for the ~14h miscalibration this caused and the fix (now implemented in `movies_index.csv`).
2. **Position limit caps scaling.** Bankroll simulations showing 100x+ multipliers assume unconstrained sizing. In practice, $25K/threshold is the ceiling. With ~5 thresholds per movie in the contested zone, max per-movie exposure is ~$125K.
3. **The fallback rule is a tail risk.** If RT's site goes down at expiration, resolution shifts to Tuesday. If it's down for a week, everything resolves No. This is an edge case but worth monitoring in live trading.
