# Trading Strategy Implications — Ridge Lambda Model

**Date:** 2026-04-19.
**Source:** `rotten-tomatoes-forecasting` library's ridge_t2 lambda model, evaluated on 143-movie cohort LOO across T-5d → T-1d.
**Audience:** `kalshi-trading` orchestrator. Drop this doc into the orchestrator repo's findings/ and use as a design input for the RT strategy.

**Context:** the forecasting library exposes `estimate_lambda()` (reviews/hour) and `estimate_p_fresh()`. Those feed into `compute_edge()` which returns `edge_cents = P(Yes) × 100 − market_price`. This document translates the lambda model's **known error structure** into **where the edge is real vs. where it's phantom**.

---

## TL;DR

- **Snap gate:** concentrate exposure at T-3d / T-2d. T-5d / T-4d predictions are too wide for reliable edge (p90|err| of 47-67 reviews). T-1d is tight but the market has mostly absorbed the information.
- **HARD filter: exclude long-gap targets.** `target_gap > 15d` → skip entirely at every snap. This is the architectural-ceiling failure category; Ridge fails worst there (T-3d h/m MAE 25 reviews vs. cohort 10).
- **Soft filter:** on targets that clear the gap gate, also skip or size-down on extreme observed-count quartiles (Q1 niche / Q4 magnet) where the model has known systematic bias direction.
- **Sizing bound:** scale position inversely to `p90|err|` at the snap. The library can return this estimate (post-integration).
- **The model is unbiased at every snap** (mean_err ≈ 0), but the error distribution has a **long negative tail** driven by critic-magnet / late-surge targets. Those targets hit the architectural residual.

---

## 1. Prediction quality by snap

```
snap   typical miss    90th-pct miss   on typical phase-1 window
T-5d       23              67          ~30-45% relative error
T-4d       15              47          ~20-35%
T-3d        8              21          ~15-20%  ← edge-computable
T-2d        3               8          ~10-15%  ← peak quality
T-1d        2               5          ~15-25%  ← accurate but little info left
```

(`typical miss` = median |pred − actual|. `90th-pct miss` = 90% of predictions are within this.)

**Interpretation:**
- **T-3d onward is where edge is real.** At T-2d the typical miss is 3 reviews on a phase-1 window that averages 10-15 reviews. Any `edge_cents` > some threshold that includes ~20% relative error is real signal.
- **T-5d / T-4d is coin-flippy.** p90|err| of 47-67 means 10% of predictions are off by half the total phase-1 volume. You can't defend sizing on that unless the bet is tiny.
- **T-1d is tight but low-yield.** The market has absorbed most of the information by then. Use for closing / correcting positions, not opening fresh ones.

**Practical rule for the orchestrator's snap gate:**

```
auto-trade allowed:     T-3d, T-2d
exploratory only:       T-5d, T-4d   (small sizes, if at all)
close / adjust only:    T-1d
```

**Snap timing convention:** T-Nd = N days before midnight ET on bet close day. So T-2d snap is ~58h before 10am ET market close (= 2 × 24h + 10h). Not 48h.

---

## 2. Target-type filtering

The library's lambda model has two known failure modes that concentrate on target extremes.

### 2.1 Low-volume niche movies — systematic over-prediction

Targets with low observed review flow (Q1 of cohort `observed_count` at snap, roughly < 50 observed critics at T-3d) get **over-predicted** by ~16 reviews on average (mean_err = +16.6 on that quartile at T-3d).

Why: the library's 10-feature rate-based prediction expects "typical" review flow. Niche movies don't match. The model's prior pulls predictions toward cohort mean.

**Consequence for trading:** if the market is correctly pricing a niche movie as low-volume, the library will appear to show **phantom Yes edge** that doesn't exist. Buying Yes on low-volume targets = buying into the model's bias.

**Filter signal:**
- `observed_count` at snap is in the bottom cohort quartile, AND
- `target_gap > 15 days`, AND
- `observed_critics` in first-24h window < some threshold (thin early arrival).

### 2.2 High-volume critic-magnet movies — systematic under-prediction

Targets with high future activity that the observation window doesn't yet reveal (tentpole releases, franchise entries, the_drama-type phenomena) get **under-predicted** by ~12 reviews on average (mean_err = −12.2 on Q4 actual-volume at T-3d).

Why: the architectural ceiling identified in `findings/path_b_lite_investigation.md` §8. When a target draws way more critics than the cohort average, Ridge's regression on observation-window features can't scale up fast enough. The finite-pool features added in tier 2 help (via `observed_top_tier_frac`) but don't eliminate the miss.

**Consequence for trading:** if the market correctly prices these as high-volume, the library shows **phantom No edge**. Buying No on magnets = buying into the model's bias.

**Filter signal:**
- `observed_count` at snap is in the top cohort quartile, AND/OR
- Short gap (first_review within 7 days of close), AND
- Observation-window `rate_last_day` is rising / in the top quartile.

### 2.3 Long-gap movies — HARD EXCLUDE from auto-trading

**Rule: `target_gap > 15 days` → skip the target entirely, at every snap.**

Rationale: long-gap movies are the category where Ridge consistently fails worst. `they_will_kill_you` at T-3d had observed_count=46 and actual phase-1 of 67, but Ridge predicted 23 — error of −42 reviews. `forbidden_fruits_2026` at T-5d: predicted total 83, actual 37 (+46). These aren't occasional bad predictions; they're the signature of the architectural-ceiling failure mode (see `findings/path_b_lite_investigation.md` §8) which no Ridge feature addresses.

The proposed-ship-stack test (2026-04-19) confirmed this: on the 5 h/m targets, MAE at T-3d is 25 reviews with per-movie errors of ±20-47. Vs. the cohort's 10-review T-3d MAE.

Operationally:
- The orchestrator computes `target_gap = (close_ts − first_review_ts) / 86400d` from its own DB. Both terms UTC, tz-aware.
- If `target_gap > 15`, skip — do not even call `estimate_lambda`. Movie is not in the strategy's trade universe.
- Revisit when h/m cohort representation grows (multi-month wait); that's when the long-gap ceiling might become addressable.

This filter is stricter than the Q1/Q4 observed-count filters in §2.1/§2.2. Gap-based exclusion happens first; the others apply to the surviving set.

### 2.4 The sweet-spot target profile

Trade confidently where:

- **target_gap in [5, 12] days** — the cohort middle; close-anchor approximates embargo-anchor; gap-features work.
- **observed_count at T-3d in Q2-Q3** — moderate, matches where Ridge's feature coefficients are best-fit.
- **At least 20 observed critics** — features like `pub_diversity`, `top_critic_frac` have stable input.
- **`rate_last_day` in cohort mid-range** — momentum feature isn't in a regime where the model extrapolates poorly.

Roughly: "a movie of moderate expected popularity on a normal release cadence." The majority of the 143-movie cohort falls here; predictions on these targets have typical miss of 7-8 reviews at T-3d — enough to compute real edge.

---

## 3. Sizing

### 3.1 The unbiased-but-wide problem

At T-3d, the model's mean error is ~0 but individual predictions can be wrong by up to 42 reviews (the `they_will_kill_you` case). In aggregate, over- and under-predictions cancel. For a single bet, you can still be wildly wrong.

**Implications:**
- Relying on point `edge_cents` for sizing = exposed to the long tail.
- Fixed-fraction-of-bankroll sizing (e.g., 10%) doesn't account for per-movie prediction variance.
- Kelly-style sizing using the LOO residual distribution at the snap IS a defensible upgrade.

### 3.2 Sizing formula sketch

```
position_size_cap = bankroll × base_fraction × (snap_quality_factor) × (target_quality_factor)

snap_quality_factor:
  T-3d:  1.0  (reference)
  T-2d:  1.3  (tighter predictions → more confidence)
  T-1d:  0.8  (low remaining info → less time for price discovery)
  T-4d:  0.4  (wide predictions)
  T-5d:  0.2  (very wide)

target_quality_factor:
  sweet-spot profile (§2.4):  1.0
  extreme quartile:            0.3
  long-gap late-surge:         0.0  (skip)
```

The library can return `p90_abs_err_estimate` alongside the point prediction post-integration. Then sizing becomes:

```
position_size ≤ bankroll_risk_tolerance / (p90_abs_err × conversion_factor_to_edge_cents)
```

### 3.3 Position limit interaction

Per `findings/kalshi_rt_contract_rules.md`: $25K per member per contract (per threshold). With ~5 thresholds per movie in the contested zone, max exposure is ~$125K per movie regardless of bankroll. For bankrolls under ~$250K, position limits don't bind; fraction-of-bankroll sizing dominates. Above that, the $25K/threshold cap is the binding constraint — your sizing formula should max out at the cap, not scale linearly.

---

## 4. Bias-correction hook (orchestrator option)

If the orchestrator chooses NOT to filter extreme-quartile targets and instead trade them with adjusted predictions, here's a rough correction table derived from per-quartile mean_err at T-3d:

```
target profile                                 adjust phase-1 pred by
low-volume niche (Q1 observed_count)           -5 to -10 reviews (less Yes edge)
mid-volume standard (Q2-Q3)                     0 (no correction)
high-volume magnet (Q4 observed_count)         +5 to +12 reviews (less No edge)
long-gap late-surge (gap > 15d, thin obs)      +20 to +40 reviews (major No-edge reduction)
```

**This is rough.** n=30-35 per quartile. Use as a safety margin, not a precise calibration. Better: exclude extreme targets from auto-trade and only hand-pick them when other strategy inputs confirm.

---

## 5. What the forecasting library does and doesn't cover

**Covers:**
- `estimate_lambda` (review arrival rate, as reviews/hour)
- `estimate_p_fresh` (probability each remaining review is "positive")
- `compute_edge` (mechanical Poisson-binomial roll-up into Yes/No probability and edge_cents)

**Does NOT cover (lives in orchestrator):**
- Market price fetching / orderbook snapshots
- Position sizing / Kelly fraction
- Target filtering / skip rules (the rules in §2 of this doc)
- Snap-timing triggers (when to actually call `compute_edge`)
- Slippage modeling
- Partial fills / multi-threshold allocation
- Risk caps / correlation across simultaneous positions

The library is "given these inputs, here's the math." Everything strategic lives in the orchestrator.

---

## 6. Open deployment questions

1. **Does Kalshi's price itself correlate with these target profiles?** If the market correctly prices low-volume movies AND correctly prices magnets, our filters are just "skip markets where we can't beat the house." If the market has its own opposite biases, the library's errors might partially cancel out. Empirical question — answerable only with live P&L.

2. **Correlation of errors across simultaneous positions.** If Ridge's under-prediction residuals concentrate in a specific week (all long-gap movies releasing together), multiple simultaneous positions on that week's slate have correlated downside. Sizing formulas should account for this. Not characterized on our cohort; worth checking post-integration.

3. **Post-integration: orderbook snapshot data.** The orchestrator should be recording orderbook snapshots at each live call (per existing memory note). That backtest data will tell us whether the model's errors map to realizable P&L.

---

## 7. Validation status

- **Lambda model: 143-movie LOO cohort validation (2026-04-19).** 17 features, Ridge(α CV), per-snap phase-1 MAE 2-32 across T-1d to T-5d.
- **Phase-2 composition validation (h/m subset, n=5):** Ridge + C=1 composition at T-2d has MAE 2.45, me +0.67. At T-1d: MAE 1.48, me −0.38. Earlier snaps dominated by architectural-ceiling long-gap movies (they_will_kill_you, forbidden_fruits) — per §2.3 those should be excluded from trade universe.
- **Cohort composition MAE note:** the library's `LambdaPrediction.phase2_pred` is a constant C=1. On the 143-movie day-level cohort, composition MAE appears ~10-14 with systematic me ≈ -10. **This is not a calibration problem.** It's a noon-shift preprocessing artifact on day-level `actual_phase_2` reconstruction. Use h/m composition numbers above for calibration monitoring. Don't auto-trigger retrain on cohort composition MAE drift — it's measuring the wrong thing.
- **Trading-quality validation: not yet performed.** No P&L backtests with the Ridge model. The recommendations in §2 and §3 are derived from lambda-MAE structure, not from realized trading returns.
- **Priority work in orchestrator:** use Ridge predictions in a price-history backtest, compare to the prior KDE-based backtest results, validate or falsify §2's target-filter claims.
