# PARAMETERS.md

All tunable parameters, split into two independent categories:
- **Model parameters** control how we estimate probabilities (lambda, p_fresh, edge).
- **Strategy parameters** control when and how we act on those probabilities (what to bet, when, how much).

These are independent axes — the same model can run with different strategy filters, and the same strategy can run with different model estimators.

---

# Model Parameters

These live in `edge.py` and `critic_model.py`. They determine the probability estimates that feed into `compute_edge()`.

## Binomial parameters (p_fresh estimation)

| Parameter | Value | Location | Description |
|---|---|---|---|
| `n_prior` | 20.0 | `critic_model.estimate_p_fresh()` | Pseudo-count for blending observed fresh rate with critic-profile prior. At `total_count = n_prior`, the blend is 50/50. Higher values trust the prior longer; lower values let observed data dominate sooner. |
| `n_training` | 20 | `critic_model.default_training_slugs()` | Number of most-recent resolved movies used to build critic profiles (base_rate, fresh_rate). Shared with lambda estimation. Larger = more stable profiles but less recency. |

**How p_fresh works:** `p_fresh = w * observed_rate + (1-w) * prior_rate`, where `w = total_count / (total_count + n_prior)`. The prior is a base_rate-weighted average of remaining (unreviewed) critics' historical fresh rates.

## Poisson parameters (lambda estimation)

| Parameter | Value | Location | Description |
|---|---|---|---|
| `n_training` | 20 | `critic_model.default_training_slugs()` | Same training pool as p_fresh. Determines which movies' review timing data gets used to fit critic KDEs. |
| `shrinkage_k` | 3.0 | `critic_model.build_kde_lambda_model()` | Shrinkage toward population prior. Each critic's KDE is blended: `(n/(n+k)) * empirical + (k/(n+k)) * population`. At k=3, a critic with 3 reviews gets 50/50 blend; with 10 reviews, 77% empirical. |
| `bandwidth_floor` | 0.5 days | `critic_model._fit_critic_kde()` | Minimum KDE bandwidth. Prevents overfitting to tight clusters of review times. If a critic's Scott's-rule bandwidth falls below this, it's forced up. |
| `scaling_threshold` | 40 | `critic_model._compute_scaling()` | Minimum expected reviews before observed/expected scaling is applied. Below this (common at T-7d+ where KDE tail mass is thin), scaling is unreliable so the ratio is forced to 1.0. |
| `scaling_clamp` | [0.5, 2.0] | `critic_model._compute_scaling()` | Bounds on the observed/expected scaling ratio. Prevents wild overcorrection when the KDE expectations are far off. |

**How lambda works:** For each unreviewed critic, `expected_remaining += base_rate * KDE_integral(0, days_before_close)`. This is optionally scaled by `observed_count / expected_so_far` (if above threshold), then divided by `hours_to_close` to get reviews/hour.

## Edge computation parameters

| Parameter | Value | Location | Description |
|---|---|---|---|
| Poisson tail cutoff | 1e-10 | `edge.compute_edge()` | `k_max = poisson.ppf(1 - 1e-10, mu)`. Determines how far into the Poisson tail we sum. Effectively exact for any practical mu. |

## Naive estimator parameters (fallback, no --kde)

| Parameter | Value | Location | Description |
|---|---|---|---|
| Lambda window | 6 hours | `edge.naive_lambda()` | Counts reviews in last 6h, divides by 6. Simple rate estimate. |
| p_fresh | raw `fresh/total` | `edge.naive_p_fresh()` | No blending, no prior. Just the running average. |

---

# Strategy Parameters

These live in the orchestrator (or backtest notebooks). They filter and size bets based on model output. The model doesn't know about these — it just produces edge estimates.

## Direction filter

| Parameter | Value | Status | Description |
|---|---|---|---|
| Direction | No-only | Validated | Only take No-side signals (edge_cents < 0). Buy Yes loses money in backtest. The model's systematic conservatism (underpredicting remaining reviews) is a feature for No bets — it overweights current score, which makes it good at predicting "this score won't recover." |

## Score margin filter

| Parameter | Value | Status | Description |
|---|---|---|---|
| `margin_band` | ~[-3, +3] | **Validated in bankroll simulation** | Only take No signals where `current_score - threshold` falls within a band around zero. A tight band around zero (the "contested zone" where score is close to the threshold) compounds dramatically better than no filter. |

**Per-trade analysis** (from `notebooks/threshold_fragility.ipynb`, No-only, min_edge=10c, T-5d to T-1d):

| Score Margin | Trades | Win Rate | Mean P&L/trade |
|---|---|---|---|
| < -10 (well below threshold) | 20 | 100% | 21.0c |
| -10 to -5 | 30 | 80% | 17.8c |
| -5 to 0 (just below) | 180 | 90% | 27.7c |
| 0 to +5 (just above) | 92 | 40% | 13.3c |
| +5 to +10 | 1 | 0% | -17.8c |

Per-trade stats suggested a ceiling filter (cut the 40% WR "above threshold" trades). But bankroll simulation (`notebooks/margin_bankroll_sim.ipynb`) told a different story: cutting those trades halves the multiplier because they're still +EV and provide compounding opportunities. The real win is a **band** filter that cuts both tails.

**Bankroll simulation results** (start=$1K, risk=10%/movie, T-5d to T-1d):

| min_edge | Band | Movies | Multiplier |
|---|---|---|---|
| 20c | -3 to +3 | 70 | **249.0x** |
| 5c | -5 to +2 | 103 | 207.9x |
| 15c | -3 to +3 | 79 | 192.7x |
| 10c | -3 to +3 | 92 | 188.4x |
| 20c | no filter | 76 | 147.1x |
| 15c | no filter | 88 | 141.2x |
| 10c | no filter | 99 | 94.3x |
| 20c | ≤ 0 | 66 | 45.7x |

Key insights from single-path simulation:
- A tight band around zero beats no filter at every min_edge level (249x vs 147x at 20c).
- Cutting only the ceiling (≤ 0) **hurts** compounding — those above-threshold trades are +EV and provide opportunities.
- Cutting the floor (removing extreme negative margins) helps — "easy wins" far below threshold have small payoffs that dilute compounding.
- The robust finding is "tight band around zero." Exact boundaries (±3 vs ±5) are suggestive but optimized on 136 movies — don't overfit.

**Bootstrap robustness analysis** (`notebooks/margin_robustness.ipynb`, 10K resamples per config):

| Config | Median | p5 (bad luck) | p25 | Std | Med/Std |
|---|---|---|---|---|---|
| 15c, no filter | 132x | 29.7x | 69.5x | 403x | 0.33 |
| 20c, no filter | 139x | 26.4x | 68.2x | 467x | 0.30 |
| 20c, [-3,+3] | 244x | 31.3x | 107.2x | 1605x | 0.15 |
| 20c, [-1,+1] | 633x | 34.1x | 177.2x | 516116x | 0.00 |

Key findings:
- **Zero ruin risk across all configs.** Even the worst 1st-percentile outcomes are 8-16x. Every strategy is profitable.
- **[-1,+1] is a lottery ticket.** Median 633x but std 516K — a few lucky resamples dominate. Not a reliable strategy despite the high median.
- **[-3,+3] improves the median (244x vs 139x) but inflates variance more than proportionally.** Med/std drops from 0.30 to 0.15. The band filter mostly inflates the upside, not the downside.
- **Downside floors are similar across configs** (p5 ranges 26-34x). The filter doesn't meaningfully change worst-case outcomes.
- **No filter has the best risk-adjusted metrics** (med/std=0.30-0.33). If optimizing for outcome predictability, skip the filter. If optimizing for typical outcome, the band filter wins.

**Why this works:** The contested zone (score near threshold) is where the market misprices most. Scores far below threshold are "obvious No" — the market already prices them low, so payoffs are small. Scores far above threshold are "obvious Yes" — betting No against them is risky. The edge concentrates where the outcome is genuinely uncertain and the Poisson-binomial math disagrees with market intuition about how fragile the current score is.

## Edge threshold

| Parameter | Value | Status | Description |
|---|---|---|---|
| `min_edge` | 20c | **Validated — peak at 20c** | Minimum absolute edge (cents) to trigger a bet. Tested at 5, 10, 15, 20, 25, 30, 35, 40c. Compounding peaks at 20c (147x no filter, 249x with [-3,+3] band). Above 20c, the movie pool thins too fast (61 movies at 25c, 51 at 30c) and compounding degrades monotonically. |

## Action window

| Parameter | Value | Status | Description |
|---|---|---|---|
| Window start | T-5d (120h before close) | Validated | Best per-trade returns in T-5d to T-1d window. Earlier signals (T-7d+) have unreliable lambda scaling. |
| Window end | T-1d (24h before close) | Validated | Very close to close, the market is efficient and edges shrink. |

## Position sizing

| Parameter | Value | Status | Description |
|---|---|---|---|
| Risk per movie | 10% of bankroll | Tested in simulation | Bankroll fraction allocated per movie. Tested in `notebooks/bankroll_simulation.ipynb`. |
| Max positions per movie | Not set | TBD | How many threshold/time combinations can be open for the same movie. |

## Threshold filter

| Parameter | Value | Status | Description |
|---|---|---|---|
| `min_threshold` | None (no filter) | Tested — **not useful** | Tested filtering No bets to thresholds >= X. Per-trade quality is flat from 45-75 and actually declines at 80+. Threshold level alone doesn't predict trade quality. The score margin filter is strictly better. |

---

# Open Questions

## Model parameters
- **`n_prior`**: Hardcoded guess. Probably insensitive in the action window (total_count >> 20), but not validated.
- **`n_training`**: Why 20? Tradeoff between profile stability and recency. Not explored.
- **`shrinkage_k`**: Why 3? Affects how quickly sparse critics' KDEs converge to their own data vs population shape. Not explored.
- **`scaling_threshold` and `scaling_clamp`**: Tuned once during validation (raised from 5 to 40, tightened from [0.3, 3.0] to [0.5, 2.0]). Could revisit.
- **`bandwidth_floor`**: 0.5 days seems reasonable given timestamp noise (~98% day-level), but not explored.

## Strategy parameters
- **`min_edge`**: Swept 5-40c. Peaks at 20c, degrades monotonically above (movie pool too thin). Validated.
- **`margin_band`**: [-3, +3] improves median 75% (244x vs 139x at 20c) but doubles variance. [-1, +1] is a lottery ticket (median 633x, std 516Kx). Fine-grained sweep and bootstrap done — the band filter mostly inflates upside, not downside. Decision depends on whether you optimize for typical outcome (use band) or predictability (skip filter). Risk fraction (`bankroll_frac`) interaction not yet explored.
- **Position sizing**: 10% risk/movie was tested but not optimized. Kelly criterion or other sizing strategies not explored.
- **Evaluation frequency**: How often does the orchestrator check for signals? Daily snapshots are validated; hourly needs optimization work.
