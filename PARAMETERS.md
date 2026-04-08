# PARAMETERS.md

All tunable parameters in the model, organized by which top-level input they feed into.

---

## Binomial parameters (p_fresh estimation)

These control how we estimate the probability that each future review is positive.

| Parameter | Value | Location | Description |
|---|---|---|---|
| `n_prior` | 20.0 | `critic_model.estimate_p_fresh()` | Pseudo-count for blending observed fresh rate with critic-profile prior. At `total_count = n_prior`, the blend is 50/50. Higher values trust the prior longer; lower values let observed data dominate sooner. |
| `n_training` | 20 | `critic_model.default_training_slugs()` | Number of most-recent resolved movies used to build critic profiles (base_rate, fresh_rate). Shared with lambda estimation. Larger = more stable profiles but less recency. |

**How p_fresh works:** `p_fresh = w * observed_rate + (1-w) * prior_rate`, where `w = total_count / (total_count + n_prior)`. The prior is a base_rate-weighted average of remaining (unreviewed) critics' historical fresh rates.

---

## Poisson parameters (lambda estimation)

These control how we estimate the rate of future review arrivals (reviews/hour).

| Parameter | Value | Location | Description |
|---|---|---|---|
| `n_training` | 20 | `critic_model.default_training_slugs()` | Same training pool as p_fresh. Determines which movies' review timing data gets used to fit critic KDEs. |
| `shrinkage_k` | 3.0 | `critic_model.build_kde_lambda_model()` | Shrinkage toward population prior. Each critic's KDE is blended: `(n/(n+k)) * empirical + (k/(n+k)) * population`. At k=3, a critic with 3 reviews gets 50/50 blend; with 10 reviews, 77% empirical. |
| `bandwidth_floor` | 0.5 days | `critic_model._fit_critic_kde()` | Minimum KDE bandwidth. Prevents overfitting to tight clusters of review times. If a critic's Scott's-rule bandwidth falls below this, it's forced up. |
| `scaling_threshold` | 40 | `critic_model._compute_scaling()` | Minimum expected reviews before observed/expected scaling is applied. Below this (common at T-7d+ where KDE tail mass is thin), scaling is unreliable so the ratio is forced to 1.0. |
| `scaling_clamp` | [0.5, 2.0] | `critic_model._compute_scaling()` | Bounds on the observed/expected scaling ratio. Prevents wild overcorrection when the KDE expectations are far off. |

**How lambda works:** For each unreviewed critic, `expected_remaining += base_rate * KDE_integral(0, days_before_close)`. This is optionally scaled by `observed_count / expected_so_far` (if above threshold), then divided by `hours_to_close` to get reviews/hour.

---

## Edge computation parameters

| Parameter | Value | Location | Description |
|---|---|---|---|
| Poisson tail cutoff | 1e-10 | `edge.compute_edge()` | `k_max = poisson.ppf(1 - 1e-10, mu)`. Determines how far into the Poisson tail we sum. Effectively exact for any practical mu. |

---

## Trading / backtest parameters

| Parameter | Value | Location | Description |
|---|---|---|---|
| `min_edge` | 5c+ | `notebooks/kde_backtest.ipynb`, `notebooks/bankroll_simulation.ipynb` | Minimum absolute edge (in cents) to trigger a bet. Tested at 5, 10, 15, 20c in bankroll simulation (No-only). Higher thresholds (15-20c) produced better compounding results despite fewer trades. Not rigorously optimized. |
| Action window | T-5d to T-1d | Backtest finding | Best per-trade returns observed in this window. Not a hard parameter in code but a finding that shapes when we'd trade. |
| `EVERY_N_HOURS` | 24 (daily) | `notebooks/kde_backtest.ipynb` | Snapshot interval for backtest evaluation. 24 = daily, 1 = hourly (slower, needs optimization). |

---

## Naive estimator parameters (fallback, no --kde)

| Parameter | Value | Location | Description |
|---|---|---|---|
| Lambda window | 6 hours | `edge.naive_lambda()` | Counts reviews in last 6h, divides by 6. Simple rate estimate. |
| p_fresh | raw `fresh/total` | `edge.naive_p_fresh()` | No blending, no prior. Just the running average. |

---

## Open questions

- **`n_prior`**: Hardcoded guess. Probably insensitive in the action window (total_count >> 20), but not validated.
- **`n_training`**: Why 20? Tradeoff between profile stability and recency. Not explored.
- **`shrinkage_k`**: Why 3? Affects how quickly sparse critics' KDEs converge to their own data vs population shape. Not explored.
- **`scaling_threshold` and `scaling_clamp`**: Tuned once during validation (raised from 5 to 40, tightened from [0.3, 3.0] to [0.5, 2.0]). Could revisit.
- **`bandwidth_floor`**: 0.5 days seems reasonable given timestamp noise (~98% day-level), but not explored.
- **`min_edge`**: Tested at 5, 10, 15, 20c in bankroll simulation. 15-20c outperformed lower thresholds on compounding returns (fewer but higher-quality trades). Not rigorously optimized — the bankroll sim is one lens, not a grid search.
