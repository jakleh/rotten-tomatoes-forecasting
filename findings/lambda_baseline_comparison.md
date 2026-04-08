# Lambda Baseline Comparison

**Date:** 2026-04-08
**Notebook:** `notebooks/lambda_baseline_comparison.ipynb`
**Plans:** `plans/plan_lambda_baseline_comparison.md`, `plans/plan_blended_kde_lambda.md`

## Question

Does the per-critic KDE lambda model outperform naive baselines? Is the complexity worth it? And does blending the KDE prior with observed rate beat the current scaling approach?

## Setup

Four lambda estimators, everything else held constant (same KDE-based p_fresh, same action window T-1d to T-5d, same daily snapshots, No-only):

- **KDE (scaled):** Per-critic KDE model (`critic_model.py`) with multiplicative observed/expected scaling. Estimates remaining reviews from unreviewed critics' timing distributions.
- **Blended KDE:** Unscaled KDE rate blended with rolling rate, weighted by total_count / (total_count + 20). Like p_fresh blend but for lambda.
- **Naive rolling:** Reviews in last 24h / 24. Zero complexity.
- **Blended rolling:** Prior rate (average reviews/hour across training movies) blended with rolling average, weighted by total_count / (total_count + 20).

136 movies, 33,776 evaluations per estimator.

## P&L Results (No-only, T-1d to T-5d)

| Estimator | min_edge | Trades | Win rate | Total P&L | Mean P&L/trade |
|---|---|---|---|---|---|
| KDE (scaled) | 5c | 487 | 79.9% | 8,989c | 18.5c |
| KDE (scaled) | 10c | 325 | 75.7% | 7,176c | 22.1c |
| KDE (scaled) | 15c | 232 | 72.8% | 5,827c | 25.1c |
| KDE (scaled) | 20c | 173 | 71.1% | 4,679c | 27.0c |
| Blended KDE | 5c | 490 | 74.3% | 7,883c | 16.1c |
| Blended KDE | 10c | 332 | 69.0% | 6,329c | 19.1c |
| Blended KDE | 15c | 245 | 66.5% | 5,312c | 21.7c |
| Blended KDE | 20c | 187 | 64.7% | 4,459c | 23.8c |
| Naive rolling | 5c | 584 | 71.9% | 8,767c | 15.0c |
| Naive rolling | 10c | 406 | 66.0% | 7,004c | 17.3c |
| Naive rolling | 15c | 320 | 62.2% | 5,696c | 17.8c |
| Naive rolling | 20c | 250 | 60.0% | 4,911c | 19.6c |
| Blended rolling | 5c | 487 | 73.5% | 7,755c | 15.9c |
| Blended rolling | 10c | 332 | 67.5% | 6,340c | 19.1c |
| Blended rolling | 15c | 250 | 65.2% | 5,188c | 20.8c |
| Blended rolling | 20c | 185 | 63.2% | 4,333c | 23.4c |

## Lambda Accuracy (action window)

| Estimator | MAE (reviews) | Mean error (bias) | Median error |
|---|---|---|---|
| KDE (scaled) | 15.3 | -4.3 (slight underprediction) | -1.0 |
| Blended KDE | 41.7 | +33.8 (overprediction) | +18.2 |
| Naive rolling | 50.7 | +40.2 (massive overprediction) | +21.7 |
| Blended rolling | 45.5 | +40.6 (massive overprediction) | +29.2 |

## Key Findings

1. **KDE (scaled) wins on per-trade quality.** Higher win rate (76% vs 64-69%) and higher mean P&L per trade (22.1c vs 15.0-19.1c) at every min_edge threshold. KDE takes fewer, better trades.

2. **Total P&L is surprisingly close.** At min_edge=10c, KDE gets 7,176c vs naive's 7,004c — only 2.4% more. The naive baseline compensates for lower quality by taking more trades (406 vs 325). At min_edge=5c, KDE's advantage is slightly wider (8,989c vs 8,767c).

3. **KDE's advantage is concentrated at T-3d to T-4d.** Mean P&L per trade at this horizon: KDE 28.9c, blended KDE 23.6c, naive 18.1c. Near close (T-1d to T-2d), naive slightly beats KDE (18.4c vs 16.5c). KDE's value is in the far-from-close window where the rolling average lacks signal.

4. **Naive baselines massively overpredict remaining reviews.** Bias of +40 reviews means they extrapolate yesterday's rate as if it continues for days. Review rates decelerate as close approaches — the KDE model captures this curve shape, the rolling average can't.

5. **Naive rolling's overprediction accidentally helps.** Overpredicting lambda creates excess uncertainty, which systematically lowers P(Yes), which generates more No signals. Since the market structurally overprices Yes, those extra No signals are often correct. Naive rolling is profitable for the wrong reason.

6. **Zero-lambda is rare.** Only 3.1% of naive rolling snapshots (17/549) in the action window. 14/136 movies affected. Less of a problem than anticipated.

7. **Blended KDE doesn't beat scaled KDE.** The blend replaces the KDE's temporal shape with a flat rolling rate as total_count grows (blend weight → 1). The scaled approach preserves the shape and just adjusts magnitude, which is better suited here because the KDE shape is already being refined in real-time by critic dropping.

8. **Blended rolling doesn't beat naive rolling.** Despite addressing the zero-lambda failure mode, the blended estimator's total P&L is actually lower than naive (6,340c vs 7,004c at 10c). The prior-blending adds complexity without improving results.

## Interpretation

The KDE model is a better estimator by every statistical measure (3x lower MAE, near-zero bias vs +40 bias). But translating estimation quality into P&L is not linear — the naive baseline's overprediction creates more uncertainty in the model, which triggers more No signals, some of which happen to be correct.

KDE's real value is **trade selectivity**: fewer trades, higher win rate, higher per-trade return. This matters for a compounding strategy where per-trade quality drives bankroll growth. A 76% win rate compounds very differently from a 66% win rate.

The scaling approach works better than blending for lambda because the KDE shape is already accurate (critics are dropped as reviews arrive, continuously refining the shape). Scaling adjusts the magnitude of a good shape; blending replaces the shape with flat rate data. This is different from p_fresh estimation, where blending works well because there's no temporal structure to preserve.

## Conclusion

KDE with scaling is the best approach. It earns its keep through trade selectivity (76% win rate, 22.1c/trade) which matters most for compounding. The scaling mechanism is validated — it preserves the KDE's temporal shape while correcting magnitude, which is the right tool for this problem. No reason to switch to blending or a simpler model.
