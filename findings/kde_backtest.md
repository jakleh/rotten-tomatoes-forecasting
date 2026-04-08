# Findings: KDE Model Backtest

**Date:** 2026-04-07
**Notebook:** `notebooks/kde_backtest.ipynb`
**Plan:** `plans/plan_kde_backtest.md`

---

## Setup

- 141 movies with hourly price data, 136 produced trade evaluations (5 skipped due to <5 training movies)
- Daily snapshots (every 24h) — hourly timed out at 30 min, needs `estimate_lambda` optimization
- Training set per movie: 20 most recent movies with Bet Close Date before test movie (no lookahead)
- Resolution from terminal prices (>=90 Yes, <=10 No), 29 ambiguous threshold-resolutions excluded
- Effective close time inferred from last price CSV timestamp (Bet Close Date is date-only / midnight UTC)
- 33,776 total trade evaluations across 4,912 snapshots

---

## P&L summary

### By minimum edge threshold

| min_edge | Trades | Win rate | Total P&L | Mean P&L/trade | Movies |
|----------|--------|----------|-----------|----------------|--------|
| 3c | 24,381 | 66% | +67,384c | +2.76c | 136 |
| 5c | 21,554 | 63% | +67,190c | +3.12c | 136 |
| 10c | 17,399 | 58% | +61,878c | +3.56c | 136 |
| 15c | 14,440 | 54% | +59,638c | +4.13c | 136 |
| 20c | 12,094 | 50% | +55,490c | +4.59c | 136 |

Positive P&L at every threshold. Mean P&L per trade increases with min_edge — higher-conviction bets pay better.

### By time horizon (min_edge = 5c)

| Horizon | Trades | Win rate | Total P&L | Mean P&L/trade |
|---------|--------|----------|-----------|----------------|
| T-1d | 136 | 61.0% | +1,389c | +10.2c |
| **T-3d** | **495** | **70.5%** | **+6,816c** | **+13.8c** |
| **T-5d** | **1,044** | **67.1%** | **+10,099c** | **+9.7c** |
| T-7d | 1,399 | 58.2% | +417c | +0.3c |
| T-7d+ | 18,480 | 62.6% | +48,469c | +2.6c |

**T-3d is the sweet spot:** highest win rate (70.5%) and best per-trade return (+13.8c). Confirms the model validation finding that the action window is T-3d to T-1d.

T-7d is essentially breakeven. T-7d+ has volume but thin per-trade edge.

### By edge magnitude

| Bucket | Trades | Win rate | Mean P&L | Mean edge |
|--------|--------|----------|----------|-----------|
| 0-3c | 9,511 | 94% | +0.22c | 1.5c |
| 3-5c | 2,726 | 88% | -0.03c | 4.2c |
| 5-10c | 4,141 | 85% | +1.27c | 7.6c |
| 10-15c | 2,958 | 77% | +0.75c | 12.5c |
| 15-20c | 2,347 | 71% | +1.78c | 17.5c |
| 20-30c | 3,417 | 67% | +3.39c | 24.7c |
| 30-50c | 4,008 | 55% | +4.16c | 39.4c |
| 50+c | 4,668 | 33% | +5.83c | 67.2c |

Small edges are noise (high win rate but near-zero P&L). Large edges (50+c) have low win rate but large payoffs when right. Best risk-adjusted zone: 20-50c edges.

---

## Direction asymmetry (key finding)

At min_edge = 5c:

| Direction | Trades | Win rate | Total P&L | Mean P&L/trade |
|-----------|--------|----------|-----------|----------------|
| Buy No | 10,156 | 69.2% | **+132,491c** | +13.0c |
| Buy Yes | 11,398 | 57.0% | **-65,301c** | -5.7c |

**All profit comes from the No side.** Buy Yes loses money.

The model is systematically more conservative than the market — it predicts lower P(Yes) than the market price implies. When the model says "the market is overpricing Yes" (buy No), it's consistently right. When it says "the market is underpricing Yes" (buy Yes), it's consistently wrong.

This is consistent with the known systematic underprediction of remaining reviews at T-3d (median error -12.4 from validation). Fewer expected reviews = overweight current score = conservative = lower P(Yes). This conservatism happens to be profitable on the No side.

**Implication for a live strategy:** only trade No, or require a much higher min_edge for Yes bets.

---

## Calibration

| Metric | Model | Market |
|--------|-------|--------|
| **Brier score (all)** | 0.1932 | 0.1172 |

The market is a better probabilistic forecaster overall. By horizon:

| Horizon | Model Brier | Market Brier | Delta |
|---------|-------------|--------------|-------|
| T-1d | 0.0134 | 0.0091 | +0.0043 |
| **T-3d** | **0.0206** | **0.0208** | **-0.0002** |
| T-5d | 0.0785 | 0.0605 | +0.0180 |
| T-7d | 0.1877 | 0.1041 | +0.0836 |
| T-7d+ | 0.2411 | 0.1445 | +0.0966 |

**T-3d is the one horizon where the model matches the market** on calibration. At T-7d+ the model is substantially worse. The model's edge is not from being a better overall forecaster — it's structural (conservatism that pays off on the No side).

---

## Top and bottom movies (min_edge = 5c)

### Top 10

| Movie | Total P&L | Trades | Mean P&L |
|-------|-----------|--------|----------|
| gladiator_ii | +30,745c | 882 | +34.9c |
| mission_impossible_the_final_reckoning | +21,322c | 397 | +53.7c |
| wicked_for_good | +12,519c | 240 | +52.2c |
| deadpool_and_wolverine | +10,629c | 572 | +18.6c |
| the_running_man_2025 | +6,736c | 176 | +38.3c |

### Bottom 10

| Movie | Total P&L | Trades | Mean P&L |
|-------|-----------|--------|----------|
| borderlands | -16,183c | 506 | -32.0c |
| joker_folie_a_deux | -12,732c | 622 | -20.5c |
| reagan_2024 | -8,888c | 348 | -25.5c |
| sinners_2025 | -6,504c | 177 | -36.7c |
| melania | -5,738c | 245 | -23.4c |

Wide per-movie dispersion. A few big losers and winners dominate the total.

---

## Known limitations

1. **Daily snapshots only.** Hourly backtest timed out. `estimate_lambda` needs vectorization (precompute KDE integrals over a time grid) before hourly is feasible.
2. **Correlated trades.** Daily snapshots for the same movie are not independent. The 21K "trades" overstate the number of independent bets.
3. **No fees.** Kalshi fee structure unknown — net P&L = gross - fees (BACKLOG §5.1).
4. **No execution realism.** Assumes we can trade at the hourly price. No bid-ask spread, no liquidity constraints.
5. **Close time approximation.** Effective close time = last price CSV timestamp. Actual market close may differ by up to ~1h.
6. **Training data timing bias.** KDE timing data is built relative to midnight-UTC Bet Close Date, but backtest evaluates relative to actual close time (~14h offset). Effect is small at T-3d+ horizons.

---

## Position-level analysis (No-only strategy)

De-duplicated to one entry per (movie, threshold). Enter the first time the No edge exceeds the threshold in the T-5d to T-1d window, hold to resolution.

Dataset spans 2024-03-25 to 2026-03-30 = 735 days (2.01 years), ~70 movies/year.

### By minimum edge threshold

| min_edge | Positions | Movies | Per movie | Win rate | Avg cost | Avg P&L | ROI |
|----------|-----------|--------|-----------|----------|----------|---------|-----|
| 5c | 325 | 112 | 2.9 | 80.6% | 60.5c | +20.1c | 33.2% |
| 10c | 231 | 99 | 2.3 | 78.4% | 54.7c | +23.7c | 43.4% |
| 15c | 174 | 88 | 2.0 | 76.4% | 49.7c | +26.7c | 53.8% |
| 20c | 136 | 76 | 1.8 | 75.7% | 46.1c | +29.7c | 64.3% |

Win rate is higher at position level (76-81%) than snapshot level (69%) — the first signal in the action window tends to be higher quality.

### Per-movie economics (min_edge = 10c)

- 99 movies have at least 1 position, **81% are profitable**
- Median per-movie P&L: +48.6c (mean +55.3c, std 73.2c)
- Average capital at risk per movie: 128c ($1.28) across 2.3 positions
- Worst movie: -79.7c, best movie: +302.8c
- Annualized (~49 movies/yr with signals): ~+2,721c/yr ($27/yr at $1/contract)

ROI scales linearly with position size. At 100 contracts/position: ~$2,700/yr. Constraint is Kalshi liquidity.

### Bankroll compounding simulation

**Notebook:** `notebooks/bankroll_simulation.ipynb`

Simulates bankroll trajectories replaying actual movie outcomes chronologically (single pass, no resampling or replacement), with all positions within a movie treated as perfectly correlated (worst case). Starting bankroll: $1,000. Note: with fractional betting, order doesn't affect the final multiplier — the product of per-movie returns is commutative.

| min_edge | risk/movie | Final bankroll | Multiplier | Movies |
|----------|-----------|----------------|------------|--------|
| 5c | 10% | $93,811 | 93.8x | 112 |
| 10c | 10% | $94,260 | 94.3x | 99 |
| 15c | 10% | $141,210 | 141.2x | 88 |
| 20c | 10% | $147,052 | 147.1x | 76 |
| 15c | 15% | $1,133,180 | 1,133x | 88 |

Higher min_edge (15-20c) compounds better despite fewer movies — higher per-position ROI dominates. Bankroll never dipped below starting value in any configuration (0% drawdown from start). **Updated:** min_edge=20c is the peak (see `findings/score_margin_and_robustness.md` for extended sweep through 40c and score margin band filter analysis with bootstrap robustness testing).

---

## Conclusions

1. **The model has a real, profitable edge.** +67K cents at min_edge=5c across 136 movies is not noise.
2. **The edge is concentrated in T-5d to T-1d** with the best per-trade returns at T-3d (+13.8c, 70.5% win rate).
3. **The edge is one-directional: No only.** Buy Yes loses money. The model's conservatism (underpredicting remaining reviews) is a feature on the No side, not a bug.
4. **The model is not a better forecaster than the market** (higher Brier score). The edge comes from a structural bias, not superior information.
5. **Position-level ROI is 43-64%** (depending on min_edge threshold) with 76-81% win rate. 81% of movies are profitable.
6. **A disciplined live strategy should:** trade T-5d to T-1d only, buy No only (or require very high min_edge for Yes), require min_edge >= 10-15c.

---

## Next steps

1. **Optimize for hourly backtest.** Vectorize `estimate_lambda` KDE integrals over a time grid. Would increase snapshot count ~24x and give finer-grained P&L curves.
2. **Investigate direction asymmetry.** Why does Buy Yes lose? Is it the underprediction of lambda, or is p_fresh also biased? Could a direction-aware strategy (No-only) be formalized?
3. **Per-movie P&L drivers.** What makes gladiator_ii (+30K) different from borderlands (-16K)? Movie characteristics that predict model accuracy.
4. **Fee estimation.** Kalshi fee structure needed to compute net P&L (BACKLOG §5.1).
5. **Live validation.** Test on 5-10 upcoming movies at modest size before scaling up.
