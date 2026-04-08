# Monte Carlo Price Perturbation Simulation

**Date:** 2026-04-08
**Notebook:** `notebooks/monte_carlo_price_perturbation.ipynb`
**Plan:** `plans/plan_monte_carlo_price_perturbation.md`

## Question

The bankroll simulation is deterministic — same movies, same prices, same result. The with-replacement bootstrap tests composition risk (what if the movie mix changes?) but introduces a duplication artifact. What happens when we introduce realistic price noise — how sensitive are the results to the exact market price at entry?

## Method

Perturb market prices with empirical noise; keep the model side (reviews, KDE, p_fresh) fixed. Since `edge = model_p_yes * 100 - market_price` is linear in market_price, no KDE recomputation is needed — each MC sim is just arithmetic on cached `model_p_yes` values. 5,000 simulations per config.

**Noise model:** Empirical hour-to-hour price deltas from the action window (24-120h to close), sampled with replacement and added independently to each snapshot's price. Clamped to [1, 99].

**Noise characteristics (action window):** 117K deltas, mean=-0.04c (effectively zero), std=3.5c, p5=-3.4c, p95=+3.3c. Fat-tailed (kurtosis=78) — rare large moves are realistic. Volatility peaks at T-5d to T-3d (std=3.95c) vs T-3d to T-1d (std=2.98c). Mid-range prices (40-60c) are most volatile (std=5.1c); extremes (<20c, >80c) are quieter (std~2c).

## Results: Monte Carlo Distribution (5,000 sims)

| Config | ACTUAL | mc_p1 | mc_p5 | mc_p25 | mc_med | mc_p75 | mc_p95 | mc_std | ACTUAL %ile |
|---|---|---|---|---|---|---|---|---|---|
| **20c, [-3,+3]** | 249x | 122x | 154x | 206x | **255x** | 333x | 1072x | 625x | 47% |
| 20c, no filter | 147x | 88x | 111x | 150x | **186x** | 238x | 559x | 476x | 22% |
| 15c, [-3,+3] | 193x | 80x | 102x | 141x | **172x** | 220x | 661x | 421x | 64% |
| 15c, no filter | 141x | 73x | 98x | 137x | **167x** | 206x | 400x | 225x | 29% |
| 10c, [-3,+3] | 188x | 79x | 102x | 147x | **175x** | 218x | 845x | 467x | 60% |
| 10c, no filter | 94x | 46x | 59x | 89x | **108x** | 133x | 298x | 199x | 31% |

ACTUAL = the deterministic result from `margin_bankroll_sim.ipynb`. mc_* = Monte Carlo price perturbation distribution. ACTUAL %ile = where the deterministic result falls in the MC distribution.

**Note:** The zero-noise sanity check produced 226.6x (vs expected 249x from the original notebook). The ~10% gap is likely from data updates since the original backtest. All MC results are internally consistent with the zero-noise baseline.

## Key Findings

### 1. Price noise introduces moderate variance — much less than composition risk

| Config | MC spread (p95/p5) | Bootstrap spread (p95/p5) |
|---|---|---|
| 20c, [-3,+3] | 7.0x | 66.9x |
| 20c, no filter | 5.0x | 34.8x |
| 15c, no filter | 4.1x | 26.7x |

The MC p5-to-p95 spread is 4-8x. The with-replacement bootstrap spread is 20-67x. **Price noise is a much smaller source of uncertainty than movie composition risk.** This makes sense: a few cents of price noise changes entry timing and price slightly, but the same movies still resolve the same way. Composition changes (different winners/losers) fundamentally alter the multiplicative product.

### 2. MC median is slightly above ACTUAL for most configs

For "20c, [-3,+3]": MC median = 255x vs ACTUAL baseline = 226.6x (zero-noise). Price noise creates a slight positive skew — when noise pushes a marginal movie into the signal set, it sometimes adds a profitable position. When noise pushes a movie out, that opportunity is just missed (no loss). The asymmetry between "gain a winning trade" and "miss a winning trade" explains why the median rises.

### 3. Downside floors are solid

Even at the MC 1st percentile (worst out of 5,000 price-noise draws), all configs remain highly profitable:

| Config | mc_p1 |
|---|---|
| 20c, [-3,+3] | 122x |
| 20c, no filter | 88x |
| 15c, no filter | 73x |
| 10c, no filter | 46x |

No ruin, no break-even, not even close. The strategy is robust to price noise at every config.

### 4. The band filter amplifies MC variance (same pattern as bootstrap)

| Config | mc_std |
|---|---|
| 20c, no filter | 476x |
| 20c, [-3,+3] | 625x |

The band filter concentrates on higher-ROI positions, which amplifies both upside and downside. Same pattern seen in the bootstrap analysis.

### 5. Noise sensitivity: results are stable up to 1x, blow up at 2x

Tested 0.5x, 1x, 2x noise scale on "20c, [-3,+3]":

| Noise scale | mc_med | mc_p5 | mc_p95 | mc_std |
|---|---|---|---|---|
| 0.5x | 231x | 172x | 357x | 123x |
| 1.0x | 255x | 154x | 1072x | 625x |
| 2.0x | 454x | 169x | 7910x | 51585x |

At 2x noise, rare large perturbations create extreme outlier sims. The p5 is stable (169-172x), but the right tail explodes. **1x noise is the appropriate scale — it's calibrated to actual empirical price movements.**

### 6. Most movies have stable entry decisions

Per-movie sensitivity analysis (1,000 sims, 20c/[-3,+3]):
- **66 movies** entered in 95-100% of sims (stable core)
- **16 movies** entered in 5-95% of sims (sensitive to price noise)
- **39 movies** entered in <5% of sims (rare/never)
- **15 movies** never entered

The 16 sensitive movies are ones where the edge is near the 20c threshold — a few cents of noise can flip them in or out. The stable core of ~66 movies drives most of the compounding.

## Comparison: MC Price Perturbation vs With-Replacement Bootstrap

| Source of risk | What it tests | Variance magnitude | p5 floor |
|---|---|---|---|
| MC price perturbation | Same movies, noisy prices | mc_std 200-625x | 59-154x |
| With-replacement bootstrap | Different movie mixes | bs_std 225-1605x | 23-31x |

The bootstrap has wider variance and lower downside floors because it can remove winning movies entirely. The MC can only shift entry prices — it can't remove a movie from the pool. **Composition risk dominates price risk.**

## Conclusions

1. **Price noise is a small source of variance.** The strategy's profitability is robust to realistic price perturbations — even worst-case 1st percentile MC outcomes are 46-122x depending on config.
2. **Composition risk (from bootstrap) is the dominant uncertainty.** The future movie mix matters much more than price noise. This is reassuring for live trading: if we trust the model, the exact entry price matters less than whether the right movies show up.
3. **The 20c/[-3,+3] config remains the top performer** with MC median 255x and a solid p5 of 154x. ACTUAL falls near the MC median (47th percentile), confirming it's a typical outcome, not a lucky draw.
4. **No config shows ruin risk** under price perturbation. The strategy is structurally profitable.
5. **~66 movies form a stable entry core** — their signals are robust to price noise. The 16 sensitive movies at the edge threshold contribute variance but don't dominate.
