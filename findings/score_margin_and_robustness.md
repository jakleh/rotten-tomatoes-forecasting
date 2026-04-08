# Score Margin Filter & Robustness Analysis

**Date:** 2026-04-08
**Notebooks:** `notebooks/threshold_fragility.ipynb`, `notebooks/margin_bankroll_sim.ipynb`, `notebooks/margin_robustness_v2.ipynb`

## Question

Does the No-side edge concentrate by threshold level (fragility of high percentages) or by some other dimension? Can we improve compounding with a filter? How robust are the results?

## Threshold Fragility Hypothesis

**Hypothesis:** High Tomatometer percentages are mathematically fragile — at 80% fresh, a rotten review moves the score down 4x as much as a fresh review moves it up (`p/(1-p)` asymmetry). The market underprices this fragility, so No edge should concentrate at high thresholds.

**Result:** Partially wrong. Edge does NOT concentrate by threshold level — per-trade quality is flat from threshold 45 to 75 and actually declines at 80+. Filtering to high thresholds doesn't help.

## Score Margin: Where the Edge Actually Concentrates

The edge concentrates by **score margin** (current_score - threshold at time of signal):

| Score Margin | Trades | Win Rate | Mean P&L/trade |
|---|---|---|---|
| < -10 (well below threshold) | 20 | 100% | 21.0c |
| -10 to -5 | 30 | 80% | 17.8c |
| -5 to 0 (just below) | 180 | 90% | 27.7c |
| 0 to +5 (just above) | 92 | 40% | 13.3c |
| +5 to +10 | 1 | 0% | -17.8c |

*(No-only, min_edge=10c, T-5d to T-1d, 325 trades)*

The model excels at "this score won't recover above the threshold" (90% WR) but struggles with "this score will drop below the threshold" (40% WR). All margin buckets except 0-to-+5 and +5-to-+10 have positive mean P&L — removing them improves per-trade stats but costs compounding opportunities.

**Why:** The market overestimates recovery probability for scores just below a threshold. The Poisson-binomial math correctly computes how hard recovery is, especially at high percentages where the `p/(1-p)` fragility amplifies the difficulty.

## Bankroll Simulation: Band Filter

Tested score margin as a band filter [floor, ceiling] on compounding returns:

| min_edge | Band | Movies | Multiplier |
|---|---|---|---|
| 20c | -3 to +3 | 70 | **249.0x** |
| 5c | -5 to +2 | 103 | 207.9x |
| 15c | -3 to +3 | 79 | 192.7x |
| 10c | -3 to +3 | 92 | 188.4x |
| 20c | no filter | 76 | 147.1x |
| 15c | no filter | 88 | 141.2x |
| 10c | no filter | 99 | 94.3x |
| 20c | <= 0 | 66 | 45.7x |

*(Single-path, 10% risk/movie, $1K start, No-only, T-5d to T-1d)*

Key findings:
- A tight band around zero beats no filter at every min_edge level.
- Cutting only the ceiling (<= 0) **hurts** — those above-threshold trades are still +EV.
- Cutting the floor helps — "easy wins" far below threshold have small payoffs that dilute compounding.
- Fine-grained sweep found [-1,+1] at 672x, but with only 59 movies and 64% WR this is a thin, volatile pool.

## min_edge Peak

Tested 5c through 40c. Compounding peaks at 20c and degrades monotonically above:

| min_edge | No filter | [-3,+3] | Movies (no filter) |
|---|---|---|---|
| 15c | 141.2x | 192.7x | 88 |
| **20c** | **147.1x** | **249.0x** | **76** |
| 25c | 84.3x | 149.7x | 61 |
| 30c | 68.9x | 115.3x | 51 |

Above 20c, the movie pool thins too fast for compounding to overcome lost opportunities.

## Robustness Analysis

### Order doesn't matter

With fixed fractional betting (10% of bankroll per movie), the final bankroll is:

`start * product(1 + frac * roi_i for each movie i)`

Multiplication is commutative. Shuffling without replacement produces the exact same multiplier every time. The "ACTUAL" column below is the one deterministic outcome.

### With-replacement bootstrap (composition risk)

Resampling with replacement tests: "what if the future movie pool had a different mix of winners and losers?" This allows the same movie to appear multiple times, which can't happen in reality — but it stress-tests the strategy against pessimistic (and optimistic) compositions. 10,000 simulations per config.

| config | mov | WR | ACTUAL | repl_p1 | repl_p5 | repl_p25 | repl_med | repl_p75 | repl_p95 | repl_std |
|---|---|---|---|---|---|---|---|---|---|---|
| 10c, no filter | 99 | 81% | 94.3x | 13.3x | 22.7x | 50.6x | 92.1x | 171.8x | 458.8x | 247x |
| 15c, no filter | 88 | 78% | 141.2x | 16.3x | 29.7x | 69.5x | 131.8x | 261.9x | 793.4x | 403x |
| 20c, no filter | 76 | 78% | 147.1x | 14.0x | 26.4x | 68.2x | 138.5x | 292.6x | 919.7x | 467x |
| 10c, [-3,+3] | 92 | 75% | 188.4x | 14.0x | 28.0x | 82.8x | 182.4x | 405.6x | 1405.7x | 935x |
| 15c, [-3,+3] | 79 | 73% | 192.7x | 14.6x | 29.4x | 86.0x | 186.9x | 417.4x | 1453.9x | 844x |
| 20c, [-3,+3] | 70 | 73% | 249.0x | 13.7x | 31.3x | 107.2x | 243.9x | 583.6x | 2094.0x | 1605x |
| 15c, [-1,+1] | 67 | 64% | 322.2x | 8.1x | 21.7x | 99.0x | 295.2x | 967.1x | 5788.9x | 8901x |
| 20c, [-1,+1] | 59 | 64% | 672.1x | 11.9x | 34.1x | 177.2x | 633.1x | 2372.7x | 16640.1x | 516116x |

ACTUAL = the one deterministic outcome for each config. With fractional betting, the final bankroll is `start * product(1 + frac * roi_i)` — multiplication is commutative, so movie order doesn't change the result. Shuffling without replacement (10K permutations) confirmed this: zero variance, every sim identical to ACTUAL.

repl_* = with-replacement bootstrap, 10K sims. Each sim draws N movies from the pool of N **with duplication allowed** — the same movie can appear multiple times and others can be missing entirely. This changes the *composition* of the pool (not just the order), which is why repl results differ from ACTUAL. It tests: "what if the future had more movies like the losers and fewer like the winners (or vice versa)?" The duplication is artificial (you can't bet the same position twice), so the variance is overstated, but the p1/p5 columns give a rough lower bound on performance under pessimistic future movie mixes.

### Interpretation

- **Zero ruin across all 80K simulations.** Even the worst 1st-percentile draws (duplicating losers) stayed profitable. Worst outcome: 8.1x (15c/[-1,+1]).
- **Losses are capped, wins are not.** At 10% risk/movie, a losing movie costs ~10% of bankroll. A winning movie with 120% ROI adds 12%. This asymmetry makes ruin nearly impossible at these win rates.
- **Band filters increase both median and variance.** [-3,+3] raises ACTUAL from 147x to 249x at 20c, but replacement std goes from 467x to 1605x. The filter concentrates on higher-ROI movies, amplifying both upside and downside composition draws.
- **[-1,+1] is a lottery ticket.** 672x ACTUAL but replacement std of 516Kx from only 59 movies at 64% WR.
- **Downside floors are similar across configs.** repl_p5 ranges 22-34x. The band filter doesn't meaningfully worsen the pessimistic case.
- **The with-replacement bootstrap overstates variance** because it allows duplicate positions on the same movie. The actual risk is somewhere between the deterministic ACTUAL (no variance) and the replacement bootstrap (inflated variance). The truth is: the ACTUAL number is what you get from these 136 movies; the replacement spread indicates sensitivity to the movie mix changing in the future.

## Conclusions

1. **Score margin, not threshold level, determines trade quality.** The contested zone (score near threshold) is where the market misprices most.
2. **A [-3,+3] band filter improves compounding** (249x vs 147x at 20c) by cutting low-value trades at both extremes. But the improvement comes with higher variance under composition risk.
3. **20c min_edge is the peak.** Above it, the movie pool thins too fast.
4. **All configs are robustly profitable.** Even pessimistic with-replacement draws (p1) stay above 8x across all configs.
5. **The conservative choice is 15-20c, no filter** (141-147x, tight distribution, 78% WR). The aggressive choice is **20c, [-3,+3]** (249x, wider distribution, 73% WR). Both are defensible.
