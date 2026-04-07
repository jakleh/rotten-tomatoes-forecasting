# Prompts

Handoff prompts for starting new conversations. Read CLAUDE.md and BACKLOG.md for full context before using any prompt.

---

## Context for All Prompts

The Poisson-binomial betting function is built (`edge.py`). It computes edge in cents for "Above X" Kalshi RT bets given 7 inputs (threshold, market_price, fresh_count, total_count, hours_to_close, lambda_rate, p_fresh). The CLI accepts `--lambda` and `--p-fresh` overrides; without them, it falls back to naive defaults. `get_movie_state()` returns raw counts (fresh/total split by top/non-top critic + recent timestamps) — parameter estimation is decoupled.

Key data notes:
- The scraper runs every 50 minutes; `edge.py` queries the Neon PostgreSQL DB for live review counts.
- 20/141 movies have review data that doesn't match ground truth (day-level timestamp noise near close).
- Top critics are systematically ~6pp more negative — early review baskets overweight them.
- Score ranges in `movies_index.csv` are fractions (e.g., 0.8750 = 87.5%). Price CSVs use tz-aware UTC timestamps and cents.

---

## Active Priorities

### Prompt 1: High-frequency score polling (Backlog §1.1)

```
Read CLAUDE.md and BACKLOG.md §1.1. Build a lightweight poller that scrapes the displayed Tomatometer score (just the integer) for each active movie every 1-5 minutes. Log timestamp + score. This detects review arrivals between the scraper's 50-minute runs and enables real-time lambda estimation for edge.py.

This does NOT need the full review page parse — just the displayed score number. Keep it simple: one script, one output format.
```

### Prompt 2: Kalshi API client (Backlog §1.2)

```
Read CLAUDE.md and BACKLOG.md §1.2. Build a minimal Kalshi API client that fetches live prices for all RT Tomatometer markets. No order placement — just price retrieval. Output: for each movie, current price for each threshold. This is the foundation for comparing edge.py output to market prices.
```

**Prereqs:** Kalshi API credentials and API docs.

### Prompt 3: Per-critic KDE lambda model (Backlog §3, brainstorm/brainstorm_critic_kde_lambda.md)

```
Read CLAUDE.md, BACKLOG.md §3, and brainstorm/brainstorm_critic_kde_lambda.md (the full writeup including the original conversation transcript).

We're replacing the aggregate lambda (review arrival rate) with a sum of per-critic KDEs. Every critic gets a 1D Gaussian KDE fitted to their historical review timing (days before bet close). At time t, lambda = sum of KDE integrals from 0 to t for all critics who haven't reviewed yet, weighted by their base rate (movies_reviewed / movies_available). When a review is observed, that critic's KDE is dropped from the sum.

Current state (notebooks/critics_index.ipynb):
- Critic frequency analysis complete: top critic reviewed 18/20 recent movies in the 96h-24h window. ~138 critics account for 50% of reviews, ~300 for 75%.
- Cumulative review share curve plotted (Pareto/CDF).
- Box+strip plots for top 30 critics' review timing (days before close).
- Per-critic KDE grid (top 16 critics, 4x4) — shapes look reasonable at daily resolution.
- Aggregate lambda curve prototype working — monotonically decreasing, steepest drop at 2-3 days before close, inflection point from concave-down to concave-up near close.

Known issues to address:
- Day-level timestamp resolution (98% of data): causes degenerate KDEs (zero variance) for some critics and artificially tight bandwidths for others. Currently skipping zero-variance critics. Mitigations to explore: jitter, bandwidth floor, population prior blending.
- Only plotting top 16 KDEs — should expand to include sparser critics to see how bandwidth naturally widens with less data (the core thesis of the approach).
- 1-review critics can't get KDEs at all — need the population prior / shrinkage approach from the brainstorm doc.
- Lambda curve doesn't yet account for 1-review and zero-variance critics (they're excluded). Need to add their contribution via a fallback model.

Next steps:
1. Expand KDE grid to show sparse critics alongside prolific ones (bandwidth contrast).
2. Quantify how many critics are affected by the zero-variance / degenerate KDE issue.
3. Implement population-prior blending for sparse critics.
4. Validate: for a resolved movie, simulate the model at T-7d, T-3d, T-1d and compare predicted remaining reviews to actual.
5. Eventually integrate with compute_edge() — replace scalar lambda + p_fresh with per-critic vectors feeding a Poisson binomial.

Also see brainstorm/brainstorm_finite_pool_model.md and brainstorm/brainstorm_reviewer_graph.md for related ideas this model unifies.
```

### Prompt 4: Parameter refinement — general (Backlog §3)

```
Read CLAUDE.md and BACKLOG.md §3. The betting function (edge.py) takes lambda_rate and p_fresh as inputs. The CLI accepts --lambda and --p-fresh overrides. The current defaults are naive placeholders (recent rate, running average). notebooks/parameter_exploration.ipynb has data loading, cross-movie arrival curves, snapshot helpers, and edge trajectory tools already set up. Open the notebook and continue developing better estimators. See brainstorm/ for approach ideas.
```

---

## Future Explorations (Deferred)

These prompts investigate ideas from the brainstorm phase. They may feed into parameter refinement or model extensions later.

### Overdispersion check (Backlog §3)

```
Read BACKLOG.md §3. Measure actual variance of fresh counts in rolling windows vs. binomial prediction across resolved movies. Is beta-binomial worth exploring, or is the i.i.d. binomial close enough?
```

### Volume-review count correlation (Backlog §7, market strategies)

```
Read BACKLOG.md §7 (market-based strategies). Using the price history CSVs, compute minute-row count as an activity proxy. Correlate with total review count from reviews.csv. Is there structure? Can we predict total review count from early trading activity?
```

### Embargo-lift divergence (Backlog §7, market strategies)

```
Read BACKLOG.md §7 and brainstorm/brainstorm_market_strategies.md (strategy 1). For resolved movies, compare the initial review basket (first 20 reviews) to the Kalshi market price. How often do they disagree, and who's right?
```

### Backtesting the betting function (Backlog §7, operational)

```
Read CLAUDE.md and BACKLOG.md §7. Replay edge.py's compute_edge() against resolved markets using historical review data and price histories. For each movie at each hourly snapshot: compute the edge, record whether the bet would have been profitable. Measure calibration (predicted P(Yes) vs. actual outcomes), Brier scores, and retroactive P&L.
```
