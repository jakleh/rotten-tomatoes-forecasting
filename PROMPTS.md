# Prompts

Handoff prompts for starting new conversations. Read CLAUDE.md and BACKLOG.md for full context before using any prompt.

---

## Context for All Prompts

The Poisson-binomial betting function (`edge.py`) and per-critic KDE model (`critic_model.py`) are built. `edge.py` computes edge in cents for "Above X" Kalshi RT bets given 7 inputs (threshold, market_price, fresh_count, total_count, hours_to_close, lambda_rate, p_fresh). The `--kde` flag uses per-critic KDE-based lambda and p_fresh estimates. Manual `--lambda` and `--p-fresh` overrides always take precedence.

`critic_model.py` has three layers: (1) CriticProfiles — per-critic base_rate, fresh_rate, timing_data; (2) KDELambdaModel — Gaussian KDEs per critic with population prior + shrinkage (k=3, bw floor 0.5d), lambda = sum of weighted KDE integrals for unreviewed critics, optional observed/expected scaling; (3) estimate_p_fresh — base_rate-weighted average of remaining critic fresh rates blended with observed rate.

KDE model backtest complete (`findings/kde_backtest.md`). No-only strategy in T-5d to T-1d window: 43-64% ROI per movie, 76-81% win rate, 81% of movies profitable. Buy Yes loses money — the model's conservatism is a feature on the No side. Bankroll simulation (`notebooks/bankroll_simulation.ipynb`): $1K → $141K at min_edge=15c, 10% risk/movie over 88 movies. Optimal compounding at min_edge=15c.

Key data notes:
- The scraper runs every 50 minutes; `edge.py` queries the Neon PostgreSQL DB for live review counts.
- 20/141 movies have review data that doesn't match ground truth (day-level timestamp noise near close).
- Top critics are systematically ~6pp more negative — early review baskets overweight them.
- Score ranges in `movies_index.csv` are fractions (e.g., 0.8750 = 87.5%). Price CSVs use tz-aware UTC timestamps and cents.
- 98% of review timestamps are day-level resolution. Use `format='ISO8601'` for reviews.csv, `utc=True` for price CSVs.

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

### Prompt 3: Per-critic KDE lambda model ~~— review plan and implement~~ (COMPLETE)

Implemented in `critic_model.py`. Validated in `notebooks/critic_model_validation.ipynb`. Findings in `findings/critic_kde_model_validation.md`. See BACKLOG §3.1.

### Prompt 5: KDE model backtest — ~~review plan and implement~~ (COMPLETE — daily snapshots)

```
Read these files in order before doing anything:
1. CLAUDE.md (project overview, conventions, file structure, how to run things)
2. PROTOCOL.md (plan → implement → validate workflow)
3. plans/plan_kde_backtest.md (the backtest plan — this is your spec)
4. findings/critic_kde_model_validation.md (model validation results — understand what the model gets right and wrong before backtesting its P&L)
5. edge.py (the betting function — compute_edge() is the core calculation you'll call at every snapshot)
6. critic_model.py (the KDE model — build_critic_profiles(), build_kde_lambda_model(), estimate_lambda(), estimate_p_fresh())
7. One hourly price CSV to understand the format: rt-price-histories/they_will_kill_you/kalshi-price-history-kxrt-wil-hour.csv

Your job: review the plan for correctness and completeness, flag anything that looks wrong or underspecified, then implement as a notebook (notebooks/kde_backtest.ipynb). Do NOT blindly follow the plan — verify the methodology makes sense before writing code.

How the model works (so you can verify the backtest is using it correctly):
- build_critic_profiles(reviews_df, movies_df, training_slugs) → CriticProfiles with per-critic base_rate, fresh_rate, timing_data (days before close)
- build_kde_lambda_model(profiles) → KDELambdaModel with population prior + per-critic KDEs (shrinkage k=3, bandwidth floor 0.5d)
- estimate_lambda(model, days_before_close, hours_to_close, observed_critics, observed_count, first_review_dbc) → reviews/hour. Sum of weighted KDE integrals for unreviewed critics, optionally scaled by observed/expected ratio.
- estimate_p_fresh(profiles, observed_critics, fresh_count, total_count) → float. Base_rate-weighted average of remaining critic fresh rates blended with observed rate.
- compute_edge(threshold, market_price, fresh_count, total_count, hours_to_close, lambda_rate, p_fresh) → dict with edge_cents, p_yes, p_no. Positive edge = buy Yes is +EV.

Key things the backtest must get right:
- NO LOOKAHEAD. Training set for movie X = 20 most recent movies with Bet Close Date < X's Bet Close Date. Use default_training_slugs(movies_df, exclude_slug=slug, before_date=bet_close_date) to get the correct training set.
- Resolution from terminal prices (not reviews.csv). Avoids the 20-movie data quality issue. Last price >= 90 → Yes, <= 10 → No.
- Forward-fill NaN prices in the hourly CSVs (last traded price persists).
- Review state at each snapshot = reviews with estimated_timestamp <= snapshot_time.
- 98% of timestamps are day-level, so review state changes ~once per day for most movies. Cache the review state and only recompute when a new review's timestamp is crossed.

Known model characteristics (from validation):
- Lambda scaling overcorrects at T-7d (guard rails fixed: min expected 40, clamp [0.5, 2.0]). At T-7d, scaling doesn't engage — falls back to unscaled.
- Systematic underprediction of remaining reviews at T-3d (MAE=19.5, median err=-12.4). Conservative for betting.
- p_fresh is excellent (T-1d MAE=0.031, correlation=0.990).
- The action window is T-3d to T-1d where the model is most accurate.

Performance: ~620K edge calculations (141 movies × 400h × 11 thresholds). May need 10-30 min. Consider starting with daily snapshots or a subset of movies to iterate quickly, then run the full hourly backtest.

After running, write findings to findings/kde_backtest.md and update BACKLOG.md §7 status.
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
