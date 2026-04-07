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

### Prompt 3: Parameter refinement (Backlog §3)

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
