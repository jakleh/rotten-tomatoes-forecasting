# Prompts

Handoff prompts for starting new conversations. Read CLAUDE.md and BACKLOG.md for full context before using any prompt.

---

## What We Learned (Context for All Prompts)

The dataset survey (23/141 movies mispriced, median 57¢ edge) confirmed the opportunity is real. The systematic misprice backtest showed that deterministic bounds (worst/best case from remaining reviews) are too conservative — by the time they lock the outcome, the market has already self-corrected (max 7¢ edge on clean-data movies). A probabilistic approach (Poisson-binomial) is needed to capture edges before the market corrects.

Key data notes:
- 20/141 movies have review data that doesn't match ground truth (day-level timestamp noise near the 10 AM ET close cutoff). Only movies with minute-level timestamps are reliable for boundary-adjacent analysis.
- Top critics are systematically ~6pp more negative — early review baskets overweight them.
- The scraper runs every 50 minutes; fresh/total counts come from the Neon PostgreSQL DB.
- Score ranges in `movies_index.csv` are fractions (e.g., 0.8750 = 87.5%). Price CSVs use tz-aware UTC timestamps and cents.

---

## Active Priorities

### Prompt 1: Build the Poisson-binomial betting function (Backlog §1.1)

```
Read CLAUDE.md and BACKLOG.md §1.1. Build a function that computes the expected edge (in cents) for a Kalshi RT bet.

Inputs (7): threshold, market_price, fresh_count, total_count, hours_to_close, lambda_rate, p_fresh.

Method: Poisson-binomial. For each possible number of additional reviews k (Poisson with rate lambda * hours_to_close), and for each possible fresh count among those k (binomial with p_fresh), compute final score and check threshold crossing. Sum to get P(resolve Yes) and P(resolve No). Edge = P(Yes) * (100 - price) - P(No) * price.

Start simple: lambda = recent review rate, p_fresh = current running average. The function should be a standalone module (not a notebook) that can be called from anywhere. Include a DB query helper to fetch current fresh/total counts per movie from the Neon PostgreSQL database (same one the RT scraper writes to — see CLAUDE.md for schema).
```

**Prereqs:** DATABASE_URL environment variable for review counts.

### Prompt 2: High-frequency score polling (Backlog §1.2)

```
Read CLAUDE.md and BACKLOG.md §1.2. Build a lightweight poller that scrapes the displayed Tomatometer score (just the integer) for each active movie every 1-5 minutes. Log timestamp + score. This detects review arrivals between the scraper's 50-minute runs and enables real-time lambda estimation.

This does NOT need the full review page parse — just the displayed score number. Keep it simple: one script, one output format.
```

### Prompt 3: Kalshi API client (Backlog §1.3)

```
Read CLAUDE.md and BACKLOG.md §1.3. Build a minimal Kalshi API client that fetches live prices for all RT Tomatometer markets. No order placement — just price retrieval. Output: for each movie, current price for each threshold. This is the foundation for comparing model output to market prices.
```

**Prereqs:** Kalshi API credentials and API docs.

---

## Future Explorations (Deferred)

These prompts investigate ideas from the brainstorm phase. They're not blocking the betting function but may feed into parameter refinement later.

### Volume-review count correlation (Backlog §5, market strategies)

```
Read BACKLOG.md §5 (market-based strategies). Using the price history CSVs, compute minute-row count as an activity proxy. Correlate with total review count from reviews.csv. Is there structure? Can we predict total review count from early trading activity?
```

### Embargo-lift divergence (Backlog §5, market strategies)

```
Read BACKLOG.md §5 and brainstorm/brainstorm_market_strategies.md (strategy 1). For resolved movies, compare the initial review basket (first 20 reviews) to the Kalshi market price. How often do they disagree, and who's right?
```

### Overdispersion check (Backlog §5, parameter refinement)

```
Read BACKLOG.md §5 (parameter refinement). Measure actual variance of fresh counts in rolling windows vs. binomial prediction. Is beta-binomial worth exploring, or is the binomial close enough?
```

### Hierarchical p_fresh diagnostic (Backlog §5, parameter refinement)

```
Read BACKLOG.md §5 and brainstorm/brainstorm_hierarchical_p_fresh.md. Fit p_forecast ~ f(p_pre, n_reviews) across historical movies. Does pre-forecast freshness predict forecast-window freshness?
```
