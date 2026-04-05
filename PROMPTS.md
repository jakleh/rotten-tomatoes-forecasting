# Prompts

Handoff prompts for starting new conversations. Organized by phase: data infrastructure first, then explorations, then operational. Each prompt references the relevant backlog items and brainstorm files.

---

## Data Infrastructure

### Prompt 0: Repo setup — nbstripout + git hygiene (Backlog §4.3)

```
Read CLAUDE.md and BACKLOG.md §4.3. Before we start committing, set up nbstripout as a git filter so notebook outputs (which may contain DB credentials in tracebacks) are automatically stripped on commit. Do NOT commit notebook outputs.
```

**Prereqs:** None. Do this before anything else.

### Prompt 1: Local DB dump for joint analysis (Backlog §1.2)

```
Read CLAUDE.md and BACKLOG.md §1.2. The RT scraper has finished backfilling ~141 movies into the Neon database. I want to dump the reviews table locally so we can join review data with the Kalshi price history CSVs in rt-price-histories/ (folder names = movie slugs = join key). Figure out the best format — CSV, parquet, or DuckDB — and set it up so we can run ad-hoc queries across both datasets without hitting Neon.
```

**Prereqs:** Backlog §1.1 (historical backfill) must be complete.

### Prompt 2: High-frequency score polling (Backlog §1.3)

```
Read CLAUDE.md and BACKLOG.md §1.3. I want to detect review arrivals between scraper runs by polling the displayed Tomatometer score at high frequency (every 1-5 minutes). Design a lightweight poller that scrapes just the score number for each active movie and logs timestamp + score. This doesn't need the full review page parse — just the displayed integer.
```

### Prompt 3: Kalshi API client (Backlog §5.1)

```
Read CLAUDE.md and BACKLOG.md §5.1. Build a minimal Kalshi API client that fetches live prices for all RT markets. No order placement — just price retrieval. This is the foundation for comparing any model/strategy output to market prices and for the eventual execution pipeline.
```

**Prereqs:** Need Kalshi API credentials and familiarity with their API docs (SOURCES.md §1.4, §1.5).

---

## Explorations — Gut-Checks on the Joint Dataset

These prompts are for the brainstorm-and-count phase. Each one investigates a specific question from the data. Results go in notebooks (one per exploration).

### Prompt 4: Systematic backtest of forecast-score divergence (Backlog §2.8)

```
Read CLAUDE.md, BACKLOG.md §2.8, and brainstorm/brainstorm_market_strategies.md (especially "Recommended next step: systematic backtest"). For each of the ~141 resolved movies, I want to know: at the point where the actual score was "knowable" from accumulated reviews, what was the Kalshi market pricing? Compute the edge in cents and how long it persisted. Segment by market activity level (minute-row count as proxy for volume). This tells us how big the opportunity is and where it concentrates.
```

**Prereqs:** Local DB dump (Prompt 1) must be done so reviews and price histories can be joined.

### Prompt 5: Volume-review count correlation (Backlog §2.10)

```
Read CLAUDE.md and BACKLOG.md §2.10. Using the price history CSVs in rt-price-histories/, compute the minute-row count for each movie as a proxy for trading activity. Once the review DB is available locally, correlate activity with total review count. Plot a scatter. Is there structure? If so, can we predict total review count from early trading activity — which would give us a review ceiling for free?
```

**Prereqs:** Can start the activity-proxy side immediately. Full correlation needs Prompt 1.

### Prompt 6: Embargo-lift divergence (Backlog §2.9)

```
Read CLAUDE.md, BACKLOG.md §2.9, and brainstorm/brainstorm_market_strategies.md (strategy 1). For resolved movies, compare the initial review basket (first 20 reviews) to the Kalshi market price at that time. How often do they disagree? When they disagree, who's right — the basket or the market? This tells us whether early review data has informational edge over the market.
```

**Prereqs:** Local DB dump (Prompt 1).

### Prompt 7: Price trace anomaly detection (Backlog §2.11)

```
Read CLAUDE.md, BACKLOG.md §2.11, and brainstorm/brainstorm_market_strategies.md (price trace anomaly section). For resolved markets, identify price spikes that coincide with no new reviews in the DB. Measure mean reversion: how far and how fast do these spikes revert? This tells us whether "amateur spike" reversion is a real, exploitable pattern or just noise.
```

**Prereqs:** Local DB dump (Prompt 1) for aligning price timestamps with review arrivals.

### Prompt 8: Overdispersion check (Backlog §2.1)

```
Read CLAUDE.md and BACKLOG.md §2.1. Measure the actual variance of fresh counts in rolling windows across movies and compare to binomial prediction. Is sentiment overdispersion real and material, or is the binomial close enough? This determines whether beta-binomial is worth exploring. See SOURCES.md §3.1 and §4.
```

**Prereqs:** Local DB dump (Prompt 1).

### Prompt 9: Diagnostic regression for hierarchical p_fresh (Backlog §2.6)

```
Read CLAUDE.md, BACKLOG.md §2.6, and brainstorm/brainstorm_hierarchical_p_fresh.md (Step 1: diagnostic regression). For each historical movie, compute p_pre (freshness rate in the pre-forecast window) and p_forecast (freshness rate in the forecast window). Fit a simple regression: p_forecast ~ f(p_pre, n_reviews). Plot the scatter. Does pre-forecast freshness predict forecast-window freshness? If it's a noisy mess, the hierarchical model isn't worth building.
```

**Prereqs:** Local DB dump (Prompt 1). Need 30-50+ movies with full review timelines. Also need to define forecast window boundaries.

---

## Explorations — Deeper Dives (Conditional on Gut-Check Results)

### Prompt 10: Hierarchical Bayesian p_fresh model (Backlog §2.6, conditional)

```
Read CLAUDE.md, BACKLOG.md §2.6, and brainstorm/brainstorm_hierarchical_p_fresh.md (Steps 2-3). The diagnostic regression (Prompt 9) showed signal — p_pre does predict p_forecast. Now build the hierarchical model: use the cross-movie regression as an informed Beta prior, update with the current movie's recent reviews via exponential forgetting. See the brainstorm doc for full design.
```

**Prereqs:** Prompt 9 must show a meaningful relationship.

### Prompt 11: Review-integrity uncertainty (Backlog §2.7)

```
Read CLAUDE.md, BACKLOG.md §2.7, and brainstorm/brainstorm_review_integrity.md. Compare sentiment for the same reviews across multiple scrapes of tracked movies to estimate the empirical flip rate. Then run sensitivity analysis: perturb k reviews and measure how much model output changes. Almost certainly immaterial, but good to confirm.
```

**Prereqs:** Need multi-scrape data for tracked movies (already have this).
