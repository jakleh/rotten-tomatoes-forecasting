# Prompts

Handoff prompts for starting new conversations. Each maps to one or more backlog items. Use in order unless you have a reason to skip ahead.

---

## Next up

### Prompt 0: Repo setup — nbstripout + git init (5.1)

```
Read CLAUDE.md and BACKLOG.md §5.1. Before we start committing, set up nbstripout as a git filter so notebook outputs (which may contain DB credentials in tracebacks) are automatically stripped on commit. Then git init, and make the initial commit with everything we have so far. Do NOT commit notebook outputs.
```

**Prereqs:** None. Do this before anything else.

### Prompt 1: Backtesting notebook (2.8)

```
Read CLAUDE.md and BACKLOG.md. I want to start working through the backlog in priority order. The recommended next step is 2.8 (backtesting & calibration via resolved markets). I have Kalshi price history CSVs ready to download. Let's figure out the data format, plan how to store it, and build a backtesting notebook that replays the model against resolved markets and measures calibration (predicted probability vs. actual outcome). Start by reading the backlog entry for 2.8 and the current notebook to understand what we're working with.
```

**Prereqs:** Download at least one Kalshi price history CSV first so we can inspect the format. Also check SOURCES.md §1.2 (resolution rules) and §1.1 (fee schedule) — we'll need those for accurate backtesting.

---

## Upcoming

### Prompt 2: Full score PMF + range betting (2.6)

```
Read CLAUDE.md and BACKLOG.md §2.6. Refactor the notebook's core calculation to output the full PMF over possible final scores instead of a single P(break). Then add a cell that evaluates every threshold in a list and computes the range bet P&L for adjacent threshold pairs. See the backlog for the math.
```

### Prompt 3: Kalshi API integration (1.5 + 2.7 MVP)

```
Read CLAUDE.md and BACKLOG.md §1.5 and §2.7. Build a minimal Kalshi API client that fetches live prices for all RT markets, then add a cell to the notebook that compares model output to market prices and prints the edge for each market. This is steps 1-3 of the 2.7 MVP — no automated order placement yet.
```

**Prereqs:** Need Kalshi API credentials and familiarity with their API docs (SOURCES.md §1.4, §1.5).

### Prompt 4: Model improvements — beta-binomial + Bayesian p_fresh (1.1 + 1.4)

```
Read CLAUDE.md and BACKLOG.md §1.1 and §1.4. Upgrade the notebook: (1) replace the binomial with a beta-binomial to handle sentiment overdispersion, and (2) replace the 7-day-window p_fresh estimate with a Beta posterior with exponential time-weighting. See SOURCES.md §3.1 for references. Run the overdispersion test from SOURCES.md §4 first to confirm the beta-binomial is warranted.
```

### Prompt 5: Cross-movie empirical Lambda (1.8)

```
Read CLAUDE.md and BACKLOG.md §1.8. Build a notebook or script that computes Lambda empirically from a cohort of similar movies rather than KDE extrapolation. Start by analyzing the review-count-over-time data we have in the DB for our tracked movies — plot the rate curves overlaid to see if lifecycle stage is a universal predictor. Then design the cohort similarity metric. See SOURCES.md §2.5 for data requirements.
```

**Prereqs:** Need historical review data across multiple movies. May need to extend the scraper for a one-time backfill.

### Prompt 6: Database backfill (2.10)

```
Read CLAUDE.md and BACKLOG.md §2.10. I want to backfill the Neon database with historical review data from a broad set of movies (50-100+, across release tiers, prioritizing movies that had Kalshi RT markets). This unblocks cross-movie analysis (1.8, 1.9, 2.3, 2.5). The existing RT scraper is at ~/Desktop/rotten-tomatoes-analysis/. Let's figure out the best approach — extend the scraper or build a standalone backfill script — and plan the implementation.
```

**Prereqs:** Decide which movies to backfill. Prioritize movies with resolved Kalshi RT markets (for backtesting and drift calibration). Get a list of those movies from Kalshi's settled markets.

### Prompt 7: Diagnostic regression for p_fresh stationarity (1.9, Step 1)

```
Read CLAUDE.md and BACKLOG.md §1.9 (especially the "Step 1 — Diagnostic regression" section). Also read brainstorm_hierarchical_p_fresh.md for full context on the hierarchical model design. The database backfill is done. Now I want to run the diagnostic: for each historical movie, compute p_pre (freshness rate in the pre-forecast window) and p_forecast (freshness rate in the forecast window), then fit a simple regression p_forecast ~ f(p_pre, n_reviews). Plot the scatter. The goal is to answer: does pre-forecast freshness actually predict forecast-window freshness? This determines whether the hierarchical model (Step 2) is worth building.
```

**Prereqs:** Database backfill (Prompt 6) must be complete. Need 30-50+ movies with full review timelines. Also need to define the forecast window boundaries (e.g., 48h before Kalshi market close).

### Prompt 8: Hierarchical Bayesian p_fresh model (1.9, Step 2)

```
Read CLAUDE.md and BACKLOG.md §1.9 (Step 2 section) and brainstorm_hierarchical_p_fresh.md. The diagnostic regression (Step 1) showed signal — p_pre does predict p_forecast. Now build the hierarchical model: use the cross-movie regression as an informed Beta prior, then update with the current movie's recent reviews via exponential forgetting. The cross-movie data should be the base prediction, not a floor — see the brainstorm doc for why. Integrate this into the existing notebook's pipeline.
```

**Prereqs:** Prompt 7 must show a meaningful relationship (not a noisy mess). If it doesn't, skip this and revisit the approach.

### Prompt 9: Review-integrity uncertainty analysis (1.10)

```
I'm building models that use backfilled Rotten Tomatoes review data to backtest
betting strategies on Kalshi movie markets. I need help accounting for a known
source of uncertainty in my historical review data.

Context:
- I scrape RT reviews and store them in a time-series database (reviewer, publication,
  sentiment, subjective score, estimated timestamp)
- For backtesting, I use backfilled historical reviews — scraped after the fact, not
  in real time
- RT critics can edit their reviews after publication: they can change their subjective
  score (e.g., 3/5 → 4/5) or, more rarely, flip their tomatometer sentiment
  (Fresh ↔ Rotten)
- My backfilled data captures the review state at scrape time, not necessarily the
  state at the time the review was originally published
- This means my historical "score at time T" reconstruction may be slightly off

What I care about:
- Tomatometer percentage (% Fresh) is the primary signal — derived from sentiment,
  not subjective scores
- Sentiment flips are rare but would change the percentage
- Subjective score changes are more common but don't affect the tomatometer
- I also have Kalshi price/score forecasting data by day/hour/minute as a
  complementary signal

What I need:
- Statistical methods to quantify and account for this review-integrity uncertainty
  in backtesting (e.g., confidence intervals, sensitivity analysis, noise modeling)
- How to propagate this uncertainty through models that use historical tomatometer
  percentage as an input feature
- Whether this uncertainty is likely to be material given that sentiment flips are
  rare and individual review changes move the percentage minimally on large review counts

Read CLAUDE.md and BACKLOG.md §1.10 for full context on the problem, materiality
analysis, and proposed approaches. Start by estimating the empirical flip rate from
our multi-scrape data on tracked movies, then run sensitivity analysis on the
backtesting pipeline.
```

**Prereqs:** Database backfill (Prompt 6) and backtesting setup (Prompt 1) should be done first. The empirical flip rate estimation (approach 4 in §1.10) requires multiple scrapes of the same movie over time — we have this for currently-tracked movies.
