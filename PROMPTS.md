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
