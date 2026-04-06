# CLAUDE.md

## Project Overview

Finding profitable edges in Kalshi prediction markets for Rotten Tomatoes Tomatometer scores. Combines a scraped review database with Kalshi market price histories to identify where and when the market misprices movie scores.

This project connects to the same Neon PostgreSQL database used by the RT scraper (separate repo at `~/Desktop/rotten-tomatoes-analysis/`), but is fully independent -- no shared code, no shared config.

## Build Protocol

Follow `PROTOCOL.md` for all non-trivial work. Do not write code before writing and presenting a plan doc. See `PROTOCOL.md` for what qualifies as non-trivial.

## Development Philosophy

The goal is `data -> max profit`. Premature formalization (investing in specific models before understanding where the edge actually is) is the main risk. The workflow:

1. **Build dataset** — scrape, dump, enrich. The richer and more unique the dataset, the better the eventual strategies.
2. **Observe informally** — stare at charts, brainstorm, look for patterns. This is a creative exercise. Brainstorm notes go in `brainstorm/` (gitignored).
3. **Cheap quantitative gut-check** — not modeling, just counting. "How many of 141 movies show this?" / "What's the median edge in cents?" / "Does this scatter plot have any structure?" Kills bad ideas in 20 minutes.
4. **Iterate 1-3** until concepts survive the gut-checks.
5. **Formalize** the survivors into modular functions. By this point the ideas are crystal clear, so the math is just implementation.

**Current phase:** Step 5 — formalizing the Poisson-binomial betting function. The dataset is built, observations are done, the surviving approach has been identified.

## File Structure

```
├── pyproject.toml              # Dependencies (uv managed)
├── .gitignore
├── CLAUDE.md                   # This file
├── PROTOCOL.md                 # Build protocol (plan → implement → validate)
├── BACKLOG.md                  # Priorities, ideas, infrastructure, platform mechanics
├── SOURCES.md                  # Literature, data, and hard numbers to gather
├── PROMPTS.md                  # Handoff prompts for new conversations
├── .env                        # DATABASE_URL (gitignored)
├── reviews.csv                 # Local dump of reviews table (gitignored)
├── movies_index.csv            # Per-movie metadata: volume, score range, dates (gitignored)
├── notebooks/
│   ├── dataset_survey.ipynb               # Broad survey of the joint dataset (complete)
│   ├── misprice_backtest.ipynb            # Systematic misprice backtest — bounds approach (complete)
│   ├── misprice_backtest_deep_dive.ipynb  # Deep dive on clean-data movies (complete)
│   └── poisson_binomial_threshold.ipynb   # Original Poisson-binomial model exploration
├── rt-price-histories/         # Kalshi market price CSVs (~141 movies, minute/hour/day)
├── plans/                      # Implementation plans (gitignored)
└── brainstorm/                 # Strategy brainstorms (gitignored)
```

## Tech Stack

- **Language**: Python >= 3.13
- **Package manager**: uv
- **Database**: Neon (serverless PostgreSQL) via SQLAlchemy + psycopg2-binary
- **Analysis**: pandas, numpy, scipy, matplotlib
- **Notebooks**: Jupyter via ipykernel

## Database Connection

The project connects to the same Neon PostgreSQL instance used by the scraper. Connection is via `DATABASE_URL` environment variable.

```bash
# Launch notebook with database access
DATABASE_URL="postgresql://..." uv run jupyter notebook
```

- Neon cold-starts take ~1-3s on first connection
- `sslmode=require` is needed for Neon connections
- Connection strings starting with `postgres://` must be rewritten to `postgresql://` for SQLAlchemy

## Database Schema

### `reviews` table (Neon PostgreSQL)

This table is populated by the RT scraper (separate project). It is **insert-only** -- reviews are never deleted or updated. The scraper runs every 50 minutes via Cloud Run Jobs.

| Field | Type | Description |
|---|---|---|
| id | SERIAL | Auto-increment primary key |
| unique_review_id | TEXT (UNIQUE) | MD5 hash of (movie_slug + name + publication + rating). Dedup key. |
| movie_slug | TEXT | Movie identifier (e.g., "project_hail_mary") |
| reviewer_name | TEXT | Critic's name |
| publication_name | TEXT | Publication (e.g., "The Guardian") |
| top_critic | BOOLEAN | True if scraped from top-critics filter |
| tomatometer_sentiment | TEXT | **"positive" or "negative"** -- this is the fresh/rotten signal |
| subjective_score | TEXT | Critic's rating in their own scale (e.g., "3/5", "A-", "8/10") |
| written_review | TEXT | Review snippet text |
| site_timestamp_text | TEXT | Raw RT relative timestamp as scraped (e.g., "5m", "3h", "2d", "Mar 20") |
| scrape_time | TIMESTAMPTZ | UTC datetime when the scrape ran |
| estimated_timestamp | TIMESTAMPTZ | Computed absolute time: scrape_time minus the offset in site_timestamp_text |
| timestamp_confidence | TEXT | Timestamp granularity: "m" (minute-level), "h" (hour-level), "d" (day-level or date-format) |
| page_position | INTEGER | 0-indexed position in scrape result (0 = newest review on page) |

**Indexes**: `idx_reviews_movie_slug` on `(movie_slug)`, `idx_reviews_movie_timestamp` on `(movie_slug, estimated_timestamp)`

### Key data characteristics

- **Tomatometer score** = count of "positive" sentiment / total reviews for a movie
- **Deduplication**: Each review appears exactly once (enforced by UNIQUE on `unique_review_id`)
- **Two-pass scraping**: The scraper runs `top-critics` then `all-critics` filters. A review scraped as a top critic has `top_critic = True`. The same review is NOT duplicated when scraped again in the all-critics pass (dedup catches it).
- **Timestamp precision varies**: "m"-confidence timestamps are accurate to ~1 minute, "h" to ~1 hour, "d" to ~1 day. KDE-based analysis should be aware of this noise.
- **Scrape frequency**: Every 50 minutes. Reviews with relative timestamps ("5m", "3h") get more precise `estimated_timestamp` values than date-format ones ("Mar 20").
- **Movies tracked**: Configured in the scraper's `movies.json`. All ~141 movies that had Kalshi RT markets have been backfilled. Originally live-tracked (with minute-level timestamps): `project_hail_mary`, `ready_or_not_2_here_i_come`, `forbidden_fruits_2026`, `they_will_kill_you`.

## Approach: Poisson-Binomial Betting Function

The surviving approach after exploration: a **Poisson-binomial model** that computes the probability of a movie's final score crossing a Kalshi threshold, given accumulated reviews and expected future review arrivals.

**Inputs (7):** threshold, market price, current fresh count, current total count, time until close, lambda (review arrival rate), p_fresh (success rate).

**Output:** Edge in cents = expected profit per contract. Positive = bet, negative = pass.

**What was tried and why alternatives were abandoned:**
- *Deterministic bounds (worst/best case from remaining reviews)*: Too conservative — by the time bounds lock the outcome, the market has already corrected. See `notebooks/misprice_backtest.ipynb` and `notebooks/misprice_backtest_deep_dive.ipynb`.
- *Market-price-only strategies (spike detection, anomaly reversion)*: Interesting ideas in `brainstorm/brainstorm_market_strategies.md` but require the structural model first to define "fair value."

**Where the alpha lives:** Lambda and p_fresh estimation. Everything else is observable. These start simple (lambda = recent review rate, p_fresh = running average) and get refined with cross-movie data, top-critic correction, time-varying rates, etc. See `brainstorm/` for refinement ideas.

## Backlog & Strategy

`BACKLOG.md` has the full priority list. `PROMPTS.md` has handoff prompts for new conversations. `SOURCES.md` lists data and literature to gather.

**Current state:** Dataset is built (reviews.csv, movies_index.csv, ~141 price histories). Three exploration notebooks complete (dataset survey, misprice backtest, deep dive). The Poisson-binomial approach has survived gut-checks. Now formalizing.

**What we learned from exploration:**
- 23/141 movies had clear misprices (median 57¢ edge) — the opportunity is real and not volume-dependent.
- Deterministic bounds approach was too conservative — max 7¢ edge on clean-data movies. The market corrects before bounds can prove it wrong.
- 20/141 movies have review data that doesn't match ground truth (day-level timestamp noise near close). Only movies with minute-level late-window timestamps are reliable for boundary analysis.
- Conclusion: a *probabilistic* approach is needed to capture edges before the market self-corrects.

**Recommended next steps (in order):**
1. **Build the betting function.** Poisson-binomial model: 7 inputs → edge in cents. Fresh/total counts come from the existing scraper's DB. Lambda and p_fresh start simple.
2. **High-frequency score polling.** Detect review arrivals between scraper runs (every 1-5 min). Needed for real-time lambda estimation and future backtesting.
3. **Kalshi API client.** Fetch live prices for automated comparison. Foundation for the eventual execution pipeline.

## How to Run

```bash
# Install dependencies
cd ~/Desktop/rt-analysis
uv sync

# Launch notebook
DATABASE_URL="postgresql://..." uv run jupyter notebook
```

## Dependencies

`pyproject.toml`: sqlalchemy, psycopg2-binary, pandas, numpy, scipy, matplotlib, ipykernel.
