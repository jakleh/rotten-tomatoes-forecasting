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

**Current phase:** Step 5 complete for v1. The Poisson-binomial betting function is built (`edge.py`). Now refining parameters (lambda, p_fresh) and building operational infrastructure (score polling, Kalshi API).

## File Structure

```
├── edge.py                     # Poisson-binomial betting function (core module)
├── pyproject.toml              # Dependencies (uv managed)
├── CLAUDE.md                   # This file
├── PROTOCOL.md                 # Build protocol (plan → implement → validate)
├── BACKLOG.md                  # Priorities, ideas, infrastructure, platform mechanics
├── SOURCES.md                  # Literature, data, and hard numbers to gather
├── PROMPTS.md                  # Handoff prompts for new conversations
├── .env                        # DATABASE_URL (gitignored)
├── reviews.csv                 # Local dump of reviews table (gitignored)
├── movies_index.csv            # Per-movie metadata: volume, score range, dates (gitignored)
├── notebooks/                  # Exploration notebooks
│   ├── parameter_exploration.ipynb  # Active: lambda + p_fresh estimator workspace
│   ├── dataset_survey.ipynb         # Archived: broad dataset exploration
│   ├── misprice_backtest.ipynb      # Archived: deterministic bounds backtest
│   ├── misprice_backtest_deep_dive.ipynb  # Archived: clean-data movie deep dive
│   └── poisson_binomial_threshold.ipynb   # Archived: original probability model
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

## Core Module: `edge.py`

The Poisson-binomial betting function. Computes P(final score crosses threshold) and the expected edge in cents for an "Above X" Kalshi RT bet.

**Usage:**
```bash
uv run python edge.py <movie_slug> <threshold> <market_price> <hours_to_close>
uv run python edge.py the_drama 75 42 24                          # naive lambda + p_fresh from DB
uv run python edge.py the_drama 75 42 24 --lambda 1.5 --p-fresh 0.72  # override with your own estimates
```

**Components:**
- `compute_edge()` — pure math, no I/O. Takes 7 inputs (threshold, market_price, fresh_count, total_count, hours_to_close, lambda_rate, p_fresh) → returns edge_cents, p_yes, p_no. Takes lambda and p_fresh as given; makes no assumptions about how they were estimated.
- `get_movie_state()` — queries the Neon PostgreSQL DB for raw review counts: fresh/total overall and split by top/non-top critic, plus recent timestamps. Returns data only — no parameter estimation.
- `naive_lambda()`, `naive_p_fresh()` — v1 default estimators (reviews in last 6h / 6, fresh/total). Placeholders for better estimators.

**Resolution math:** "Above X" resolves Yes when `round(fresh/total * 100) >= X+1`, i.e., `fresh/total >= (X + 0.5) / 100`. See `brainstorm/brainstorm_rounding_and_resolution.md` for derivation and empirical confirmation.

**Output:** `edge_cents = P(Yes) * 100 - market_price`. Positive = buy Yes is +EV, negative = buy No is +EV.

**Where the alpha lives:** Lambda and p_fresh estimation. Everything else is observable. v1 uses simple placeholders (lambda = recent rate, p_fresh = running average). The CLI accepts `--lambda` and `--p-fresh` overrides so you can plug in better estimates. See `notebooks/parameter_exploration.ipynb` for the workspace and BACKLOG §3 for refinement directions.

**What was tried and why alternatives were abandoned:**
- *Deterministic bounds (worst/best case from remaining reviews)*: Too conservative — by the time bounds lock the outcome, the market has already corrected. See `notebooks/misprice_backtest.ipynb` and `notebooks/misprice_backtest_deep_dive.ipynb`.
- *Market-price-only strategies (spike detection, anomaly reversion)*: Require the structural model first to define "fair value." See `brainstorm/brainstorm_market_strategies.md`.

## Backlog & Strategy

`BACKLOG.md` has the full priority list. `PROMPTS.md` has handoff prompts for new conversations. `SOURCES.md` lists data and literature to gather.

**Current state:** Betting function v1 is built (`edge.py`). Dataset is complete (reviews.csv, movies_index.csv, ~141 price histories). Four exploration notebooks archived. Now building operational infrastructure and refining model parameters.

**Next steps (in order):**
1. **High-frequency score polling.** Detect review arrivals between scraper runs (every 1-5 min). Needed for real-time lambda estimation and future backtesting.
2. **Kalshi API client.** Fetch live prices for automated comparison. Foundation for the eventual execution pipeline.
3. **Parameter refinement.** Cross-movie lambda, top-critic correction, time-varying p_fresh. See BACKLOG §3.

## How to Run

```bash
# Install dependencies
cd ~/Desktop/rt-analysis
uv sync

# Run the betting function (naive defaults)
DATABASE_URL="postgresql://..." uv run python edge.py <movie_slug> <threshold> <market_price> <hours_to_close>

# Run with custom parameter estimates
DATABASE_URL="postgresql://..." uv run python edge.py <movie_slug> <threshold> <market_price> <hours_to_close> --lambda 1.5 --p-fresh 0.72

# Launch notebooks (for exploration/backtesting)
DATABASE_URL="postgresql://..." uv run jupyter notebook
```

## Dependencies

`pyproject.toml`: sqlalchemy, psycopg2-binary, pandas, numpy, scipy, matplotlib, ipykernel.
