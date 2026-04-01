# CLAUDE.md

## Project Overview

Probability analysis for Kalshi bets on Rotten Tomatoes Tomatometer scores. Uses a nonhomogeneous Poisson process to model review arrival rates, then computes the probability that a movie's score will cross a bet threshold within a forecast window.

This project connects to the same Neon PostgreSQL database used by the RT scraper (separate repo at `~/Desktop/rotten-tomatoes-analysis/`), but is fully independent -- no shared code, no shared config.

## File Structure

```
├── pyproject.toml              # Dependencies (uv managed)
├── .gitignore
├── CLAUDE.md                   # This file
├── BACKLOG.md                  # Model improvements, strategies, scaling plans
├── SOURCES.md                  # Literature, data, and hard numbers to gather
├── PROMPTS.md                  # Handoff prompts for new conversations, mapped to backlog items
└── notebooks/
    └── kalshi_analysis.ipynb   # Main analysis notebook
```

## Tech Stack

- **Language**: Python >= 3.13
- **Package manager**: uv
- **Database**: Neon (serverless PostgreSQL) via SQLAlchemy + psycopg2-binary
- **Analysis**: pandas, numpy, scipy, matplotlib
- **Notebooks**: Jupyter via ipykernel

## Database Connection

The notebook connects to the same Neon PostgreSQL instance used by the scraper. Connection is via `DATABASE_URL` environment variable.

```bash
# Launch notebook with database access
DATABASE_URL="postgresql://..." uv run jupyter notebook notebooks/kalshi_analysis.ipynb
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
- **Movies tracked**: Configured in the scraper's `movies.json`. Currently enabled: `project_hail_mary`, `ready_or_not_2_here_i_come`, `forbidden_fruits_2026`, `they_will_kill_you`.

## Analysis Methodology

### Step 1: Nonhomogeneous Poisson Process (NHPP)
- Model review arrivals as a point process with time-varying rate lambda(t)
- Fit lambda(t) using Gaussian KDE on historical `estimated_timestamp` values
- Rate function: lambda(t) = N_observed * kde(t)
- Integrate lambda(t) over the forecast window to get Lambda (expected review count)

### Step 2: N-ceiling
- Reviews in the forecast window ~ Poisson(Lambda)
- N-ceiling = smallest n where P(N >= n) < alpha (significance level)
- Truncates the infinite Poisson tail for tractable enumeration

### Step 3: Threshold-Breaking Probability
- For each n from 1 to n_ceiling, weighted by P_poisson(N=n):
  - For each i (fresh reviews) from 0 to n:
    - new_score = (current_fresh + i) / (current_total + n)
    - If new_score crosses the Kalshi threshold: accumulate P(N=n) * Binom(i | n, p)
- p = estimated probability of a fresh review (from recent sentiment trend or overall score)
- Total = probability the score breaks the threshold before the bet closes

## Backlog & Strategy

`BACKLOG.md` tracks model improvements, new strategies, and scaling plans. Read it before starting new work — it has a priority matrix at the bottom.

**Current state:** The base notebook (`notebooks/kalshi_analysis.ipynb`) implements the Poisson/binomial model end-to-end. It works but has known gaps documented in the backlog (sections 1.x). The scaling strategy (sections 2.6-2.9) is designed but not yet implemented.

`SOURCES.md` lists specific data, hard numbers, and literature needed to resolve open assumptions. Many backlog items are blocked or weakened by unknowns that can be resolved by gathering the sources listed there (Kalshi fee schedule, RT rounding rule, resolved market count, etc.).

`PROMPTS.md` has ready-to-use handoff prompts for new conversations, each mapped to specific backlog items. Use these when starting a fresh session.

See also `BACKLOG.md` §5 for operational security notes (credential handling, notebook output hygiene).

**Recommended next steps (in order):**
1. **2.8 — Backtesting on resolved markets.** Download Kalshi price histories for resolved RT markets, replay the model, measure calibration. This validates whether the model is good enough to bet on before investing in improvements.
2. **2.6 — Full score PMF + range betting.** Refactor the notebook's core loop to output the full score distribution instead of a single P(break). This unlocks multi-threshold evaluation and range bets.
3. **1.5 — Compare model output to Kalshi market prices.** Fetch live prices via Kalshi API and compute edge = model price - market price (adjusted for slippage and fees).
4. **1.1 + 1.4 — Beta-binomial + Bayesian p_fresh.** Low-effort model improvements that sharpen the estimate.

## How to Run

```bash
# Install dependencies
cd ~/Desktop/rt-analysis
uv sync

# Launch notebook
DATABASE_URL="postgresql://..." uv run jupyter notebook notebooks/kalshi_analysis.ipynb
```

## Dependencies

`pyproject.toml`: sqlalchemy, psycopg2-binary, pandas, numpy, scipy, matplotlib, ipykernel.
