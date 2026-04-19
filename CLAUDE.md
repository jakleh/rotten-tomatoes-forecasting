# CLAUDE.md

## Current Conventions (AUTHORITATIVE — read before writing any timestamp / snap / phase logic)

These supersede any conflicting convention in `findings/`, `notebooks/`, `brainstorm/`, or `plans/` docs older than 2026-04-19. If a historical doc describes a different convention, that doc is outdated — this section wins. Many historical docs use UTC-midnight snap convention, 14h phase-2 window, `C=2`, or silent noon-shift — **do not copy those patterns into new code.**

**Ship lambda model:** Ridge regression on 17 features (`ridge_t2` stack). Replaces the older KDE-based `estimate_lambda`. Library version 0.2.0. See `findings/ridge_lambda_investigation.md` for validation + `plans/plan_ridge_integration.md` for integration spec.

**Snap convention:** midnight **ET** on close−N days (Eastern-midnight anchored). T-Nd snap = midnight ET on (close_date − N). For a typical 10am ET close, T-Nd snap is `N × 24 + 10` hours before market close.
- Not midnight UTC. Not `close_ts − N days`.
- DST-aware via `tz_convert('US/Eastern').normalize()`.

**Phase-1 window:** `(midnight ET on close day, snap_time]`. Exactly `N × 24` hours for snap at T-Nd. Ridge predicts reviews in this window.

**Phase-2 window:** `(midnight ET on close day, close_ts]`. ~10h for typical 10am ET close. 9h on DST spring-forward days, 11h on fall-back days, other values for non-10am closes (e.g., 2h for 2am ET close). Computed dynamically from `close_ts` per `compute_close_day_phase2(close_ts, C)`.

**Phase-2 constant:** `C = 1.0` reviews. Empirical mean on 5 h/m targets with full scraper coverage: `{2, 1, 1, 1, 0}`. NOT `C=2` (that was an older value for a wider 14h UTC-midnight window, superseded).

**Noon-shift preprocessing:** day-level reviews (`timestamp_confidence == 'd'`) shifted from midnight UTC to 12:00 UTC before profile build. Purpose: de-spike KDE boundaries and center day-level review timestamps in-day. Applied **ONCE at data ingest by the caller** — `extract_lambda_features` defaults `apply_noon_shift=False` and does not silently mutate timestamps.

**`dbc` (days before close) anchor:** always measured against `close_ts` (10am ET = 14:00 UTC EDT / 15:00 UTC EST). Never against midnight of anything. `midnight_et_dbc = (close_ts − midnight_et_close).total_seconds() / 86400` ≈ 0.417 during EDT, ≈ 0.458 during EST.

**Model artifact serialization:** JSON at `_artifacts/default_regressor.json`. NOT pickle (avoids sklearn version-drift risk). Contains per-snap Ridge coefficients + StandardScaler stats + LOO residuals + metadata.

**Snap routing in public API:** `estimate_lambda` takes `snap_days: int` (keyword-only, values in {1, 2, 3, 4, 5}). `snap_dbc` is an internal concept only — never appears in public signatures. Out-of-range snap_days raises.

**Target filter for deployment (orchestrator):** `target_gap > 15d` → skip entirely (architectural ceiling failure zone). See `findings/trading_strategy_from_ridge_errors.md` §2.3.

See `plans/plan_ridge_integration.md` for the full library spec.

---

## Project Overview

Pure forecasting library for Rotten Tomatoes Tomatometer prediction markets. Computes the probability that a movie's final Tomatometer score crosses a given threshold, and the expected edge (in cents) against a market price.

This is a library, not a trading system. Its contract ends at `compute_edge() -> (edge_cents, p_yes, p_no)`. Strategy, backtesting, execution, and position sizing live in the orchestrator repo (`~/Desktop/kalshi-trading/`).

This project connects to the same Neon PostgreSQL database used by the RT scraper (separate repo at `~/Desktop/rotten-tomatoes-analysis/`), but is fully independent -- no shared code, no shared config.

## Build Protocol

Follow `PROTOCOL.md` for all non-trivial work. Do not write code before writing and presenting a plan doc. See `PROTOCOL.md` for what qualifies as non-trivial.

## File Structure

```
├── rotten_tomatoes_forecasting/                # The package
│   ├── __init__.py             # Public API: 8 symbols + __version__
│   ├── edge.py                 # compute_edge(), naive_lambda(), naive_p_fresh(), EdgeResult
│   └── critic_model.py         # KDE model: CriticProfiles, KDELambdaModel, estimate_lambda, estimate_p_fresh, etc.
├── pyproject.toml              # Dependencies (uv managed), package config
├── CLAUDE.md                   # This file
├── PROTOCOL.md                 # Build protocol (plan -> implement -> validate)
├── BACKLOG.md                  # Model validation and improvement priorities
├── PROMPTS.md                  # Handoff prompts for new conversations
├── PARAMETERS.md               # All tunable parameters + pending-integration values
├── .env                        # DATABASE_URL (gitignored)
├── reviews.csv                 # Local dump of reviews table (gitignored)
├── movies_index.csv            # Movie slugs + bet close dates (gitignored)
├── notebooks/                  # Model validation notebooks
│   ├── critic_model_validation.ipynb    # KDE model validation on historical movies
│   ├── kde_lambda_calibration.ipynb     # Volume prediction gut-checks for KDE model
│   ├── critics_index.ipynb              # Critic frequency analysis and KDE prototyping
│   ├── parameter_exploration.ipynb      # Lambda + p_fresh estimator workspace
│   ├── dataset_survey.ipynb             # Broad dataset exploration
│   ├── misprice_backtest.ipynb          # Deterministic bounds backtest (led to KDE approach)
│   ├── misprice_backtest_deep_dive.ipynb  # Clean-data movie deep dive
│   ├── poisson_binomial_threshold.ipynb   # Original probability model
│   ├── stratified_training_validation.ipynb  # Stratified training + bandwidth cap + piecewise + Phase A/B (74 cells)
│   └── .cache/                          # LOO results caches (gitignored)
├── findings/                   # Model validation findings
│   ├── critic_kde_model_validation.md   # KDE model accuracy (lambda, p_fresh, calibration)
│   ├── kalshi_rt_contract_rules.md      # Contract rules: resolution, position limits, fallbacks
│   ├── embargo_anchor_investigation.md  # Embargo-anchor rebuild: rejected (2026-04-17)
│   └── stratified_training_investigation.md  # Current validated stack: combined_score + ceil=0.7 + piecewise (F=1.0)
├── rt-rules-contract.pdf       # Kalshi RT contract rules (source document)
├── plans/                      # Implementation plans (gitignored)
└── brainstorm/                 # Model design brainstorms (gitignored)
```

## Public API

```python
from rotten_tomatoes_forecasting import (
    compute_edge,           # Pure math: 7 inputs -> EdgeResult dict
    build_critic_profiles,  # DataFrame -> CriticProfiles
    build_kde_lambda_model, # CriticProfiles -> KDELambdaModel
    estimate_lambda,        # Model + observed state -> float (reviews/hr)
    estimate_p_fresh,       # Profiles + observed state -> float
    default_training_slugs, # DataFrame -> list of slugs
    CriticProfiles,         # Dataclass: per-critic base rates, fresh rates, timing data
    KDELambdaModel,         # Dataclass: KDE-based lambda estimator
)
```

**Internal (not re-exported, accessible via submodule):**
- `rotten_tomatoes_forecasting.edge.naive_lambda()`, `naive_p_fresh()` — v1 fallback estimators
- `rotten_tomatoes_forecasting.critic_model._compute_scaling()`, `_blended_integral()`, etc. — internal helpers

This library is pure DataFrame-in / numbers-out. It does NOT access a database. Callers own DB access and pass review DataFrames to the public API. See `~/Desktop/kalshi-trading/src/series/rotten_tomatoes/db.py` for the reference consumer.

## Tech Stack

- **Language**: Python >= 3.13
- **Package manager**: uv
- **Database**: No direct DB access. Consumers own DB reads. Notebooks use SQLAlchemy + psycopg2-binary for ad-hoc analysis.
- **Analysis**: pandas, numpy, scipy, matplotlib
- **Notebooks**: Jupyter via ipykernel

## Installation

```bash
# Development (from this repo)
uv sync

# As a dependency (from another project)
pip install -e ~/Desktop/rotten-tomatoes-forecasting
# or: pip install git+https://github.com/jakleh/rotten-tomatoes-forecasting.git
```

## How to Run

```bash
# Launch notebooks
DATABASE_URL="postgresql://..." uv run jupyter notebook
```

Consumers (e.g., `~/Desktop/kalshi-trading/`) import the public API and pass review DataFrames to functions like `compute_edge` and `build_critic_profiles`.

## Database Connection

The library itself does NOT connect to a database. Notebooks and consumers that need the reviews table use SQLAlchemy directly via `DATABASE_URL`.

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
- **Movies tracked**: Configured in the scraper's `movies.json`. All ~143 movies that had Kalshi RT markets have been backfilled. Originally live-tracked (with minute-level timestamps): `project_hail_mary`, `ready_or_not_2_here_i_come`, `forbidden_fruits_2026`, `they_will_kill_you`, `the_drama`, `the_super_mario_galaxy_movie`.

## Core Module: `rotten_tomatoes_forecasting/edge.py`

The Poisson-binomial betting function. Computes P(final score crosses threshold) and the expected edge in cents for an "Above X" bet.

**`compute_edge()`** — pure math, no I/O. Takes 7 inputs (threshold, market_price, fresh_count, total_count, hours_to_close, lambda_rate, p_fresh) -> returns `EdgeResult` dict with edge_cents, p_yes, p_no, expected_reviews, k_max. Takes lambda and p_fresh as given; makes no assumptions about how they were estimated.

**Resolution math:** "Above X" resolves Yes when `round(fresh/total * 100) >= X+1`, i.e., `fresh/total >= (X + 0.5) / 100`. See `brainstorm/brainstorm_rounding_and_resolution.md` for derivation.

**Output:** `edge_cents = P(Yes) * 100 - market_price`. Positive = buy Yes is +EV, negative = buy No is +EV.

**Where the alpha lives:** Lambda and p_fresh estimation. Everything else is observable.

## Per-Critic KDE Model: `rotten_tomatoes_forecasting/critic_model.py`

Replaces naive lambda/p_fresh with estimators grounded in per-critic historical data. Three layers:

1. **CriticProfiles** — per-critic base_rate (P(reviews this movie)), fresh_rate, and timing_data (days-before-close). Built from training set of 20 most recent resolved movies.
2. **KDELambdaModel** — fits Gaussian KDEs to each critic's timing data. Expected remaining reviews = sum of base_rate x KDE integral for unreviewed critics. Scaled in real-time via observed/expected ratio. Sparse critics (0-1 reviews) and degenerate KDEs (zero variance) fall back to a population prior with shrinkage (k=3, bandwidth floor 0.5d).
3. **estimate_p_fresh()** — base_rate-weighted average of remaining critic fresh rates, blended with movie's running observed rate (prior sample size n=20). No KDEs involved.

**Key functions:** `build_critic_profiles()`, `build_kde_lambda_model()`, `estimate_lambda()`, `estimate_p_fresh()`, `default_training_slugs()`.

**Design:** See `plans/plan_critic_kde_lambda.md` for the full spec and `brainstorm/brainstorm_critic_kde_lambda.md` for design rationale. Validated in `notebooks/critic_model_validation.ipynb`.

**What was tried and why alternatives were abandoned:**
- *Deterministic bounds (worst/best case from remaining reviews)*: Too conservative -- by the time bounds lock the outcome, the market has already corrected. See `notebooks/misprice_backtest.ipynb`.
- *Market-price-only strategies (spike detection, anomaly reversion)*: Require the structural model first to define "fair value."

## Known Issues

**Close-day lambda bias:** The KDE model drops close-day reviews because of a UTC midnight vs actual close time mismatch. Partial fix applied. See `brainstorm/brainstorm_close_day_lambda_bias.md` and BACKLOG.md 1.3.

## Dependencies

`pyproject.toml`: sqlalchemy, psycopg2-binary, pandas, numpy, scipy, matplotlib, ipykernel.
