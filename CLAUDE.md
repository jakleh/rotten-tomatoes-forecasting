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

**Target filter for deployment (orchestrator):** `target_gap > 15d` → skip entirely (architectural ceiling failure zone — long-lead / festival-film volume is unpredictable from a pre-snap snapshot). Rationale was in `findings/trading_strategy_from_ridge_errors.md`, removed at the 2026-06-02 prune; recoverable from git history.

See `plans/plan_ridge_integration.md` for the full library spec.

---

## Project Overview

The focused RT-modeling workspace for Rotten Tomatoes Tomatometer prediction markets. Computes the probability that a movie's final Tomatometer score crosses a given threshold, and the expected edge (in cents) against a market price; the math is pure DataFrame-in / numbers-out, ending at `compute_edge() -> (edge_cents, p_yes, p_no)`.

**Direction (2026-06-02):** this repo is where the RT model gets refined until it works or the project is killed. Eventual deployment is a thin **script → Vultr VM** that imports the edge-calc here plus the execution/infra helpers in `~/Desktop/kalshi-trading/` — *not* a multi-series orchestrator (that abstraction is being deprecated as premature). Strategy/execution are deliberately kept out of this repo for now. Today `kalshi-trading` (branch `feat/kalshi-har-replacement`) still imports this as a package via `src/series/KXRT/`.

**Active roadmap:** minute-level review timestamps (~20 movies as of 2026-06-02, up from 2) unlock self-labeling the final 10am-ET score from reviews, measuring the actual close-day review count (the piecewise λ model's final-~10h "phase-2" term, today the flat `C=1` constant), and fitting a single smooth λ(t). The next step is the VoI **Gate 1 / Gate 2** calibration — is the market beatable, and does the Poisson×Binomial architecture clear it with perfect inputs — *before* any model upgrade. See `BACKLOG.md`.

**Gate-1 status (2026-06-07):** Gate 1 ran directionally on the 16-movie settled KXRT cohort (code in the new `gates/`; analyses in `notebooks/gate1_calibration.ipynb` + `gate1b_incremental_info.ipynb`). Result: the market **prices the observed review state** (no current-state edge) and is **stale/thin** — most order books go one-sided ~4 days before close, and contested markets have no live two-sided quote near close — so **tradeability, not forecasting, is the binding constraint**. Full result: memory `project_gate1_findings` + `plans/plan_gate_1_2_calibration.md`.

**Arena-map status (2026-06-09):** the tradeable-edge-surface check ran (`notebooks/arena_map.ipynb`): Kalshi candles are activity-gated but the order book persists through silent gaps (probe: P(state identical across gap)=1.00000) → per-minute LOCF book reconstruction is valid. The contested∧tight-spread arena **exists and is early** — at T-3d all 16/16 movies have a contested ≤10¢-spread book with median 28% of reviews still to come; near-close (≤12h) it is ~empty. **The tradeable window is T-2d..T-5d (center T-3d)**. See the "Arena map result (2026-06-09)" section of `plans/plan_gate_1_2_calibration.md` + memory `project_arena_map`.

**Gate-2 status (2026-06-09, REVISED 2026-06-10 to the data-ready cohort): STRONG PASS (directional).** After the pre-registered dense-cohort STOP-gate (`notebooks/gate2_density.ipynb`) and the 2026-06-10 data-readiness revision (cells exclude `animal_farm_2025`, whose DB history is provably incomplete — settlement-consistency check), the canonical run (`notebooks/gate2_oracle.ipynb`, as_of_id=648979, ex-ANI): realized λ/p_fresh through `compute_edge` beats the state-at-snap book on **both Brier and spread-crossing PnL net of taker fees** — T-3d Brier diff +0.0883 [+0.0367, +0.1377], PnL +27.1¢/contract [+16.5, +36.3], 86% win; **pooled +0.1247 / +32.1¢ [+22.4, +41.7], 91% win**; robust ex-billie; wins on BOTH trade sides; lagged(scrape-time) ≈ pure(publication-time) oracle, so scraper cadence isn't binding. (Original 13-movie figures — T-3d +0.0775/+24.1¢, pooled +0.0966/+27.5¢ — kept as labeled history in the plan; the revision only removes the defective movie, which was a drag.) The market prices the current review state (Gate 1b) but **not the flow**. Oracle math: `gates/oracle.py`, test-validated. Full result + caveats: the "GATE 2 RESULT — REVISED" section of `plans/plan_gate_1_2_calibration.md` + memory `project_gate2_result`.

**Integrity incident + revision (2026-06-10):** the recorder's settlement-consistency check exposed a sentiment-case switch (UPPERCASE rows scraped ≥ ~2026-06-02; raw table preserved, processing layer now case-insensitive — see the schema note) and coverage-thin movies (DB missing reviews RT counted: animal_farm_2025, power_ballad pre-05-01, backrooms, in_the_grey — backfill = BACKLOG §1.9, high priority). `notebooks/gate2_integrity_recheck.ipynb` first proved **zero sentiment-case drift** in all 134 original Gate-2 cell rows (no uppercase pre-close row touched any cell movie); the operator then promoted the readiness criterion into the canonical cell definition → Gate 2 + Gate 3a **re-executed ex-animal_farm** (the revised numbers above/below; every conclusion survives stronger). Gate 1/arena not re-run by recorded reasoning (plan addendum). animal_farm + power_ballad are `data_not_ready` for Gate 3b (effective cohort 17).

**Gate-3a status (2026-06-09, re-swept 2026-06-10 on the data-ready cells): p_fresh is the binding input.** Error-tolerance sweep (`notebooks/gate3_tolerance.ipynb`, 36 markets/12 movies, anchor = the revised pooled +32.1¢): the PnL edge survives λ mis-estimation from **0.55× to 3×** at δ=0 (±170% random noise → 48/50 draws clear) and **δ ∈ [−0.10, 0]** of p_fresh error at m=1 — but δ=+0.05 kills the CI (random ±0.05 → 34/50; **over-estimating freshness remains the failure mode**; under-estimating is nearly free → the conservative-shade option strengthens). Shipped-0.2.0 proxies: Ridge λ error (m ∈ [0.62, 1.38] worst-snap) comfortably inside; `estimate_p_fresh` (~±0.03–0.05) in at −0.05, out at +0.05 → **the improvement priority inverts to p_fresh** (untouched since 0.1.x). Next: **Gate 3b** — run the actual `estimate_lambda`/`estimate_p_fresh` on a re-derived ET-midnight cell grid (A1-pool review cache; 17-movie data-ready cohort) → deployable-stack verdict vs the +32.1¢ oracle ceiling. See "GATE 3a RESULT — REVISED" in the plan.

This project reads the same Neon PostgreSQL database the RT scraper populates (separate repo at `~/Desktop/rotten-tomatoes-analysis/`) but shares no code or config.

## Build Protocol

Follow `PROTOCOL.md` for all non-trivial work. Do not write code before writing and presenting a plan doc. See `PROTOCOL.md` for what qualifies as non-trivial.

## File Structure

```
├── rotten_tomatoes_forecasting/                # The package (v0.2.0, Ridge lambda model)
│   ├── __init__.py             # Public API + __version__
│   ├── edge.py                 # compute_edge(), naive_lambda(), naive_p_fresh(), EdgeResult
│   ├── lambda_model.py         # Ridge model: fit_lambda_regressor, estimate_lambda,
│   │                           #   compute_close_day_phase2, save/load_regressor,
│   │                           #   load_default_regressor, LambdaRegressor, LambdaPrediction
│   ├── features.py             # extract_lambda_features (17 features), apply_noon_shift,
│   │                           #   midnight_et_of_close, FEATURE_NAMES
│   ├── pool.py                 # A1 pool context + shared base_rate primitive
│   ├── p_fresh.py              # estimate_p_fresh (migrated from critic_model.py)
│   └── _artifacts/             # Shipped fit artifacts
│       └── default_regressor.json   # Default LambdaRegressor JSON (~23KB)
├── scripts/
│   └── fit_default_regressor.py     # Refit script for the shipped artifact
├── gates/                           # Gate-calibration support layer (NOT part of the shipped package)
│   ├── kalshi_data.py          # Public Kalshi/KXRT fetcher (no auth; stdlib urllib)
│   ├── db_facts.py             # Read-only as_of_id-pinned reviews queries
│   ├── oracle.py               # Gate-2 oracle λ/p_fresh decomposition (placement rules, two boundaries)
│   ├── build_cohort.py         # Driver: settled cohort + 1-min candles -> _cache/ (network)
│   ├── build_snap_state.py     # Driver: per-(market,snap) mid + observed state -> _cache/ (DB)
│   ├── build_density.py        # Driver: dense-cohort-guard density facts -> _cache/ (DB)
│   ├── build_reviews_cache.py  # Driver: pinned per-review cohort rows -> _cache/ (DB)
│   ├── probe_candle_open.py    # Driver: candle bid/ask open+close probe (LOCF validation)
│   ├── recorder.py             # §1.7 weekly settled-market recorder -> recorded/ (idempotent; --check staleness)
│   ├── validate_recorded.py    # Cross-check recorded/ vs _cache/ (the recorder's rerunnable Phase-3 audit)
│   ├── slug_map.py             # Shared Kalshi-title -> DB-slug mapping (build_cohort + recorder)
│   ├── _make_*.py              # nbformat codegen for each analysis notebook
│   ├── _cache/                 # Cached CSVs/PNGs incl. arena_spans.csv, gate2_cells.csv (gitignored)
│   └── recorded/               # COMMITTED system of record: settled-market ledger + per-event 1-min candles (.csv.gz)
├── notebooks/                       # Gate analyses (cache-only, sandbox-safe; the citable numbers)
│   ├── gate1_calibration.ipynb / gate1b_incremental_info.ipynb      # Gate 1 (2026-06-07)
│   ├── arena_map.ipynb              # Tradeable-edge arena map (2026-06-09)
│   ├── gate2_density.ipynb / gate2_oracle.ipynb                    # Gate 2 STOP-gate + result (2026-06-09)
│   ├── gate2_integrity_recheck.ipynb    # 2026-06-10 sentiment-case/coverage recheck (zero drift; ex-ANI stronger)
│   └── gate3_tolerance.ipynb        # Gate 3a λ/p_fresh error-tolerance band (2026-06-09)
├── tests/                           # 648 tests (98 package: edge/features/lambda_model/p_fresh/pool/package
│                                    #  + 550 gate-support: oracle placement/invariants, compute_edge battery, recorder)
├── pyproject.toml              # Dependencies (uv managed), package-data config
├── CLAUDE.md                   # This file
├── PROTOCOL.md                 # Build protocol (plan -> implement -> validate)
├── BACKLOG.md                  # Model validation and improvement priorities
├── PROMPTS.md                  # Handoff prompts for new conversations
├── PARAMETERS.md               # All tunable parameters (0.2.0 Ridge)
├── .env                        # DATABASE_URL (gitignored)
├── reviews.csv                 # Local dump of reviews table (gitignored)
├── movies_index.csv            # Movie slugs + bet close dates (gitignored)
├── findings/                   # Live findings only (KDE-era investigations pruned 2026-06-02)
│   ├── ridge_lambda_investigation.md   # Ship candidate (ridge_t2); 3-tier optimization; pool robustness
│   ├── kalshi_rt_contract_rules.md     # Contract: resolution, expiration (1st Monday after wide release), fallbacks
│   └── archive/                        # Superseded KDE-era investigations (banner'd; replay via git history)
├── rt-rules-contract.pdf       # Kalshi RT contract rules (source document)
├── plans/                      # Implementation plans (gitignored)
└── brainstorm/                 # Model design brainstorms (gitignored)
```

## Public API (0.2.0)

```python
from rotten_tomatoes_forecasting import (
    compute_edge,               # Pure math: 7 inputs -> EdgeResult dict
    estimate_lambda,            # LambdaRegressor + features + snap_days + close_ts -> LambdaPrediction
    estimate_p_fresh,           # reviews_df + training_slugs + observed state -> float
    fit_lambda_regressor,       # Cohort -> LambdaRegressor (5-snap × α CV × LOO fit)
    load_default_regressor,     # Load shipped artifact from _artifacts/default_regressor.json
    extract_lambda_features,    # 17-feature vector per (target, snap) under ET-midnight convention
    compute_close_day_phase2,   # Dynamic phase-2 hours + count for close_ts (DST-aware)
    LambdaRegressor,            # Dataclass: fitted Ridge per snap + residuals + metadata
    LambdaPrediction,           # Dataclass: rate_per_hour + phase1/phase2/total + p90|err|
    naive_lambda, naive_p_fresh,  # v1 fallback estimators
)
```

**Internal (not re-exported, accessible via submodule):**
- `rotten_tomatoes_forecasting.lambda_model.SnapModel`, `LambdaRegressorMetadata`, `save_regressor`, `load_regressor` — artifact plumbing.
- `rotten_tomatoes_forecasting.pool.A1Context`, `build_a1_pool_context`, `compute_critic_base_rates` — pool primitives used by `extract_lambda_features` and `estimate_p_fresh`.
- `rotten_tomatoes_forecasting.features.FEATURE_NAMES`, `VALID_SNAP_DAYS`, `midnight_et_of_close`, `apply_noon_shift` — extraction helpers + constants.

This package is pure DataFrame-in / numbers-out. It does NOT access a database. Callers own DB access and pass review DataFrames to the public API. Reference consumer: `~/Desktop/kalshi-trading/src/series/KXRT/db.py` (branch `feat/kalshi-har-replacement`).

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
```

`~/Desktop/kalshi-trading/` currently imports this via a local path dependency (an optional `rt` extra; `rotten-tomatoes-forecasting @ file://...`). Per the scripts direction, the eventual consumer is a script that imports the edge-calc directly rather than installing a package — don't invest in the packaging/publish path.

## How to Run

```bash
# Launch notebooks
DATABASE_URL="postgresql://..." uv run jupyter notebook
```

Consumers (e.g., `~/Desktop/kalshi-trading/`) import the public API and pass review DataFrames to functions like `compute_edge`, `extract_lambda_features`, and `estimate_lambda`.

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
| tomatometer_sentiment | TEXT | **"positive"/"negative"; UPPERCASE in rows scraped ≥ ~2026-06-02** -- the fresh/rotten signal. Raw rows preserved as-is (operator call 2026-06-10); **always compare case-insensitively** (`lower()` in SQL / `.str.lower()` on DataFrames — `gates/db_facts.py`, `gates/oracle.py`, `p_fresh.py` already do) |
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
- **Timestamp precision varies**: "m"-confidence timestamps are accurate to ~1 minute, "h" to ~1 hour, "d" to ~1 day. Day-level reviews land at midnight UTC by default; `apply_noon_shift` centers them mid-day so they don't all cluster on day boundaries.
- **Scrape frequency**: Every 50 minutes. Reviews with relative timestamps ("5m", "3h") get more precise `estimated_timestamp` values than date-format ones ("Mar 20").
- **Movies tracked**: Configured in the scraper's `movies.json`. All ~143 movies that had Kalshi RT markets have been backfilled. Originally live-tracked (with minute-level timestamps): `project_hail_mary`, `ready_or_not_2_here_i_come`, `forbidden_fruits_2026`, `they_will_kill_you`, `the_drama`, `the_super_mario_galaxy_movie`.

## Core Module: `rotten_tomatoes_forecasting/edge.py`

The Poisson-binomial betting function. Computes P(final score crosses threshold) and the expected edge in cents for an "Above X" bet.

**`compute_edge()`** — pure math, no I/O. Takes 7 inputs (threshold, market_price, fresh_count, total_count, hours_to_close, lambda_rate, p_fresh) -> returns `EdgeResult` dict with edge_cents, p_yes, p_no, expected_reviews, k_max. Takes lambda and p_fresh as given; makes no assumptions about how they were estimated.

**Resolution math:** "Above X" resolves Yes when `round(fresh/total * 100) >= X+1`, i.e., `fresh/total >= (X + 0.5) / 100`. See `brainstorm/brainstorm_rounding_and_resolution.md` for derivation.

**Output:** `edge_cents = P(Yes) * 100 - market_price`. Positive = buy Yes is +EV, negative = buy No is +EV.

**Where the alpha lives:** Lambda and p_fresh estimation. Everything else is observable.

## Ridge Lambda Model: `rotten_tomatoes_forecasting/lambda_model.py`

Replaces the per-critic KDE architecture (0.1.x). Per-snap Ridge regression on 17 features predicts phase-1 review volume; phase-2 is a DST-aware close-day constant.

1. **`extract_lambda_features`** (`features.py`) — builds a 17-feature vector for one (target, snap): 10 observation-window statistics (counts, rates, top-critic fraction, publication diversity/entropy, low-activity critic fraction), 4 nonlinear transforms of the dominant count/rate features, and 3 finite-pool aggregates (`remaining_base_rate_sum`, `pool_mass_consumed`, `observed_top_tier_frac`) computed against the LOO-clean A1 pool (20 most recent resolved movies before target close).
2. **`fit_lambda_regressor`** (`lambda_model.py`) — for each snap in `{1, 2, 3, 4, 5}`: feature extraction across the cohort, 5-fold CV α selection over `{0.01, 0.1, 1, 10, 100, 1000}`, LOO residual capture, then a full-cohort fit. Produces a `LambdaRegressor` with per-snap `SnapModel` (scaler stats + ridge coefficients + intercept + α) and metadata (fit date, cohort size, sklearn version, phase-2 C).
3. **`estimate_lambda`** (`lambda_model.py`) — scores a feature row through the snap's `SnapModel`, adds `compute_close_day_phase2(close_ts, C)` for the close-day window (DST-aware), and returns a `LambdaPrediction` with the full phase-1 / phase-2 / total breakdown, rate-per-hour, and a p90|err| estimate from training LOO residuals.
4. **`estimate_p_fresh`** (`p_fresh.py`) — base_rate-weighted average of remaining critic fresh rates, blended with the running observed rate (prior sample size `n_prior=20`). Behavior unchanged from 0.1.x; takes `reviews_df` + `training_slugs` explicitly instead of the removed `CriticProfiles`.

**Artifact.** Ships at `_artifacts/default_regressor.json` (~23KB). Version-agnostic JSON (not pickle) — reconstructs `SnapModel` dataclasses at load time. Re-fittable via `scripts/fit_default_regressor.py`.

**Performance vs 0.1.x KDE baseline** (cohort LOO, ET-midnight convention):
- T-5d: 37.96 → 32.14 MAE (+15%), mean_err -17.19 → -0.45
- T-3d: 17.86 → 9.96 MAE (+44%), mean_err -9.58 → -0.01
- T-1d: 3.87 → 2.22 MAE (+43%), mean_err +2.73 → -0.03

See `findings/ridge_lambda_investigation.md` for the full validation and `plans/plan_ridge_integration.md` for integration rationale.

**What was tried and why alternatives were abandoned:**
- *KDE per-critic model* (0.1.x): architectural ceiling on critic-magnet / late-surge movies — 14 interventions couldn't break through. Ridge bypasses the `base_rate × KDE × exclusion` sum by regressing directly on observable features. See `findings/archive/path_b_lite_investigation.md`.
- *Deterministic bounds (worst/best case)*: too conservative; by the time bounds lock the outcome the market has already corrected. (Validated in the now-pruned `misprice_backtest.ipynb`; in git history.)

## Known Issues

**None tracked.** The KDE-era notebooks + `notebooks/_helpers.py` — which imported the removed `critic_model` and so couldn't run under 0.2.0 — were deleted in the 2026-06-02 prune (replayable from git history). New analysis notebooks should import the 0.2.0 public API directly.

## Dependencies

`pyproject.toml`: sqlalchemy, psycopg2-binary, pandas, numpy, scipy, matplotlib, ipykernel, scikit-learn.
