# Backlog

## 1. Current Priorities: Operational Infrastructure

The betting function v1 is built (`edge.py`). The immediate focus is operational infrastructure to support live betting: real-time review detection and market price retrieval.

### 1.1 High-frequency score polling

Detect review arrivals between scraper runs (50-minute gaps) by polling the displayed Tomatometer score every 1-5 minutes. One number per movie, not a full page parse.

**Why it's now a priority:** Lambda estimation needs to know when reviews arrive. The scraper gives 50-minute resolution; the poller gives 1-5 minute resolution. Also enables future backtesting of the probability model with real-time data.

**Status:** Not started.

### 1.2 Kalshi API client

Fetch live prices for all RT markets. No order placement — just price retrieval. Foundation for comparing model output to market prices and for the eventual execution pipeline.

**Status:** Not started. Need API credentials and docs (see SOURCES.md §1.4, §1.5).

---

## 2. Betting Function (Complete)

### 2.1 Poisson-binomial edge calculation

**Status: Complete.** Implemented in `edge.py`. See `plans/plan_betting_function.md` for the full plan and validation results.

**What it does:** For each possible number of additional reviews k (Poisson-distributed), computes P(final score crosses threshold) using binomial CDF, then derives edge in cents. Seven inputs → edge_cents, p_yes, p_no.

**v1 parameter estimates (intentionally simple):**
- `lambda_rate`: reviews in last 6 hours / 6 (flat recent rate from DB)
- `p_fresh`: current `fresh_count / total_count` (running average from DB)

**Known limitations:** Lambda is bursty (not constant-rate), p_fresh has top-critic bias early in lifecycle. These are addressed by parameter refinement (§3).

---

## 3. Parameter Refinement (Feeds into §2)

### 3.1 Per-critic KDE lambda model — **Complete**

Replace naive lambda and p_fresh with a per-critic KDE model. Every critic gets a KDE fitted to their historical review timing (days before close), weighted by their base rate (movies reviewed / total movies). Lambda = sum of weighted KDE integrals for unreviewed critics. p_fresh = weighted average of per-critic fresh rates blended with the movie's running observed rate. Real-time scaling via observed/expected ratio.

**Status: Complete.** Implemented in `critic_model.py`. Integrated into `edge.py` via `--kde` flag. Validation notebook at `notebooks/critic_model_validation.ipynb`.

**Key design decisions (2026-04-07):**
- Build KDEs for ALL critics (not just "active" ones). Long-tail critics self-average into a smooth background rate.
- Scale lambda in real-time via observed/expected ratio. No pre-market predictor (Kalshi window R²=0.058, raw day-1 count R²=0.312 — neither sufficient).
- `compute_edge()` interface unchanged — aggregate to scalar lambda + p_fresh.
- Population prior + shrinkage (k=3) for sparse/degenerate KDEs. Bandwidth floor 0.5d.

### 3.2 Other refinement ideas (deferred)

- **Time-varying p_fresh:** Freshness rate may shift over a movie's lifecycle. `brainstorm/brainstorm_time_varying_p_fresh.md`
- **Top-critic correction:** Early reviews overweight top critics who are ~6pp more negative. Adjust p_fresh accordingly.
- **Overdispersion (beta-binomial):** If sentiment is clustered (not i.i.d.), the binomial underestimates variance. `brainstorm/brainstorm_poisson_binomial_threshold.md`

**Note:** Cross-movie lambda (`brainstorm_cross_movie_lambda.md`) and hierarchical p_fresh (`brainstorm_hierarchical_p_fresh.md`) are superseded by the per-critic KDE model, which handles both.

---

## 4. Data Infrastructure (Complete)

### 4.1 Historical review database backfill

**Status: Complete.** All ~141 movies backfilled into Neon PostgreSQL.

### 4.2 Local data dump + movies index

**Status: Complete.** `reviews.csv` (23K+ reviews) and `movies_index.csv` (141 movies with volume, dates, score ranges, review counts) at project root. See `plans/plan_populate_movies_index.md` for details.

### 4.3 Trading volume for resolved markets

**Status: Complete.** Recorded in `movies_index.csv` from Google Sheet.

---

## 5. Platform Mechanics

### 5.1 Kalshi fee schedule

Per-contract trading fee and any settlement fees. Required for realized edge calculation (net edge = gross edge - fees).

**Status:** Open. See SOURCES.md §1.1.

### 5.2 RT rounding rules

**Status: Resolved.** RT uses standard rounding (round half up). Kalshi resolves against the displayed score. See `brainstorm/brainstorm_rounding_and_resolution.md`.

### 5.3 Kalshi resolution rules

**Status: Resolved.** "Above X" = displayed score >= X+1. Snapshot at 10:00 AM ET on expiration date. All Critics Tomatometer. See `brainstorm/brainstorm_rounding_and_resolution.md`.

---

## 6. Exploration History (Archived)

Completed explorations that informed the current approach. Notebooks and findings preserved for reference. These led to the decision to build the Poisson-binomial model.

### 6.1 Dataset survey (Complete — 2026-04-06)

Broad survey of the joint dataset. Found 25 misprices across 23/141 movies with median 57¢ edge. Edge does NOT concentrate in low-volume markets. Score is knowable early (93%+ reviews in by T-24h). Top critics are systematically ~6pp more negative.

**Notebook:** `notebooks/dataset_survey.ipynb` · **Findings:** `brainstorm/brainstorm_dataset_survey_findings.md`

### 6.2 Systematic misprice backtest (Complete — 2026-04-06)

Tested deterministic worst-case/best-case bounds across 141 movies. Found 74 episodes across 52 movies but median edge only 9¢. The biggest survey misprices (Joker 97¢, Snow White 61¢) fell in a 20-movie data-quality gap where review counts don't match ground truth (day-level timestamp noise). 33 lock/resolution mismatches traced to this cause.

**Conclusion:** Deterministic bounds too conservative — by the time they lock, the market has already corrected. Probabilistic approach needed.

**Notebook:** `notebooks/misprice_backtest.ipynb` · **Plan:** `plans/plan_misprice_backtest.md`

### 6.3 Deep dive on clean-data movies (Complete — 2026-04-06)

Focused analysis on Forbidden Fruits and They Will Kill You — the two resolved movies with minute/hour-level timestamps in the critical 48h window. Max edge was 7¢ (Forbidden Fruits "Above 80") and 4¢ (They Will Kill You "Above 90"). Market was well-calibrated by the time bounds locked.

**Confirmed:** The edge is real but you need a probabilistic approach to capture it before the market self-corrects.

**Notebook:** `notebooks/misprice_backtest_deep_dive.ipynb`

### 6.4 Data quality findings

- 20/141 movies have review-implied final scores that don't match the movies_index score range (derived from market resolution). Root cause: day-level timestamps attribute reviews to the wrong side of the 10 AM ET close cutoff.
- Only 6 movies have significant minute-level timestamp data: `the_drama` (145m), `the_super_mario_galaxy_movie` (162m), `they_will_kill_you` (42m), `forbidden_fruits_2026` (17m), `project_hail_mary` (25m), `ready_or_not_2_here_i_come` (23m).
- The Drama and Super Mario Galaxy have 99-100% minute+hour coverage but are still active (no resolved Kalshi markets yet, no price histories).

---

## 7. Future Ideas

Ideas from brainstorming that survived initial gut-checks but are not the current priority. Brainstorm files preserved in `brainstorm/` (gitignored).

### Model extensions

- **Finite-pool / remaining-reviewer model:** Replace binomial with Poisson-binomial using per-reviewer p_i. `brainstorm/brainstorm_finite_pool_model.md`
- **Reviewer graph model:** Model critics as a graph with influence edges. `brainstorm/brainstorm_reviewer_graph.md`
- **Full score PMF and range betting:** Produce full distribution over final scores, evaluate every threshold simultaneously. `brainstorm/brainstorm_score_pmf_range_betting.md`
- **Early score trajectory forecasting:** Match partial trajectories to historical library. Useful early in lifecycle (<30 reviews).

### Market-based strategies

- **Embargo-lift divergence:** Compare initial review basket to market price. `brainstorm/brainstorm_market_strategies.md`
- **Price trace anomaly detection:** Detect amateur-driven spikes, bet on reversion. `brainstorm/brainstorm_market_strategies.md`
- **Volume-review count correlation:** Predict review ceiling from trading volume. `brainstorm/brainstorm_market_strategies.md`
- **Market price as leading signal:** Between scraper runs, price moves may signal new reviews. Circularity risk.

### Operational (when ready to bet)

- **Backtesting framework:** Replay probability model against resolved markets. Measure calibration, Brier scores, retroactive P&L. The backtest notebooks provide a foundation but need the probabilistic model plugged in.
- **Automated execution pipeline:** Model → edge detection → Kelly sizing → order placement via Kalshi API.
- **Credential handling:** DATABASE_URL and Kalshi API keys via environment variables, never committed.

---

## 8. Priority Matrix

| # | Item | Impact | Effort | Status |
|---|------|--------|--------|--------|
| 2.1 | Poisson-binomial betting function | Very High | Medium | **Complete** |
| 1.1 | High-frequency score polling | High | Medium | **Next build** |
| 1.2 | Kalshi API client | High | Low-Medium | Not started (needs credentials) |
| 3.1 | Per-critic KDE lambda model | Very High | Medium-High | **Complete** |
| 3.2 | Other parameter refinements | Medium | Ongoing | After §3.1 |
| 5.1 | Kalshi fee schedule | Medium | Low | Open |
| 7.x | Backtesting framework (probabilistic) | Very High | Medium | After §1 |
