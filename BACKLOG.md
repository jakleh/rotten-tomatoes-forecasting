# Backlog

## 1. Current Priority: Poisson-Binomial Betting Function

The surviving approach after dataset survey, systematic backtest, and deep-dive analysis. Reviews accumulate with inertia; the market lags behind what a probability model can infer from the review count. A deterministic bounds approach was too conservative (see exploration history below). The probabilistic Poisson-binomial model captures edges earlier.

### 1.1 Build the betting function

**Inputs (7):**
- `threshold` — Kalshi threshold (e.g., 75 for "Above 75")
- `market_price` — current price in cents
- `fresh_count` — current number of positive reviews (from DB)
- `total_count` — current number of total reviews (from DB)
- `hours_to_close` — time remaining until bet close
- `lambda_rate` — expected reviews per hour in the remaining window (Poisson parameter)
- `p_fresh` — probability each future review is positive (binomial parameter)

**Output:** Edge in cents. Positive = profitable bet, negative = pass.

**Method:** For each possible number of additional reviews `k` (Poisson-distributed with rate `lambda_rate * hours_to_close`), and for each possible number of fresh reviews among those `k` (binomial with `p_fresh`), compute the resulting final score and check if it crosses the threshold. Sum up: `P(resolve Yes)` and `P(resolve No)`. Edge = `P(Yes) * (100 - price) - P(No) * price`.

**Starting simple:**
- `lambda_rate`: reviews in last 6 hours / 6 (flat recent rate)
- `p_fresh`: current `fresh_count / total_count` (running average)
- Fresh/total counts: query the existing Neon PostgreSQL DB (populated by the RT scraper every 50 minutes)

**Status:** Not started. This is the immediate next build.

### 1.2 High-frequency score polling

Detect review arrivals between scraper runs (50-minute gaps) by polling the displayed Tomatometer score every 1-5 minutes. One number per movie, not a full page parse.

**Why it's now a priority:** Lambda estimation needs to know when reviews arrive. The scraper gives 50-minute resolution; the poller gives 1-5 minute resolution. Also enables future backtesting of the probability model with real-time data.

**Status:** Not started.

### 1.3 Kalshi API client

Fetch live prices for all RT markets. No order placement — just price retrieval. Foundation for comparing model output to market prices and for the eventual execution pipeline.

**Status:** Not started. Need API credentials and docs (see SOURCES.md §1.4, §1.5).

---

## 2. Data Infrastructure (Complete)

### 2.1 Historical review database backfill

**Status: Complete.** All ~141 movies backfilled into Neon PostgreSQL.

### 2.2 Local data dump + movies index

**Status: Complete.** `reviews.csv` (23K+ reviews) and `movies_index.csv` (141 movies with volume, dates, score ranges, review counts) at project root. See `plans/plan_populate_movies_index.md` for details.

### 2.3 Trading volume for resolved markets

**Status: Complete.** Recorded in `movies_index.csv` from Google Sheet.

---

## 3. Platform Mechanics

### 3.1 Kalshi fee schedule

Per-contract trading fee and any settlement fees. Required for realized edge calculation (net edge = gross edge - fees).

**Status:** Open. See SOURCES.md §1.1.

### 3.2 RT rounding rules

**Status: Resolved.** RT uses standard rounding (round half up). Kalshi resolves against the displayed score. See `brainstorm/brainstorm_rounding_and_resolution.md`.

### 3.3 Kalshi resolution rules

**Status: Resolved.** "Above X" = displayed score >= X+1. Snapshot at 10:00 AM ET on expiration date. All Critics Tomatometer. See `brainstorm/brainstorm_rounding_and_resolution.md`.

---

## 4. Exploration History

Completed explorations that informed the current approach. Notebooks and findings preserved for reference.

### 4.1 Dataset survey (Complete — 2026-04-06)

Broad survey of the joint dataset. Found 25 misprices across 23/141 movies with median 57¢ edge. Edge does NOT concentrate in low-volume markets. Score is knowable early (93%+ reviews in by T-24h). Top critics are systematically ~6pp more negative.

**Notebook:** `notebooks/dataset_survey.ipynb` · **Findings:** `brainstorm/brainstorm_dataset_survey_findings.md`

### 4.2 Systematic misprice backtest (Complete — 2026-04-06)

Tested deterministic worst-case/best-case bounds across 141 movies. Found 74 episodes across 52 movies but median edge only 9¢. The biggest survey misprices (Joker 97¢, Snow White 61¢) fell in a 20-movie data-quality gap where review counts don't match ground truth (day-level timestamp noise). 33 lock/resolution mismatches traced to this cause.

**Conclusion:** Deterministic bounds too conservative — by the time they lock, the market has already corrected. Probabilistic approach needed.

**Notebook:** `notebooks/misprice_backtest.ipynb` · **Plan:** `plans/plan_misprice_backtest.md`

### 4.3 Deep dive on clean-data movies (Complete — 2026-04-06)

Focused analysis on Forbidden Fruits and They Will Kill You — the two resolved movies with minute/hour-level timestamps in the critical 48h window. Max edge was 7¢ (Forbidden Fruits "Above 80") and 4¢ (They Will Kill You "Above 90"). Market was well-calibrated by the time bounds locked.

**Confirmed:** The edge is real but you need a probabilistic approach to capture it before the market self-corrects.

**Notebook:** `notebooks/misprice_backtest_deep_dive.ipynb`

### 4.4 Data quality findings

- 20/141 movies have review-implied final scores that don't match the movies_index score range (derived from market resolution). Root cause: day-level timestamps attribute reviews to the wrong side of the 10 AM ET close cutoff.
- Only 6 movies have significant minute-level timestamp data: `the_drama` (145m), `the_super_mario_galaxy_movie` (162m), `they_will_kill_you` (42m), `forbidden_fruits_2026` (17m), `project_hail_mary` (25m), `ready_or_not_2_here_i_come` (23m).
- The Drama and Super Mario Galaxy have 99-100% minute+hour coverage but are still active (no resolved Kalshi markets yet, no price histories).

---

## 5. Future Ideas

Ideas from brainstorming that survived initial gut-checks but are not the current priority. Brainstorm files preserved in `brainstorm/` (gitignored).

### Parameter refinement (feeds into §1.1)

- **Cross-movie lambda:** Use historical movies' review rates to predict this movie's rate. `brainstorm/brainstorm_cross_movie_lambda.md`
- **Hierarchical p_fresh:** Use cross-movie regression as a prior for freshness rate. `brainstorm/brainstorm_hierarchical_p_fresh.md`
- **Time-varying p_fresh:** Freshness rate may shift over a movie's lifecycle. `brainstorm/brainstorm_time_varying_p_fresh.md`
- **Top-critic correction:** Early reviews overweight top critics who are ~6pp more negative. Adjust p_fresh accordingly.
- **Overdispersion (beta-binomial):** If sentiment is clustered (not i.i.d.), the binomial underestimates variance. `brainstorm/brainstorm_poisson_binomial_threshold.md`

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

## 6. Priority Matrix

| # | Item | Impact | Effort | Status |
|---|------|--------|--------|--------|
| 1.1 | Poisson-binomial betting function | Very High | Medium | **Next build** |
| 1.2 | High-frequency score polling | High | Medium | Not started |
| 1.3 | Kalshi API client | High | Low-Medium | Not started (needs credentials) |
| 3.1 | Kalshi fee schedule | Medium | Low | Open |
| 5.x | Parameter refinement (lambda, p_fresh) | High | Ongoing | After §1.1 |
| 5.x | Backtesting framework (probabilistic) | Very High | Medium | After §1.1 |
