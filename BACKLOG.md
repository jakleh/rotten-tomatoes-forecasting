# Backlog

## 1. Data Infrastructure

Things that expand what we can observe. Build these first — everything else depends on having richer data.

### 1.1 Historical review database backfill

Backfill the Neon PostgreSQL database with review data for all ~141 movies that had Kalshi RT markets, plus a broader set across release tiers. This unblocks nearly every exploration item below.

**Approach:** Extend the existing RT scraper (in `~/Desktop/rotten-tomatoes-analysis/`) to do a one-time historical pass. It already handles parsing and deduplication. Timestamps will be day-level precision for backfilled reviews (`timestamp_confidence` = "d").

**Status:** In progress — scraper is being expanded to cover all ~141 movies.

### 1.2 Local DB dump for joint analysis

Dump the reviews table locally so price histories and review data can be joined by `movie_slug` without hitting Neon. Folder names in `rt-price-histories/` match DB slugs, so the join key is direct — no mapping table needed. Consider DuckDB for SQL over local files without running a server.

### 1.3 High-frequency score polling

Between scraper runs (50-minute gaps), RT updates the displayed score in real time. Scraping just the displayed Tomatometer number every 1-5 minutes would detect review arrivals in near-real-time. Cheap to build — it's one number per movie, not a full page parse.

### 1.4 Final trading volume for resolved markets

Collect the final cumulative trading volume for each of the ~141 resolved Kalshi RT markets. This is visible on the Kalshi UI for closed bets. One-time manual or scripted capture. Gives us a real activity measure per market (stronger than the minutes-with-price-activity proxy from the price CSVs).

### 1.5 Kalshi volume scraper (live markets)

For active/future markets, periodically scrape cumulative trading volume from the Kalshi UI. A capture at embargo lift gives a useful feature: how much money is in the market before any reviews exist. This is a proxy for pre-release attention/hype and may correlate with total review count (see exploration item 2.10).

---

## 2. Ideas to Explore

Things to brainstorm, observe in the data, and gut-check. Framed as questions, not implementations. Items with substantial prior thinking have brainstorm files linked; short items are self-contained.

### Review-counting model ideas

#### 2.1 Does sentiment overdispersion matter?

Real critics aren't independent coins — sentiment clusters by publication, screening event, and timing. The binomial may underestimate variance. **Gut-check:** Measure the actual variance of fresh counts in rolling windows across movies and compare to binomial prediction. If the overdispersion is small, this doesn't matter.

See SOURCES.md §3.1 (beta-binomial references) and §4 (empirical overdispersion test).

#### 2.2 How sensitive is Lambda to the rate model?

The Poisson/binomial model's Lambda comes from KDE extrapolation, which is fragile (designed for interpolation, not forecasting). **Gut-check:** For resolved movies, compute Lambda via KDE at various points in time and compare to the actual review count that materialized. How wrong is it, and does the error correlate with anything observable?

#### 2.3 Do review arrivals actually look Poisson?

Reviews arrive in bursts (embargo lifts, wide release waves, scraper batching artifacts). This means variance > mean, violating the Poisson assumption. **Gut-check:** Plot observed review count distributions across forecast windows and compare to Poisson. If overdispersed, negative binomial may be needed.

#### 2.4 Is the p_fresh estimate fragile enough to matter?

The notebook uses a 7-day window or overall score fallback. A Bayesian Beta posterior with time-weighting would be smoother. **Gut-check:** How much does p_fresh actually vary depending on the estimation method? If the difference is < 2 percentage points, this isn't worth the complexity.

#### 2.5 Can other movies' review rates predict this movie's?

Instead of extrapolating from one movie's KDE, use actual review counts from similar historical movies in the same time window. **Gut-check:** Regress review_count_in_window on features (current total, days since peak, release type) across movies. See which features predict.

Full design: `brainstorm/brainstorm_cross_movie_lambda.md`

#### 2.6 Does pre-forecast freshness predict forecast-window freshness?

The hierarchical p_fresh model uses cross-movie data to predict how freshness evolves in the final forecast window. **First step:** Diagnostic regression — p_forecast ~ f(p_pre, n_reviews). If the scatter plot is a noisy mess, the hierarchical model isn't worth building.

Full design: `brainstorm/brainstorm_hierarchical_p_fresh.md`, `brainstorm/brainstorm_time_varying_p_fresh.md`

#### 2.7 How much noise do post-publication review edits introduce?

Backfilled reviews capture state at scrape time, not publication time. Sentiment flips are rare but could matter near thresholds. **Gut-check:** Compare sentiment for the same reviews across multiple scrapes of tracked movies to estimate the flip rate. Almost certainly immaterial, but worth confirming.

Full design: `brainstorm/brainstorm_review_integrity.md`

### Market-based strategy ideas

#### 2.8 Forecast-score divergence near close

Market prices don't fully reflect the actual score even right before bet close, especially in low-volume markets (e.g., The Housemaid: ~86% forecast, <75% actual score). **Gut-check:** For each of 141 movies, identify when the actual score was "knowable" from reviews, compare to market price at that time. Compute edge in cents and how long it persisted. Segment by market activity level.

Full design: `brainstorm/brainstorm_market_strategies.md`

#### 2.9 Embargo-lift divergence strategy

After reviews start, compare the initial review basket to the Kalshi market price. When they diverge, the basket may be more informative. **Gut-check:** Across 141 movies, how often does the initial basket (first 20 reviews) disagree with the market, and who's right?

Full design: `brainstorm/brainstorm_market_strategies.md`

#### 2.10 Volume-review count correlation and ceiling strategy

If trading volume correlates with total review count, you can estimate a review ceiling early and compute worst-case scores. **Gut-check:** Use minute-row count in price CSVs as activity proxy. Plot against total reviews (once DB has all movies). Is there structure?

Full design: `brainstorm/brainstorm_market_strategies.md`

#### 2.11 Price trace anomaly detection (amateur arbitrage)

Detect amateur-driven price movements (spikes without new information) and bet on reversion. **Gut-check:** For resolved markets, measure mean reversion speed after price spikes that coincide with no new reviews. How far and how fast does it revert?

Full design: `brainstorm/brainstorm_market_strategies.md`

#### 2.12 Market price as a leading signal

Between scraper runs, someone else may see a new review and trade. The price moves but our model hasn't updated. **Caution:** circularity risk — the market price should inform confidence (how much to bet), not direction (which side). Our structural model drives direction independently.

### Structural model ideas

#### 2.13 Finite-pool / remaining-reviewer model

The critic pool is finite. Once a critic has reviewed, they're removed. This means the arrival rate has a hard ceiling and p_fresh depends on who's *left*, not who's already reviewed. Replaces Binom(n, p) with a Poisson binomial (per-reviewer p_i). Hardest part: defining the pool for a given movie.

Full design: `brainstorm/brainstorm_finite_pool_model.md`

#### 2.14 Reviewer graph model

Model critics as graph nodes with edges encoding influence, co-occurrence, and correlation. Predicts *who* reviews next and *what* they'll say. Replaces i.i.d. Bernoulli with structurally-aware conditional distributions.

Full design: `brainstorm/brainstorm_reviewer_graph.md`

#### 2.15 Early score trajectory forecasting

Match a movie's partial trajectory to a library of historical trajectories (k-nearest or latent model). Forecast terminal score distribution directly, without Poisson/binomial machinery. Most useful early in lifecycle (< 30 reviews) when the score can still swing 10+ points.

#### 2.16 Full score PMF and range betting

Any model that produces a full PMF over final scores (not just P(break) for one threshold) can evaluate every Kalshi market simultaneously. Range bets (selling volatility across two thresholds) are a natural application. Position sizing via Kelly criterion, adjusted for slippage and fees.

Full design: `brainstorm/brainstorm_score_pmf_range_betting.md`

#### 2.17 Clustering movies by review rate / market size

Group movies by (total reviews at close) vs. (reviews in last 48h). Use cluster membership to pick better rate parameters. Could replace Poisson entirely: "no movie with 120+ reviews at this point had more than X reviews in the final 48h" = direct ceiling without parametric assumptions.

Full design: `brainstorm/brainstorm_market_strategies.md`

---

## 3. Platform Mechanics

Factual things to look up. Model-independent — needed regardless of which approach we pursue.

### 3.1 Compare model output to market price

Any model output is only useful if compared to the Kalshi market price. Edge = model price - market price (adjusted for slippage and fees). This is a universal need, not specific to any model.

### 3.2 RT rounding rules

How does RT round the displayed Tomatometer? (Round half up? Truncate? Banker's rounding?) Kalshi resolves against the *displayed* score, so the rounding rule determines the outcome near boundaries.

See SOURCES.md §2.1.

### 3.3 Top critic vs. all critic distinction

RT has separate scores. Some Kalshi markets may resolve on one or the other. Need to check resolution rules.

See SOURCES.md §1.2.

### 3.4 Kalshi fee schedule

Per-contract trading fee and any settlement fees. Required for realized edge calculation.

See SOURCES.md §1.1.

### 3.5 Kalshi resolution rules

Exact resolution language: displayed score or exact fraction? All critics or top critics? Score at market close or a snapshot? What if RT changes the score after close?

See SOURCES.md §1.2.

---

## 4. Operational

For when we're ready to bet. Not the current phase.

### 4.1 Backtesting framework

Replay any strategy against resolved markets. For each market at each point in time: what did our approach say, what did the market say, what actually happened? Measure calibration, Brier scores, retroactive P&L. This is the single most important thing before risking real money.

**Reframing note:** Backtesting applies to whatever strategies survive the brainstorm/gut-check phase — it's not specific to the Poisson/binomial model. The framework should be modular: plug in any strategy, get calibration metrics out.

### 4.2 Automated execution pipeline

Close the loop: model -> edge detection -> Kelly sizing -> order placement via Kalshi API. MVP: model + API client + comparison loop that prints edges for human review. Automation comes later once calibration is trusted.

### 4.3 Credential handling

- `DATABASE_URL`: Pass via environment variable, never commit. Notebook tracebacks can leak it — use nbstripout or clear outputs before committing.
- Kalshi API keys (future): Same treatment. Keys that can place orders = bank password.

---

## 5. Data Sources to Investigate

### 5.1 Kalshi API / data exports
- **Price history** — downloaded for ~141 markets (in `rt-price-histories/`). Done.
- **Order book depth** — needed for slippage computation. Full book, not just best bid/ask.
- **Fee schedule** — see §3.4.
- **Market metadata** — resolution rules, thresholds, settlement times.
- **Resolution outcomes** — final results for backtesting labels.

### 5.2 Rotten Tomatoes (additional scraping)
- **High-frequency Tomatometer** — see §1.3.
- **Audience score** — may correlate with critic sentiment direction.
- **"Critics Consensus" text** — editorial summary changes may signal framing shifts.

### 5.3 External
- **Embargo lift dates** — sometimes announced on social media / press sites.
- **Festival/press screening dates** — leading indicators for review timing.
- **Social media sentiment** (Letterboxd, Twitter/X) — audience reactions from early screenings.
- **Box office tracking** (Box Office Mojo, The Numbers) — tracking numbers correlate with review activity.
- **Metacritic scores** — divergence between RT and Metacritic signals polarized reception.

---

## 6. Priority Matrix

Current phase is data infrastructure + informal observation. Priorities reflect that.

| # | Item | Category | Impact | Effort | Status/Dependencies |
|---|------|----------|--------|--------|---------------------|
| 1.1 | Historical review database backfill | Infrastructure | Very High | Medium | In progress |
| 1.2 | Local DB dump for joint analysis | Infrastructure | High | Low | Blocked on 1.1 |
| 2.8 | Forecast-score divergence (systematic backtest) | Exploration | Very High | Medium | Needs 1.2 |
| 2.10 | Volume-review count correlation | Exploration | High | Low | Can start now (price CSVs + activity proxy) |
| 2.9 | Embargo-lift divergence | Exploration | High | Medium | Needs 1.2 |
| 2.11 | Price trace anomaly detection | Exploration | High | Medium | Can start now (price CSVs only) |
| 1.4 | Kalshi volume scraper | Infrastructure | Medium | Medium | Independent |
| 1.3 | High-frequency score polling | Infrastructure | Medium | Medium | Independent |
| 3.2 | RT rounding rules | Platform | Low | Low | Independent |
| 3.3 | Top critic distinction | Platform | Low | Low | Independent |
| 3.4-3.5 | Kalshi fees + resolution rules | Platform | Medium | Low | Independent |
| 4.1 | Backtesting framework | Operational | Very High | Medium | After explorations mature |
