# Analysis Backlog

## 1. Holes in the Current Model

### 1.1 Independence assumption on sentiment (Binomial)

The inner loop models each new review as an independent Bernoulli(p_fresh) draw. Real critics aren't coins:

- **Publication-level correlation.** If the NYT pans a film, the Washington Post review (often written by someone in the same screening) is not independent. Sentiment clusters by publication tier, geography, and screening event.
- **Temporal correlation.** Early embargo-lift reviews skew toward outlets with advance access (often friendlier to studios). Late-arriving reviews skew toward smaller outlets whose sentiment distribution may differ. p is not stationary across the forecast window — it shifts as the *type* of reviewer changes.
- **Sentiment contagion.** Critics read each other. A high-profile pan can shift the Overton window for subsequent reviews. The probability of the 51st review being fresh is not independent of the 50th.

**Impact:** The binomial likely underestimates the variance of the fresh count. The true distribution of (fresh out of n) has fatter tails than Binom(n, p) — meaning extreme outcomes (all fresh or all rotten batches) are more likely than the model predicts. This matters most when the score is near the threshold.

**Possible fix:** Replace the binomial with a beta-binomial, where p itself is drawn from Beta(a, b). The extra dispersion parameter captures the "reviews come in correlated clusters" effect. Fit a and b from historical movie data by measuring the overdispersion of fresh counts in rolling windows.

**Literature:** See `SOURCES.md` §3.1 (beta-binomial references) and §4 (empirical overdispersion test we can run on our own data).

### 1.2 KDE extrapolation into the future

Gaussian KDE is an interpolation tool. We're fitting it on [0, t_now] and evaluating it on [t_now, t_end]. The KDE has no concept of "the future" — it places a Gaussian bump around every observed arrival time and sums them. When we evaluate past the last data point, we're riding the tail of the rightmost kernel bumps.

- **If the movie is past its review peak**, the KDE tail may still carry substantial mass — overestimating Lambda.
- **If there's an upcoming event** (wide release, awards season), the KDE has no way to anticipate a rate increase it hasn't seen yet.
- **Bandwidth sensitivity.** Silverman's rule was designed for unimodal densities. Review arrivals are often multimodal (embargo lift spike, wide release spike, awards bump). The bandwidth may oversmooth or undersmooth these modes, and the error propagates directly into Lambda.

**Impact:** Lambda is the single number that drives the entire downstream calculation. If it's wrong by 2x, everything shifts.

**Possible fixes:**
- Use a parametric rate model (e.g., sum of log-normals for each known event) instead of KDE.
- Fit a time-series model (exponential decay from peak, with change-point detection for new spikes).
- Ensemble: average the KDE estimate with the simple "reviews in last 48h" count, weighted by how far past peak we are.
- **Cross-movie empirical Lambda (see 1.8):** Instead of extrapolating from this movie's KDE, use the actual review counts from the same forecast window of similar movies. This sidesteps the KDE extrapolation problem entirely.

**Literature:** See `SOURCES.md` §3.5 (KDE bandwidth for multimodal data), §3.6 (Hawkes processes as a NHPP alternative).

### 1.8 Cross-movie empirical Lambda

The current model derives Lambda entirely from the target movie's own review history via KDE. This is a single-trajectory estimate that must extrapolate into the future. An alternative: use the *actual observed* review counts from the equivalent time window of other movies as a direct empirical estimate of Lambda.

**Approach:**

For a target movie with a 48h forecast window closing at time T:
1. Identify a cohort of "similar" historical movies.
2. For each cohort movie, count how many reviews arrived in the 48h window before *its* equivalent close time (or the same lifecycle stage).
3. The distribution of those counts across the cohort *is* the distribution of N — no Poisson assumption needed. Or, fit a Poisson (or negative binomial) to the cohort counts to get a parametric Lambda estimate.

This replaces KDE extrapolation with direct empirical observation: "movies like this one got 3-8 reviews in their final 48h."

**What makes movies "relevantly similar" for Lambda estimation:**

This is the key open question. Candidate similarity features, roughly ordered by suspected importance:

1. **Review count at the same lifecycle stage.** A movie with 200 reviews at 5 days before close probably behaves like other movies with ~200 reviews at ~5 days out. This may be the single strongest predictor — it captures both the movie's overall "size" and where it is in the decay curve.
2. **Time since first review / time since peak.** The rate decay pattern may be fairly universal. A movie 3 days past its review peak may trickle at a similar rate to any other movie 3 days past peak, regardless of genre.
3. **Recent rate trajectory.** Is the rate currently decelerating, flat, or re-accelerating? A movie in steady decay behaves differently from one experiencing a second wave.
4. **Release type / scale.** Wide release vs. limited vs. platform release determines how many outlets cover the film and how long the tail lasts.
5. **Genre.** May matter less for *rate* than for *sentiment*. An indie horror and a blockbuster sequel might trickle at similar rates once they're past peak, even though their total review counts differ.
6. **Studio / distributor.** Major studios may generate longer review tails due to PR activity.

**Empirical approach to answering "what matters":**

Rather than guessing which features matter, measure it:
1. Build a dataset of (movie, window_start, window_length, review_count_in_window) across many movies and time windows.
2. Regress review_count_in_window on candidate features (current total reviews, days since first review, days since peak, genre, release type, etc.).
3. See which features have predictive power. Features that don't help predict the rate can be dropped from the similarity definition.

This regression could be as simple as a linear model or as flexible as a random forest — the goal is feature selection, not a production model.

**Relationship to other approaches:**

- **vs. KDE (current):** Cross-movie Lambda uses real data from the right time window instead of extrapolating a density estimate. Stronger when we have enough cohort movies; weaker when the target movie is unusual (no good matches).
- **vs. finite-pool model (2.5):** Complementary. The finite-pool model gives a structural ceiling on how many reviews are possible (based on who's left in the pool). Cross-movie Lambda gives an empirical base rate for how many actually show up. The finite-pool ceiling can truncate the cross-movie distribution.
- **vs. Hawkes process (`SOURCES.md` §3.6):** Hawkes captures within-movie burst dynamics (embargo lift → cascade). Cross-movie Lambda captures between-movie rate patterns (movies at this stage tend to get X reviews). An ensemble of both would be ideal.

**Data requirements:**
- Historical review data across many movies (not just the ones we're currently tracking). Needs enough movies to build meaningful cohorts.
- Release metadata: release date, release type (wide/limited), genre, studio. Needed for similarity features.
- The scraper DB accumulates this naturally over time, but we may need to backfill historical movies to bootstrap the cohort. See `SOURCES.md` §2.3 and §2.4.

**Pending sources:** See `SOURCES.md` §2.3 (typical review counts by movie type), §2.4 (review lifecycle timing). Also need: a dataset of release dates and types for historical movies to compute lifecycle-stage features.

### 1.3 Poisson assumption (independent arrivals)

The NHPP assumes that, conditional on the rate λ(t), arrivals are independent. But reviews arrive in bursts:

- **Embargo lifts** drop 10-30 reviews within an hour.
- **Scraper batching** means we observe reviews in 50-minute clusters, injecting artificial burstiness into `estimated_timestamp`.
- **Wide release** triggers a second wave.

These bursts mean the actual variance of the review count exceeds the Poisson's mean = variance property (overdispersion).

**Impact:** Like the binomial issue, the Poisson underestimates the probability of extreme counts. P(N=0) and P(N >> Lambda) are both higher than the model thinks.

**Possible fix:** Use a negative binomial distribution for N instead of Poisson. The NB has a dispersion parameter that lets variance > mean. Fit it from historical count data across movies.

### 1.4 p_fresh estimation is fragile

The notebook uses a 7-day window (if ≥5 reviews) or falls back to the overall score. Problems:

- **7 days is arbitrary.** If the movie is in week 1, this is the entire history. If it's in week 6, the last 7 days might have 2 reviews.
- **Small-sample noise.** With 5-10 recent reviews, the sample proportion has high variance. A single rotten review shifts p by 10-20%.
- **No shrinkage toward a prior.** A Bayesian estimate (Beta posterior) would regularize small samples toward the overall score instead of hard-switching between two point estimates.

**Possible fix:** Beta(fresh + 1, rotten + 1) posterior using all reviews, optionally with exponential time-weighting so recent reviews count more. This is smooth, handles small samples, and degrades gracefully.

### 1.9 Time-varying p_fresh via hierarchical cross-movie model

**Core idea:** Replace the static p_fresh with a time-varying p_fresh(t) that uses cross-movie data as the *base prediction*, not just a conservative floor. The key insight: if we have data from many historical movies, that data should inform what we expect p_fresh to be during the forecast window — and the current movie's own reviews then update that expectation.

**The three-level structure:**

**Level 1 — Within-movie relationships:** For each historical movie m, observe:
- p_pre(m) = freshness rate during the pre-forecast window (e.g., [bet_open, forecast_start], typically ~1 week)
- p_forecast(m) = freshness rate during the forecast window (e.g., [forecast_start, bet_close], typically 48h)
- The "relationship" is how one maps to the other: did freshness hold steady, drift down, spike up?

**Level 2 — Across-movie patterns:** Collect (p_pre(m), p_forecast(m)) pairs across all historical movies. This reveals the population-level pattern: how does pre-forecast freshness relate to forecast-window freshness? Is there systematic drift? Does it depend on features (review count, score level, genre, lifecycle stage)? This is a regression surface: E[p_forecast | p_pre, features].

**Level 3 — Prediction for current movie:** The current movie has an observed p_pre. Plug it into the Level 2 regression to get a predicted p_forecast with uncertainty. This is the base prediction — informed by all historical movies, not just this one.

This is a **hierarchical model**: the population-level distribution of freshness trajectories (Level 2) provides an informed prior for any individual movie. The individual movie's own data (Level 1) then updates toward its specific trajectory.

**Implementation roadmap:**

**Step 1 — Diagnostic regression (do this first):**
Before building any sophisticated model, answer: does pre-forecast freshness actually predict forecast-window freshness? For each historical movie, compute (p_pre, p_forecast, n_reviews_at_forecast_start). Fit a simple regression: p_forecast ~ f(p_pre, n_reviews). Plot p_pre vs. p_forecast. This tells you whether the signal exists at all:
- Tight cloud along the diagonal → p_fresh is roughly stationary, fancy time-varying stuff has limited upside.
- Systematic drift (e.g., p_forecast < p_pre on average) → the signal the model needs to capture.
- Noisy mess with no pattern → cross-movie data won't help much; rely on the individual movie's own reviews.

**Step 2 — Hierarchical Bayesian model (if Step 1 shows signal):**
Use the cross-movie regression as the *prior* for a Beta-Binomial model of the current movie:
- The Level 2 regression gives E[p_forecast | p_pre, features] and its uncertainty → convert to Beta(α₀, β₀) prior parameters.
- Update with the current movie's recent reviews using exponential forgetting: α(t) = α₀ + Σ y_i · exp(-γ(t - t_i)), β(t) = β₀ + Σ (1 - y_i) · exp(-γ(t - t_i)).
- One hyperparameter (γ) controls how fast old reviews fade.
- When movie-specific data is thin, the prediction defaults to the cross-movie prior. When strong, it overrides.

**Step 3 — Trajectory feature enrichment (if enough movies):**
Enrich the feature set for the Level 2 regression using trajectory-shape features:
- **Functional PCA (FPCA):** Treat each movie's freshness-over-time curve as a function. FPCA finds the principal modes of variation (e.g., PC1 = overall level, PC2 = trend direction, PC3 = volatility). Project the current movie's partial trajectory onto this basis to get features. Python: `scikit-fda`.
- **Latent trajectory clustering:** Cluster movies into discrete trajectory types ("starts high, stays high" vs. "starts high, decays" vs. "volatile"). Assign the current movie to a cluster and use that cluster's forecast behavior as the prediction. More interpretable than FPCA.
- These are exploratory/preprocessing tools — they enrich the regression inputs, not replace the model itself.

**Conditioning variables for Level 2 regression:**
- **Review count at forecast start** — likely the strongest predictor. 200 reviews won't drift; 30 can swing wildly.
- **Current score relative to threshold** — different regimes for margin of safety.
- **Time since release** — early-lifecycle vs. long-tail movies drift differently.
- **p_pre itself** — the pre-forecast freshness rate.
- **FPCA scores or cluster assignment** (Step 3) — trajectory shape features.

**Conservative floor (supplementary, not primary):**
The conditioned drift distribution from Level 2 also provides a conservative bound. Use a quantile (e.g., 5th percentile of p_forecast among movies with similar features) rather than the raw worst case (a single order statistic that gets more extreme as sample grows). This floor is useful for risk management but is a byproduct of the model, not a separate component.

**Drift should be measured in absolute percentage points**, not as a ratio. A movie at 95% drifting to 94% (ratio 0.989) and a movie at 55% drifting to 50% (ratio 0.909) are very different as ratios but both represent small absolute movements. Thresholds are on an absolute scale, so absolute drift is what matters.

**Other mathematical methods to explore for p_fresh(t):**

Several formal methods could model p_fresh as a function of time. These range from simple to heavy and could substitute for or enhance the Beta-Binomial component:

- **Logistic regression with time features:** Model log-odds(p_fresh) = f(t). Linear f(t) captures monotonic drift; spline basis functions (GAM-style) capture nonlinear patterns without hand-picking polynomial degree. Natural choice for binary outcome modeling.
- **Kernel-smoothed rate (Nadaraya-Watson):** p_fresh(t) = Σ K((t - t_i)/h) · y_i / Σ K((t - t_i)/h). Smooth local average — same spirit as the KDE for lambda(t). Poor at extrapolation.
- **Gaussian Process on the logit scale:** GP prior on logit(p_fresh(t)) with Bernoulli observations. Gives full posterior with uncertainty bands and minimal structural assumptions. Heavier machinery; may not buy much over Beta-Binomial given data sparsity.
- **Changepoint-aware Beta-Binomial:** Bayesian online changepoint detection (Adams & MacKay 2007) for the Beta posterior. Instead of smooth exponential forgetting, detects sudden regime shifts in p_fresh (e.g., a wave of negative reviews after controversy). More complex but handles discontinuities.

**Data requirements:**
- Historical review data across many resolved movies (the backfill — see 2.10).
- Kalshi resolved market data with close times (for aligning forecast windows).
- Enough movies to stratify by features. Minimum ~30-50 movies for meaningful conditioning; more is better for Step 3.

**Relationship to other items:**
- **1.4 (Bayesian p_fresh):** This subsumes 1.4. The hierarchical model is a strict upgrade over the static Beta posterior proposed there.
- **1.1 (Beta-binomial for overdispersion):** Complementary. 1.1 addresses overdispersion in the *count* of fresh reviews for a given n. This item addresses the *estimation of p itself*.
- **2.8 (Backtesting):** The cross-movie component depends on having resolved market data. Build 2.8 first, then use the same dataset here.
- **2.10 (Database backfill):** Needs a broad movie dataset. Blocked on backfill.

### 1.10 Review-integrity uncertainty in backfilled data

Backfilled reviews capture the review state at scrape time, not at the time of original publication. RT critics can edit their reviews after publishing — changing their subjective score (e.g., 3/5 → 4/5) or, more rarely, flipping their tomatometer sentiment (Fresh ↔ Rotten). This means our historical "score at time T" reconstruction may be slightly off: we're assuming the sentiment recorded in the backfill was the sentiment at the time the review was originally published, but it may have been edited between publication and scrape.

**What changes and what doesn't:**

- **Subjective score edits** (more common): Do not affect the tomatometer. The tomatometer is derived from the binary Fresh/Rotten sentiment, not the numeric score. These edits are noise we can ignore for tomatometer modeling.
- **Sentiment flips** (rare): A critic changes their review from Fresh to Rotten or vice versa. This *does* affect the tomatometer. One flip on a movie with N total reviews shifts the percentage by 1/N (one review changes sides) or 2/N (if the flip also changes the denominator boundary — but it doesn't; the review count stays the same). For N = 100, that's a 1 percentage point shift. For N = 200, it's 0.5 pp.

**Impact on backtesting and modeling:**

The concern is that our reconstructed historical tomatometer trajectory (used in 2.8 backtesting, 1.9 drift calibration, 2.3 trajectory forecasting) may contain small errors from post-publication edits. If we compute "the score was 87% at hour T" but a review had been flipped since then, the true score at hour T was 86% or 88%.

- **Backtesting (2.8):** If the model's predicted probability was based on a slightly wrong historical score, the calibration analysis inherits that noise. On any single market this is negligible, but systematic bias (e.g., critics tend to flip toward Rotten more than toward Fresh) could accumulate.
- **Cross-movie models (1.9, 2.3):** Models trained on historical score trajectories absorb this noise into their training data. If the noise is small and unbiased, it washes out. If there's a directional tendency, it biases the learned relationships.

**Materiality assessment:**

Likely small, but should be quantified rather than assumed away:

- Sentiment flips are rare by all available evidence (critics occasionally revise scores but rarely change their binary recommendation).
- On movies with 100+ reviews, even multiple flips move the tomatometer by low single-digit percentage points.
- The model's forecast uncertainty (from Poisson count variance + binomial sentiment variance) is typically much larger than the data-integrity noise — the signal-to-noise ratio of the edit problem is low.
- **However:** Near thresholds (e.g., the true score is 89.5%), even a 1 pp error can flip the binary outcome. This is exactly where we're betting, so even small systematic bias could matter at the margin.

**Approaches to quantify and account for it:**

1. **Sensitivity analysis.** For each backtesting scenario, perturb k reviews (flip their sentiment) and re-run the model. Measure how much the model's output probability changes. If flipping 1-3 reviews barely moves the needle, the uncertainty is immaterial. Sweep k from 1 to some reasonable upper bound.

2. **Noise model on historical sentiments.** Treat each backfilled sentiment as observed with probability (1 - ε) and flipped with probability ε, where ε is the estimated flip rate. Propagate this through the score calculation: the "true" fresh count at time T is not deterministic but has a distribution centered on the observed count with binomial noise from ε. This widens the confidence interval on historical scores.

3. **Confidence intervals on reconstructed scores.** Instead of treating the backfilled score at time T as exact, compute a confidence interval: score ± δ, where δ depends on the total review count and the assumed flip rate ε. Use these intervals in backtesting — if the model's prediction falls within the interval, it's consistent with the data even if the point estimate differs.

4. **Empirical flip rate estimation.** If we have multiple scrapes of the same movie over time (which we do for currently-tracked movies), compare the sentiment for the same `unique_review_id` across scrapes. Any change is a detected flip. This gives a direct empirical estimate of ε. For backfilled movies (scraped once), we can't detect flips, but we can apply the rate estimated from tracked movies.

5. **Propagation through models.** For models that take historical tomatometer percentage as input (1.9 regression, 2.3 trajectory matching), add the noise model as an observation-error term. Instead of conditioning on "score was exactly X% at time T," condition on "score was observed as X% ± δ." This is standard measurement-error modeling.

**Relationship to other items:**
- **2.8 (Backtesting):** This is a known source of error in backtesting. Quantify it as part of backtesting validation.
- **2.10 (Database backfill):** Backfilled data is the primary source of this uncertainty. Real-time tracked movies have less exposure (shorter gap between publication and scrape).
- **1.9 (Drift calibration):** The cross-movie regression absorbs this noise. If unbiased, it averages out; if biased, it shifts the learned drift.
- **2.3 (Trajectory forecasting):** Same — trajectory library inherits the noise.

**Priority:** Low effort to quantify (approach 1 or 4), and the answer likely confirms it's immaterial. Worth doing once backtesting is set up to put a number on it rather than hand-waving.

### 1.5 No consideration of Kalshi market price

The model outputs P(break) but the notebook doesn't compare it to the market price. The bet is only +EV if:

```
P(break) > market_implied_probability    (for buying "yes")
P(break) < market_implied_probability    (for buying "no")
```

Without this comparison, you know the probability but not whether there's an edge.

### 1.6 Score rounding / RT calculation edge cases

RT's Tomatometer displays a rounded integer percentage. The precise rounding rule (round half up? truncate? banker's rounding?) affects whether a calculated 89.5% displays as 89% or 90%. Kalshi resolves against the *displayed* score. If the bet threshold is 90% and the true ratio is 89.7%, the rounding rule determines the outcome. We don't model this.

**Pending sources:** See `SOURCES.md` §2.1 (RT rounding rule) and §1.2 (Kalshi resolution rules).

### 1.7 Top critic vs. all critic score distinction

RT has separate Tomatometer scores for "all critics" and "top critics." Some Kalshi markets may resolve against one or the other. The model doesn't distinguish — it treats all reviews equally. If the market resolves on the top-critic score, filtering to `top_critic = True` changes the base counts, p_fresh, and possibly the arrival rate.

**Pending source:** See `SOURCES.md` §1.2 (Kalshi resolution rules — which score do they use?).

---

## 2. New Strategies

### 2.1 Reviewer graph model

**Core idea:** Model critics as nodes in a graph. Edges encode influence, co-occurrence, and correlation. Use the graph structure to make better predictions about *who* will review next and *what* they'll say.

**Node features:**
- Historical fresh rate (per genre, per studio, overall)
- Typical review timing (days after release, days after embargo lift)
- Publication tier (top critic or not)
- Review volume (prolific vs. occasional)
- Sentiment volatility (always positive? highly variable?)

**Edge features / relationships:**
- **Co-screening correlation:** Critics who attend the same screenings (same embargo lift batch) have correlated sentiment. Detect these clusters from timestamp proximity.
- **Publication correlation:** Same-outlet critics rarely contradict each other on the same film.
- **Influence propagation:** If a high-profile critic publishes a pan, does the fresh rate of subsequent reviews drop? Measure this with lagged cross-correlation.
- **"Follows" pattern:** Some critics consistently review after a specific other critic (e.g., a niche outlet waits for the NYT take). Detect from historical ordering.

**External triggers to model:**
- **Embargo lifts** — the single biggest driver of review clustering. If we can detect or predict the embargo lift time, we can predict both the *count* and *sentiment profile* of the first batch.
- **Wide release date** — triggers a second wave of reviews from general-audience critics.
- **Awards/festival appearances** — can reactivate reviews for older films.
- **Controversy / viral moments** — can trigger a burst of reviews with atypical sentiment.

**How it improves the model:** Instead of a single p_fresh for all incoming reviews, you'd estimate a conditional distribution: "given who has already reviewed and when, the next k reviewers are most likely [these critics], whose sentiment on this genre is [this distribution]." This replaces the i.i.d. Bernoulli assumption with something structurally aware.

**Data requirements:**
- Historical review data across many movies (not just the current one) to build reviewer profiles.
- Embargo lift dates (may need to be inferred from the first review cluster timestamp).
- Publication metadata (tier, geography, typical genre coverage).

### 2.2 Kalshi price trace anomaly detection (amateur bettor arbitrage)

**Core idea:** The Kalshi market price is set by supply and demand. Sophisticated bettors (with models like ours) push the price toward the "true" probability. Amateur bettors push it away — briefly. If we can detect amateur-driven price movements in real time, we can bet against them and profit from the reversion.

**What amateur bettor activity looks like:**
- **Overreaction to single reviews.** A new high-profile rotten review drops; amateurs panic-sell "yes" contracts. The price dips below fair value, then reverts as sophisticated bettors buy.
- **Round-number clustering.** Amateurs anchor to the displayed Tomatometer percentage. If it shows 91%, they think "above 90% is safe" and overbuy, even if the margin is razor-thin.
- **Time-of-day patterns.** Retail bettor activity likely peaks during US evening hours. Institutional/model-driven activity may be more uniform.
- **Volume spikes without information.** A burst of volume with no new reviews suggests sentiment-driven trading, not information-driven.

**Strategy:**
1. **Store historical Kalshi price/volume data** at high frequency (minute-level or trade-level).
2. **Define "fair value"** at each point in time using our Poisson/binomial model (or an improved version).
3. **Detect divergences:** When market price deviates from model price by more than X%, flag it.
4. **Classify the divergence:** Is it driven by new information (a review dropped) or by noise (no new reviews, just trading activity)?
5. **If noise:** Bet on reversion. The price should return to fair value.

**Shape/trace features to extract:**
- Volatility in rolling windows (high vol without new reviews = noise)
- Mean reversion speed after price shocks
- Volume-price correlation (high volume + price move = possibly informed; high volume + no sustained move = noise)
- Price behavior around known events (review drops, embargo lifts) vs. quiet periods

**Data availability:** Kalshi provides downloadable price history for each market. This means the historical data collection that originally made this strategy high-effort is largely already done — we don't need to build real-time price capture infrastructure first. We can download resolved market histories retroactively and start analysis immediately.

**Data requirements:**
- Download price history CSVs for all RT-related Kalshi markets (active and resolved).
- Align price timestamps with review arrival timestamps from our DB (`estimated_timestamp`).
- For resolved markets: record the final outcome to label which price movements were "right" and which reverted.

**Immediate action items:**
1. Download all available RT market price histories from Kalshi.
2. Overlay review arrival times from our DB onto the price series — visually inspect how prices react to review drops.
3. Measure mean reversion: after a price spike that coincides with no new review, how far and how fast does it revert?
4. Measure information incorporation: after a review drops, how long until the price stabilizes at its new level? This is the window for informed trading.

### 2.3 Early score trajectory forecasting

**Core idea:** Instead of just asking "will the score cross X in the next 48 hours?", model the *trajectory* of the score over its full lifecycle. Early in a movie's review cycle, the score is volatile. Late, it's nearly locked in. The Poisson/binomial handles the late-stage well (few reviews expected, each one barely moves the needle). But early-stage forecasting — before the score has stabilized — is a different problem.

**Approach:**
- Use historical movie data to build a library of score trajectories: score vs. cumulative review count.
- Classify trajectories by shape (starts high and holds, starts high and decays, starts low and recovers, volatile throughout).
- Given a movie's current partial trajectory, match it to the historical library (k-nearest trajectories or a latent trajectory model) and forecast the terminal score distribution.
- This gives P(final score > threshold) directly, without the Poisson/binomial machinery.

**When it's better than the current model:** Very early in the lifecycle (< 30 reviews). The Poisson/binomial is most useful when the score is semi-stable and you're modeling small perturbations. The trajectory model handles the regime where the score can still swing 10+ percentage points.

### 2.4 Conditional updates from partial information

**Core idea:** Between scraper runs (50-minute gaps), partial information leaks:
- RT's site updates the displayed score in real time. If the score changed between scrapes, new reviews arrived.
- Kalshi's price moves may reflect information we haven't scraped yet (other bettors saw the review first).

By scraping just the displayed score (a single number, much cheaper than full review data) at high frequency (every 1-5 minutes), we'd detect review arrivals in near-real-time and could update our model / place bets faster.

### 2.5 Finite-pool / remaining-reviewer model

**Core idea:** The critic pool for any given movie is finite — maybe 200-400 critics who plausibly review major releases. Once a critic has reviewed, they're removed from the pool. This has two structural consequences the current model ignores entirely:

1. **The arrival rate has a hard ceiling.** If 250 out of 300 likely critics have already reviewed, there are at most 50 reviews left, ever. The KDE/Poisson doesn't know this — it extrapolates from historical rate patterns as if the pool were infinite. Late in a movie's lifecycle, the model may predict reviews that *cannot come from anyone*.

2. **p_fresh depends on who's left, not who's already reviewed.** Each remaining critic has a personal historical fresh rate (across movies, by genre, etc.). If the remaining pool skews harsh — because the friendlier critics reviewed early — the true forward-looking p_fresh is lower than the sample average suggests. The current model's p estimate is backward-looking; this makes it forward-looking.

**Mathematical upgrade — Poisson binomial:**

The current model uses Binom(n, p) — n i.i.d. Bernoulli trials with the same p. The finite-pool model replaces this with a **Poisson binomial distribution**: n independent but *non-identically distributed* Bernoulli trials, where each trial has its own p_i (the remaining critic's historical fresh rate).

The PMF of the Poisson binomial (number of fresh reviews out of n) doesn't have a closed form for general p_i, but it can be computed efficiently via FFT or recursive convolution. For n < 100 (typical for a forecast window), direct computation is fast. (See `SOURCES.md` §3.2 for implementation references.)

**Full pipeline replacement:**

| Current model | Finite-pool model |
|---|---|
| KDE → Lambda → Poisson(Lambda) for review count | Per-reviewer arrival probability → multinomial sampling over remaining pool |
| Single p_fresh → Binom(n, p) | Per-reviewer p_i → Poisson binomial |
| Anonymous reviews | Named reviewers with history |

Instead of "how many reviews arrive?" (Poisson) then "how many are fresh?" (binomial), the finite-pool model asks: "For each remaining critic, what's the probability they review in the next 48h, and if they do, what's the probability they go fresh?" Then you enumerate or simulate the joint distribution.

**Data requirements:**
- Historical review data across many movies to build per-reviewer profiles (fresh rate by genre, typical review timing relative to release).
- A way to identify the "plausible reviewer pool" for a given movie (genre, release size, distributor). Not every critic reviews every movie.
- The scraper already captures `reviewer_name` and `publication_name`, so building profiles from accumulated data is straightforward — it just takes time to build the corpus.

**Open questions — pool boundary definition:**

The hardest part of this strategy isn't the math, it's defining *who is in the pool* for a given movie. Some considerations:

- **Total credentialed critics vs. active critics vs. movie-specific pool.** RT may have 2000-3000 credentialed critics, but most are ghosts — approved but inactive. The actually-active critic population is probably 500-1000. For any given movie, the pool that will plausibly review is a further subset: maybe 100-500 depending on release size, genre, and studio. (These numbers are rough guesses — see `SOURCES.md` §2.2 and §2.3 for how to get hard numbers.)
- **Publication gating.** Some critics can only submit RT-qualifying reviews through a specific publication. If a critic leaves a publication or a publication stops covering film, they drop out of the effective pool even though they're still credentialed. The reviewer identity isn't fully separable from the publication.
- **Genre/release-size segmentation.** A critic who reviews every Marvel movie may never review an A24 indie. The pool is genre- and scale-dependent. We'd need to segment historical data to build movie-type-specific pools rather than one global pool.
- **Pool estimation approaches:**
  - *Empirical:* For past movies of similar type (genre, studio, release size), look at the set of critics who actually reviewed. The union of those sets is the plausible pool for a new similar movie.
  - *Activity-based:* Any critic who has reviewed at least N movies in the last M months is "active." Intersect with genre relevance.
  - *Probabilistic:* Don't define a hard pool boundary. Instead, assign each credentialed critic a probability of reviewing this movie (based on their historical coverage patterns). Critics with P(review) < epsilon are effectively out of the pool. This is cleaner than a binary in/out classification and feeds directly into the per-reviewer arrival probability model.

**Relationship to 2.1 (reviewer graph):** This is complementary. The finite-pool model treats remaining reviewers as independent (each with their own p_i). The graph model (2.1) adds correlation structure between them. You'd build the finite-pool model first (it's simpler), then layer the graph correlations on top.

### 2.6 Multi-market portfolio / range betting

**Core idea:** Kalshi lists multiple threshold markets per movie (above 65%, above 70%, above 80%, above 90%, etc.). The current notebook evaluates one (movie, threshold, direction) at a time. But the underlying math — the Poisson over review counts and the binomial over fresh counts — produces a **full PMF over possible final scores**. From that single distribution, every threshold market can be evaluated simultaneously.

**Range bets (the double-win hedge):**

When the score sits between two thresholds and the model says it's unlikely to escape, you can bet both sides:

- Example: score at 67%, markets at "above 65%" and "below 70%".
- Buy YES on "above 65%" at price $X, buy YES on "below 70%" at price $Y.
- If score stays in [65%, 70%]: both pay $1 → profit = $2 - X - Y.
- If score escapes the range: one pays, one doesn't → profit = $1 - X - Y (which may still be positive if X + Y < $1).
- The bet is +EV when the market underprices P(score stays in range).

This is selling volatility — you profit when the score is stable. The model is well-suited for this because it explicitly computes the probability of the score *not* moving, which is exactly what the range bet pays on.

**Full score PMF (key implementation):**

Instead of computing P(crosses threshold T) for a single T, compute the full distribution over final scores. For each (n, i) scenario:

```
final_score(n, i) = (current_fresh + i) / (current_total + n)
```

Weight by P(N=n) · P(i fresh | n). Bin the results into score buckets. This gives P(final score in [a, b]) for any interval, which directly answers: "What is every Kalshi market worth?"

**Multi-movie scanning:**

Run the model for every active movie with a Kalshi market. For each movie, evaluate all available threshold markets. Rank opportunities by edge (model price minus market price). The best bets surface automatically.

**Slippage and fees — the real cost of a bet:**

The theoretical edge (model price minus market price) is not the realized edge. Two frictions eat into it:

1. **Slippage.** Kalshi is an order-book market. The price you see is the best available — but it's only available at a certain depth. As your wager size increases, you fill through the order book and get progressively worse prices. A $10 bet might fill at 65¢; a $200 bet might average 68¢ across the fills. Your own wager moves the market against you. This means edge is a *function of bet size*, not a constant — and it shrinks as you scale up.

2. **Trading fees.** Kalshi charges a per-contract fee on trades (and sometimes on settlement). This is a fixed cost per contract that further reduces the realized payout. The fee schedule should be checked against the Kalshi API or fee page, as it may vary by market type or volume tier.

**Implication for edge calculation:** The raw edge formula `edge = model_price - market_price` must be adjusted:

```
realized_edge(size) = model_price - average_fill_price(size) - fees_per_contract
```

Where `average_fill_price(size)` is computed by walking the order book: sum up available contracts at each price level until the desired size is filled, then take the weighted average. If `realized_edge(size) <= 0`, the bet is -EV at that size even if the raw edge is positive.

**Optimal bet size** is the size that maximizes expected profit, not the size with the largest edge:

```
expected_profit(size) = size · realized_edge(size)
```

This function typically rises, peaks, then falls as slippage overtakes the edge. The peak is the maximum profitable bet size. Kelly sizing should be capped at this value even if the Kelly formula suggests a larger position.

**Position sizing — Kelly criterion (adjusted for friction):**

For a binary bet with model probability p, average fill price q(size), and fee f per contract:

```
net_payout = 1 - q(size) - f     (what you receive on a win, minus cost and fees)
net_cost = q(size) + f            (what you pay to enter)
f* = (p · net_payout - (1-p) · net_cost) / net_payout
```

In practice, use **fractional Kelly** (half or quarter Kelly) because:
- The model isn't perfectly calibrated.
- Slippage makes the actual edge smaller than the point estimate.
- Survival matters more than growth rate.

For range bets (two correlated contracts), the Kelly calculation is more involved — you need the joint distribution of outcomes, not just marginals. The full score PMF gives you this. Slippage compounds for range bets since you're filling two order books.

**Liquidity as the binding constraint:**

The bottleneck for scaling isn't model capacity — it's how much money the market absorbs before your own bets close the edge. Practical implications:
- Walk the order book before sizing. A 10% raw edge means nothing if it vanishes after $50 of fills.
- Spread bets across movies and thresholds to diversify liquidity exposure.
- The amateur-detection strategy (2.2) helps here: amateurs flooding in = temporarily higher liquidity + temporarily wrong prices = the best window for larger bets with less slippage.
- Consider limit orders instead of market orders. A limit order at your target price avoids slippage entirely, at the cost of uncertain fill. In slow-moving RT markets, patience may be rewarded.

### 2.7 Automated execution pipeline

**Core idea:** Close the loop from model output to bet placement. The system runs on every scraper cycle (every 50 minutes) without human intervention.

**Pipeline:**

```
Scraper updates DB (every 50 min)
    → Model recomputes full score PMF for each active movie
    → Fetch live Kalshi order books for all threshold markets (Kalshi API)
    → For each market:
        → raw_edge = model_price - best_ask
        → Walk order book to compute realized_edge(size) and optimal_size
        → Subtract fees per contract
    → Filter: realized_edge(optimal_size) > minimum_threshold (e.g., 3%)
    → Size: min(fractional Kelly, optimal_size, order book depth)
    → Place orders via Kalshi API (limit orders preferred over market orders)
    → Log everything (model state, order book snapshot, orders, fills, fees paid)
```

**Minimum viable version:**
1. A function that takes model state and returns the full score PMF.
2. A Kalshi API client that fetches prices for all RT markets.
3. A comparison loop that prints: "Market X: model says Y%, market says Z%, edge = W%."
4. Human reviews and places bets manually.

Steps 1-3 are pure code with no risk. Step 4 is where you add automation later once you trust the model's calibration.

**What to log for backtesting:**
- Every model run: timestamp, movie, current score, Lambda, p_fresh, full score PMF.
- Every Kalshi price snapshot: timestamp, market ID, bid, ask, last trade, volume.
- Every bet placed: timestamp, market, direction, price, size, fill.
- Every market resolution: final score, which bets won/lost, P&L.

This log is the training data for everything else — calibration analysis, amateur detection, strategy refinement.

### 2.8 Backtesting & model calibration via resolved markets

**Core idea:** Kalshi's downloadable price history for resolved RT markets gives us ground truth: at every point in time, what did the market believe, and what actually happened? We can retroactively run our model against the same data and compare.

**Calibration analysis:**

For each resolved market, replay history:
1. At each point in time t, compute our model's P(break) using only reviews available at time t.
2. Record the market price at time t (from Kalshi price history).
3. Record the actual outcome (did the score cross the threshold?).

Then measure:
- **Model calibration:** When our model said 30%, did the event happen ~30% of the time? Plot a calibration curve (predicted probability vs. observed frequency). A well-calibrated model falls on the diagonal.
- **Market calibration:** Same analysis for the market price. Is the market well-calibrated? If not, where does it systematically misprice?
- **Relative calibration:** Where is our model better than the market, and where is it worse? This directly identifies which situations have the most exploitable edge.
- **Brier scores:** Quantify overall forecast accuracy for both our model and the market. Decompose into calibration, resolution, and uncertainty components.

**Retroactive P&L simulation:**

Using the full pipeline (model → edge detection → Kelly sizing → slippage from order book), simulate what our P&L would have been on resolved markets. This answers: "If we'd been running this system for the last N months, how much would we have made (or lost)?"

This is the single most important thing to do before risking real money at any scale.

**Literature:** See `SOURCES.md` §3.4 (prediction market calibration methods, Brier score decomposition) and §1.6 (how many resolved RT markets exist — determines sample size).

**What we need:**
- Kalshi price history downloads for every resolved RT market.
- Our review DB with timestamps (already have this for movies we've been tracking).
- A way to replay the model at historical points in time (run the notebook's math against a time-filtered subset of reviews).

### 2.9 Market price as a leading signal

**Core idea:** The Kalshi price aggregates the beliefs of all market participants — including people who may have information we don't have yet. Between our scraper runs (50-minute gaps), someone else may have seen a new review and traded on it. The price moved, but our model hasn't updated.

**Use cases:**

- **Price move with no new reviews in our DB:** Either (a) a review dropped that we haven't scraped yet, or (b) it's noise. If we cross-reference with high-frequency Tomatometer polling (2.4), we can distinguish the two. If the displayed score changed, a review dropped and the price is incorporating real information — update our model's prior accordingly. If the score didn't change, the price move is likely noise — potential reversion opportunity.
- **Price *doesn't* move after a review drops in our DB:** Either the market already priced it in (someone saw it before us) or the market is slow. If the review is material (rotten review on a movie near a threshold), this is a potential information edge — we see it, the market hasn't reacted yet.
- **Ensemble signal:** Use the market price as a Bayesian prior and update it with our model's structural estimate. This hedges against model errors — if the market and our model agree, high confidence; if they disagree, size down.

**Caution — circularity risk:** If we use the market price as an input to our model and then bet against the market price, we're partially betting against our own input. The market price should inform *confidence* (how much to bet), not *direction* (which side to bet). Our structural model (Poisson/binomial/finite-pool) should drive the direction independently.

**Literature:** See `SOURCES.md` §3.7 (Glosten-Milgrom, informed vs. uninformed trading) and §3.4 (prediction market efficiency — are these markets efficient enough that the price is a reliable signal?).

### 2.10 Historical review database backfill

**Core idea:** Backfill the Neon PostgreSQL database with review data from a broad set of historical movies (beyond the ones currently tracked by the scraper). Many backlog items are bottlenecked on having cross-movie data — this unblocks them.

**What to backfill:**
- Review-level data (reviewer, publication, sentiment, timestamp) for 50-100+ movies across release tiers:
  - Wide-release blockbusters (~20 movies)
  - Mid-tier releases (~20 movies)
  - Limited/indie releases (~20 movies)
  - Movies that had active Kalshi RT markets (all available — critical for 2.8 backtesting and 1.9 drift calibration)
- Release metadata per movie: release date, release type (wide/limited/platform), genre, studio/distributor.

**Approach options:**
1. **Extend the existing RT scraper** (in `~/Desktop/rotten-tomatoes-analysis/`) to do a one-time historical pass. The scraper already knows how to parse RT review pages — it just needs to be pointed at historical movie slugs. Timestamps will be less precise (old reviews show dates, not relative times like "5m ago"), so `timestamp_confidence` will be "d" for most backfilled reviews.
2. **Standalone backfill script** in this repo. Simpler, no coupling to the scraper's scheduling/config. Downside: duplicates parsing logic.
3. **Academic/public datasets.** Check Kaggle and academic repositories for existing RT review datasets with timestamps. May save scraping effort but likely missing recent movies and `top_critic` flags.

**Recommendation:** Option 1 (extend the scraper) is cleanest. The scraper already handles deduplication via `unique_review_id`, so re-scraping movies that overlap with current tracking is safe.

**Items this unblocks:**
- **1.8** Cross-movie empirical Lambda — needs review arrival data across many movies.
- **1.9** Empirical score drift calibration — needs resolved-market movies with full review timelines.
- **2.1** Reviewer graph model — needs multi-movie review history to build per-reviewer profiles.
- **2.3** Early score trajectory forecasting — needs a library of score trajectories.
- **2.5** Finite-pool / remaining-reviewer model — needs per-reviewer history across movies.

**Priority:** High. This is infrastructure that multiplies the value of many other items. Should be done before or in parallel with 2.8 (backtesting).

---

## 3. Data Sources to Investigate

### 3.1 Kalshi API / data exports
- **Downloadable price history** — available per market. Contains timestamped price data for active and resolved markets. This is the foundation for backtesting (2.8), amateur detection (2.2), and market-as-signal (2.9). **Action item: download all RT-related market histories immediately — this is free data with high value.**
- **Order book depth** — reveals where liquidity sits and determines slippage. Need full book (not just best bid/ask) to compute average fill price at a given size. Thin books are easier to move (and revert).
- **Fee schedule** — per-contract trading fee and any settlement fees. May vary by market type or volume tier. Required to compute realized edge accurately.
- **Market metadata** — resolution rules, exact threshold definitions, settlement times.
- **Resolution outcomes** — for resolved markets, the final result. Needed to label backtesting data and measure calibration.

### 3.2 Rotten Tomatoes (additional scraping)
- **Displayed Tomatometer at high frequency** — just the number, scraped every 1-5 min. Detects review arrivals between full scraper runs.
- **Audience score** — may be a leading indicator of critic sentiment or at least correlated.
- **"Critics Consensus" text** — RT's editorial summary. Changes to this text may signal a shift in the editorial framing.
- **Review count by day** for historical movies — to build the trajectory library (2.3).

### 3.3 External
- **Embargo lift dates** — sometimes announced on social media or press sites. Knowing the embargo date in advance lets us predict the first review burst.
- **Festival screening dates / press screening invitations** — leading indicators of when reviews will start appearing.
- **Social media sentiment** (Letterboxd, Twitter/X) — audience reactions from early screenings may preview critic sentiment direction.
- **Box office tracking sites** (Box Office Mojo, The Numbers) — tracking numbers correlate with review activity and studio PR pushes.
- **Metacritic scores** — a different aggregation (weighted average, not fresh/rotten binary). Divergence between RT and Metacritic can signal that a movie has polarized reception, which affects the variance of p_fresh.

---

## 4. Priority Matrix

| # | Item | Impact | Effort | Dependencies |
|---|------|--------|--------|-------------|
| 2.10 | Historical review database backfill | Very High | Medium | Scraper extension or standalone script |
| 1.5 | Compare to Kalshi market price | High | Low | Kalshi API |
| 1.1 | Beta-binomial for sentiment overdispersion | High | Low | None |
| 1.4 | Bayesian p_fresh estimation | Medium | Low | Subsumed by 1.9 if 1.9 is built |
| 1.9 | Time-varying p_fresh via empirical drift calibration | Very High | Medium | 2.10 (backfill), 2.8 (resolved markets) |
| 1.8 | Cross-movie empirical Lambda | High | Medium | 2.10 (backfill) |
| 1.2 | Parametric rate model (replace KDE extrapolation) | High | Medium | Historical data across movies |
| 2.4 | High-frequency score polling | High | Medium | Additional scraping |
| 2.8 | Backtesting & calibration via resolved markets | Very High | Medium | Kalshi price history downloads, review DB |
| 2.2 | Price trace anomaly detection | High | Medium | Kalshi price history downloads (available now) |
| 1.3 | Negative binomial for arrival counts | Medium | Low | Historical data to fit dispersion |
| 2.1 | Reviewer graph model | High | High | 2.10 (backfill) |
| 2.3 | Early score trajectory forecasting | Medium | High | 2.10 (backfill) |
| 2.5 | Finite-pool / remaining-reviewer model | Very High | Medium | 2.10 (backfill) |
| 2.6 | Full score PMF + range betting + multi-movie scanning | Very High | Low-Medium | Kalshi API |
| 2.7 | Automated execution pipeline | High | Medium | 2.6, Kalshi API |
| 2.9 | Market price as leading signal | Medium | Low | Kalshi price history downloads |
| 1.6 | RT rounding rule investigation | Low | Low | Empirical testing |
| 1.7 | Top critic score separation | Low | Low | Market resolution rule check |
| 1.10 | Review-integrity uncertainty in backfilled data | Low | Low | 2.10 (backfill), 2.8 (backtesting) |

---

## 5. Operational Security

### 5.1 DATABASE_URL in notebook tracebacks

The notebook connects via `DATABASE_URL` from the shell environment. The connection string is never written to a file. However, if a connection fails, Python/SQLAlchemy may print the full URL (including password) in the stack trace. If the notebook is committed with outputs, the traceback — and the credentials — end up in git history.

**Mitigations:**
- Always clear notebook outputs before committing (`Cell → All Output → Clear` in Jupyter, or `jupyter nbconvert --clear-output`).
- The `.gitignore` already excludes `.ipynb_checkpoints/`, but not the notebook itself. Be careful with `git add`.
- Consider adding a pre-commit hook that strips notebook outputs automatically (e.g., `nbstripout`).
- If credentials are ever accidentally committed, rotate the Neon DB password immediately — git history is permanent even if the file is later cleaned up.

### 5.2 Kalshi API credentials (future)

When we implement the automated execution pipeline (2.7), we'll need Kalshi API keys. Same principles apply:
- Store in environment variables, never in code or config files.
- Never commit `.env` files. Add `.env` to `.gitignore` now (preemptively).
- API keys that can place orders are high-risk — treat them like a bank password.
