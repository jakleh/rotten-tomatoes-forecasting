# Sources & Data to Gather

Resources, hard numbers, and literature needed to resolve assumptions and open questions across the backlog. Items marked **[GATHER]** are things to look up or download. Items marked **[READ]** are papers or references to study.

---

## 1. Kalshi — Platform Mechanics

### 1.1 Fee schedule [GATHER]
**Resolves:** Slippage/fee calculations in 2.6, Kelly sizing, edge computation.
**What to find:**
- Per-contract trading fee (cents per contract, or percentage?)
- Is there a fee on settlement/payout, or only on entry?
- Does the fee vary by market type, volume tier, or account level?
- Are there maker/taker fee differences (limit order vs. market order)?
**Where to look:** Kalshi fee page, account settings, or API docs.

### 1.2 RT market resolution rules [GATHER]
**Resolves:** 1.6 (rounding), 1.7 (top critic vs. all critic).
**What to find:**
- Exact resolution language: does Kalshi resolve on the *displayed* integer Tomatometer, or the exact fraction?
- Which score: all critics or top critics?
- What is the resolution source — the RT page URL at a specific time? A snapshot? The score at market close?
- What happens if RT changes its score after market close (e.g., a review is removed)?
**Where to look:** The fine print / rules section on any active Kalshi RT market page.

### 1.3 Price history CSV format [GATHER]
**Resolves:** Data pipeline design for 2.2, 2.8, 2.9.
**What to find:**
- Column names and types (timestamp, price, volume, bid, ask?)
- Time granularity (tick-level? minute? hourly?)
- Does it include volume per period, or just price?
- Does it include bid/ask spread, or just last trade price?
- Is this the same for active and resolved markets?
**Where to look:** Download one CSV from any Kalshi RT market and inspect it.

### 1.4 Order book API [GATHER]
**Resolves:** Slippage computation in 2.6, automated execution in 2.7.
**What to find:**
- API endpoint for order book depth (full book, not just best bid/ask).
- Rate limits on the order book endpoint.
- Format: price levels and quantities at each level.
- Can you get a snapshot of the book, or is it streaming only?
**Where to look:** Kalshi API docs (https://trading-api.readme.io/reference or similar).

### 1.5 Kalshi API — order placement [GATHER]
**Resolves:** 2.7 (automated execution pipeline).
**What to find:**
- API endpoint for placing limit and market orders.
- Authentication (API key? OAuth?).
- Rate limits on order placement.
- Can you query open positions and P&L programmatically?
- Sandbox/paper-trading environment available?
**Where to look:** Kalshi API docs.

### 1.6 Resolved RT market inventory [GATHER]
**Resolves:** 2.8 (backtesting sample size).
**What to find:**
- How many RT Tomatometer markets has Kalshi listed historically (active + resolved)?
- For how many can we download price history?
- What movies/thresholds do they cover?
- This determines whether we have enough resolved markets to do meaningful calibration analysis or if the sample is too small.
**Where to look:** Browse Kalshi's resolved/settled markets filtered to Rotten Tomatoes or entertainment category.

---

## 2. Rotten Tomatoes — Platform Mechanics

### 2.1 Tomatometer rounding rule [GATHER]
**Resolves:** 1.6.
**What to find:**
- How does RT round the displayed integer Tomatometer? Round half up? Truncate? Banker's rounding?
- Test empirically: find a movie where fresh/total produces a .5 fraction and see what RT displays.
- Example: if 45/50 = 90.0% and 44/49 = 89.796...%, what does RT show for 89.796%?
**Where to look:** RT website, by finding movies near a rounding boundary and comparing the displayed score to fresh_count/total_count from our DB.

### 2.2 Critic credentialing and pool size [GATHER]
**Resolves:** 2.5 (finite pool boundary definition).
**What to find:**
- How many total Tomatometer-approved critics exist? (RT may publish this somewhere.)
- What are the approval criteria? (RT has published these — look for the "Tomatometer Criteria" or "Approved Tomatometer Critics" page.)
- Is there a public list of approved critics and their publications?
- Does RT distinguish between "individual" critics and "publication" approval? (i.e., can a critic submit without a qualifying publication?)
**Where to look:** RT's "About" or "Critic Submission" pages. Also: past news articles about RT changing their approval criteria (~2018 they expanded access).

### 2.3 Typical review counts by movie type [GATHER]
**Resolves:** 2.5 (pool size estimation), 2.3 (trajectory forecasting).
**What to find:**
- For wide-release blockbusters (Marvel, etc.): how many total reviews typically? (Expectation: 300-400+)
- For mid-tier releases: how many? (Expectation: 100-250)
- For limited/indie releases: how many? (Expectation: 30-100)
- How does review count correlate with box office? With studio?
- What fraction of total reviews come from top critics?
**Where to look:** Browse RT pages for ~20 movies across tiers and record review counts. Or query our own DB if we have enough movies tracked.

### 2.4 Review lifecycle timing [GATHER]
**Resolves:** 1.2 (KDE extrapolation), 2.5 (reviewer arrival modeling).
**What to find:**
- How many days before wide release do embargo-lift reviews appear? (Typically 1-7 days, but varies.)
- How long after release do reviews continue to trickle in? When does the rate effectively hit zero?
- Is there a standard "embargo lift" → "wide release" → "tail" three-phase pattern, or does it vary more than that?
**Where to look:** Our own DB — plot review arrival curves for the movies we've tracked. Supplement with Box Office Mojo release dates.

### 2.5 Historical review data for cross-movie Lambda [GATHER]
**Resolves:** 1.8 (cross-movie empirical Lambda).
**What to find:**
- Review-count-over-time data for a broad set of historical movies (not just the ones our scraper tracks). We need enough movies (ideally 50+) to build meaningful cohorts for Lambda estimation.
- Release dates, release types (wide/limited/platform), genre, studio/distributor for each movie — needed for similarity feature analysis.
- Ideally: timestamped review counts (reviews per day or per hour), not just final totals. The RT page shows review dates, so this is scrapable.
**Where to look:**
- RT pages for historical movies — review count + date of each review is visible on the reviews tab.
- Could extend the existing scraper to do a one-time historical backfill of review timestamps for ~50-100 movies across different tiers.
- Box Office Mojo / TMDb for release metadata (date, budget tier, genre).
- Alternative: academic datasets of RT reviews may exist (check Kaggle, academic data repositories).
**Key question:** Is the effort of backfilling historical movies worth it, or can we bootstrap with a smaller cohort from movies we naturally accumulate over the next few weeks/months?

### 2.6 Audience score mechanics [GATHER]
**Resolves:** 3.2 (audience score as a signal).
**What to find:**
- Is the audience score a useful leading indicator for critic sentiment, or are they independent?
- How is the audience score computed? (Verified vs. unverified ratings? Weighted?)
- When does audience score become available relative to critic reviews?
**Where to look:** RT FAQ/about page for audience score methodology.

---

## 3. Statistical Methods — Literature

### 3.1 Beta-binomial for overdispersed counts [READ]
**Resolves:** 1.1 (sentiment correlation).
**What to read:**
- Hoff, P. (2009). *A First Course in Bayesian Statistical Methods*, Ch. 3 — Beta-binomial model as a conjugate Bayesian framework. Clear treatment of when and why to use beta-binomial over binomial.
- Gelman et al. (2013). *Bayesian Data Analysis*, Ch. 5 — Hierarchical models. The beta-binomial is a special case of a hierarchical binomial model.
- For implementation: `scipy.stats.betabinom` exists — check if it's sufficient or if we need custom fitting.
**Key question to answer:** How to estimate the overdispersion parameter from our data. Method of moments (match observed variance of fresh counts in rolling windows to the beta-binomial variance formula) is the simplest approach.

### 3.2 Poisson binomial distribution [READ]
**Resolves:** 2.5 (finite-pool model).
**What to read:**
- Hong, Y. (2013). "On computing the distribution function for the Poisson binomial distribution." *Computational Statistics & Data Analysis*. — The DFT/FFT method for computing the exact PMF efficiently.
- Fernandez, M. & Williams, S. (2010). "Closed-form expression for the Poisson-binomial probability density function." *IEEE Transactions*. — Alternative recursive approach.
- For implementation: the `poibin` Python package, or direct FFT implementation via numpy.
**Key question to answer:** For n < 100 remaining reviewers, is direct recursion fast enough, or do we need FFT? (Likely yes for direct recursion — it's O(n^2) which is fine at n=100.)

### 3.3 Kelly criterion with transaction costs [READ]
**Resolves:** 2.6 (position sizing with slippage and fees).
**What to read:**
- Thorp, E. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market." — The classic reference. Section on fractional Kelly and practical adjustments.
- MacLean, Thorp & Ziemba (2011). *The Kelly Capital Growth Investment Criterion*. — Comprehensive treatment including extensions for correlated bets (relevant to range bets) and transaction costs.
**Key question to answer:** For correlated binary bets (range bets across two thresholds on the same movie), how does multivariate Kelly work? The joint distribution from the full score PMF gives us the correlation, but the Kelly formula for multiple simultaneous bets requires matrix algebra.

### 3.4 Prediction market calibration and efficiency [READ]
**Resolves:** 2.8 (backtesting framework design), 2.2 (is the market actually inefficient?).
**What to read:**
- Manski, C. (2006). "Interpreting the Predictions of Prediction Markets." — Caution: market prices are not always well-calibrated probabilities. Relevant to our calibration analysis design.
- Rothschild, D. (2009). "Forecasting Elections: Comparing Prediction Markets, Polls, and Their Biases." — Methodology for measuring calibration of prediction markets. Directly applicable to our Brier score analysis.
- Wolfers, J. & Zitzewitz, E. (2004). "Prediction Markets." — Survey of prediction market efficiency and where mispricings occur (thin markets, low-information events — sounds like RT bets).
**Key question to answer:** Are niche prediction markets like Kalshi RT bets efficient, or is there systematic mispricing we can exploit? The literature on political prediction markets suggests thin markets are less efficient — good news for us.

### 3.5 KDE bandwidth selection for multimodal data [READ]
**Resolves:** 1.2 (KDE extrapolation sensitivity).
**What to read:**
- Sheather & Jones (1991). "A Reliable Data-Based Bandwidth Selection Method for Kernel Density Estimation." — The SJ bandwidth selector, often better than Silverman for multimodal data. Available in R; would need to port or find a Python equivalent.
- Botev, Grotowski & Kroese (2010). "Kernel Density Estimation via Diffusion." — Adaptive bandwidth that handles multimodal densities well. May be overkill, but worth knowing about.
**Key question to answer:** Is Silverman's rule actually causing problems for our review arrival data, or is it fine in practice? Test by comparing Silverman, SJ, and cross-validated bandwidth on our actual data and seeing how much Lambda changes.

### 3.6 Nonhomogeneous Poisson process — alternatives to KDE [READ]
**Resolves:** 1.2, 1.3 (rate model and overdispersion).
**What to read:**
- Daley & Vere-Jones (2003). *An Introduction to the Theory of Point Processes*, Vol. I. — The definitive reference. Heavy, but Ch. 7 (conditional intensities) is relevant if we move toward Hawkes processes (self-exciting point processes that model review bursts).
- For a lighter introduction: Laub, Taimre & Pollett (2015). "Hawkes Processes." — Hawkes processes naturally capture the "burst then decay" pattern of review arrivals around embargo lifts. They model the rate as: λ(t) = μ + Σ g(t - t_i), where each past event boosts the rate temporarily.
**Key question to answer:** Would a Hawkes process give materially better Lambda estimates than KDE, or is the added complexity not worth it for our use case?

### 3.7 Market microstructure — mean reversion and informed trading [READ]
**Resolves:** 2.2 (amateur detection), 2.9 (market-as-signal).
**What to read:**
- Glosten & Milgrom (1985). "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders." — The theoretical foundation for distinguishing informed from uninformed trades based on price impact and reversion.
- Easley & O'Hara (1987). "Price, Trade Size, and Information in Securities Markets." — How trade size signals informed vs. uninformed activity. Relevant to detecting amateur vs. sophisticated trading on Kalshi.
- For a practical/modern take: Cartea, Jaimungal & Penalva (2015). *Algorithmic and High-Frequency Trading*, Ch. 2-3. — Market impact models and optimal execution. Directly applicable to slippage modeling.
**Key question to answer:** Can we apply standard market microstructure models (designed for equities) to prediction markets with much lower liquidity and discrete information events?

---

## 4. Empirical Data — Quick Wins

These are things we can measure or gather quickly from data we already have or can easily access.

| Item | What | How | Resolves |
|---|---|---|---|
| Review count histogram | Total reviews per movie across ~20 RT pages | Browse RT, record counts | 2.5 pool sizing |
| Embargo lift timing | Days between first review and wide release for tracked movies | Query our DB + Box Office Mojo | 1.2, 2.1 |
| Overdispersion test | Variance of fresh counts in rolling 10-review windows vs. binomial prediction | Compute from our DB | 1.1 beta-binomial fit |
| Kalshi price history sample | Download 1 CSV, inspect format | Kalshi website | 2.2, 2.8, 2.9 pipeline design |
| Kalshi fee confirmation | Exact fee per contract | Kalshi fee page or place a $1 test bet | 2.6 edge calculation |
| RT rounding test | Find a movie at a .5 boundary, compare displayed vs. computed | RT website + our DB | 1.6 |
| Resolved market count | How many RT markets has Kalshi resolved? | Browse Kalshi settled markets | 2.8 sample size |
| Cross-movie rate comparison | For 5-10 RT pages, count reviews in final 48h before score stabilized | Browse RT review pages, note dates | 1.8 Lambda cohort feasibility |
| Similarity feature gut-check | For tracked movies, plot reviews-per-day curves overlaid — do they share a shape? | Query our DB | 1.8 — is lifecycle stage a universal predictor? |
