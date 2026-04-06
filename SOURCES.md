# Sources & Data to Gather

Resources and hard numbers needed for the betting function and platform mechanics. Items marked **[GATHER]** need to be looked up. Items marked **[RESOLVED]** are done.

---

## 1. Kalshi — Platform Mechanics

### 1.1 Fee schedule [GATHER] — NEEDED FOR BETTING FUNCTION
**What to find:** Per-contract trading fee (cents or percentage?), settlement fees, maker/taker differences, volume tier discounts.
**Why it matters:** Net edge = gross edge - fees. Without this, we can't size bets properly.
**Where to look:** Kalshi fee page, account settings, or API docs.

### 1.2 RT market resolution rules [RESOLVED]
Resolved 2026-04-05. "Above X" = displayed score >= X+1. Snapshot at 10:00 AM ET on expiration date. All Critics Tomatometer. See `brainstorm/brainstorm_rounding_and_resolution.md`.

### 1.3 Price history CSV format [RESOLVED]
~141 movie price histories in `rt-price-histories/`. Columns: timestamp + one per threshold ("Above X"), prices in cents. Minute/hour/day granularity. No volume data — just prices.

### 1.4 Order book API [GATHER]
**What to find:** API endpoint for order book depth, rate limits, snapshot vs. streaming.
**Why it matters:** Slippage computation for the execution pipeline.
**Where to look:** Kalshi API docs.

### 1.5 Kalshi API — order placement [GATHER]
**What to find:** Order placement endpoints, authentication, rate limits, paper trading environment.
**Why it matters:** Automated execution pipeline (Backlog §1.3, future).
**Where to look:** Kalshi API docs.

---

## 2. Rotten Tomatoes — Platform Mechanics

### 2.1 Tomatometer rounding rule [RESOLVED]
RT uses standard rounding (round half up). Confirmed empirically. See `brainstorm/brainstorm_rounding_and_resolution.md`.

### 2.2 Critic credentialing and pool size [GATHER]
**What to find:** Total Tomatometer-approved critics, approval criteria, public list.
**Why it matters:** Finite-pool model (future refinement, Backlog §5).
**Where to look:** RT's "About" or "Critic Submission" pages.

### 2.3 Review lifecycle timing [PARTIALLY RESOLVED]
**Known:** Our DB has review arrival curves for ~141 movies. Embargo-to-close is typically 7 days (median). 93% of reviews arrive by T-24h.
**Still useful:** Formalizing the review arrival rate curve shape for lambda estimation.

---

## 3. Reference Library

Statistical methods and literature for current and future work. Not blocking anything — read as needed.

### Directly relevant to the betting function
- **Poisson binomial distribution:** Hong (2013), Fernandez & Williams (2010). The `poibin` Python package or direct FFT via numpy. For n < 100, O(n^2) recursion is fine.
- **Kelly criterion with transaction costs:** Thorp (2006), MacLean, Thorp & Ziemba (2011). Needed for position sizing once the function works.

### For future parameter refinement
- **Beta-binomial for overdispersion:** Hoff (2009) Ch. 3, Gelman et al. (2013) Ch. 5. `scipy.stats.betabinom`.
- **KDE bandwidth selection:** Sheather & Jones (1991), Botev et al. (2010). For lambda estimation if we move beyond flat rates.
- **Nonhomogeneous Poisson / Hawkes processes:** Daley & Vere-Jones (2003), Laub et al. (2015). Models review bursts around embargo lifts.

### For market strategy ideas
- **Prediction market efficiency:** Manski (2006), Rothschild (2009), Wolfers & Zitzewitz (2004). Thin markets are less efficient — supports our approach.
- **Market microstructure:** Glosten & Milgrom (1985), Easley & O'Hara (1987). For amateur/informed trader detection.

---

## 4. Empirical Data — Quick Wins

| Item | Status | Notes |
|---|---|---|
| Kalshi price history (~141 markets) | **Done** | In `rt-price-histories/` |
| Resolved market inventory | **Done** | ~141 markets |
| Review count histogram | **Done** | In `movies_index.csv` (26–375 reviews) |
| Embargo lift timing | **Done** | In `movies_index.csv` (median 7 days before close) |
| Kalshi fee confirmation | **Open** | See §1.1 |
| RT rounding test | **Done** | Standard rounding confirmed |
