# Implementation Plan: Kalshi Analysis Notebook

## What to Build

A Jupyter notebook (`notebooks/kalshi_analysis.ipynb`) that computes the probability a movie's Rotten Tomatoes score will cross a Kalshi bet threshold within a forecast window.

## Implementation Steps

### 1. Create `pyproject.toml`

```toml
[project]
name = "rt-analysis"
version = "0.1.0"
description = "Kalshi bet probability analysis for Rotten Tomatoes movies"
requires-python = ">=3.13"
dependencies = [
    "sqlalchemy>=2.0",
    "psycopg2-binary>=2.9",
    "pandas>=2.0",
    "numpy>=2.0",
    "scipy>=1.14",
    "matplotlib>=3.9",
    "ipykernel>=6.29",
]
```

### 2. Create `.gitignore`

```
.ipynb_checkpoints/
.venv/
__pycache__/
*.py[cod]
.python-version
```

### 3. Run `uv sync`

### 4. Create `notebooks/kalshi_analysis.ipynb`

#### Cell 1 (markdown): Title
"Kalshi Bet Probability Analysis -- Rotten Tomatoes Tomatometer"

#### Cell 2 (code): Imports and Parameters
- Imports: os, warnings, datetime, matplotlib, numpy, pandas, scipy.stats, sqlalchemy
- Parameters: MOVIE_SLUG, THRESHOLD, DIRECTION ("below"/"above"), HOURS_UNTIL_CLOSE (48), ALPHA (0.05), KDE_BANDWIDTH ("silverman"), optional OVERRIDE_CURRENT_SCORE and OVERRIDE_TOTAL_REVIEWS

#### Cell 3 (markdown): "Step 1: Load Review Data"

#### Cell 4 (code): Connect and query
- `create_engine(DATABASE_URL)` -- handle `postgres://` -> `postgresql://` prefix
- Query: SELECT reviewer_name, tomatometer_sentiment, estimated_timestamp, scrape_time, timestamp_confidence, page_position, top_critic FROM reviews WHERE movie_slug = :slug AND estimated_timestamp IS NOT NULL ORDER BY estimated_timestamp ASC
- Load into DataFrame, derive `is_fresh` from tomatometer_sentiment == "positive"

#### Cell 5 (code): Display current state
- total_reviews, fresh_count, current_score (fresh/total)
- Date range, reviews per day
- Respect manual overrides if set

#### Cell 6 (markdown): "Step 2: Nonhomogeneous Poisson Process"

#### Cell 7 (code): Convert timestamps, fit KDE
- hours_since_start = (estimated_timestamp - min_timestamp) in hours
- t_now = (now - min_timestamp) in hours
- t_end = t_now + HOURS_UNTIL_CLOSE
- KDE: scipy.stats.gaussian_kde(arrival_times, bw_method=KDE_BANDWIDTH)
- lambda(t) = N_observed * kde(t)
- Lambda = np.trapz(lambda_t over [t_now, t_end])
- Also compute Lambda_simple = count of reviews in most recent 48h (fallback/comparison)
- Warn if Lambda < 1

#### Cell 8 (code): Visualize lambda(t)
- Full history rate curve
- Shaded forecast window with Lambda annotation
- Rug plot of actual arrivals
- Vertical "now" line

#### Cell 9 (markdown): "Step 3: N-ceiling"

#### Cell 10 (code): Compute n-ceiling
- poisson_dist = stats.poisson(mu=Lambda)
- n_ceiling = int(poisson_dist.ppf(1 - ALPHA)) + 1
- Print Lambda, n_ceiling, P(N >= n_ceiling)

#### Cell 11 (code): Visualize Poisson PMF
- Bar chart of PMF, n_ceiling highlighted in orange/red

#### Cell 12 (markdown): "Step 4: Estimate Fresh Review Probability"

#### Cell 13 (code): Estimate p
- p_recent = mean of is_fresh for reviews in last 7 days (if >= 5 reviews)
- p_naive = overall current score
- Use p_recent if available, else p_naive

#### Cell 14 (markdown): "Step 5: Probability of Breaking Threshold"

#### Cell 15 (code): Core calculation
```python
def compute_break_probability(current_fresh, current_total, threshold, direction, p_fresh, poisson_dist, n_ceiling):
    basket_matrix = np.zeros((n_ceiling + 1, n_ceiling + 1))
    total_prob = 0.0
    for n in range(1, n_ceiling + 1):
        p_n = poisson_dist.pmf(n)
        if p_n < 1e-15:
            continue
        binom_dist = stats.binom(n, p_fresh)
        for i in range(0, n + 1):
            new_score = (current_fresh + i) / (current_total + n)
            breaks = (new_score < threshold) if direction == "below" else (new_score > threshold)
            if breaks:
                p_scenario = p_n * binom_dist.pmf(i)
                basket_matrix[n][i] = p_scenario
                total_prob += p_scenario
    return total_prob, basket_matrix
```
- Call it, print headline probability

#### Cell 16 (code): Heatmap of (n, i) probability contributions

#### Cell 17 (code): Threshold boundary table
- For each n, show the critical i where score crosses threshold

#### Cell 18 (markdown): "Summary"

#### Cell 19 (code): Final summary printout
- All parameters + headline probability

## Verification
1. `uv sync` installs without errors
2. `DATABASE_URL="..." uv run jupyter notebook notebooks/kalshi_analysis.ipynb` opens
3. All cells run without errors, visualizations render
4. Sanity: score well above threshold + few reviews -> near 0%
5. Sanity: score near threshold + active reviews -> meaningful probability
