# Brainstorm: Hierarchical Cross-Movie p_fresh Model

*Conversation date: 2026-03-31*

*Continues from the time-varying p_fresh brainstorm. That conversation arrived at a "Beta-Binomial for base prediction + empirical drift for conservative floor" approach. This conversation challenges and improves that framing.*

---

## The challenge — why "base + floor" is the wrong framing

The previous conversation concluded with:
- Beta-Binomial with exponential forgetting answers: "Given this movie's own review history, what's my best estimate for p_fresh going forward?"
- Empirical drift from resolved markets provides a conservative floor.
- Combine: take the more conservative of the two, or use the drift as the prior.

**The problem:** Why should the base prediction only use this movie's own data? If we have cross-movie data, it should inform the *base prediction itself*, not just the safety net.

The freshness rate during a forecast window isn't just a property of the individual movie — it's also a property of *movies like this one at this stage*. A movie with 40 reviews at 78% fresh in its second week isn't just "a movie at 78%." It's a member of a population of movies that looked like that, and those movies had forecast-window freshness rates that followed a pattern. That pattern should be the starting point, not an afterthought.

---

## The three-level structure

### Level 1 — Within-movie relationships

For each historical movie m, observe:
- **p_pre(m)** = freshness rate during the pre-forecast window (e.g., [bet_open, forecast_start], typically ~1 week)
- **p_forecast(m)** = freshness rate during the forecast window (e.g., [forecast_start, bet_close], typically 48h)

The "relationship" is how one maps to the other for that movie. Maybe a movie ran 85% fresh in the pre-forecast week and then 70% fresh in the final 48h. That's a downward shift.

### Level 2 — Across-movie patterns

Collect all the (p_pre(m), p_forecast(m)) pairs across many movies. Now you're looking at the *population-level* pattern:
- Across all movies, how does pre-forecast freshness relate to forecast-window freshness?
- Is there a systematic drift?
- Does it depend on features (review count, score level, genre)?

This is a regression surface: E[p_forecast | p_pre, features].

### Level 3 — Prediction for current movie

The current movie has an observed p_pre. Plug it into the population-level relationship from Level 2 to get a predicted p_forecast — and crucially, the *uncertainty* around that prediction.

**This is a hierarchical Bayesian model (multilevel model):**
- There's a population-level distribution of "how freshness evolves during the forecast window" — learned from all historical movies.
- Each individual movie is a draw from that population, with its own idiosyncrasies.
- For the current movie, the population distribution is the prior. The current movie's own recent reviews update it.

The key difference from the previous framing: the prior isn't generic (Beta(1,1) or whatever). It's *learned from data* — specifically, from the cross-movie relationships described above. A movie whose pre-forecast freshness looks like X gets a prior that reflects "movies with pre-forecast freshness X historically had forecast freshness distributed like Y."

---

## Implementation roadmap

### Step 1 — Diagnostic regression (do this first)

Before building any sophisticated model, answer the question: **does pre-forecast freshness actually predict forecast-window freshness?**

For each historical movie, compute:
- p_pre = freshness rate during the pre-forecast window
- p_forecast = freshness rate during the forecast window
- n_reviews = total review count at forecast start

Fit a regression: p_forecast as a function of p_pre and n_reviews. Plot p_pre on the x-axis, p_forecast on the y-axis.

**What the scatter plot tells you:**
- **Tight cloud along the diagonal** → p_fresh is roughly stationary and the simple constant-p model isn't that wrong. The fancy time-varying stuff has limited upside.
- **Systematic drift** (e.g., p_forecast tends to be lower than p_pre) → that's the signal your model needs to capture.
- **Noisy mess with no pattern** → cross-movie data won't help much and you're better off relying on the individual movie's own reviews.

The point of this step is purely diagnostic. No Bayesian machinery, no time-weighting, no forgetting. Just: "across 50+ movies, when p_pre was 80% and the movie had 150 reviews, what did p_forecast tend to be?"

### Step 2 — Hierarchical Bayesian model (if Step 1 shows signal)

Use the cross-movie regression as the *prior* for a Beta-Binomial model of the current movie:
- The Level 2 regression gives E[p_forecast | p_pre, features] and its uncertainty → convert to Beta(alpha_0, beta_0) prior parameters.
- Update with the current movie's recent reviews using exponential forgetting.
- When movie-specific data is thin, the prediction defaults to the cross-movie prior. When strong, it overrides.

### Step 3 — Trajectory feature enrichment (if enough movies)

This is where the exploratory methods come in. Enrich the feature set for the Level 2 regression using trajectory-shape features extracted from the full freshness curve, not just the p_pre summary statistic.

---

## Extracting trajectory features — FPCA, clustering, and PCA

Jake raised the question of whether we could extract features/characteristics of movies that drive the nonstationarity (the peaks and troughs in freshness rate over time) and use those to improve the prediction.

### Functional PCA (FPCA)

Regular PCA works on fixed-length vectors. If you represent each movie's freshness trajectory as a vector (e.g., daily freshness rate over 14 days), PCA finds the principal modes of variation:
- PC1 = overall freshness level (high vs. low across the whole window)
- PC2 = trend direction (rising vs. falling freshness)
- PC3 = curvature (U-shape vs. inverted-U)

**Limitations of regular PCA here:** Trajectories aren't the same length across movies. The sampling is irregular (reviews don't arrive on a clock). PCA is linear — finds linear combinations of time points, not nonlinear patterns.

**FPCA (Functional PCA)** is the continuous-function version. Instead of treating each trajectory as a discrete vector, it treats it as a smooth function and finds the principal *functions* of variation. Handles irregular sampling and different-length trajectories naturally. Output is the same kind of thing (principal modes of freshness trajectory variation), but mathematically cleaner. Python: `scikit-fda`.

### Latent trajectory modeling / growth mixture models

Rather than finding continuous principal components, this clusters movies into discrete trajectory *types*:
- "Starts high, stays high"
- "Starts high, decays"
- "Starts low, recovers"
- "Volatile throughout"

For the current movie, you estimate which cluster it belongs to and use that cluster's forecast behavior as the prediction. Gives interpretable groupings rather than abstract principal components.

### How these slot into the framework

Both FPCA and clustering are **exploratory tools** — they identify structure in the data but don't directly give you p_forecast for the current movie. They're a preprocessing step that feeds into the hierarchical regression:

1. Run FPCA (or clustering) on historical freshness trajectories to extract features.
2. Use those features as inputs to the Level 2 regression: p_forecast ~ f(p_pre, FPCA_scores, review_count, ...).
3. For the current movie, compute its partial trajectory, project it onto the FPCA basis (or assign it to a cluster), and run the regression.

PCA/FPCA enriches the feature set for the cross-movie regression — it's not the model itself.

---

## Key design decisions

### Drift measurement: absolute points, not ratios

A movie at 95% drifting to 94% has ratio 0.989. A movie at 55% drifting to 50% has ratio 0.909. Very different ratios, both small absolute movements. Since thresholds are on an absolute scale, absolute percentage-point drift (Delta = score_end - score_start) is the right metric.

### Conditioning variables for the Level 2 regression

Roughly ordered by suspected importance:
1. **Review count at forecast start** — likely the strongest predictor. 200 reviews won't drift; 30 can swing wildly.
2. **p_pre itself** — the pre-forecast freshness rate.
3. **Current score relative to threshold** — different regimes for margin of safety.
4. **Time since release** — early-lifecycle vs. long-tail.
5. **FPCA scores or cluster assignment** (Step 3) — trajectory shape.

### The conservative floor is a byproduct, not a separate model

The conditioned drift distribution from Level 2 naturally provides a conservative bound: use a quantile (e.g., 5th percentile of p_forecast among movies with similar features). This replaces the "empirical drift floor" from the previous conversation — it's the same idea, but now it falls out of the main model rather than being bolted on as a separate component.

---

## Relationship to previous brainstorm

The previous conversation (see `brainstorm_time_varying_p_fresh.md`) arrived at:
- Beta-Binomial with exponential forgetting for the single-movie estimate
- Empirical score drift from resolved markets for a conservative floor
- Combined: use the drift distribution as the prior, or take the more conservative of the two

This conversation refines that into a cleaner architecture:
- The cross-movie data *is* the base prediction (via the hierarchical model), not a separate floor
- The single-movie Beta-Binomial update is how the current movie's own data adjusts the cross-movie prior
- The conservative floor is a quantile of the model's own output, not a separate mechanism
- Feature enrichment (FPCA, clustering) is an optional Step 3 that improves the cross-movie regression

---

## Blocker

Same as before: the database backfill (backlog item 2.10) must happen first. The diagnostic regression (Step 1) needs 30-50+ movies with complete review timelines. The trajectory analysis (Step 3) needs more.

## Open questions

1. How many movies do we need before the Level 2 regression is stable enough to trust?
2. Is the pre-forecast window definition (bet_open to forecast_start) the right one, or should we use a different anchor (e.g., time since first review, time since peak review rate)?
3. For the Beta-Binomial update (Step 2), how to set gamma (the forgetting rate) — cross-validate, or derive from the Level 2 residuals?
4. Is logit(p_forecast) ~ f(logit(p_pre), features) a better regression specification than the raw probability scale? (Probably yes — keeps predictions in (0,1) and handles the boundary behavior.)
