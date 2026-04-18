# Backlog

Priorities for the rotten-tomatoes-forecasting forecasting library. Strategy, backtesting, and execution concerns live in the orchestrator repo (`~/Desktop/kalshi-trading/`).

## 1. Model Validation

### 1.1 Kalshi-independent lambda validation

The current lambda validation uses Kalshi market resolution (price >= 90 or <= 10) as ground truth. This conflates model accuracy with market behavior. Once we have enough movies with minute-level review timestamps, we can validate lambda purely against review arrival rates -- predicting how many reviews arrive in a window vs. how many actually did -- without needing Kalshi thresholds or resolution data.

**Why this matters:** Minute-level movies give us the exact RT score at any point in time. We can test whether `estimate_lambda()` accurately predicts the number of remaining reviews from any snapshot, regardless of what Kalshi markets did. This is a cleaner test of the model's core claim.

**Prerequisite:** Enough minute-level movies to form a meaningful test set. Currently only `the_drama` and `the_super_mario_galaxy_movie` have useful pre-close minute-level data.

### 1.2 p_fresh calibration audit

Verify that `estimate_p_fresh()` is well-calibrated across different movie profiles (blockbusters vs indie, sequel vs original). The current validation (`findings/critic_kde_model_validation.md`) shows excellent aggregate calibration (MAE=0.031 at T-1d) but hasn't been broken down by movie characteristics.

### 1.3 Close-day lambda bias patch

The KDE model drops close-day reviews because `Bet Close Date` was stored as midnight UTC while actual close is ~14:00 UTC. Partial fix applied (full UTC datetimes in movies_index.csv), but ~98% of reviews have day-level timestamps that still land at midnight. See `brainstorm/brainstorm_close_day_lambda_bias.md` for patch approaches. This is less urgent now that the orchestrator repo owns P&L evaluation, but the model should still be correct.

### 1.4 Bandwidth selection should adapt to timestamp granularity

Current `bandwidth_floor=0.5d` works for our cohort (~98% day-level timestamps) but is not the right value as cohort granularity shifts. Two adjustments worth tracking:

- **Add a bandwidth ceiling** (~1.0d for day-level data). Scott's rule can produce effective bandwidths of 2-3 days for sparse, spread-out training data, which over-smooths across days that day-level timestamps shouldn't bridge. See `findings/stratified_training_investigation.md` (KDE quality test, \u00a710-11) for the over-prediction this contributes to.
- **As cohort granularity improves** (more h/m-confidence reviews from live-tracking), lower both the floor and the ceiling. A future heuristic could auto-detect the dominant granularity in the training data and pick bounds accordingly.

For now, the documented assumption is: bandwidth bounds are calibrated to a mostly-day-level cohort. Re-tune when granularity meaningfully changes.

### 1.5 Recommended parameter values (pending library integration)

Per `findings/stratified_training_investigation.md` §10.8, the deployment-recommended stack composes four interventions. None are in library code yet — all live in `notebooks/stratified_training_validation.ipynb`. When integrating, use these validated values:

**KDE bandwidth (§9.4 of findings):**
- `bandwidth_floor = 0.5d` (current default — keep)
- `bandwidth_ceiling = 0.7d` (NEW — add as parameter to `build_kde_lambda_model`)

**Training-set selector (§10.5-10.8 of findings):**
- Method: `combined_score` ranking. Score = `α × exp(−|gap_diff| / σ_gap) + (1 − α) × jaccard(target_critics, candidate_critics_in_aligned_window)`. Take top `k`.
- `α = 0.5` (plateau across [0.3, 0.7], so anywhere in that range is fine)
- `σ_gap = 8.0` (matches cohort gap IQR ~10.94, so gap_score and Jaccard are on comparable scales)
- `k = 20` (same as current stratified)
- Skip rule: `first_review_dbc < snap_dbc + 1` OR `len(observed_critics) < 3` → fall back to gap-only `matched_training_slugs(band=3.0, n=20)`. If that also fails, fall back to `default_training_slugs`.
- Aligned-window definition for Jaccard: target's window length = `first_review_dbc_target − snap_dbc`. For each candidate training movie, take critics in the first `window_length` days after that movie's first review.

**Close-day piecewise patch (§8 of findings):**
- `F = 1.0` (estimated from h/m close-day reviews on 4 live-tracked movies, n=21, all pre-market — possibly biased high by pull timing; defensible range ∈ [0.5, 1.0])
- Phase 2 prediction = `F × mean(close_day_count(slug) for slug in training_slugs)` added on top of Phase 1
- Phase 2 actual (for validation) = `F × close_day_count(target)` added on top of `actual_remaining`
- Apply symmetrically — both predicted and actual get the F-weighted close-day component

**`_compute_scaling` parameters (kept at current defaults, per §9.3 falsification):**
- `threshold = 40.0` (relaxed scaling test §9.3 falsified the hypothesis that this should be lower; loosening it slightly hurts)
- `clamp = (0.5, 2.0)` (lower clamp barely binds; upper clamp binds 14-38% but loosening doesn't help — root cause was bandwidth, not scaling)

These are documentation-only until the integration plan lands. Library currently uses defaults that don't include any of these new values.

### 1.6 Triggered re-curation for live deployment

Per-snap rebuild of profiles+KDE is fine for backtest (~500ms each, ~1144 builds total in our LOO) but expensive for live trading where decisions might fire many times per minute. Implement a `should_rebuild()` heuristic in the orchestrator: rebuild only when the observed critic set changes by Jaccard < 0.9 since the last build (i.e., ≥10% set turnover). Reduces rebuild frequency 5-10× in practice without sacrificing model freshness. Lives in the orchestrator (deployment concern), not this library.

### 1.7 Long-term direction: richer training-set curation, retire the piecewise patch

The close-day piecewise patch (§1.5) is a tactical bridge, not a permanent fix. It exists because we have no h/m close-day data outside ~4 live-tracked movies. Once enough h/m close-day data accumulates (months away), the KDE itself can model the close-day window directly and the piecewise add-on can be retired.

**The KDE will eventually be the entire model.** That makes pre-bet-close MAE the right primary objective, not just "MAE at the snapshot we happen to evaluate." Better KDE shape and scale, achieved via better training-set curation, is the path.

**Current curation features (Phase A validated):** gap (close − first_review) and observed-critic-set Jaccard. Combined as `α × exp(−|gap_diff|/σ_gap) + (1−α) × jaccard`. α=0.5 is robust across [0.3, 0.7] plateau.

**Candidate features for future work** (in rough order of signal-to-engineering-cost):

1. **Early-arrival shape statistic** (Phase B candidate). Rate of arrivals or slope of cumulative count in the observed window. Encodes "how fast did this movie attract reviews."
2. **Top-critic-to-all-critic ratio in observed reviews.** Distinguishes prestige (high ratio) from mass (low ratio). Uses existing `top_critic` flag.
3. **Publication mix.** Aggregate observed critics to publications, compute Jaccard at publication level. Stabler than per-critic Jaccard.
4. **First-review timestamp's UTC hour.** US morning vs evening embargo lift differ in arrival patterns.
5. **Aggregate sentiment so far** (positive/negative ratio in observed reviews). Confounds with `p_fresh` but might encode movie type.

External (require new data):
- Genre, distributor, budget tier, festival vs wide release. Would need to enrich `movies_index.csv`. Highest signal per feature but biggest data investment.

**Eventual formulation: learned-weights similarity model with time-varying coefficients.**

```
similarity(target, candidate, snap) = Σ_i w_i(n_obs) × feature_i(target, candidate)
```

Where `w_i(n_obs)` is a learned function of how much we've observed (a proxy for confidence in the data signal vs the prior). Bayesian intuition: feature weight on observation-derived signals (Jaccard, shape) should grow with `n_obs`; weight on prior signals (gap, possibly external metadata) should shrink.

Tuning via gradient descent on cohort MAE loss with LOO. Constrain to ~5 features max before overfitting becomes a concern (n=143 cohort).

**Build order:**
- Add features one at a time; validate each on the existing harness with the established decision rule (≥3% MAE improvement to keep).
- If/when ≥3 useful features accumulate, graduate from manual weighting to learned weights with cross-validation.
- If feature accumulation plateaus before that point, stop — we've hit the irreducible floor given current cohort.

**What this means for the validated stack:** ship `combined_score (α=0.5) + bandwidth_cap (0.7d) + piecewise (F=1.0)` now (per §1.5). The long-term work above is layered on top — feature additions are non-breaking, learned weights replace fixed α only when motivated by feature growth.

## 2. Model Improvements (Deferred)

### 2.1 Finite pool model

Replace the Poisson arrival assumption with a Poisson-binomial model using per-critic remaining-pool. See `brainstorm/brainstorm_finite_pool_model.md`. Would improve accuracy near close when the "remaining critic pool" is well-characterized.

### 2.2 Time-varying p_fresh

Allow p_fresh to change as a function of time-to-close (early reviews are more negative due to top-critic overweighting). See `brainstorm/brainstorm_time_varying_p_fresh.md`.

### 2.3 Hierarchical p_fresh

Cross-movie shrinkage for p_fresh estimates. See `brainstorm/brainstorm_hierarchical_p_fresh.md`.

## 3. Package Maintenance

### 3.1 Test suite (DONE)

59 tests in `tests/` covering `compute_edge()` math, `estimate_lambda()` / `estimate_p_fresh()` behavior, public API surface, and cross-repo import patterns. Run with `uv run python -m pytest tests/`.

### 3.2 CI

Set up GitHub Actions to run tests on push.
