# PARAMETERS.md

All tunable parameters in the library, grouped by component. Parameters currently under active investigation (not yet in library code) are documented at the bottom under "Pending library integration."

---

## Binomial parameters (p_fresh estimation)

| Parameter | Value | Location | Description |
|---|---|---|---|
| `n_prior` | 20.0 | `critic_model.estimate_p_fresh()` | Pseudo-count for blending observed fresh rate with critic-profile prior. At `total_count = n_prior`, the blend is 50/50. Higher values trust the prior longer; lower values let observed data dominate sooner. |
| `n` (training) | 20 | `critic_model.default_training_slugs()` | Number of most-recent resolved movies used to build critic profiles (base_rate, fresh_rate). Shared with lambda estimation. Larger = more stable profiles but less recency. |

**How p_fresh works:** `p_fresh = w * observed_rate + (1-w) * prior_rate`, where `w = total_count / (total_count + n_prior)`. The prior is a base_rate-weighted average of remaining (unreviewed) critics' historical fresh rates.

## Poisson parameters (lambda estimation)

| Parameter | Value | Location | Description |
|---|---|---|---|
| `n` (training) | 20 | `critic_model.default_training_slugs()` | Same training pool as p_fresh. Determines which movies' review timing data gets used to fit critic KDEs. |
| `shrinkage_k` | 3.0 | `critic_model.build_kde_lambda_model()` | Shrinkage toward population prior. Each critic's KDE is blended: `(n/(n+k)) * empirical + (k/(n+k)) * population`. At k=3, a critic with 3 reviews gets 50/50 blend; with 10 reviews, 77% empirical. |
| `bandwidth_floor` | 0.5 days | `critic_model.build_kde_lambda_model()` | Minimum KDE bandwidth. Prevents overfitting to tight clusters of review times. If a critic's Scott's-rule bandwidth falls below this, it's forced up. Calibrated for a cohort that is ~98% day-level timestamps (see BACKLOG.md §1.4). |
| `scaling_threshold` | 40 | `critic_model._compute_scaling()` (hardcoded) | Minimum expected reviews before observed/expected scaling is applied. Below this (common at T-7d+ where KDE tail mass is thin), scaling is unreliable so the ratio is forced to 1.0. Relaxed-scaling test falsified the hypothesis that this should be lower (findings §9.3). |
| `scaling_clamp` | (0.5, 2.0) | `critic_model._compute_scaling()` (hardcoded) | Bounds on the observed/expected scaling ratio. Prevents wild overcorrection when the KDE expectations are far off. Lower clamp barely binds (0-10%); upper clamp binds 14-38% but loosening doesn't help. |

**How lambda works:** For each unreviewed critic, `expected_remaining += base_rate * KDE_integral(0, days_before_close)`. This is optionally scaled by `observed_count / expected_so_far` (if above threshold), then divided by `hours_to_close` to get reviews/hour.

## Edge computation parameters

| Parameter | Value | Location | Description |
|---|---|---|---|
| Poisson tail cutoff | 1e-10 | `edge.compute_edge()` | `k_max = poisson.ppf(1 - 1e-10, mu)`. Determines how far into the Poisson tail we sum. Effectively exact for any practical mu. |

## Naive estimator parameters (fallback, used if consumer passes naive rates)

| Parameter | Value | Location | Description |
|---|---|---|---|
| `hours` (naive lambda window) | 6.0 | `edge.naive_lambda()` | Counts reviews in last 6h, divides by 6. Simple rate estimate. |
| p_fresh | raw `fresh/total` | `edge.naive_p_fresh()` | No blending, no prior. Just the running average. |

---

## Pending library integration (validated, not yet in library code)

Per `findings/stratified_training_investigation.md` §10.8 and `BACKLOG.md` §1.5. These parameters are the deployment-recommended stack. All currently live in `notebooks/stratified_training_validation.ipynb` as inline helpers. Integration into library code is gated on the pre-ship tuning pass (`brainstorm/brainstorm_pre_ship_tuning.md` / `PROMPTS.md` Prompt 4) which may update these values.

### Bandwidth ceiling (new parameter on `build_kde_lambda_model`)

| Parameter | Proposed value | Description |
|---|---|---|
| `bandwidth_ceiling` | 0.7 days | Upper cap on effective KDE bandwidth. Scott's rule can produce bandwidths of 2-3 days for sparse, spread-out training data — over-smooths day-level data into multi-day humps. Capping at 0.7d was validated (findings §9.4) as the single highest-leverage change (baseline MAE −37%, stratified −26% on middle window). Re-tune as cohort shifts toward h/m timestamps. |

### Training-set selector (new function, replaces `default_training_slugs` as recommended default)

| Parameter | Proposed value | Description |
|---|---|---|
| Selector method | `combined_score` | Score = `α × exp(−|gap_diff| / σ_gap) + (1 − α) × jaccard(target_critics, candidate_critics_in_aligned_window)`. Take top `k`. Ranks candidates by a weighted blend of gap match and observed-critic-set overlap. |
| `α` | 0.5 | Weight between gap_score and Jaccard. Plateau across [0.3, 0.7] so α=0.5 is defensible (findings §10.7). |
| `σ_gap` | 8.0 | Gaussian decay scale for `gap_score`. Matches cohort gap IQR ~10.94 so gap_score and Jaccard are on comparable scales. **Under re-examination** in pre-ship pass (σ_gap → 0 is equivalent to exact-match filter; worth testing). |
| `k` | 20 | Number of training movies selected. Matches existing `default_training_slugs`. Under re-examination in pre-ship pass (sweep {5, 10, 15, 20, 25, 30, 50}). |
| Aligned-window definition | `target_window_days = first_review_dbc_target − snap_dbc` | For each candidate, take its critics from the first `target_window_days` after that candidate's first review. Matches target's observation window length, anchored to each movie's own embargo-lift proxy. |
| Fallback chain | `combined_score` → `matched_training_slugs(band=3.0, n=20)` → `default_training_slugs(n=20)` | Skip rule: if `first_review_dbc < snap_dbc + 1` OR `len(observed_critics) < 3`, fall back to gap-only. If that also fails, fall back to recency-only. |

### Close-day piecewise patch (new helper function)

| Parameter | Proposed value | Description |
|---|---|---|
| `F` | 1.0 (ship at 0.7 to hedge pull-timing bias) | Fraction of close-day reviews that arrive pre-market-close (12am UTC → 10am EST, ~14h window). Estimated from h/m close-day reviews on 4 live-tracked movies (n=21, all pre-market — possibly biased high). Piecewise T-1d MAE improvement survives across F ∈ [0.5, 1.0] (findings §8.4), so exact value isn't load-bearing. Re-tune when fresh reviews.csv with more h/m close-day movies is available. |
| `PRE_MARKET_HOURS` | 14 | 10am EST as UTC offset. ±1h depending on DST; using 14h (EDT) as the default since DST covers most of the year. |
| Phase 2 prediction | `F × mean(close_day_count(slug) for slug in training_slugs)` | Added on top of Phase 1 (existing snap-to-midnight-UTC prediction). Composable; retire by setting `F=0` or skipping the call entirely once h/m close-day data is sufficient for a KDE-only close-day model. |

### Snapshot-state helper (promoted from notebook)

| Parameter | Proposed value | Description |
|---|---|---|
| Skip rule: minimum first_review_dbc | `snap_dbc + 1` | A snapshot must be at least 1 day after the target's first review for the Jaccard computation to have usable signal. |
| Skip rule: minimum observed critics | 3 | A target must have ≥3 observed critics at snap for Jaccard to discriminate. Below this, fall back to gap-only selector. |

---

## Parameters under active investigation (pre-ship tuning pass)

Per `brainstorm/brainstorm_pre_ship_tuning.md` and `PROMPTS.md` Prompt 4. The pre-ship tuning pass may update the values above before library integration. Specifically:

- **`σ_gap`** — sweep ∈ {2, 4, 8, 16, ∞} at T-3d/T-5d/T-7d full window. σ_gap→0 is equivalent to `gap_overlap_ranked` (exact-match filter); σ_gap=8 is current recommendation. Decision rule: ≥3% T-3d MAE improvement with bootstrap CI95 lower > 0 → replace.
- **`k` (n_training)** — sweep ∈ {5, 10, 15, 20, 25, 30, 50} using σ_gap winner. Decision rule: same.
- **`F`** — re-estimate after fresh reviews.csv pull from Neon (gated). If new estimate differs from current by > 0.2, update.
- **Re-pick frequency** (backtest methodology) — measure MAE-vs-rebuild-frequency curve on h/m target movies. Informs but does not set the live-deployment trigger (which lives in the orchestrator per BACKLOG.md §1.6).

Out of scope for this pass: **`α`** tuning (plateau too flat, keep at 0.5 per findings §10.7).

---

## Deferred: Path B (learned-weights similarity model)

Per `plans/plan_learned_similarity_model.md`. A future investigation may replace the `combined_score` selector entirely with a learned-weights model over ~17 features (including TMDb movie metadata). Would introduce time-varying coefficients parameterized by `n_observed_target`. Out of scope for this pass; documented for continuity.
