# PARAMETERS.md

All tunable parameters in the library, grouped by component. Parameters currently under active investigation (not yet in library code) are documented at the bottom under "Pending library integration."

> **2026-04-19 note:** the KDE-related parameters documented in this file are **shipping out** as part of the Ridge lambda integration (`plans/plan_ridge_integration.md`). Sections below are accurate for the current (pre-0.2.0) library; post-integration this file will be rewritten around Ridge's feature list and per-snap α table. See `findings/ridge_lambda_investigation.md` for the new parameter story.

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

Per `findings/stratified_training_investigation.md` §10.8 and `findings/path_b_lite_investigation.md` §12. These parameters are the deployment-recommended stack. Helpers factored in `notebooks/_helpers.py`. Integration plan pending per `PROMPTS.md` Prompt 5.

**Revisions since original spec (2026-04-18, per Path B-lite):**
- **Weighted KDE build** added as a new variant — per-data-point weights = combined_score values. Validated +7.7% cohort MAE.
- **Midnight+noon snap convention** adopted — snap_time = `midnight UTC on close-N`, day-level reviews shifted to 12:00 UTC before profile build. Clean day-boundary semantics, halved h/m bias.

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

### Close-day piecewise patch (REVISED form per G1 audit 2026-04-18)

The `F × mean_close_day_count` form was replaced with a constant after the G1 audit showed it was wrong by ~4×. Findings §17 has the full story.

| Parameter | Proposed value | Description |
|---|---|---|
| Phase 1 integration bound | `dbc_to = midnight_utc_dbc` | Per-target; `midnight_utc_dbc = (close_ts − close_ts.floor('D')).total_seconds() / 86400` ≈ 0.583 for 14:00 UTC closes. **Correctness fix:** the prior implementation used `dbc_to = 0.0`, which integrated the KDE into the pre-market window where it has no real training signal. |
| Phase 2 constant `C` | `2.0` | Expected pre-market arrivals in the `(midnight UTC of close day, 10am EDT]` window. Based on G1 audit of 3 h/m-observable movies with scraper running past close (forbidden_fruits=2, they_will_kill_you=2, you_me_and_tuscany=0; mean 1.33). C=1 wins MAE on this sample; C=2 is shipped as safer-against-under-prediction midpoint. |
| `PRE_MARKET_HOURS` | 14 | Used by the F-audit script to define market close in UTC. 10am EDT = 14:00 UTC. ±1h depending on DST. |
| Phase 2 formula | `phase2 = C` (constant) | No training-set aggregation, no F parameter, no per-movie scaling. Trivial to implement; trivial to re-tune as more h/m data arrives. |

### Snapshot-state helper (promoted from notebook)

| Parameter | Proposed value | Description |
|---|---|---|
| Skip rule: minimum first_review_dbc | `snap_dbc + 1` | A snapshot must be at least 1 day after the target's first review for the Jaccard computation to have usable signal. |
| Skip rule: minimum observed critics | 3 | A target must have ≥3 observed critics at snap for Jaccard to discriminate. Below this, fall back to gap-only selector. |

### Weighted KDE build (ADDED per Path B-lite §4, 2026-04-18)

New function `build_kde_lambda_model_weighted(profiles, scores, ...)` that takes per-movie similarity scores.

| Parameter | Proposed value | Description |
|---|---|---|
| Weighting source | `combined_score_with_scores()` output | Returns `{slug: score}` for the top-20 training. Those scores become per-data-point weights in `gaussian_kde(timing_data, weights=...)`. |
| base_rate formula | `sum_of_weights_of_movies_this_critic_reviewed / sum_of_all_weights` | Replaces the unweighted `n_reviewed / n_training`. Critics in more-similar training movies get higher base_rates. |
| Normalization | `raw_weights × (n_movies / sum(raw_weights))` | Scales weights so their sum matches unweighted convention (base_rate range comparable). Falls back to uniform if all scores are zero. |

Validated: cohort-wide +7.7% MAE vs equal-weight KDE at same selector, bandwidth, and snap. See `findings/path_b_lite_investigation.md` §4.

### Snap-time convention (REVISED per Path B-lite §7, 2026-04-18)

| Parameter | Proposed value | Description |
|---|---|---|
| `snap_time` formula | `close_ts.floor('D') − SNAP_DAYS × pd.Timedelta(days=1)` | **Midnight UTC on close−N**, not `close_ts − Nd`. Cleaner day-boundary semantics. |
| Day-level timestamp shift | `ts += 12h` for `timestamp_confidence == 'd'` reviews | Shifts day-level reviews from midnight UTC to noon UTC at profile-build time. Eliminates the "spike-on-boundary" artifact that naive midnight-aligned snap creates. H/m reviews unchanged. |
| `snap_dbc_effective` | `SNAP_DAYS + midnight_utc_dbc` | E.g., for SNAP_DAYS=3 and 14:00-UTC close: snap_dbc_effective ≈ 3.583. |
| Phase_1 window | `(midnight_utc_dbc, snap_dbc_effective]` | 3 full calendar days before close day. |

Tradeoff: cohort MAE higher (13.45 vs ship 6.58) but mean_err near-calibrated (+1.56 vs +4.98) and h/m bias halved (−6.78 vs −14.75). Bias is correctable downstream; variance is not. See `findings/path_b_lite_investigation.md` §7.

---

## Parameters under active investigation (pre-ship tuning pass)

Per `brainstorm/brainstorm_pre_ship_tuning.md` and `PROMPTS.md` Prompt 4. The pre-ship tuning pass may update the values above before library integration. Specifically:

- **`σ_gap`** — sweep ∈ {2, 4, 8, 16, ∞} at T-3d/T-5d/T-7d full window. σ_gap→0 is equivalent to `gap_overlap_ranked` (exact-match filter); σ_gap=8 is current recommendation. Decision rule: ≥3% T-3d MAE improvement with bootstrap CI95 lower > 0 → replace.
- **`k` (n_training)** — sweep ∈ {5, 10, 15, 20, 25, 30, 50} using σ_gap winner. Decision rule: same.
- **`F`** — RESOLVED 2026-04-18 via G1 audit (findings §17): F × count form replaced by constant C=2. See "Close-day piecewise patch" table above.
- **Re-pick frequency** (backtest methodology) — measure MAE-vs-rebuild-frequency curve on h/m target movies. Informs but does not set the live-deployment trigger (which lives in the orchestrator per BACKLOG.md §1.6).

Out of scope for this pass: **`α`** tuning (plateau too flat, keep at 0.5 per findings §10.7).

---

## Deferred: Path B (learned-weights similarity model)

Per `plans/plan_learned_similarity_model.md`. A future investigation may replace the `combined_score` selector entirely with a learned-weights model over ~17 features (including TMDb movie metadata). Would introduce time-varying coefficients parameterized by `n_observed_target`. Out of scope for this pass; documented for continuity.
