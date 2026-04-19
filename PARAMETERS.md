# PARAMETERS.md

All tunable parameters in the 0.2.0 library, grouped by component. Validation + rationale lives in `findings/ridge_lambda_investigation.md` and `plans/plan_ridge_integration.md`.

---

## Ridge lambda model

| Parameter | Shipped value | Location | Description |
|---|---|---|---|
| `alpha_grid` | `{0.01, 0.1, 1, 10, 100, 1000}` | `lambda_model.DEFAULT_ALPHA_GRID` | Candidate L2 penalties swept per snap by 5-fold CV in `fit_lambda_regressor`. |
| `alpha*` per snap (shipped) | `T-5d=100, T-4d=10, T-3d=10, T-2d=100, T-1d=100` | `_artifacts/default_regressor.json` (`snap_alphas`) | CV-selected alpha per snap on the 144-movie cohort. |
| `snap_days_list` | `{5, 4, 3, 2, 1}` | `lambda_model.DEFAULT_SNAPS` / `features.VALID_SNAP_DAYS` | Snap horizons with shipped models. Out-of-range `snap_days` raises `ValueError`. |
| `cv_folds` | 5 | `lambda_model.DEFAULT_CV_FOLDS` | K-fold count for alpha CV. |
| `cv_seed` | 42 | `lambda_model.DEFAULT_CV_SEED` | KFold seed — same seed produces the same alpha selection. |
| `phase2_C` | 1.0 | `lambda_model.DEFAULT_PHASE2_C` / artifact metadata | Empirical mean phase-2 arrivals across 5 h/m targets `{2, 1, 1, 1, 0}`. Supersedes the earlier `C=2` for a 14h UTC window. |
| Snap routing | `snap_days: int ∈ {1,2,3,4,5}` | `estimate_lambda` (keyword-only) | Out-of-range raises. `snap_dbc` is not a public concept. |

**How lambda works:** `phase1 = scale(features) @ ridge_coef + intercept` via the per-snap `SnapModel`; `phase2 = phase2_C × (phase2_hours / 10.0)` via `compute_close_day_phase2(close_ts)`; `rate_per_hour = (phase1 + phase2) / hours_to_close`.

---

## Feature extraction

17 features total per (target, snap). See `findings/ridge_lambda_investigation.md` §4.1 for the coefficient table.

| Group | Features |
|---|---|
| Observation-window (10) | `observed_count`, `first_review_dbc`, `target_gap`, `observed_rate`, `rate_last_day`, `rate_first_day`, `top_critic_frac`, `pub_diversity`, `pub_entropy`, `low_activity_frac` |
| Nonlinear transforms (4) | `log_observed_count`, `log_rate_last_day`, `sqrt_rate_last_day`, `rate_delta` |
| Finite-pool (3) | `remaining_base_rate_sum`, `pool_mass_consumed`, `observed_top_tier_frac` |

| Parameter | Shipped value | Location | Description |
|---|---|---|---|
| A1 pool size | 20 | `features.A1_POOL_SIZE` | Number of most-recent resolved movies (excluding target) used to compute per-critic base_rate for the finite-pool features. |
| Top-tier cutoff | 30 | `features.TOP_TIER_N` | Top-30 A1 critics by base_rate form the "workhorse" set used for `observed_top_tier_frac`. |
| `LOW_ACTIVITY_THRESHOLD` | 5 | `features.LOW_ACTIVITY_THRESHOLD` | Critics with fewer than 5 cohort-wide reviews count as low-activity for `low_activity_frac`. |
| `apply_noon_shift` default | `False` | `extract_lambda_features` | Caller owns noon-shift preprocessing. Library does not silently mutate timestamps on every call. |

**Skip rules** (return `None` instead of a feature dict):
- A1 pool has fewer than 5 eligible training movies.
- `first_review_dbc < snap_dbc_eff + 1.0` (first review too close to snap for features to be meaningful).
- `observed_count == 0` or `len(observed_critics) < 3`.

---

## p_fresh

Unchanged from 0.1.x. Now consumes the shared `compute_critic_base_rates` primitive over the caller's training pool.

| Parameter | Shipped value | Location | Description |
|---|---|---|---|
| `n_prior` | 20.0 | `p_fresh.estimate_p_fresh` | Pseudo-count for blending observed fresh rate with the critic-profile prior. 50/50 blend at `total_count == n_prior`. |
| Fallback prior | 0.65 | `p_fresh._DEFAULT_FALLBACK_PRIOR` | Returned when no remaining critics have positive base_rate (e.g., all observed or empty training pool). |

**How p_fresh works:** `p_fresh = blend_weight × observed_rate + (1 − blend_weight) × prior`, where `blend_weight = total_count / (total_count + n_prior)` and `prior = Σ(base_rate × fresh_rate) / Σ(base_rate)` over remaining critics.

---

## Edge computation

Unchanged from 0.1.x.

| Parameter | Shipped value | Location | Description |
|---|---|---|---|
| Poisson tail cutoff | 1e-10 | `edge.compute_edge` | `k_max = poisson.ppf(1 − 1e-10, mu)`. Effectively exact for any practical mu. |

---

## Naive estimator fallbacks

Still exposed. Callers that don't want a Ridge model can pass naive rates directly to `compute_edge`.

| Parameter | Value | Location | Description |
|---|---|---|---|
| `hours` (naive lambda window) | 6.0 | `edge.naive_lambda` | Review count in the last 6h divided by 6. |
| `naive_p_fresh` | raw `fresh/total` | `edge.naive_p_fresh` | No blending, no prior. |

---

## Conventions (authoritative — see `CLAUDE.md`)

| Concept | Value |
|---|---|
| Snap anchor | Midnight **ET** on close−N days (DST-aware via `tz_convert('US/Eastern').normalize()`). |
| Phase-1 window | `(midnight ET on close day, snap_time]` — exactly `N × 24h`. |
| Phase-2 window | `(midnight ET on close day, close_ts]` — ~10h for 10am ET closes; dynamic DST handling. |
| Noon-shift | Day-level reviews shifted from midnight UTC to 12:00 UTC. Applied **once at ingest by the caller** via `apply_noon_shift(reviews_df)`. |
| Artifact format | JSON at `_artifacts/default_regressor.json`. Per-snap `{alpha, scaler_mean, scaler_scale, ridge_coef, ridge_intercept}` + LOO residuals + metadata. |

---

## Deferred directions

Tracked in `BACKLOG.md`. Short list of parameters that might change in a future release:

- **Artifact refresh cadence.** Each release re-fits via `scripts/fit_default_regressor.py` on the then-current cohort. Consumers who want fresher fits call `fit_lambda_regressor` with their own up-to-date reviews DataFrame.
- **Snap coverage.** If deployment ever needs T-6d/T-7d, re-fit with a larger `snap_days_list` — the shipped artifact will raise on unsupported snaps.
- **Tier 3 (stacking).** Out of scope per `findings/ridge_lambda_investigation.md` §7; revisit only with a different model class (GBM, quantile regression) that shows structurally different errors.
- **TMDb metadata features.** Deferred. Would add orthogonal signal on critic-magnet / late-surge targets that Ridge can't predict from observation-window features alone.
