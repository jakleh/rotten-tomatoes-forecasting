# Changelog

## 0.2.0 — 2026-04-19

### Changed (breaking)

- **Lambda estimator architecture:** replaced per-critic KDE with Ridge regression on 17 features. See `findings/ridge_lambda_investigation.md`.
- **`estimate_lambda` signature:** now takes a fitted `LambdaRegressor` + `features` dict + keyword-only `snap_days`, `close_ts`, `hours_to_close` (and optional `phase2_C`). Returns a `LambdaPrediction` dataclass instead of a scalar.
- **`estimate_p_fresh` signature:** now takes `reviews_df` + `training_slugs` directly (the removed `CriticProfiles` is no longer required). Behavior unchanged.
- **Removed public symbols:** `build_critic_profiles`, `build_kde_lambda_model`, `CriticProfiles`, `KDELambdaModel`, `default_training_slugs`. The submodule `rotten_tomatoes_forecasting.critic_model` is deleted.
- **`estimate_p_fresh` module move:** now at `rotten_tomatoes_forecasting.p_fresh`. Top-level re-export is unchanged; consumers doing `from rotten_tomatoes_forecasting.critic_model import estimate_p_fresh` need to update the import path.

### Added

- `fit_lambda_regressor(cohort_reviews, close_date_map, ...)` — per-snap CV α + LOO Ridge fit pipeline.
- `load_default_regressor()` — loads the shipped JSON artifact via `importlib.resources`.
- `extract_lambda_features(target_slug, *, snap_days, close_ts, reviews_df, close_date_map, ...)` — 17-feature extractor under ET-midnight convention (all args after `target_slug` are keyword-only).
- `compute_close_day_phase2(close_ts, C=1.0)` — dynamic DST-aware phase-2 hours + count.
- `LambdaRegressor`, `LambdaPrediction` dataclasses in the top-level package.
- `_artifacts/default_regressor.json` shipped default regressor (~23KB) + `scripts/fit_default_regressor.py` refit script.
- `apply_noon_shift(reviews_df)` helper (migrated from notebook) for optional day-level timestamp centering.

### Unchanged

- `compute_edge` signature and behavior.
- `naive_lambda` and `naive_p_fresh` fallbacks.

### Performance

Phase-1 MAE improvements over 0.1.x (143-movie cohort LOO, ET-midnight convention):

```
T-5d: 37.96 → 32.14  (+15%)   mean_err -17.19 → -0.45
T-4d: 30.82 → 21.09  (+32%)   mean_err -17.85 → -0.28
T-3d: 17.86 →  9.87  (+44%)   mean_err  -9.58 → -0.01
T-2d:  8.14 →  3.44  (+58%)   mean_err  +6.51 → -0.02
T-1d:  3.87 →  2.24  (+43%)   mean_err  +2.73 → -0.03
```

Shipped composition (phase-1 + C=1) on h/m subset (n=2–5):

```
T-5d: MAE 30.39, me +15.08  (n=2)
T-4d: MAE 14.69, me  +1.92  (n=4)
T-3d: MAE 25.48, me  -5.15  (n=5, long-gap architectural ceiling)
T-2d: MAE  2.45, me  +0.67  (n=5, calibrated)
T-1d: MAE  1.48, me  -0.38  (n=5, calibrated)
```

### Migration (orchestrator)

```python
# Before (0.1.x)
profiles = build_critic_profiles(reviews, close_date_map, training_slugs)
kde = build_kde_lambda_model(profiles)
lam = estimate_lambda(kde, days_before_close, hours_to_close, observed_critics,
                      observed_count, first_review_dbc)
p_fresh = estimate_p_fresh(profiles, observed_critics, fresh_count, total_count)

# After (0.2.0)
regressor = load_default_regressor()
features = extract_lambda_features(
    target_slug,
    snap_days=snap_days, close_ts=close_ts,
    reviews_df=reviews, close_date_map=close_date_map,
)
if features is None:
    pass  # skip rules not met
pred = estimate_lambda(regressor, features, snap_days=snap_days,
                       close_ts=close_ts, hours_to_close=hours_to_close)
p_fresh = estimate_p_fresh(reviews, training_slugs, observed_critics,
                           fresh_count, total_count)
edge = compute_edge(..., lambda_rate=pred.rate_per_hour, p_fresh=p_fresh)
```

See `plans/plan_ridge_integration.md` §6 for the full migration walkthrough.

### Dependencies

- `scikit-learn>=1.8.0` is now load-bearing (Ridge, StandardScaler, KFold, Pipeline at fit time). No new dependencies.
- JSON artifact format removes sklearn-version-drift risk at load time.
