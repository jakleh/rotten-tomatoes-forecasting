# Backlog

Priorities for the rotten-tomatoes-forecasting forecasting library. Strategy, backtesting, and execution concerns live in the orchestrator repo (`~/Desktop/kalshi-trading/`).

**Last reorder:** 2026-04-19. Primary active work is now Ridge lambda integration (§1). Historical KDE-era items retained below under §3-§4.

## 1. Active work

### 1.1 Ridge lambda integration (primary, in progress — see `plans/plan_ridge_integration.md`)

Per `findings/ridge_lambda_investigation.md`, ship lambda estimator is Ridge regression on 17 features. Replaces the KDE-based `estimate_lambda` entirely at 0.2.0 (breaking API bump).

**Ship spec:**
- Model: `StandardScaler → Ridge(α=snap-specific)` pipeline. α via 5-fold CV per snap over {0.01, 0.1, 1, 10, 100, 1000}. Validated α*: 10 for T-3d/T-4d; 100 for T-5d/T-2d/T-1d.
- Features (17): 10 observation-window + 4 nonlinear transforms + 3 finite-pool aggregates. Full list in `findings/ridge_lambda_investigation.md` §4.1.
- Base_rate source for finite-pool features: A1 pool (20 most recent resolved movies before target close, LOO-clean).
- **Snap / phase convention: Eastern-midnight** anchored. Snap at midnight ET on close−N. Phase-1 = (midnight ET close, snap_time]. Phase-2 constant `C = 1` over (midnight ET close, close_ts] ≈ 10h. See `CLAUDE.md` "Current Conventions" + `plans/plan_ridge_integration.md` §3.6–§3.7.
- Artifact: JSON at `_artifacts/default_regressor.json`. NOT pickle.

**Performance vs current library 0.1.x (cohort LOO):**
```
snap   library MAE   ridge_t2 MAE    Δ%      library me   ridge_t2 me
T-5d     37.96         32.14        +15%      -17.19       -0.45
T-4d     30.82         21.10        +32%      -17.85       -0.28
T-3d     17.86          9.96        +44%       -9.58       -0.01
T-2d      8.14          3.42        +58%       +6.51       -0.02
T-1d      3.87          2.22        +43%       +2.73       -0.03
```

**Integration:** `plans/plan_ridge_integration.md`, phases A-E library-side (~4 days), Phase F orchestrator migration. Version 0.1.0 → 0.2.0.

### 1.2 Test suite expansion for 0.2.0

Current `tests/` has ~58 tests covering 0.1.x. At 0.2.0 the new surface needs test coverage per `plans/plan_ridge_integration.md` §8.1:

- `test_lambda_model.py` — load/fit/predict/calibration/skip-rule/snap_days-range/composition-arithmetic/artifact-json-roundtrip/sklearn-version-tolerance/DST-edge-cases.
- `test_features.py` — feature parity with notebook, noon-shift idempotency, pool-LOO-cleanliness, sparse cohort handling, skip-rule edge cases.
- `test_pool.py` — A1 pool build, base_rate primitive, top-tier determinism.
- `test_calibration_regression.py` — shipped artifact cohort mean_err stays near 0; h/m composition MAE stays tight at T-2d/T-1d.

Lands alongside Phase B/C of the integration plan.

### 1.3 CI

Set up GitHub Actions to run tests on push. Currently manual: `uv run python -m pytest tests/`. Belt-and-suspenders for the 0.2.0 ship.

### 1.4 p_fresh calibration audit

`estimate_p_fresh` is unchanged at 0.2.0 (moves from `critic_model.py` to `p_fresh.py` but behavior identical). Current validation (`findings/critic_kde_model_validation.md`) shows excellent aggregate calibration (MAE=0.031 at T-1d) but hasn't been broken down by movie characteristics (blockbuster vs indie, sequel vs original, long-gap vs short-gap). Worth auditing post-0.2.0 ship to catch any systematic bias that correlates with deployment target-type filters.

## 2. Deferred model improvements

### 2.1 Finite-pool model (partially addressed at 0.2.0)

Original proposal in `brainstorm/brainstorm_finite_pool_model.md`: Poisson-binomial arrival model using per-critic remaining-pool. **Partially addressed by Ridge's finite-pool features** at 0.2.0 — `remaining_base_rate_sum`, `pool_mass_consumed`, `observed_top_tier_frac` bring pool-composition information into the regression. What's NOT done: the full per-critic Poisson-binomial architecture. Still a valid direction for eventual `compute_edge` upgrade if per-critic prediction becomes relevant (e.g., for fresh/rotten probability on a per-review basis). Low priority; no clear evidence the approximation loss of aggregate features over per-critic modeling is material.

### 2.2 Time-varying p_fresh

Allow p_fresh to change as a function of time-to-close (early reviews are more negative due to top-critic overweighting). See `brainstorm/brainstorm_time_varying_p_fresh.md`. Not yet validated; Q of whether the bias is material enough to warrant complexity.

### 2.3 Hierarchical p_fresh

Cross-movie shrinkage for p_fresh estimates. See `brainstorm/brainstorm_hierarchical_p_fresh.md`. Same status as §2.2 — brainstorm exists; needs empirical go/no-go.

## 3. Historical / resolved

### 3.1 Kalshi-independent lambda validation — RESOLVED

Was: validate `estimate_lambda()` purely against review arrival rates (no Kalshi market resolution dependency), gated on enough minute-level movies to form a meaningful test set.

**Resolved 2026-04-19:** Ridge investigation validated lambda prediction via LOO on 143 movies across T-5d → T-1d, measuring predicted-vs-actual phase-1 review counts. No Kalshi market dependency. Full results: `findings/ridge_lambda_investigation.md`. The h/m minute-level subset (n=5) serves as the deployment-representative calibration sample per `findings/trading_strategy_from_ridge_errors.md` §7.

### 3.2 Close-day lambda bias — RESOLVED at 0.2.0

Was: the 0.1.x KDE dropped close-day reviews because `Bet Close Date` was stored as midnight UTC while actual close is ~14:00 UTC. Partial fix (full UTC datetimes in movies_index.csv) applied; ~98% of reviews still had day-level timestamps landing at midnight.

**Resolved 2026-04-19:** at 0.2.0, Ridge's explicit phase-1 / phase-2 decomposition handles close-day review flow natively. Phase-2 is `compute_close_day_phase2(close_ts, C=1.0)` covering the midnight-ET-to-close window dynamically. No close-day review is silently dropped. See `CLAUDE.md` "Current Conventions" and `plans/plan_ridge_integration.md` §3.6.

### 3.3 Bandwidth selection — MOOT

Was: `bandwidth_floor=0.5d` hardcoded for day-level cohort; no ceiling; should adapt to cohort timestamp granularity.

**Moot 2026-04-19:** Ridge replaces the KDE entirely at 0.2.0. No KDE bandwidth to tune.

### 3.4 KDE ship stack — SUPERSEDED

Was §1.5 (now moved here): `combined_score (α=0.5, σ_gap=8) + bandwidth_cap (0.7d) + weighted KDE + midnight+noon snap + piecewise(C=2)` — the validated-but-never-shipped KDE stack.

**Superseded 2026-04-19:** Ridge (§1.1) beats this at every snap on cohort LOO. KDE stack is not shipping. Details of the KDE stack are preserved in `findings/stratified_training_investigation.md` (with CONVENTION WARNING banner) and `findings/path_b_lite_investigation.md` as historical context.

### 3.5 Triggered re-curation for live deployment — LARGELY MOOT UNDER RIDGE

Was: per-snap profile+KDE rebuild is expensive at live-trading frequency. Propose a Jaccard-based rebuild heuristic.

**Moot 2026-04-19:** Ridge doesn't need per-target profile rebuilds. One `LambdaRegressor` fit at release time, applied to any target via cheap `extract_lambda_features` calls (<50ms). The artifact staleness concern is different in shape: `fit_lambda_regressor` can be re-run periodically against a growing cohort. Released artifacts document their fit_date + cohort_size in metadata so consumers can choose to refit.

### 3.6 KDE-era long-term roadmap — HISTORICAL

Was §1.7: roadmap for retiring the piecewise patch, growing KDE feature set, eventually a learned-weights similarity model. Written under the assumption KDE was the permanent architecture.

**Historical 2026-04-19:** superseded by Ridge at 0.2.0. Ridge is already "the whole model" in the sense §1.7 imagined the KDE eventually becoming. The candidate features listed there (early-arrival shape, top-critic ratio, publication mix, first-review UTC hour, aggregate sentiment) are absorbed or made moot by Ridge's feature set. External metadata (TMDb) remains an untapped direction (see §3.7).

### 3.7 Architectural ceiling on critic-magnet / late-surge movies — BYPASSED BY RIDGE, PARTIALLY PERSISTS

Was §1.8: KDE under-predicts by 14-34 reviews on high-volume short-gap (`the_drama`, `super_mario`) and long-gap late-surge (`they_will_kill_you`, `forbidden_fruits`) targets. Root cause: `base_rate × KDE × exclusion` sum can't scale up for atypical high-activity movies.

**Bypassed 2026-04-19:** Ridge doesn't have the sum mechanism. Cohort-wide MAE improves by 15-58%. But the residual h/m error at T-3d/T-5d persists at similar magnitude (per `findings/ridge_lambda_investigation.md` §4.3) — it's in a different failure mode (observation-window features don't predict late surges) that is data-limited, not architecture-limited. The orchestrator-side `target_gap > 15d` exclusion rule (`findings/trading_strategy_from_ridge_errors.md` §2.3) handles this deployment-side rather than in the model.

**Remaining unexplored directions** (for post-0.2.0 if cohort grows or data sources expand):
- External metadata (TMDb). Declined by Jake previously (config-heavy, unclear feature win). Worth revisiting if h/m cohort data plateaus.
- Hier-Bayes with log-normal or mixture-Gamma shape (Gamma was ruled out). Shape-family choice is itself a research problem.
- More h/m cohort data. Multi-month passive wait; as h/m representation grows, retest the ruled-out interventions under different distribution.

## 4. Kept for reference

### 4.1 Prior KDE ship spec (historical)

The full KDE-stack recommendation that was §1.5 before 2026-04-19 — parameter values, selector spec, snap convention, piecewise patch — is preserved in `findings/stratified_training_investigation.md` §10.8 and `findings/path_b_lite_investigation.md` §15. Both docs have CONVENTION WARNING banners at the top. Kept for archaeological context (e.g., if an agent needs to understand what UTC-midnight snap convention looked like).

### 4.2 Prior-investigation ruled-out mitigations for KDE ceiling

See `findings/path_b_lite_investigation.md` §13 for the full list of 14 ruled-out interventions (scaling clamp sweeps, σ_gap narrowing, pool expansion, volume features, tiered base_rate multipliers, time-series extrapolation, shape-similarity selection, oracle end-shape, shape × scale, Ridge + KDE blend, active/generic magnet boost, hier-Bayes-Gamma). All retain their rejection rationale; the Ridge investigation added tier 3 stacking as additionally-ruled-out (tier 2b/2c showed pool-composition features are pool-invariant across A1/A3/hybrid, so stacking KDE + Ridge aggregate features would duplicate the signal).
