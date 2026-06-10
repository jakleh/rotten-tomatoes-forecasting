# Backlog

Priorities for the rotten-tomatoes-forecasting forecasting library. Strategy, backtesting, and execution concerns live in the orchestrator repo (`~/Desktop/kalshi-trading/`).

**Last reorder:** 2026-04-19. Primary active work is now Ridge lambda integration (§1). Historical KDE-era items retained below under §3-§4.

## 1. Active work

### 1.1 Ridge lambda integration — SHIPPED at 0.2.0

Per `findings/ridge_lambda_investigation.md`, Ridge regression on 17 features replaced the KDE-based `estimate_lambda` entirely. Library version 0.2.0.

**Shipped spec:**
- Model: `StandardScaler → Ridge(α=snap-specific)` pipeline. α via 5-fold CV per snap over {0.01, 0.1, 1, 10, 100, 1000}. Fit α on the shipped 144-movie cohort: **T-5d=100, T-4d=10, T-3d=10, T-2d=100, T-1d=100**.
- Features (17): 10 observation-window + 4 nonlinear transforms + 3 finite-pool aggregates. `rotten_tomatoes_forecasting.features.FEATURE_NAMES`.
- Base_rate source for finite-pool features: A1 pool (20 most recent resolved movies before target close, LOO-clean). `rotten_tomatoes_forecasting.pool.build_a1_pool_context`.
- Snap / phase convention: **Eastern-midnight** anchored. Snap at midnight ET on close−N. Phase-1 = (midnight ET close, snap_time]. Phase-2 constant `C = 1` over (midnight ET close, close_ts] ≈ 10h (DST-aware). `compute_close_day_phase2`.
- Artifact: JSON at `rotten_tomatoes_forecasting/_artifacts/default_regressor.json` (~23KB). Re-fittable via `scripts/fit_default_regressor.py`.
- Public API changes: removed `CriticProfiles`, `KDELambdaModel`, `build_critic_profiles`, `build_kde_lambda_model`, `default_training_slugs`. Added `fit_lambda_regressor`, `load_default_regressor`, `extract_lambda_features`, `compute_close_day_phase2`, `LambdaRegressor`, `LambdaPrediction`.

**Performance vs 0.1.x (cohort LOO, ET-midnight convention):**
```
snap   library MAE   ridge_t2 MAE    Δ%      library me   ridge_t2 me
T-5d     37.96         32.14        +15%      -17.19       -0.45
T-4d     30.82         21.10        +32%      -17.85       -0.28
T-3d     17.86          9.96        +44%       -9.58       -0.01
T-2d      8.14          3.42        +58%       +6.51       -0.02
T-1d      3.87          2.22        +43%       +2.73       -0.03
```

**Follow-ups:**
- Phase F (orchestrator migration) is a separate effort in `~/Desktop/kalshi-trading/`. Reference migration example in `plans/plan_ridge_integration.md` §6.

### 1.2 Test suite for 0.2.0 — DONE

98 tests covered the 0.2.0 package surface at ship (2026-04-19); gate-support tests added 2026-06-09 (`tests/test_oracle.py` + `tests/test_edge_battery.py`, +530) bring the suite to **628**. The original 98 per `plans/plan_ridge_integration.md` §8.1:

- `tests/test_package.py` — public API re-exports + removal of KDE symbols.
- `tests/test_edge.py` — unchanged; compute_edge math.
- `tests/test_pool.py` — A1 pool build, base-rate primitive, top-tier determinism.
- `tests/test_features.py` — 17-feature extraction, noon-shift, midnight-ET helper, skip-rule coverage, and **parity** against 5 hardcoded (slug, snap) feature vectors from `notebooks/proposed_ship_stack_test.ipynb` (skips gracefully when cohort CSVs are gitignored-absent).
- `tests/test_lambda_model.py` — fit/predict, phase-2 C / DST-spring / DST-fall / non-10am closes, composition arithmetic, JSON round-trip, shipped-artifact load + predict.
- `tests/test_p_fresh.py` — prior / blend / fallback behavior.

Not yet in place (deferred from plan §8.1 because LOO residual-replay against the full cohort is expensive and belongs in a separate nightly job if needed):
- `test_calibration_regression.py` — shipped artifact cohort mean_err / h/m composition MAE sanity. Defer until we have a lightweight fixture to avoid re-loading reviews.csv in unit tests.

### 1.3 CI

Set up GitHub Actions to run tests on push. Currently manual: `uv run python -m pytest tests/`. Belt-and-suspenders for the 0.2.0 ship.

### 1.4 p_fresh calibration audit — PRIORITY BUMPED 2026-06-09 (Gate 3a: p_fresh is the binding input)

`estimate_p_fresh` is unchanged at 0.2.0 (moves from `critic_model.py` to `p_fresh.py` but behavior identical). Current validation (`findings/archive/critic_kde_model_validation.md`) shows excellent aggregate calibration (MAE=0.031 at T-1d) but hasn't been broken down by movie characteristics (blockbuster vs indie, sequel vs original, long-gap vs short-gap). Worth auditing post-0.2.0 ship to catch any systematic bias that correlates with deployment target-type filters.

**2026-06-09 (Gate 3a, `notebooks/gate3_tolerance.ipynb`):** the edge's error-tolerance band is TIGHT in p_fresh (PnL band δ ∈ [−0.05, 0]; random ±0.05 per-cell noise clears CIs in only 26/50 draws; over-estimation worse than under-) while λ tolerance is huge — so this audit is now first-order, alongside Gate 3b: measure per-cell `estimate_p_fresh` error vs oracle p_fresh on the 16-movie gate cohort (bias sign matters), and re-rank §2.2 (time-varying) / §2.3 (hierarchical) by whether they close the gap.

### 1.5 Gate 1 / Gate 2 calibration — ACTIVE PRIORITY (2026-06-04)

VoI feasibility gates before any model upgrade: can the Poisson×Binomial architecture beat the Kalshi market, and is there edge at all? Full design: `plans/plan_gate_1_2_calibration.md` (gitignored) + memory `project_gate_calibration_design`. Gate 1 = market calibration + incremental-info over price (full cohort). Gate 2 = oracle λ/p_fresh (realized = MLE, the best inputs that exist) through `compute_edge` vs market (dense m/h subset); pass = **both** Brier-skill AND realized-PnL-net-of-cost beat the market (cluster-bootstrap by movie); fail → one form-ablation (scalar→time-varying p_fresh) then pivot to a direct `features→P(Yes)` regressor.

**Status:** Gate 1 DONE 2026-06-07 (market prices the observed state; stale/thin → tradeability binding — `project_gate1_findings`). Arena map DONE 2026-06-09: the contested∧≤10¢ arena exists at **T-2d..T-5d**, center T-3d (`project_arena_map`). **Gate 2 DONE 2026-06-09 — STRONG PASS (directional)** (`notebooks/gate2_oracle.ipynb`, as_of_id=648979): oracle λ/p_fresh through `compute_edge` beats the state-at-snap book on Brier AND taker-fee PnL (T-3d +0.0775 Brier diff / +24.1¢ per contract, movie-clustered CIs clear 0; both trade sides win; lagged≈pure oracle). The market prices the state, not the flow. **Gate 3a DONE 2026-06-09** (`notebooks/gate3_tolerance.ipynb`): λ-error tolerance huge (PnL band m ∈ [0.55, 3.0]), **p_fresh tolerance tight (δ ∈ [−0.05, 0]) → p_fresh is the binding input**; shipped Ridge λ error comfortably inside, `estimate_p_fresh` at the band edge. **Next: GATE 3b** — run the real 0.2.0 estimator on the Gate-2 cells (A1-pool cache + ET-midnight alignment) → deployable-stack verdict; p_fresh audit (§1.4) + §2.2/§2.3 jump the queue. See the 2026-06-09 PM handoff + `project_gate2_result`. Oracle support: `gates/oracle.py` + `tests/test_oracle.py` + `tests/test_edge_battery.py` (suite 98 → 628).

Parked (revisit iff Gates pass): the **ridge golden-fixture** regression test (artifact-output pinning; its `compute_edge` battery is folded into the Gate plan). Supporting discipline: **`db_facts`** read-only query functions pinned by serial id — memory `feedback_db_facts_verification`. Cleanup (low-priority): historical §3 + superseded PROMPTS Prompts 1–9 still cite pre-prune findings paths (now under `findings/archive/`) and the deleted `trading_strategy_from_ridge_errors.md` (git history); fix opportunistically.

### 1.6 DB access security-layer test (deferred, not blocking)

Verify `agent_neon_read_only`'s **grant-level** least-privilege by attempting the full write set — `CREATE TABLE`, `INSERT`, `UPDATE`, `DELETE` — and confirming each is **denied**. Defense-in-depth, distinct from the replica's infra read-only.

- **Test on the PRIMARY, not the replica.** The agent connects only to the read *replica* (in `.env`, sandbox-allowlisted via `.claude/settings.local.json` → `sandbox.network.allowedDomains`), where hot-standby blocks *all* writes regardless of grant — so a write-rejection on the replica proves the infra read-only but **masks** the role's actual grants. Only the primary exercises the GRANT layer.
- **Never touch `reviews`.** Use a throwaway scratch table/schema; drop it after.
- Requires a primary connection for the role (the agent normally has only the replica) → likely an operator-run one-off, or a temporary primary string the agent uses then discards.
- Expected: every write denied (`permission denied`). If any succeeds, the grant is broader than intended → tighten it.
- Why it matters: separates "replica is read-only" (infra, already true) from "role has no write grants" (the least-privilege we configured) — the grant layer is what would protect real data if the agent ever obtained a primary endpoint.

### 1.7 Weekly settled-market recorder (Gate cohort accumulation)

The Kalshi API retains only a rolling recent window of KXRT markets — as of 2026-06-07, **16 settled movies / 280 markets**, closing 2026-04-06 → 2026-06-01 (earliest market of any status: 2026-02-02). Older resolved RT markets (the 2024–2025 ones in `movies_index.csv`) are gone from the API. So the gate cohort grows only ~2 movies/week and **must be captured before it ages out**.

Build a periodic (≈weekly) recorder that snapshots, for newly-settled KXRT markets: market metadata (`floor_strike`/`result`/`close_time`/`settlement_ts`), the full 1-min candle history (mids), and the self-labeled 10am score (from the reviews DB). Append to the persistent cohort cache (`gates/_cache/`). Without it the cohort is permanently capped at the API's retention window. Reuses `gates/kalshi_data.py` + `gates/db_facts.py`. Related: `project_orderbook_snapshots` (a live orderbook-*depth* recorder is a separate, heavier want; this one only needs the public candle/result snapshot). Optional one-off: check whether `~/Desktop/kalshi-trading/` already cached older KXRT history that could backfill the cohort.

## 2. Deferred model improvements

### 2.1 Finite-pool model (partially addressed at 0.2.0)

Original proposal in `brainstorm/brainstorm_finite_pool_model.md`: Poisson-binomial arrival model using per-critic remaining-pool. **Partially addressed by Ridge's finite-pool features** at 0.2.0 — `remaining_base_rate_sum`, `pool_mass_consumed`, `observed_top_tier_frac` bring pool-composition information into the regression. What's NOT done: the full per-critic Poisson-binomial architecture. Still a valid direction for eventual `compute_edge` upgrade if per-critic prediction becomes relevant (e.g., for fresh/rotten probability on a per-review basis). Low priority; no clear evidence the approximation loss of aggregate features over per-critic modeling is material.

### 2.2 Time-varying p_fresh

Allow p_fresh to change as a function of time-to-close (early reviews are more negative due to top-critic overweighting). See `brainstorm/brainstorm_time_varying_p_fresh.md`. Not yet validated; Q of whether the bias is material enough to warrant complexity.

### 2.3 Hierarchical p_fresh

Cross-movie shrinkage for p_fresh estimates. See `brainstorm/brainstorm_hierarchical_p_fresh.md`. Same status as §2.2 — brainstorm exists; needs empirical go/no-go.

### 2.4 Multi-anchor instantaneous-rate Ridge features (data-gated)

Brainstormed 2026-04-20. Proposal: compute smoothed instantaneous review rates at several anchor timestamps (fractional-time normalized) across the first-review-to-close interval and add them as features to the 17-feature Ridge stack. Intent: some points in the arrival process carry more predictive signal about the final count than others; Ridge learns which anchors matter.

Origin framing: initially pitched as a standalone MVT-based predictor (rate at an "MVT point" × interval ≈ total count). The MVT is operationally vacuous for cross-movie prediction — the theorem's c_i varies per movie with no guarantee that a shared t* exists. The real operation is empirical search for anchor points with low cross-movie variance of `rate(t*) / mean_rate`. Reframed as features for Ridge (composable with existing pool/observation-window signal) instead of a parallel generative estimator.

**Prerequisite (hard blocker).** Need substantially more cohort movies with m/h-confidence timestamps spanning the pre-close window. Per memory, only `the_drama` and `super_mario_galaxy` had useful pre-close minute-level data at the 0.2.0 ship. n=2 is too small to discover stable anchors. Unblocking this item depends on the RT scraper accumulating more live-tracked movies over time.

**Scope when unblocked (notebook tier only):**
- Normalize time by first-review-to-close gap so non-uniform intervals align on fractional-time anchors (e.g., f ∈ {0.2, 0.4, 0.6, 0.8}).
- Pick a smoothing window to compute "instantaneous" rate. This reintroduces a bandwidth choice in miniature — document it.
- Add rate-at-anchor features to the existing 17-feature stack. Cohort LOO snap-by-snap vs 0.2.0 baseline.
- Decision rule: materially better LOO MAE at early snaps (T-5d / T-4d, where observation-window features are weakest) AND no regression at late snaps. Library integration is a separate backlog item if this wins.

**Known risks:**
- Smoothing-window choice recapitulates the KDE bandwidth problem the 0.2.0 ship escaped.
- Local burst/lull near an anchor biases its rate estimate. Mitigate by averaging over nearby anchors or short windows.
- h/m cohort may not be representative of the full 144-movie cohort; caveat any generalization claim.

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
