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

**2026-06-09 (Gate 3a, `notebooks/gate3_tolerance.ipynb`; re-swept 2026-06-10 on the data-ready cells):** the edge's error-tolerance band is asymmetric-binding in p_fresh (revised PnL band δ ∈ [−0.10, 0] at m=1; δ=+0.05 kills the CI; random ±0.05 clears 34/50 draws — over-estimation is the failure mode, under-estimation nearly free) while λ tolerance is huge — so this audit is first-order, alongside Gate 3b: measure per-cell `estimate_p_fresh` error vs oracle p_fresh on the data-ready gate cohort (bias sign matters), and re-rank §2.2 (time-varying) / §2.3 (hierarchical) by whether they close the gap.

**AUDIT DONE 2026-06-10 (Gate 3b, `notebooks/gate3b_deployable.ipynb` — measured per-cell δ̂ on the locked 60-cell midnight-ET grid, all oracle-clean ct cells, zero n_rem==0 sentinels):** only **33% of cells inside the δ ∈ [−0.10, 0] band; 42% in the kill zone δ̂ > +0.05**. Bias **flips sign with horizon** — mean δ̂ T-1d −0.063 (median −0.136), T-2d −0.018 (−0.071), **T-3d +0.078 (+0.123)**, T-4d +0.032 (−0.006); pooled mean +0.028 with signs 31+/29−. The 2×2 vs per-cell PnL: p_fresh-in-band trades **+20.8¢ (75% win)** vs out-of-band **−10.7¢** — p_fresh error IS the deployable-stack failure (λ 93% in-band; Gate 3b verdict DOES NOT CLEAR, pooled +2.4¢ [−12.6, +16.8] vs the in-grid oracle's +31.4¢). A constant −0.03 shade (post-verdict bench re-score) recovers +11.8¢ [−0.3, +26.0] — ~⅓ of the ceiling, CI-lo a hair under 0.

**REGRESSION PROGRAM RAN 2026-06-10 (same day; operator-signed v3 design — battery decides BUILT / bench rule decides SHIPS / recorder tripwire decides LIVE):** Phase-0 battery (`notebooks/pfresh_battery.ipynb`, 135-movie/529-row pull @ pin 653572): **remaining-critic priors ≈ no ranking signal** (Spearman −0.05 vs obs_rate +0.874); **intensity channel DEAD** (increment −0.0002; the score→fresh curves are step functions — 4.7% of mass in p ∈ [0.2, 0.8] — so subjective scores re-encode the binary; per-critic anchors transfer fine, +0.167, nothing to exploit); **state-dependence huge** (visible-score coef +0.99 beyond critic FE); bias is **behavior, not composition** (oracle-composition keeps +0.054/+0.074). Bench (`notebooks/pfresh_bench.ipynb`, locked rule): **C1′ +7.8¢ [−5.2, +23.0] / C2′ +10.0¢ [−4.4, +25.2] — both crush the shipped +2.4¢ and fix the δ̂ structure (kill-zone 42% → 27/30%) but neither beats the −0.03 shade (+11.8¢; paired −4.1/−1.9¢) → NO WINNER, ship nothing (bar-invariant)**. Per-snap reads are shared with the bar (post-build-review attribution catch); candidates' only paired edge is T-2d +6.0¢ [+0.0, +21.6] (n=12). §2.2/§2.3 now resolved-by-measurement (state matters; shrinkage immaterial — k∈{5,10,20} flat). **Next: operator decision (ship the shade as flagged interim — training s* ≈ −0.04 — vs wait) + the recorder-growth re-run with the K≈8 tripwire; the battery+bench machinery re-scores any variant in one pass.** Full result: `plans/plan_p_fresh_regression.md` § "RESULT". **Live-use addenda (2026-06-10 late PM, same plan):** the operator's split-half resilience test reversed the live lean to **C2′** (constant unstable across halves; C2 wins all cross-half directions); `gates/live_scorer.py` ships it (read-only; tripwire log); the **trade-conditioning addendum** records the buffer sweep (near-monotone; EV ≥ ~8¢ = the Thursday execution policy) + hypothesis-grade terciles (claimed-edge/remaining-flow/price-band structure) on the **watch-list for re-confirmation at ~25–30 recorder movies** — cutpoints are NOT filters until then.

### 1.5 Gate 1 / Gate 2 calibration — ACTIVE PRIORITY (2026-06-04)

VoI feasibility gates before any model upgrade: can the Poisson×Binomial architecture beat the Kalshi market, and is there edge at all? Full design: `plans/plan_gate_1_2_calibration.md` (gitignored) + memory `project_gate_calibration_design`. Gate 1 = market calibration + incremental-info over price (full cohort). Gate 2 = oracle λ/p_fresh (realized = MLE, the best inputs that exist) through `compute_edge` vs market (dense m/h subset); pass = **both** Brier-skill AND realized-PnL-net-of-cost beat the market (cluster-bootstrap by movie); fail → one form-ablation (scalar→time-varying p_fresh) then pivot to a direct `features→P(Yes)` regressor.

**Status:** Gate 1 DONE 2026-06-07 (market prices the observed state; stale/thin → tradeability binding — `project_gate1_findings`). Arena map DONE 2026-06-09: the contested∧≤10¢ arena exists at **T-2d..T-5d**, center T-3d (`project_arena_map`). **Gate 2 DONE 2026-06-09, REVISED 2026-06-10 to the data-ready cohort — STRONG PASS (directional)** (`notebooks/gate2_oracle.ipynb`, as_of_id=648979, ex-`animal_farm_2025` per the §1.9 readiness criterion): oracle λ/p_fresh through `compute_edge` beats the state-at-snap book on Brier AND taker-fee PnL (T-3d +0.0883 Brier diff / +27.1¢ per contract; pooled +0.1247 / +32.1¢, 91% win; movie-clustered CIs clear 0; both trade sides win; lagged≈pure oracle; original 13-movie figures kept as labeled history in the plan). The market prices the state, not the flow. **Gate 3a DONE 2026-06-09, re-swept 2026-06-10 on the data-ready cells** (`notebooks/gate3_tolerance.ipynb`): λ-error tolerance huge (PnL band m ∈ [0.55, 3.0] at δ=0; ±170% random noise clears 48/50), **p_fresh tolerance asymmetric-binding (PnL band δ ∈ [−0.10, 0] at m=1; δ=+0.05 kills the CI; random ±0.05 clears 34/50) → p_fresh is the binding input**; shipped Ridge λ error comfortably inside, `estimate_p_fresh` in at −0.05 / out at +0.05. **GATE 3b DONE 2026-06-10 — the shipped stack DOES NOT CLEAR; p_fresh isolated as THE failure input; ceiling intact** (`gates/build_gate3b.py` lock chain + `notebooks/gate3b_deployable.ipynb`, pin 648979, midnight-ET grid, 17-movie data-ready cohort, coverage 91.9%): pooled (n=35 unique mkts/13 movies) Brier diff +0.0015 [−0.0564, +0.0709] / PnL **+2.4¢ [−12.6, +16.8]** — neither CI clears; in-grid oracle on the SAME cells **+31.4¢ [+22.0, +40.4]** (paired gap +29.0¢; capture 8%); λ 93% inside its 3a band, **p_fresh 42% in the kill zone** (per-cell audit → §1.4); conservative −0.03 shade (bench re-score, fenced) +11.8¢ [−0.3, +26.0]. Per the locked fork: the gap is the improvement target → **next = the p_fresh regression model** (§1.4 action); the locked `gate3b_cells.csv` is the standing bench (re-score any variant by re-running the estimator pass only). Full result: `plans/plan_gate3b.md` § "GATE 3b RESULT". See the 2026-06-09 PM handoff + `project_gate2_result`. Oracle support: `gates/oracle.py` + `tests/test_oracle.py` + `tests/test_edge_battery.py` (suite 98 → 628 → 662 with `tests/test_gate3b.py`). **Integrity recheck 2026-06-10** (`notebooks/gate2_integrity_recheck.ipynb`, after the recorder exposed the sentiment-case switch + animal_farm coverage gap): **zero drift** in all 134 published Gate-2 cell rows under the case-insensitive recompute → published Gate-2/3a numbers stand verbatim; **ex-animal_farm is stronger** (pooled +0.1247 Brier diff / +32.1¢ vs +0.0966/+27.5¢); Gate 1/arena not re-run by recorded reasoning (plan addendum).

Parked (revisit iff Gates pass): the **ridge golden-fixture** regression test (artifact-output pinning; its `compute_edge` battery is folded into the Gate plan). Supporting discipline: **`db_facts`** read-only query functions pinned by serial id — memory `feedback_db_facts_verification`. Cleanup (low-priority): historical §3 + superseded PROMPTS Prompts 1–9 still cite pre-prune findings paths (now under `findings/archive/`) and the deleted `trading_strategy_from_ridge_errors.md` (git history); fix opportunistically.

### 1.6 DB access security-layer test (deferred, not blocking)

Verify `agent_neon_read_only`'s **grant-level** least-privilege by attempting the full write set — `CREATE TABLE`, `INSERT`, `UPDATE`, `DELETE` — and confirming each is **denied**. Defense-in-depth, distinct from the replica's infra read-only.

- **Test on the PRIMARY, not the replica.** The agent connects only to the read *replica* (in `.env`, sandbox-allowlisted via `.claude/settings.local.json` → `sandbox.network.allowedDomains`), where hot-standby blocks *all* writes regardless of grant — so a write-rejection on the replica proves the infra read-only but **masks** the role's actual grants. Only the primary exercises the GRANT layer.
- **Never touch `reviews`.** Use a throwaway scratch table/schema; drop it after.
- Requires a primary connection for the role (the agent normally has only the replica) → likely an operator-run one-off, or a temporary primary string the agent uses then discards.
- Expected: every write denied (`permission denied`). If any succeeds, the grant is broader than intended → tighten it.
- Why it matters: separates "replica is read-only" (infra, already true) from "role has no write grants" (the least-privilege we configured) — the grant layer is what would protect real data if the agent ever obtained a primary endpoint.

### 1.7 Weekly settled-market recorder — BUILT 2026-06-09 (local-first, ritual-scheduled)

The Kalshi API retains only a rolling recent window of KXRT markets — as of 2026-06-07, **16 settled movies / 280 markets**, closing 2026-04-06 → 2026-06-01 (earliest market of any status: 2026-02-02). Older resolved RT markets (the 2024–2025 ones in `movies_index.csv`) are gone from the API. So the gate cohort grows only ~2 movies/week and **must be captured before it ages out**.

**Status (2026-06-09):** `gates/recorder.py` shipped per `plans/plan_recorder.md` (infra riff + operator sign-off: `brainstorm/brainstorm_recorder_infra.md`). Incremental + idempotent: diffs the API settled list against the **committed** store `gates/recorded/` (`markets.csv` ledger, per-event `candles/<EVENT>.csv.gz`, `events_open.csv` open-listing snapshots, `runs.csv`), candles-written-before-ledger crash safety, DB join = self-label aggregates only (`as_of_id`-pinned; rows captured without DB carry `db_joined=False` and top up on later local runs), coverage watch warns on open KXRT events with zero DB reviews (untracked-movie guard for future cohort cleanliness), and an `n_aged_out` retention canary measured every run. Tests: `tests/test_recorder.py` (21 — suite 651; no network/DB; incl. regression tests from the 2026-06-10 adversarial review — dup-ticker dedupe, per-market close epochs, Int64 text stability on deferred joins — plus the settlement-consistency rejoin). **Cadence: ~weekly local runs, scheduled by the PROMPTS session ritual** (`python -m gates.recorder --check` warns past 10d); the cloud trigger is deliberately deferred — designated phase-2 is a zero-secret GitHub-Actions weekly cron per the brainstorm (Cloud Run held in reserve; would require §1.6's grant test first if a cloud DB role were ever provisioned). Related: `project_orderbook_snapshots` (live orderbook-*depth* recorder remains a separate, heavier want).

**Seeded 2026-06-10 (as_of_id=649484, `gates/recorded/runs.csv`):** 338 markets / 19 events / 1,210,586 candle rows (~9.0MB gz; 0 fetch failures), including 3 movies settled since the 2026-06-07 cache build (Masters of the Universe, Power Ballad, Scary Movie). Phase-3 cross-check vs `_cache/cohort_markets.csv` at seed time: **zero mismatches** on all five DB-join columns and **zero per-ticker candle row-count diffs** over the 280 overlapping markets; no-op re-run +0 in 2.4s; `--check` exit 0. (**2026-06-10 post-heal:** 77 rows re-joined after the sentiment-case fix — MAS/POW/SCA/backrooms; see the §1.5 integrity note — so healed join values now intentionally differ from the 06-07 cache; the seed-time zero-diff read remains the capture-parity evidence, and `validate_recorded.py`'s settlement-consistency table is the live integrity check.) First coverage-watch reading: **8 of 10 open KXRT events have no DB reviews** (The Death of Robin Hood / Girls Like Girls / Toy Story 5 close in ~12d, Supergirl ~19d; also The Odyssey, Spider-Man: Brand New Day, Avengers: Doomsday, Dune 3) — informational, not a gap: all 8 are plausibly pre-embargo (zero published reviews exist to miss yet), and review DATA is never lost regardless. The only thing at stake is **timestamp granularity**: if tracking were to start after an embargo lifts, the already-published reviews would enter as day-level backfill and the movie would fail the live-tracked-through-snap guard (Gate-2 rule (a)) for snap-anchored analyses. Operator practice (enable at embargo lift) prevents this entirely; the watch is the standing reminder list (operator-confirmed framing 2026-06-10).

### 1.9 Backfill early reviews for coverage-gapped movies — HIGH PRIORITY (2026-06-10)

The recorder's settlement-consistency check (self-label must land inside the score interval implied by the event's own strike results — `gates/validate_recorded.py` prints the table every run) exposed movies whose DB review history is **incomplete** (reviews RT counted never entered the DB — distinct from the sentiment-case issue, which is fixed at the processing layer):
- **`animal_farm_2025`** — self-label 20 vs implied [23,25]; first DB review 2026-04-22, first scrape 04-24, only 5 day-level rows → no deep history was ever backfilled (operator confirms reviews accumulated months pre-release).
- **`power_ballad`** — label matches settlement post-case-fix, but first DB review is 2026-05-01 (first scrape 05-07, 13 day-level rows) with suspected earlier accumulation → history known-incomplete.
- (DB facts probed 2026-06-10 @ as_of_id 652074 via `db_facts.movie_coverage`/`first_scrape` + per-slug confidence counts — re-derivable; score intervals from `validate_recorded.py`.)
- Also flagged: `backrooms` (86 vs [89,89]) and `in_the_grey` (42 vs [46,47]).

**Action (scraper repo):** re-run/extend scraping for these movies so the full review list (old reviews arrive as day-level date-format rows) enters the DB — `ON CONFLICT DO NOTHING` dedup makes it idempotent. **Acceptance test: the settlement-consistency table goes green** for the backfilled movie (the recorder auto-rejoins on its next run; no manual ledger edits). Until then these movies are `data_not_ready` for gate analyses: Gate 2/3a re-ran 2026-06-10 ex-animal_farm (canonical; §1.5), and animal_farm + power_ballad are excluded from the Gate-3b cohort (17 movies effective). Backfilling later UNBLOCKS their re-inclusion in future gate runs (cells are recomputed from pinned pulls, so adding them back is a re-run, not surgery).

### 1.10 Maker-execution experiment for KXRT (operator-signed riff 2026-06-10; gated on the C2′ taker tripwire)

**Why (the capacity asymmetry, probed 2026-06-10 — public Kalshi API + `gates/recorded/` candles, re-derivable; touch-depth/flow-rate figures are point-in-time live-book probes):** taker capacity = resting touch depth ≈ **hundreds of contracts** per contested strike (KXRT-DIS-80 live book: ~235–318 within 5¢ of touch per side) → ~$250–1,000/movie at the measured ~+20–25¢ T-3d edge. Maker capacity = window FLOW: the median tradeable movie put **~65K contracts through its T-3d contested strikes during the 3d→1d window** (per-movie sums re-derivable from `gates/recorded/candles/` with `secs_to_close ∈ (1d, 3d]` filtered to the gate3b T-3d ct tickers; median over the **13 movies that have T-3d ct cells** = 65,553 — the planning-relevant statistic, since a maker only quotes movies that pass the gates. The originally-quoted **38,155** reproduces two ways, coincidentally identical: (a) the median over all 19 recorded movies INCLUDING the six with zero ct cells — the original probe's computation, which dilutes with untradeable movies — and (b) the 13-movie median counting only two-sided-mid candles, a conservative "flow during quoted conditions" read. Michael 204K; DIS-80 alone 58K in one live 36h stretch) → a 5–15% fill share ≈ 2–9K contracts/movie at a *better* per-contract price: makers earn the spread instead of crossing it and pay the 1.75% (vs 7%) quadratic fee multiplier (`kalshi-trading/src/kalshi/fees.py:12-13`) ≈ +5–6¢/contract at-touch — and the ~6×-spread edge allows resting 10–15¢ INSIDE fair value, not at the touch. Rough ceiling $500–2,250/movie median vs taker's few hundred; KXRT volumes are also growing (DIS at 812K contracts lifetime vs far thinner gate-era books).

**The unmeasured term (the whole experiment):** maker PnL = taker-basis PnL + spread + fee-delta − **adverse selection**, and the last term cannot be estimated from candles (no fill/queue data) — only from OWN fills. The flow includes review-watchers; quote staleness is bounded by the ~50-min scraper cadence (fine for forecasting per lagged≈pure; NOT fine for quote safety — DIS absorbs ~80 reviews/day at embargo).

**Design (small, pre-registered before any order):** (a) execution lives in `kalshi-trading` (repo separation; `gates/live_scorer.py` stays read-only — it supplies fair values + logs); (b) ladder policy derived from the Gate-3a tolerance band: never rest deeper inside fair than the band-edge-implied fair-value shift; (c) requote rule tied to review arrival — baseline: cancel-all on any new target review, requote from the updated state; (d) tiny size; the measured deliverable = realized fill quality (fill price vs model fair at fill time) joined against `_cache/live_scores.csv` (**filter `mode=='live'`** — verify rows share the file); this dataset prices the adverse-selection term; (e) **gate: starts only after the K≈8 taker tripwire reads clean** (never stack an unmeasured execution change on an unconfirmed model).

### 1.8 Legacy KXRT price-history backfill parser (one-off, opportunistic — from the §1.7 recon)

`~/Desktop/kalshi-trading/data/KXRT/price_histories/` holds **145 movie directories / 436 git-tracked CSVs / 24MB** of 2024–2026 Kalshi *website* price-history exports (verified 2026-06-09: per-movie `kalshi-price-history-<event>-{minute,hour,day}.csv`, wide per-strike columns in cents — e.g. inside_out_2 minute data from 2024-05; event→`{open_date, close_date, rt_slug, disabled}` metadata in `configs/KXRT/movie_mapping.json`, 165 entries). **Single price series per strike, no bid/ask** → usable for calibration-class reads (Gate-1-style, contested under-pricing hint n) but NOT spread-crossing PnL / book-state work. Task: one-off parser → a legacy-fidelity-tier table alongside `gates/recorded/`, handling the two ticker eras (`rt*` → `kxrt*`); note repo B's `data/KXRT/movies_index.csv` (input to its mapping builder) is gitignored-absent there. Feeds: the contested-hint n and §1.4/§2.2/§2.3 p_fresh cohort growth. Source: `brainstorm/brainstorm_recorder_infra.md` §1.

## 2. Deferred model improvements

### 2.1 Finite-pool model (partially addressed at 0.2.0)

Original proposal in `brainstorm/brainstorm_finite_pool_model.md`: Poisson-binomial arrival model using per-critic remaining-pool. **Partially addressed by Ridge's finite-pool features** at 0.2.0 — `remaining_base_rate_sum`, `pool_mass_consumed`, `observed_top_tier_frac` bring pool-composition information into the regression. What's NOT done: the full per-critic Poisson-binomial architecture. Still a valid direction for eventual `compute_edge` upgrade if per-critic prediction becomes relevant (e.g., for fresh/rotten probability on a per-review basis). Low priority; no clear evidence the approximation loss of aggregate features over per-critic modeling is material.

### 2.2 Time-varying p_fresh — SUBSUMED 2026-06-10 by the p_fresh-regression riff (§1.4)

Allow p_fresh to change as a function of time-to-close (early reviews are more negative due to top-critic overweighting). See `brainstorm/brainstorm_time_varying_p_fresh.md`. **2026-06-10: Gate 3b measured the time-dependence directly** (per-cell δ̂ flips sign with horizon — §1.4) — the t_rem input of the p_fresh regression model subsumes this item; build that instead of a standalone variant.

### 2.3 Hierarchical p_fresh — FOLDS INTO the p_fresh-regression riff (§1.4)

Cross-movie shrinkage for p_fresh estimates. See `brainstorm/brainstorm_hierarchical_p_fresh.md`. **2026-06-10:** thin-history critic shrinkage (fresh-rates AND generosity thresholds) is a named component of the p_fresh regression design (`plans/plan_gate3b.md` § "Scope resolution") — folds in there rather than shipping separately.

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
