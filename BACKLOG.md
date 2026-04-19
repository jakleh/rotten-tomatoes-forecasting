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

### 1.5 SUPERSEDED: Recommended KDE parameter values

> **2026-04-19 update:** the KDE-based ship stack described below is **no longer the ship candidate**. The Ridge lambda investigation (`findings/ridge_lambda_investigation.md`) established that Ridge regression beats the KDE-ship-stack at every snapshot on the full cohort. Ship candidate is now `ridge_t2` (17 features, Ridge with per-snap CV α). Library integration plan: `plans/plan_ridge_integration.md`.
>
> The original §1.5 content is preserved below as historical context for what the KDE stack would have looked like had it shipped.

**Original KDE-stack recommendation (historical):** per `findings/stratified_training_investigation.md` §10.8 and the pre-ship tuning re-validation in §16 (2026-04-18), the deployment-recommended stack composed four interventions. None are in library code, and none will be — Ridge supersedes. Content retained below for reference:

**KDE bandwidth (§9.4 of findings):**
- `bandwidth_floor = 0.5d` (current default — keep)
- `bandwidth_ceiling = 0.7d` (NEW — add as parameter to `build_kde_lambda_model`)

**Training-set selector (§10.5-10.8, re-validated §16.3-16.4):**
- Method: `combined_score` ranking. Score = `α × exp(−|gap_diff| / σ_gap) + (1 − α) × jaccard(target_critics, candidate_critics_in_aligned_window)`. Take top `k`.
- `α = 0.5` (plateau across [0.3, 0.7], so anywhere in that range is fine)
- `σ_gap = 8.0` (re-validated §16.3: no value in {2, 4, 16, ∞} satisfies the ≥3% T-3d MAE + CI95_lo>0 rule; ∞ is meaningfully worse at T-5d/T-7d)
- `k = 20` (re-validated §16.4: same decision rule outcome; T-3d MAE plateau spans n=15-30 within ±1% of n=20. One SIG result at T-7d for n=5 is a future-work breadcrumb, not a ship change)
- Skip rule: `first_review_dbc < snap_dbc + 1` OR `len(observed_critics) < 3` → fall back to gap-only `matched_training_slugs(band=3.0, n=20)`. If that also fails, fall back to `default_training_slugs`.
- Aligned-window definition for Jaccard: target's window length = `first_review_dbc_target − snap_dbc`. For each candidate training movie, take critics in the first `window_length` days after that movie's first review.

**KDE build variant (added per Path B-lite investigation §4, 2026-04-18):**
- `build_kde_lambda_model_weighted(profiles, scores)` — **add as a new build function that takes per-movie similarity scores** (output of `combined_score_with_scores`).
- Each training movie's reviews contribute to the per-critic KDE with a weight proportional to its combined_score similarity (via `gaussian_kde(weights=...)`).
- Each critic's base_rate is likewise computed as `sum_of_weights_of_movies_they_reviewed / sum_of_all_weights`.
- Validated: cohort-wide +7.7% MAE vs equal-weight KDE at same selector, bandwidth, and snap. See `findings/path_b_lite_investigation.md` §4.
- Does NOT help h/m subset (architectural ceiling — see §1.8). Still worth shipping for cohort benefit.

**Snap convention (revised per Path B-lite investigation §7, 2026-04-18):**
- **`snap_time = close_ts.floor('D') − SNAP_DAYS × pd.Timedelta(days=1)`** (midnight UTC on close−N, not `close_ts − Nd`).
- Rationale: midnight-aligned snap gives consistent day-boundary semantics for both day-level and h/m targets. Phase_1 = 3 full calendar days before close.
- **Shift day-level reviews to 12:00 UTC** before building profiles (`day_mask = timestamp_confidence == 'd'`; `ts += 12h`). This removes the "spike-on-boundary" artifact that naive midnight-aligned snap creates. H/m reviews keep precise timestamps.
- Result: cohort mean_err from +5.95 to +1.56 (near-calibrated); h/m mean_err halved from −14.75 to −6.78. Cohort MAE higher (13.45 vs ship 6.58) — bias reduced at the cost of variance. Trade-off accepted because bias is easier to correct downstream than variance.

**Close-day piecewise patch (§8 of findings, ship form revised §17 via G1 audit 2026-04-18):**

SHIP FORM: **Phase 1 KDE to midnight UTC of close day; Phase 2 = constant C=2 reviews** (not proportional).

- **Phase 1:** `predict_window(dbc_from=snap_dbc, dbc_to=midnight_utc_dbc)` where `midnight_utc_dbc = (close_ts − close_ts.floor('D')).total_seconds() / 86400` (≈ 0.583d for 14:00-UTC closes, ≈ 0.625d for 15:00-UTC closes).
- **Phase 2:** `C = 2` (constant). Represents pre-market close-day arrivals in the 12am-UTC → 10am-EDT window.
- **Total predicted:** `phase1 + C`.
- **Not proportional to training close-day counts.** G1 audit on fresh reviews.csv (2026-04-18) showed:
  - Ship F=0.7 value was wrong by 4-5× — F=0.7 × mean_cd_training over-predicts phase 2 by ~7 reviews on the h/m-observable subset (MAE 8.34 vs MAE ~1 for constants in {1, 2}).
  - Per-movie pre-market counts on 3 clean movies (forbidden_fruits=2, they_will_kill_you=2, you_me_and_tuscany=0) show no positive correlation with close_day_count. Data supports a small baseline, not a proportional scale.
  - Constant C ∈ {1, 2} both well within MAE noise at n=3. Ship C=2 as the slightly-safer-against-under-prediction midpoint.
- **Prior framework was muddied:** the validation in findings §8 added `F × close_day_count(target)` to both predicted and actual symmetrically. For day-level targets (98% of cohort), this was proxy-vs-proxy; for h/m targets, it inflated actual beyond truth. The G1 audit bypassed this by targeting only h/m-observable movies and measuring against observed pre-market counts directly.
- **Gated follow-up:** more h/m-timestamped movies accumulating over time → re-estimate C. If cross-movie per-target F variance stays high (n=3 range was 0.0-0.67), consider per-target / per-genre phase 2 model.

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

**What this means for the validated stack:** ship `combined_score (α=0.5, σ_gap=8) + bandwidth_cap (0.7d) + piecewise(phase1 → midnight UTC, phase2 = C=2)` now (per §1.5). The long-term work above is layered on top — feature additions are non-breaking, learned weights replace fixed α only when motivated by feature growth.

### 1.5a Ridge lambda integration (supersedes §1.5)

Per `findings/ridge_lambda_investigation.md` (2026-04-19), the ship lambda estimator is Ridge regression on 17 features, not the KDE-based architecture documented in §1.5.

**Ship spec:**
- Model: `StandardScaler → Ridge(α=snap-specific)` pipeline. α selected via 5-fold CV per snap over {0.01, 0.1, 1, 10, 100, 1000}. Validated α*: 10 for T-3d/T-4d; 100 for T-5d, T-2d, T-1d.
- Features (17): 10 observation-window + 4 nonlinear transforms of `rate_last_day` / `observed_count` + 3 finite-pool aggregates. Full list in `findings/ridge_lambda_investigation.md` §4.1.
- Finite-pool base_rates computed from A1 pool (20 most recent resolved movies before target close, LOO-clean).
- Fit via LOO on cohort; shipped as a pickled artifact per release.
- **Snap / phase convention: Eastern-midnight** (anchored to market close). Snap at midnight ET on close−N; phase-1 window = (midnight ET on close, snap_time]; phase-2 constant `C=1` over (midnight ET on close, 10am ET] (10h). See `plans/plan_ridge_integration.md` §3.5b-c.

**Performance vs current library (cohort LOO):**
```
snap   library MAE   ridge_t2 MAE    Δ%      library me   ridge_t2 me
T-5d     37.96         32.14        +15%      -17.19       -0.45
T-4d     30.82         21.09        +32%      -17.85       -0.29
T-3d     17.86          9.87        +45%       -9.58       -0.01
T-2d      8.14          3.44        +58%       +6.51       -0.03
T-1d      3.87          2.24        +42%       +2.73       -0.03
```

**Integration plan:** `plans/plan_ridge_integration.md`. Phases A-E library-side (~4 days), Phase F orchestrator migration. Version bump 0.1.0 → 0.2.0 (breaking API).

### 1.8 High-volume & late-surge under-prediction — architectural ceiling (KDE-specific, BYPASSED by Ridge)

> **2026-04-19:** this ceiling was specific to the `base_rate × KDE × exclusion` architecture. Ridge (per §1.5a) bypasses the sum mechanism and regresses on observation-window features directly. Ridge's h/m performance under LOO is comparable to ship-stack at worst, better at most snaps. The architectural ceiling described below remains a valid description of the KDE's limitation; it just no longer blocks deployment.

Identified via G1 F-audit (2026-04-18). At T-3d, KDE under-predicts by 14-34 reviews for certain movies. **Two distinct failure modes** (disambiguated per `findings/path_b_lite_investigation.md`):

1. **High-volume short-gap** (e.g., `the_drama` gap=6.4d, `super_mario` gap=5.8d): typical-observed but high-future-activity mass releases.
2. **Late-surge long-gap** (e.g., `they_will_kill_you` gap=12.6d, `forbidden_fruits` gap=13.6d): most reviews arrive in the final 3 days; observed-window signal doesn't predict the surge.

**Mitigations ruled out (Path B-lite investigation, 2026-04-18):**

- ❌ **Scaling upper-clamp loosening** — cohort MAE monotonically worse with looser clamps; h/m subset unchanged. Clamp isn't binding for most of the affected movies.
- ❌ **Scaling threshold lowering** — 0% of targets are threshold-gated (expected_so_far always > 40 because it sums over all critics). Threshold change can't help.
- ❌ **σ_gap narrowing for long-gap targets** — long-gap targets have no gap-similar neighbors regardless of σ_gap; falls back to Jaccard-dominated ranking.
- ❌ **Pool expansion to n=50** — unweighted degrades cohort MAE −21%; weighted n=50 can't overcome.
- ❌ **Volume feature as fixed-weight addition** — Q4 improves 3-5% but Q1 regresses 20%. Signal is real but can't be deployed with fixed weights.
- ❌ **Per-target base_rate adjustment** (capped tiered multiplier) — cohort **+20.9% (!!)** from Q1 over-prediction fix, BUT h/m subset regresses −36% because `observed_count` tier signal misclassifies late-surge long-gap targets. Need better target-volume signal before shipping.
- ❌ **Time-series extrapolation** (const/last-day/exp-decay) — per-target variance too high; cohort blowups; h/m subset improvements inconsistent.
- ❌ **Shape-similarity selection** (cosine on early-arrival daily vectors) — helps `the_drama` slightly (+5 error), hurts others.
- ❌ **Oracle end-shape selection** (using actual end-shape, unavailable at inference) — `the_drama` still −19; `they_will_kill_you` still −34. **This confirms the limit is architectural**, not selection-based.
- ❌ **Shape × Scale point-estimate model** — cluster shape too rigid; cohort MAE 23.34 (vs baseline 13.45). Cluster shape doesn't transfer to targets.
- ❌ **Ridge + KDE blend** — cohort and h/m have opposite optimal blend weights. Averaging hurts per-target predictions when one model is right.
- ❌ **Active/generic critic-magnet boost** (Jake's hypothesis, testing whether generic-critic engagement signals magnets) — falsified. 4 of 5 h/m under-predictors have LOW generic-critic rates. Signal fires on wrong movies (e.g., Wicked at z=+1.80 — already well-predicted).
- ❌ **Hierarchical Bayes with Gamma shape** — cohort MAE 46.87, h/m MAE 36.96. Gamma is mis-specified; arrival distributions have heavier tails than Gamma allows. Non-parametric or mixture variants not yet tested.

**Ridge regression as architectural alternative (§9 of findings):**
Cohort MAE 8.85 (best of any intervention), near-calibrated everywhere (cohort me −0.30, h/m me +0.69). H/m MAE 26.91 is worse than KDE's 18.23 due to per-target variance. `rate_last_day` is the dominant feature (GBM importance 0.61). Not a clear replacement for KDE — it's a different failure-mode profile (calibrated aggregate, variable per-target vs biased aggregate, stable per-target).

**Root cause (confirmed via oracle test, §8.2 of findings):**
The `base_rate × KDE × observed-critics-exclusion` chain has a structural ceiling:
- When a target's `observed_count` matches expected, `_compute_scaling ≈ 1.0` — "looks normal."
- But for atypical high-activity movies, MORE critics pile on late than base_rates predict.
- The excluded "observed critics" leave behind low-base_rate critics whose KDE contribution is small.
- No selection improvement can escape this because the ceiling is in the sum mechanism, not the training set.

**Remaining mitigation directions (not yet tried or only partially tested):**

- **Finite-pool model** (`brainstorm/brainstorm_finite_pool_model.md`) — architectural rework. Requires per-target per-critic P(review | target), which is essentially the Path B features problem.
- **Hier Bayes with log-normal or mixture-Gamma shape** — Gamma null (see ruled-out list) doesn't invalidate the approach; finding the right shape family is itself a research problem.
- **Per-critic base_rate learning conditioned on target features** — learn `P(critic reviews | movie_features)`. Requires external metadata or more per-critic data.
- **External metadata (TMDb)** — deprioritized by Jake. Slug-to-ID mapping is config-heavy and we lack a clean hypothesis for which features would help.
- **Wait for more h/m cohort data** — as cohort shifts toward h/m representation, some of the ruled-out interventions may become effective. Multi-month wait.

**For current deployment:**
- **Two ship candidates** (per findings §15):
  - **Weighted-KDE + midnight+noon** — h/m MAE 18.23 (best per-target), but mean_err −6.78. Predictable per-target bias.
  - **Ridge(α=10)** — cohort MAE 8.85 (best), near-calibrated everywhere, but h/m MAE 26.91 (per-target variance).
- Choice depends on deployment priorities (bias correction vs per-target stability).
- Orchestrator applies downstream offset correction for known-bias patterns regardless.

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
