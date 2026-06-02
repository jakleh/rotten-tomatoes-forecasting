# Findings: Path B-lite Investigation

> **2026-04-19 CONVENTION WARNING:** this doc uses the older UTC-midnight snap convention and KDE-based architecture. Ship lambda model is now Ridge regression per `findings/ridge_lambda_investigation.md`; conventions are in `CLAUDE.md` (midnight ET snap, 10h phase-2 window, `C=1`). This doc remains as historical validation context — DO NOT copy timestamp, snap, phase, or architectural decisions from this doc into new code.

**Date:** 2026-04-18.
**Status:** Active investigation. Many mitigations ruled out. Midnight+noon snap convention adopted. Weighted-KDE selection candidate identified. Architectural limit for h/m under-prediction confirmed via oracle test. Time-series regression as alternative architecture under test.
**Scope:** Successor to pre-ship tuning. Focused on (a) resolving `BACKLOG §1.8` (high-volume T-3d under-prediction) and (b) validating the deployment-relevant convention when h/m timestamps are the expected target format.
**Related:** `findings/stratified_training_investigation.md` (precursor); `BACKLOG.md §1.5, §1.7, §1.8`; `plans/plan_learned_similarity_model.md` (deferred Path B).

---

## TL;DR

- **The h/m under-prediction for live-tracked movies has TWO distinct failure modes:**
  1. **High-volume short-gap** (e.g., `the_drama`, `super_mario`) — mass-release movies where training-volume doesn't match.
  2. **Late-surge long-gap** (e.g., `they_will_kill_you`, `forbidden_fruits`) — movies where most reviews arrive in the final 3 days, with observed-window signal not predicting the surge.

- **Scaling clamp/threshold mitigations: dead ends.** Diagnostic showed scaling is going the wrong direction for long-gap movies (down-scales when it should up-scale) and never reaches threshold-gating. Neither upper-clamp loosening nor threshold sweeping helps.

- **σ_gap narrowing for long-gap targets: null.** Long-gap targets (`americana_2025`, `the_apprentice` etc.) have no gap-similar neighbors regardless of σ_gap — the selector falls back to Jaccard-dominated ranking either way.

- **Volume feature in combined_score: small Q4 gain, stratum distortion.** Adding a volume-match feature (weighted by σ_vol=17.8, IQR/2) gives Q4 MAE −3-5% improvement but hurts Q1 (+20%). Net: not shippable as a fixed-weight addition.

- **Weighted KDE (per-point weights from combined_score): +7.7% cohort MAE.** Replaces equal-weight `gaussian_kde` with weighted variant where high-similarity training movies' review timings dominate the shape. Single-parameter architectural extension. Doesn't help h/m subset (MAE 14.75 → 14.75), but is a real cohort win worth shipping.

- **Pool expansion to n=50: null.** Expanding the training pool degrades cohort MAE even with weighting (7.40 → 7.56). Best is n=20.

- **Per-target base_rate adjustment with tiered lookup (Option C): strong Q1 fix, Q4 regression.** Raw multiplier matrix gives cohort −37.7% (over-corrects); asymmetric cap (allow down-scaling only) gives cohort **+20.9%** with Q1 MAE +42.6% fixed, but h/m subset MAE regresses −36.3% because the target-volume signal (`observed_count` tier) mis-classifies late-surge long-gap movies.

- **Time-series extrapolation (const rate, last-day rate, exp decay): too noisy for deployment.** H/m subset MAE varies wildly across methods; per-target cohort variance is enormous; some methods give blended h/m gains but destroy cohort.

- **Midnight-aligned snap semantics (midnight UTC on close-3) — ADOPTED.** Cleaner day-boundary semantics than `close_ts - 3d` (14:00 UTC). Naive midnight-aligned creates a "spike-on-boundary" artifact; **noon-shifting day-level reviews to 12:00 UTC fixes it.** Combined midnight+noon convention: cohort mean_err from +5.95 to +1.56 (near-calibrated), h/m mean_err from −14.75 to −6.78 (halved). MAE higher cohort-wide but better calibrated. **Ship going forward.**

- **Shape visualization: cohort contains shape-similar neighbors for each h/m target**, but combined_score picks almost entirely disjoint sets (overlap 1-11/20). The neighbors exist; our similarity metric wasn't finding them.

- **Oracle end-shape selection: architectural limit confirmed.** Training on top-20 end-shape-matched candidates (a signal we don't have at inference) for `the_drama`: prediction 40.36 vs actual 59. For `they_will_kill_you`: 33.07 vs actual 67. **Even with perfect selection, KDE under-predicts by 19-34 reviews.** Root cause: the observed-critics exclusion mechanism plus low-base_rate of remaining critics means the KDE can't escape its architectural ceiling for movies where "more critics pile on late."

- **Decision to not invest in TMDb metadata path:** slug-to-ID mapping is a config headache AND we lack a clean hypothesis for which external features would help — the shape-viz showed cohort neighbors exist, so the problem isn't "we need more metadata-like features," it's the underlying prediction mechanism.

- **Time-series regression (Ridge α=10): cohort MAE 8.85 (vs KDE 13.26), near-calibrated everywhere (cohort me −0.30, h/m me +0.69), but h/m MAE 26.91 due to per-target variance.** Wins some targets (`the_drama` err +0.38) and loses others (`super_mario` +50). Legitimate architectural alternative; `rate_last_day` is the dominant learned feature.

- **Blend of Ridge + KDE doesn't dominate** (§10). Cohort and h/m have opposite optimal blend weights; averaging per-target pulls predictions away from whichever model is right.

- **Active/generic critic-magnet boost was falsified** (§11). 4 of 5 h/m under-predictors have LOW generic-critic rates, not high. Signal fires on wrong movies.

- **Hierarchical Bayesian with Gamma shape failed** (§12). Gamma mis-specified for arrival data (too exponential-tailed). Non-parametric or mixture variants not yet tested.

- **After 14+ interventions, the architectural ceiling holds.** H/m-target under-prediction isn't fixable with currently-observable features at T-3d. Breaking through requires new information (metadata, more h/m cohort data, richer observation windows).

- **Ship candidates:** weighted-KDE (midnight+noon) for per-target stability with known h/m bias; Ridge for calibrated cohort with high per-target variance. Neither dominates — tradeoff depends on deployment priorities.

---

## 1. Origin

After pre-ship tuning locked σ_gap=8 and n_training=20 per `stratified_training_investigation.md` §16, attention shifted to `BACKLOG §1.8` — high-volume movies (specifically the 5 h/m live-tracked targets) are under-predicted at T-3d by 14-34 reviews while cohort-wide MAE stays low.

At this point the cohort understanding was:
- 142 movies total, ~95% day-level timestamps
- 5 h/m-live-tracked movies: `the_drama`, `the_super_mario_galaxy_movie`, `forbidden_fruits_2026`, `they_will_kill_you`, `you_me_and_tuscany`
- All future Kalshi-tracked movies will have h/m timestamps (scraper is running). **The h/m subset is the deployment-representative test set.**

Pre-ship tuning was optimizing cohort-MAE (day-level-dominated), not deployment-MAE (h/m). The interventions in this investigation are evaluated primarily against h/m subset, with cohort MAE as a don't-regress guard.

---

## 2. Scaling mitigations — ruled out

### 2.1 Upper-clamp sweep (§1.8 first proposal)

**Test:** sweep `_compute_scaling`'s upper clamp ∈ {2.0, 3.0, 4.0, ∞} at T-3d under original snap convention. Hypothesis: clamp at 2.0 blocks up-scaling for high-volume outliers.

**Result:** NULL. Cohort MAE monotonically increases with looser upper clamp (T-3d: 7.40 → 7.51). H/m subset: MAE 14.78 unchanged. Notebook: `scaling_clamp_sweep.ipynb`.

### 2.2 Scaling diagnostic — mechanism exposed

**Test:** log `expected_so_far`, `observed_count`, `obs/expected` ratio, and actual scaling_factor per target at T-3d. Notebook: `scaling_diagnostic.ipynb`.

**Key findings:**
- **0% of targets are threshold-gated** across all quartiles (expected_so_far is always > 40 because it sums over ALL critics, not just unreviewed). So lowering threshold wouldn't help.
- **Scaling is going the wrong direction** for 2 of 5 h/m targets (both long-gap: `forbidden_fruits` at ratio 0.21 → scaling 0.50; `they_will_kill_you` at ratio 0.48 → scaling 0.50, lower clamp hit).
- For these targets the KDE over-predicts in the observed window (so ratio < 1), but under-predicts in the phase-1 window. Scaling applies the wrong correction.

**Conclusion:** scaling mechanism cannot be fixed by clamp/threshold tuning — it's misaligned with the actual error structure for long-gap shape-mismatched targets.

### 2.3 σ_gap narrowing for long-gap targets (BACKLOG followup)

**Test:** use σ_gap=4 (tight) for targets with `target_gap > 10d`, σ_gap=8 otherwise. Hypothesis: long-gap targets might benefit from tighter gap-matching.

**Result:** NULL. Cohort MAE −0.7%, long-gap subset −1.7%, h/m 0% (many long-gap targets have no gap-similar neighbors at any σ_gap; they fall back to Jaccard-dominated ranking regardless). Notebook: `shape_fix_long_gap.ipynb`.

**Three cheap mitigations ruled out.** §1.8 should note these are dead.

---

## 3. Volume feature in combined_score — mixed

**Test:** extend combined_score with a volume-match term: `combined_v2 = w_gap × gap + w_jac × jaccard + w_vol × vol_sim`, where `vol_sim = exp(−|rate_target − rate_candidate|/σ_vol)`. σ_vol=17.8 (cohort rate IQR/2). Sweep `w_vol` ∈ {0, 0.1, 0.2, 0.33, 0.5} at T-3d. Notebook: `phase_b0_volume.ipynb`.

**Result:** mixed. Q4 improves ~3-5% at w_vol=0.5, but Q1 regresses +20% because low-rate targets get matched to similarly-low-rate training that produces even lower predictions. Net cohort: slight regression.

**Why:** rate-matching conflates "slow-start" with "low-total-volume." A long-gap target with low observed rate has lots of future activity; a short-gap small movie with low rate actually has low total activity. Using rate alone can't distinguish.

**Conclusion:** volume signal has real information but can't be integrated as a fixed-weight addition. Would need learned-weight or stratum-conditional modeling — which is Path B proper.

---

## 4. Weighted KDE — real cohort win

**Test:** replace equal-weighted `gaussian_kde(timing_data)` with `gaussian_kde(timing_data, weights=...)` where per-data-point weights = the source training movie's `combined_score` value. Same selection (top-20), same bandwidth cap (0.7d). Notebook: `path_b_lite_weighted_kde.ipynb`.

**Result:**

```
Full cohort MAE:          7.40 (ship) → 6.83 (weighted)    +7.7%
Q1 (0-7):                9.63        → 9.54                 +0.9%
Q2 (8-11):               7.05        → 6.50                 +7.8%
Q3 (12-18):              6.44        → 5.35                +16.9%  largest
Q4 (19-52):              6.31        → 5.76                 +8.8%
H/m subset:             14.75        → 14.75                +0.2%  no help
```

**Per-data-point weighting effect:** when training movies have different similarities to target, their review timings now contribute proportional to similarity. Equivalent to a soft filter: training that's less similar contributes less to the KDE shape.

**Cohort win is larger than any intervention since bandwidth cap.** Mechanically, it concentrates the KDE's density around the timing patterns of the most-similar training movies, which improves average-case predictions.

**H/m subset unchanged** because top-20 similarities for h/m movies are in a narrow range (0.55-0.65 for most neighbors), so weighted-KDE doesn't differentiate them much from equal-weight.

**SHIP** this under midnight+noon convention (see §7) as our selection-family improvement.

---

## 5. Pool expansion n=50 — null

**Test:** expand training pool to top-50 (weighted by combined_score). Hypothesis: with more candidates, weighted KDE can find better matches for outliers. Notebook: `path_b_lite_weighted_n50.ipynb`.

**Result:**

```
config            cohort MAE   delta vs ship
ship (n=20)           7.40      —
unweighted n=50       8.98      −21.4%
weighted n=20         6.83      +7.7%
weighted n=50         7.56      −2.2%
```

Pool-size expansion alone HURTS (unweighted n=50: −21.4%). Weighting partially recovers but can't overcome the damage. Best remains weighted n=20.

**Interpretation:** at n=50 weighted, the KDE is pulled toward cohort-mean shape; weighting can't concentrate tightly enough on the top-20 similar movies. n=20 was the right choice.

---

## 6. Per-target base_rate adjustment — Q1 win, Q4 regression

**Test:** tiered-lookup multiplier matrix. 3 critic tiers × 4 target tiers, multiplier = `P(review | c_tier, t_tier) / P(review | c_tier)`. At inference, adjust each critic's base_rate by `multiplier[critic_tier, target_tier]`. Notebook: `option_c_base_rate_adjustment.ipynb`.

**Raw multiplier result:** cohort MAE +58% WORSE. Multipliers > 1 for high-tier targets compound with `_compute_scaling` and push adjusted_base_rate beyond probability bounds (1.4+).

**Asymmetric-cap result** (`multiplier = min(raw, 1.0)` — down-scale only):

```
Cohort MAE:              6.83 → 5.40    +20.9%
Q1 (0-7):                9.54 → 5.47    +42.6%  ← huge win
Q2 (8-11):               6.50 → 4.41    +32.1%
Q3 (12-18):              5.35 → 4.93    +7.8%
Q4 (19-52):              5.76 → 6.90    −19.9%
H/m aggregate:          14.75 → 20.10   −36.3%  ← strong regression
```

**Mechanism analysis:** Target_tier is assigned by `observed_count` at T-3d. Low-observed targets get multipliers 0.44-0.69 → down-scale. This fixes Q1 over-prediction (mass-market critics no longer over-counted for niche movies).

**But long-gap h/m targets (they_will_kill_you, forbidden_fruits) have LOW observed_count** relative to their future activity. Misclassified as "small movie" → down-scaled → under-prediction amplified.

**Conclusion:** per-target base_rate adjustment is a real structural lever for Q1 over-prediction, but the target-tier signal (`observed_count`) is wrong for late-surge movies. Would need a better target-volume signal to deploy safely.

---

## 7. Midnight+noon snap convention — ADOPTED

**Problem:** original snap convention `snap_time = close_ts − 3d = 14:00 UTC on close-3` creates an asymmetry:
- Day-level training reviews on close-3 are at midnight UTC (dbc=3.583) → observed pre-snap.
- H/m target reviews on close-3 afternoon (e.g., 18:00 UTC, dbc=2.83) → in phase-1 window.

KDE built from day-level training has a "blind spot" in the `(2.58, 3.0)` dbc zone. H/m targets have real activity there that training can't predict.

### 7.1 Midnight-aligned snap alone (naive)

**Test:** `snap_time = close_ts.floor('D') − 3d = midnight UTC on close-3`. Clean day-boundary semantics. Notebook: `midnight_snap_test.ipynb`.

**Result:** cohort mean_err from +5.95 to **−14.18** (systematic under-prediction). Because day-level training spike at dbc=3.583 (midnight UTC close-3) lands exactly on the phase-1 upper boundary. KDE integral captures only half the spike mass via bandwidth smoothing.

### 7.2 Noon-shift fix

**Test:** shift all day-level reviews (timestamp_confidence='d') from midnight UTC to noon UTC. Day-level spike moves from dbc=3.583 to dbc=3.083, cleanly inside the `(0.583, 3.583]` phase-1 window. No boundary-on-spike. Notebook: `noon_shift_test.ipynb`.

**Result (2×2 comparison):**

| config | cohort MAE | cohort me | h/m MAE | h/m me |
|---|---|---|---|---|
| ship (orig_snap + no_shift) | **6.58** | +4.98 | 14.75 | −14.75 |
| orig_snap + noon_shift | 14.54 | +12.02 | 15.04 | −14.30 |
| midnight_snap + no_shift | 17.83 | −14.18 | 17.86 | −14.64 |
| **midnight_snap + noon_shift** | 13.45 | **+1.56** | 18.23 | **−6.78** |

**Adopted midnight+noon:**
- Cohort mean_err near-zero (+1.56 vs +4.98 in ship).
- H/m bias halved (−6.78 vs −14.75).
- Cohort MAE higher (13.45 vs 6.58) — tradeoff: bias reduction for variance.
- Cleaner semantics: 3 full calendar days of phase-1 window consistent for h/m and day-level.

**For trading:** bias matters more than variance. Consistently biased predictions can be corrected by downstream offset; calibrated predictions with higher variance average out over many bets.

### 7.3 Per-target h/m change under midnight+noon

```
target              orig_err    midnight+noon_err
forbidden_fruits    −3.09       +0.22       ← fixed
the_drama          −33.67      −20.58       ← improved by 13
super_mario         −5.34      +21.74       ← flipped
they_will_kill_you −21.59      −41.94       ← worse (actual also jumped 34 → 67)
you_me_and_tuscany −10.07       +6.68       ← improved
```

Mixed per-target but aggregate bias halved. Per-target variance exposed once the structural bias was removed.

---

## 8. Shape visualization + oracle test — architectural limit confirmed

### 8.1 Shape-similarity visualization

Plotted per-movie daily-arrival curves (day-level collapsed), aligned at first-review-day=0. H/m targets highlighted. Notebook: `shape_viz.ipynb`. Plots saved: `shape_viz_curves.png`, `shape_viz_the_drama.png`.

**Finding:** cohort DOES contain movies with visually-similar arrival shapes to each h/m target. They exist. But combined_score picks almost entirely disjoint sets:

| target | shape-top-20 ∩ combined_score-top-20 |
|---|---|
| the_drama | 11/20 |
| super_mario | 6/20 |
| forbidden_fruits | 1/20 |
| they_will_kill_you | 1/20 |
| you_me_and_tuscany | 2/20 |

For long-gap targets (forbidden_fruits, they_will_kill_you), combined_score misses nearly all shape-similar neighbors.

### 8.2 Oracle end-shape selection

**Test:** compute top-20 training by cosine-similarity on end-shape (daily counts in days close-7 to close-1) — a signal we don't have at inference time. Use those as weighted training. Predict h/m phase-1 under midnight+noon. Notebook inline: `/tmp/claude/end_shape_oracle.py`.

**Result:**

| target | actual | cs_pred | oracle_end_pred | improvement |
|---|---|---|---|---|
| the_drama | 59 | 38.42 | **40.36** | +1.94 (marginal) |
| super_mario | 33 | 54.74 | 61.17 | −6.43 (worse) |
| forbidden_fruits | 28 | 28.22 | 21.89 | −6.11 (worse) |
| they_will_kill_you | 67 | 25.06 | **33.07** | +8.01 (some) |
| you_me_and_tuscany | 35 | 41.68 | 42.38 | −0.70 (same) |

**Even with perfect end-shape matching**, the KDE under-predicts by 19-34 reviews on the big under-predicted targets.

### 8.3 Why the KDE has an architectural ceiling

```
predicted_remaining = Σ (unreviewed critics) base_rate[c] × KDE_integral[c] × scaling
```

The `observed_critics` exclusion removes critics who've already reviewed. For `the_drama` with 113 observed critics at snap, those are gone. The `remaining` ~1500 critics have LOW base_rates (they're the ones who didn't review most training movies).

For a HIGH-ACTIVITY target like `the_drama`, reality diverges: many of those low-base_rate critics DO pile on when it's a big movie. They review at HIGHER rates than their base_rate suggests.

The base_rate is a PER-CRITIC average across training. It assumes "this critic reviews at rate X regardless of target." In reality, critics' review probability depends on the target. Big/popular targets attract critics who usually wouldn't bother.

`_compute_scaling` is supposed to handle this, but it clamps at 2.0 and for `the_drama` doesn't even reach the clamp (observed=113, expected=120, scaling=0.94 — slight DOWN).

**No selection improvement can escape this** — we've verified via the oracle. The architecture itself (base_rate × KDE × exclusion) has a ceiling for this failure mode.

---

## 9. Time-series regression — promising

**Hypothesis:** bypass the per-critic architecture. Predict `actual_phase_1` directly via regression on observed features. Simpler; no `base_rate × KDE × exclusion` chain.

**Features at snap (midnight+noon T-3d):** observed_count, first_review_dbc, target_gap, observed_rate, rate_last_day, rate_first_day, top_critic_frac, pub_diversity, pub_entropy, low_activity_frac. Target: `actual_phase_1`. Notebook: `time_series_regression.ipynb`.

### 9.1 Results

```
model                 cohort_MAE_CV    h/m_MAE    h/m_mean_err
OLS                       10.09        26.59       −0.09
Ridge(alpha=1)             9.86        26.74       +0.47
Ridge(alpha=10)            9.71        26.91       +0.69
GBM(100, depth3)          10.94        22.70       −7.32
GBM(300, depth2)          11.60        23.67       −7.93

Baseline weighted-KDE:    13.45*       18.23       −6.78       *midnight+noon convention
```

### 9.2 Cohort win

**Ridge(alpha=10) beats the midnight+noon weighted-KDE on cohort MAE** (9.71 vs 13.45, a 28% improvement). Cohort is day-level-dominated; linear regression with observed features captures most of the phase-1 signal.

### 9.3 H/m calibration

**Linear models (OLS, Ridge) are near-calibrated on h/m** (mean_err −0.09 to +0.69) vs KDE's −6.78. This is a CLEANER calibration than KDE achieves.

But h/m MAE is worse (26.59 vs 18.23) because per-target predictions swing:

```
target                  actual    wKDE     OLS    Ridge(1)   GBM(100)
forbidden_fruits            28    28.22   10.43   11.82       6.57
the_drama                   59    38.42   59.38   59.84      51.36
super_mario                 33    54.74   82.42   82.93      59.50
they_will_kill_you          67    25.06   17.85   17.52      21.04
you_me_and_tuscany          35    41.68   51.45   52.25      46.94
```

**Linear models get `the_drama` nearly perfect** (err +0.38 to +1.25). This was the biggest under-predicted case for KDE (err −20.58). A big win on the hardest case.

But they overshoot `super_mario` by +50 and undershoot `they_will_kill_you` by −49. High variance.

### 9.4 Feature importance (GBM)

```
rate_last_day        0.613    ← dominant
observed_count       0.165
rate_first_day       0.050
top_critic_frac      0.040
pub_diversity        0.034
observed_rate        0.029
pub_entropy          0.027
low_activity_frac    0.023
target_gap           0.009
first_review_dbc     0.008
```

**`rate_last_day` (rate in the final day of observed window) is ~4× more important than any other feature.** Recent momentum is the primary signal. GBM learns a "high recent rate → expect high phase_1" relationship that linear models approximate less flexibly.

### 9.5 Interpretation

Regression offers a **different failure-mode profile** than KDE:
- **Cohort MAE:** regression wins under midnight+noon convention.
- **H/m calibration:** regression is materially better (me ≈ 0 vs KDE me ≈ −7).
- **H/m MAE:** regression is worse because it swings both directions per-target.

The two architectures get different things right. `the_drama` (short-gap, high-volume decaying) is solved by regression. `they_will_kill_you` (long-gap, late-surge) remains hard for both — no observable feature at T-3d predicts the surge.

### 9.6 Deployment implications

Three candidate architectures:

1. **KDE (midnight+noon):** h/m MAE 18.23, h/m me −6.78, cohort MAE 13.45. Predictable per-target, biased aggregate.
2. **Ridge:** cohort MAE 9.71, h/m MAE 26.91, h/m me +0.69. Calibrated aggregate, volatile per-target.
3. **Blend:** weight `w·KDE + (1−w)·Ridge` where `w` depends on `n_observed_target`. Could combine calibration + stability.

For trading: if the orchestrator aggregates over many bets, Ridge's calibration is attractive. If per-target prediction quality matters more, KDE's lower variance wins.

**Not yet adopted** — worth a follow-up experiment to test the blend and see if it dominates both.

---

## 10. Blend of Ridge + weighted-KDE — null result

Tested whether a weighted combination `pred = w × Ridge + (1 − w) × KDE` could combine Ridge's calibration with KDE's per-target stability. Notebook: `blend_ridge_kde.ipynb`.

### Constant-w sweep

| w_ridge | cohort MAE | h/m MAE | h/m me |
|---|---|---|---|
| 0 (KDE) | 13.67 | **18.23** | −6.78 |
| 0.5 | 10.54 | 22.03 | −3.53 |
| **1 (Ridge)** | **9.12** | 25.92 | **−0.29** |

Cohort and h/m have **opposite optimal blends**. Any w in (0, 1) is dominated by one endpoint for each metric.

### Adaptive blend (w = min(1, n_obs / threshold))

| threshold | cohort MAE | h/m MAE | h/m me |
|---|---|---|---|
| 40 | 9.56 | 24.86 | +0.76 |
| 160 | 10.45 | **21.07** | **−0.08** |
| 200 | 10.76 | 20.49 | −1.42 |

Best adaptive at threshold=160: h/m MAE 21.07 with near-zero bias. Still 2.8 worse than pure KDE on h/m MAE.

### Why blending fails

Per-target at w=0.5:
- `forbidden_fruits`: KDE 28.22 (perfect — actual 28), Ridge 12.04, blend 20.13. Ridge's error drags the blend.
- `the_drama`: KDE 38.42, Ridge 58.49 (perfect — actual 59), blend 48.45. KDE's error drags the blend.

**Blend averages per-target even when one model is right.** Classic weakness — averaging helps when errors are independent; here each model is CORRECT on different targets. A meta-model that ROUTES each target to the right base model would help, but with 5 h/m data points we can't fit one robustly.

---

## 11. Active/generic critic-magnet boost — null result

Tested Jake's hypothesis: if BOTH active (top-N) AND generic critics are observed in unusual numbers, target is a "critic magnet" → boost remaining critics' base_rates. Notebook: `critic_magnet.ipynb`.

### Setup

- `active_set = top N` critics by cohort activity (swept N ∈ {30, 50, 100})
- `generic_observed_rate = |observed ∩ generic| / |generic|`
- Boost = `1 + k × max(0, (target_rate − cohort_mean) / cohort_std)` (asymmetric — only boost up)
- Applied to remaining generic critics' base_rates

### H/m z-scores (n_active=50)

| target | z | active count / 50 |
|---|---|---|
| the_drama | +0.01 | 30 |
| super_mario | **+0.49** | 30 |
| forbidden_fruits | **−1.25** | 5 |
| they_will_kill_you | **−1.05** | 15 |
| you_me_and_tuscany | **−0.97** | 13 |

**Signal doesn't fire for the right movies.** 4 of 5 h/m targets have NEGATIVE z-scores (below cohort baseline generic rate). Only `super_mario` has meaningful positive z, and `super_mario` is already OVER-predicted by KDE — boosting makes it worse.

### What happens at k_boost=2.0

Cohort MAE: 13.26 → **57.78** (cohort worse by 335%). The boost fires on non-h/m movies with high generic-rate (e.g., Wicked at z=+1.80), inflating their already-good predictions into over-predictions.

### Why the hypothesis failed

The h/m under-predictors are **long-gap or niche movies whose embargo-rush hasn't hit at T-3d**. They have LOW generic-critic engagement in the observed window, not high. Their future surges are driven by release-timing dynamics we can't see at snap.

Jake's intuition was mechanistically coherent — "critic magnets attract generic reviewers" — but empirically, the observable signal at T-3d doesn't separate magnets from non-magnets in our cohort.

---

## 12. Hierarchical Bayesian (Gamma shape) — big failure

Tested empirical Bayes with per-target Gamma(α, β) shape, partial-pooled with combined_score-weighted cluster prior. Notebook: `hier_bayes_shape.ipynb`.

### Results

```
prior_strength   cohort_MAE   cohort_me    h/m_MAE    h/m_me
           40        46.87      −24.43      36.96    −36.96
          160        42.29      −18.80      33.09    −33.09
```

**Worst result produced in the entire investigation.** Systematic under-prediction of 25-38 reviews across cohort AND h/m.

### Root cause: Gamma is mis-specified

For `the_drama` (actual 59): posterior α=1.67, β=1.28 → Gamma mean = 1.30 days. Shape says 91% of reviews have arrived by day 3.58. V is inferred at 124 (under by ~40%); predicted future = 10.26 (actual 59).

The actual arrival pattern has a heavier tail than Gamma allows. Arrivals peak early (embargo lift) and continue at meaningful rates for many days — Gamma's exponential tail doesn't capture this.

### What could work

- **Log-normal** (heavier tail than Gamma)
- **Mixture of two Gammas** (embargo spike + sustained tail)
- **Non-parametric partial pooling** on empirical CDFs (no parametric assumption)

Each is 1-2 days of additional work. **Shape-family choice is itself a research problem.** This null doesn't invalidate the hierarchical idea; it does say "picking the right shape family is harder than it looks and not guaranteed to work."

---

## 13. What's ruled out

Mitigations that demonstrably do NOT fix the h/m under-prediction:

1. **Scaling upper-clamp loosening** (§2.1)
2. **Scaling threshold lowering** (§2.2 — threshold not gating anyway)
3. **σ_gap narrowing** for long-gap targets (§2.3)
4. **Pool expansion** to n=50 (§5)
5. **σ_gap=∞ / σ_gap→0 extremes** (pre-ship §16.3)
6. **Volume feature as fixed-weight addition** to combined_score (§3)
7. **Per-target base_rate adjustment** with `observed_count` tier signal (§6 — works for Q1 but hurts long-gap h/m)
8. **Time-series extrapolation** (const rate, last-day rate, exp decay) — too noisy per-target (`option_c_extrapolation.ipynb`)
9. **Shape-similarity selection** as replacement for combined_score (helps short-gap high-vol slightly, hurts long-gap)
10. **Oracle end-shape selection** (§8.2) — confirms architectural ceiling
11. **Shape × Scale model** (point-estimate factorization) — cluster shape too rigid; cohort MAE 23, h/m MAE 24
12. **Ridge + KDE blend** (§10) — cohort and h/m have opposite optimal blends; neither dominates
13. **Active/generic critic-magnet boost** (§11) — signal doesn't fire on the right movies (h/m targets have LOW generic rate, not high)
14. **Hierarchical Bayes with Gamma shape** (§12) — Gamma mis-specified; arrivals are heavier-tailed than Gamma allows

## 14. What's still open

1. **Time-series regression (Ridge)** — produces cohort MAE 9.71 (best so far under midnight+noon) and near-calibrated on both cohort and h/m. High per-target variance is the cost. Candidate alternative to KDE or as co-ship model (§9).
2. **TMDb metadata path** — deprioritized. Config headache, unclear features. Jake explicitly declined.
3. **Finite-pool model** (`brainstorm/brainstorm_finite_pool_model.md`) — could attack the exclusion-mechanism issue directly. Requires per-target per-critic P(review | target), which is essentially the Path B features problem.
4. **Hierarchical Bayes with different shape family** (log-normal, mixture of Gammas, non-parametric) — Gamma null doesn't invalidate the approach, but finding the right shape family is itself a research problem.
5. **Per-critic base_rate learning conditioned on movie type** — learn `P(critic reviews | movie_features)`. Requires movie features we don't have or the per-critic data we don't have.
6. **Hierarchical p_fresh** (`brainstorm/brainstorm_hierarchical_p_fresh.md`) — not directly related to lambda but adjacent; could help the binomial side of compute_edge.
7. **More h/m data** — over time the cohort fills in with representative targets. Multi-month wait.

### The architectural ceiling — confirmed stable

Across 14 distinct interventions in this investigation, h/m MAE didn't materially improve while maintaining cohort non-regression. The h/m under-prediction is a property of the **features available at T-3d**, not the architecture:
- Long-gap movies' late surges aren't predictable from pre-surge observations.
- High-volume short-gap movies' actual volumes aren't distinguishable from typical volumes via observed_count, rate, critic overlap, pub diversity, or any combination we've tested.

Breaking through this ceiling requires new information sources:
- External movie metadata (we declined to pursue TMDb)
- More h/m cohort data (wait)
- A richer observation window (pre-snap minute-level rate patterns — requires h/m data for all targets)
- A different observation snap (e.g., closer to close where more information is revealed)

## 15. Ship candidates (post-this-investigation)

```
Selector:       combined_score(α=0.5, σ_gap=8) top-20
KDE build:      weighted by combined_score values (per-data-point weights)
Bandwidth:      floor=0.5d, ceiling=0.7d
Snap semantics: midnight UTC on close-snap_days (midnight-aligned)
Timestamp:      day-level reviews shifted to 12:00 UTC before profile build
                (h/m reviews keep their precise timestamps)
Phase 1:        integrate KDE over (midnight_utc_dbc, snap_dbc_effective]
                where snap_dbc_effective = snap_days + midnight_utc_dbc
Phase 2:        C = 2 (constant, no training aggregation)
```

**Known limitations:**
- H/m subset mean_err ≈ −6.78 (halved from original but still biased).
- Per-target variance on h/m is high (range from +22 over to −42 under).
- Cohort MAE is ~13 vs ship's ~7; the tradeoff is bias reduction for variance.
- For very-high-volume / late-surge movies, no selection-family fix can fully solve under-prediction; architectural ceiling per §8.

**Deployment note:** the orchestrator can apply a downstream offset correction for known-bias patterns (h/m live-tracked movies under-predicted by ~7 on average), or widen confidence intervals on high-volume targets.

---

## 16. Process notes

**What went right:**
- Systematic ruling-out of cheap mitigations before committing to bigger changes.
- Oracle test (§8.2) produced the decisive negative result — saves future effort on "just try harder to match shapes."
- Midnight+noon convention was a clean semantics win that only became visible once we framed the asymmetry explicitly.
- Shape visualization (§8.1) confirmed cohort DOES contain neighbors — ruled out the "no similar movies exist" hypothesis.
- Time-series regression (§9) surfaced Ridge as a legitimate alternative architecture, not just a baseline.

**What went wrong:**
- Over-invested in selection-family interventions without explicitly reasoning about the `base_rate × KDE × exclusion` ceiling earlier. Could have saved time by diagnosing the architecture before more feature-engineering.
- Conflated "h/m" with "high-volume" + "late-surge" in discussion — the three are correlated in our cohort but distinct failure modes. Better disambiguation earlier would have sharpened test design.
- Option C's raw multiplier crash (cohort −37.7%) was foreseeable — `base_rate[c] × multiplier > 1.0` violates probability semantics. Should have capped from the start.
- Spent effort on Shape × Scale and Hier-Bayes-Gamma when the evidence from cohort variance was already pointing to "shape family choice is hard / parametric isn't enough." Should have started with non-parametric partial-pooling.
- The drive to "integrate into library" felt premature given the ongoing architectural questions. Jake's pushback ("why are you so eager to integrate?") correctly reframed — we didn't have a good enough lambda model to ship, and acknowledging that opened up the Ridge-as-alternative conversation.

---

## 17. Final stock-taking

After 14+ interventions in this investigation, we're at a clear stopping point:

**Two viable ship candidates, with tradeoffs:**

| metric | Weighted-KDE (midnight+noon) | Ridge(α=10) |
|---|---|---|
| Cohort MAE | 13.26 | **8.85** |
| Cohort mean_err | +1.90 | **−0.30** |
| H/m MAE | **18.23** | 26.91 |
| H/m mean_err | −6.78 | **+0.69** |
| Per-target h/m variance | Low | High |

Neither dominates. The choice depends on deployment priorities (bias correction vs per-target stability).

**What's confirmed as structural:**
- H/m-target under-prediction isn't fixable with current features at T-3d.
- The KDE architecture has a ceiling (confirmed via oracle test).
- Multiple alternative architectures (Ridge, shape×scale, hier-Bayes) each fail in different ways but none break through the ceiling.
- The cohort genuinely lacks the signal to predict late-surge long-gap movies without external metadata or future cohort evolution.

**What's recommended:**
- Ship the weighted-KDE + midnight+noon stack per §15 as the primary model (aligned with current library interface).
- Optionally expose Ridge as a secondary/alternative predictor.
- Document the known h/m under-bias so the orchestrator can apply downstream correction.
- Open follow-ups: TMDb metadata (if Jake reconsiders), finite-pool model, more h/m data over time.

**Known stopping conditions for future investigation:**
- If new h/m data accumulates (~30%+ of cohort), re-test all the "null" interventions here — some may become meaningful with different cohort distribution.
- If external metadata becomes available, the finite-pool and per-critic-conditional-base_rate ideas become tractable.
