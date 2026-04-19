# Findings: Ridge Lambda Investigation

**Date:** 2026-04-19.
**Status:** Complete. Ship candidate identified. Library integration pending per `plans/plan_ridge_integration.md`.
**Scope:** End-to-end evaluation of Ridge regression as the phase-1 lambda estimator, superseding the per-critic KDE architecture. Tested vs library v0.1 and ship-stack KDE across T-5d → T-1d on the full 143-movie cohort with LOO.
**Related:**
- `findings/path_b_lite_investigation.md` §9 (original Ridge proposal at T-3d only)
- `brainstorm/brainstorm_ridge_optimization.md` (three-tier optimization plan)
- `notebooks/phase1_three_way.ipynb` (headline comparison)
- `notebooks/phase1_ridge_tier{1,2,2b,2c}*.ipynb` (optimization cascade)

---

## TL;DR

- **Ridge beats the ship-stack KDE at every snap** on the full cohort LOO under the phase-1 evaluation convention (I2: integrate to midnight UTC of close day).
- Ship candidate is **ridge_t2: Ridge(α CV) on 17 features** (10 observation-window + 4 nonlinear transforms + 3 finite-pool aggregates), with standardization in a `StandardScaler → Ridge` pipeline.
- **Unbiased at every snap** (mean_err ≈ 0 across T-5d through T-1d). Library KDE has +15 to −18 review systematic bias that **flips sign** between T-3d and T-2d.
- **Three tiers of Ridge optimization** tested; each gave modest gains. Architectural improvement dominated the hyperparameter / feature tuning.
- **Pool-definition robustness finding:** aggregate finite-pool features (pool_mass_consumed, observed_top_tier_frac) are effectively invariant to which training pool you use (A1-recency, A3-gap, hybrid) — correlations 0.92-0.99 across pool definitions despite training-slug Jaccard of 0.18-0.51.
- **E5a (sqrt-compressed base_rate in KDE) ruled out.** Tested 2026-04-19 under per-snap optimal α; regresses cohort MAE by 31-63% at every snap (sig-neg). Inflates `Σ base_rate` → systematic over-prediction. Ship menu updated in `brainstorm/brainstorm_phase1_kde_menu.md`.
- **Jaccard contribution in combined_score is nearly dead weight.** Tested A3 α=1 (pure gap, no Jaccard) vs ship A3 α=0.5: equivalent T-3d/T-2d/T-1d; α=1 wins SIG at T-5d (+15%). The weighted-KDE mechanism (C2) is the load-bearing ship-stack piece, not the Jaccard feature.
- **Decision:** integrate ridge_t2 into the library, replacing the KDE-based `estimate_lambda`. KDE is a git-history artifact going forward.

---

## 1. Origin

`findings/path_b_lite_investigation.md` §9 established at T-3d that Ridge(α=10) on 10 observation-window features produced cohort MAE 8.85 vs. the ship-stack KDE's 13.26 — a 33% improvement, with near-calibrated mean_err (−0.30 vs. +1.90 for KDE). The investigation noted a tradeoff: Ridge's h/m MAE was 26.91 vs. KDE's 18.23. This tradeoff was left unresolved; neither was promoted.

This investigation extended the comparison to T-5d / T-4d / T-2d / T-1d under the phase-1 evaluation convention (I2: predict reviews in `(midnight_utc_dbc, snap_dbc_effective]`) and found Ridge's advantage held across horizons. The h/m concern from path_b_lite was also revisited under LOO (rather than train-on-cohort / test-on-h/m), and under LOO the apparent h/m regression largely disappears.

---

## 2. Three-way headline result

`notebooks/phase1_three_way.ipynb` compared:

1. **library** — v0.1 KDE: A1 recency selector, unweighted KDE, no bandwidth ceiling, F1 scaling. Raw reviews (no noon-shift).
2. **ship** — path_b_lite ship stack: A3 combined_score (α=0.5, σ_gap=8) + weighted KDE + bandwidth cap (0.7d) + noon-shift + midnight snap.
3. **ridge** — Ridge(α=10) on 10 features per path_b_lite §9.

All three evaluated over phase-1 window for T-5d → T-1d. Same skip rules. Same noon-shifted actuals.

### 2.1 Full cohort MAE

```
snap    n       library    ship    ridge
T-5d   73/75     37.96    48.81    34.69
T-4d   94/97     30.82    31.35    23.70
T-3d  130/135    17.86    13.42    10.80
T-2d  137/142     8.14     5.01     3.43
T-1d  139/144     3.87     2.47     2.18
```

### 2.2 Paired bootstrap CIs

```
         T-5d           T-4d           T-3d          T-2d          T-1d
lib→ship  library wins   ns             ship +25%    ship +38%    ship +36%
lib→ridge ns             ridge +28%     ridge +42%   ridge +58%   ridge +43%
ship→ridge ridge +30%    ridge +29%     ridge +22%   ridge +32%   ridge +10%
```

All ship/ridge wins SIG at the 95% level.

### 2.3 Bias (mean error)

```
snap   library     ship      ridge
T-5d   −17.19    +22.67    +0.32
T-4d   −17.85    +14.39    −0.21
T-3d    −9.58     +1.56    +0.04
T-2d    +6.51     +3.72    +0.00
T-1d    +2.73     +0.36    −0.02
```

**Ridge is calibrated at every horizon.** Library systematically under-predicts at T-5d/T-4d/T-3d by ~10-18 reviews, then over-predicts at T-2d/T-1d. The bias direction flips between T-3d and T-2d — the worst property for downstream consumers who'd need a snap-dependent correction.

### 2.4 h/m subset (n=2-5)

```
snap   n   library   ship    ridge
T-5d   2   32.61    30.35    28.01    ← ridge wins
T-4d   4   31.74    33.86    16.91    ← ridge wins (big)
T-3d   5   17.54    17.53    20.65    ← ship/library tied, ridge loses
T-2d   5    6.48     3.38     2.70    ← ridge wins
T-1d   5    2.64     2.38     1.89    ← ridge wins
```

Ridge wins h/m at 4/5 snaps under LOO. The path_b_lite §9 finding that Ridge had worse h/m (MAE 26.91 vs KDE 18.23) was an artifact of its train-on-cohort / test-on-h/m convention — a strictly harder holdout than LOO. Under LOO (the deployment-realistic fit, where past h/m resolutions inform predictions on new h/m targets), Ridge handles h/m better than either KDE.

---

## 3. Ridge optimization cascade

Three-tier plan per `brainstorm/brainstorm_ridge_optimization.md`. Each tier evaluated on the same cohort-LOO + h/m convention.

### 3.1 Tier 1: standardization + nonlinear transforms + α CV

Changes to ridge_orig:
- `StandardScaler → Ridge` pipeline (per-LOO training-set fit stats).
- 4 new features: `log(1+observed_count)`, `log(1+rate_last_day)`, `sqrt(rate_last_day)`, `rate_delta = rate_last_day − rate_first_day`.
- 5-fold CV over α ∈ {0.01, 0.1, 1, 10, 100, 1000} per snap.

**α* per snap:** 10 for T-5d through T-2d; 100 for T-1d. The path_b_lite fixed α=10 was near-optimal.

**Cohort MAE:**
```
snap    orig    t1       Δ%        bootstrap
T-5d   34.69   33.13   +4.49%    CI95 [-1.79, +6.38]   ns
T-4d   23.70   23.38   +1.35%    CI95 [-1.08, +1.81]   ns
T-3d   10.80   10.07   +6.74%    CI95 [+0.15, +1.38]   SIG
T-2d    3.43    3.40   +0.79%    CI95 [-0.08, +0.13]   ns
T-1d    2.18    2.21   −1.16%    CI95 [-0.13, +0.08]   ns
```

One SIG win at T-3d. The driver is the nonlinear transforms of `rate_last_day`; ablation showed standardization alone is essentially benign.

### 3.2 Tier 1 ablation (2x2 on std × new features)

Isolated standardization vs. new-features contributions:
```
variant    std    new-feat    T-3d MAE
A orig     no     no          10.80
B std      yes    no          10.90
C feat     no     yes          9.92    ← driver
D t1       yes    yes         10.07
```

- **New features** drive the cohort win at T-4d/T-3d (SIG).
- **Standardization** is decorative on predictions, valuable for coefficient interpretability.
- What looked like an h/m regression in tier 1 was small-sample noise; sign flipped across snaps.

### 3.3 Tier 2: finite-pool features (A1-recency pool)

Added 3 features describing who is left to review:
- `remaining_base_rate_sum` — Σ base_rate[c] for c ∉ observed
- `pool_mass_consumed` — observed_base_rate_sum / total
- `observed_top_tier_frac` — |observed ∩ A1_top_30| / 30

Base_rates computed from `default_training_slugs(n=20)` (LOO-clean: target excluded).

**Cohort MAE:**
```
snap     t1      t2       Δ% (t1→t2)    bootstrap
T-5d   33.13   32.14    +1.71%         ns
T-4d   23.38   21.09    +3.56%         ns (CI wide but near-SIG)
T-3d   10.07    9.87    −0.36%         ns (tied)
T-2d    3.40    3.44    −1.16%         ns
T-1d    2.21    2.24    −0.21%         ns
```

Directional cohort improvement at early snaps; ns everywhere.

**Crucially — repairs tier 1's noisy h/m drift:**
```
snap   orig    t1     t2
T-5d   28.01  37.15  30.38   ← t2 back near orig
T-4d   16.91  16.86  15.31   ← best variant
T-3d   20.65  23.76  23.86
T-2d    2.70   3.13   2.71
T-1d    1.89   1.97   1.99
```

### 3.4 Feature importance — what actually got learned

Ridge coefficients (standardized scale) at T-4d:
```
observed_top_tier_frac   −14.42  ← largest
log_rate_last_day         +8.78
sqrt_rate_last_day        +8.40
rate_first_day            +7.09
observed_rate             −8.67
rate_last_day             +5.97
...
remaining_base_rate_sum   −2.21
pool_mass_consumed        +0.87  ← near-zero
```

`observed_top_tier_frac` dominates at early snaps. **Direction is negative:** when A1 top-tier critics have already reviewed by the snap, expect fewer remaining phase-1 reviews. Interpretation: magnets showed up early → remaining pool is long-tail → tail tapers. This is exactly the architectural-ceiling signal that path_b_lite identified as missing from pure observation-window features.

The other two pool features (`remaining_base_rate_sum`, `pool_mass_consumed`) get small coefficients — largely redundant with `observed_count` and its transforms.

### 3.5 Tier 2b: A3 gap-only pool (alternative base_rate source)

Same features but base_rates computed from A3 α=1 weighted pool (20 best gap-match). Jake's hypothesis: embargo-anchor similarity (gap-matched neighbors) is the conceptually right definition of "critics who would review this kind of movie."

Result: **indistinguishable from tier 2.** All pairwise bootstrap CIs ns. Cohort MAE values differ by ≤ 0.15 units at every snap.

### 3.6 Tier 2c: hybrid pool (top-50 recent → top-20 by gap)

Two-stage filter: recency filter (50 most recent) then gap-match within those. Hypothesis: combine recency's stale-ecosystem avoidance with gap's embargo-similarity.

Result: also indistinguishable from tier 2 / tier 2b on paired bootstrap. Marginal T-3d SIG vs ridge_orig (+7.13%) that tier 2b missed narrowly, but vs tier 2 / tier 2b: all ns. Slight T-4d regression vs both.

### 3.7 The pool-definition robustness finding

Training-pool Jaccard (sample of 50 targets):
```
A1 (recency) vs A3 (gap-only):     median 0.18, mean 0.36
t2c (recent→gap) vs A1:             median 0.25, mean 0.33
t2c vs A3:                          median 0.51, mean 0.56
```

Pools genuinely differ — 18-51% overlap median.

But feature correlations across pool variants at T-3d:
```
observed_top_tier_frac:   A1↔A3 = 0.92   A1↔t2c = 0.92   A3↔t2c = 0.97
pool_mass_consumed:       A1↔A3 = 0.99   A1↔t2c = 0.99   A3↔t2c = 0.99
```

**The aggregate pool features are essentially pool-definition-invariant** (0.92-0.99 correlated) even though pool memberships vary substantially (0.18-0.56 Jaccard).

**Implication:** the "did the workhorses show up" / "how depleted is the pool" statistics are robust summary features. You don't need a carefully curated target-conditional pool; any reasonable 20-movie recent-ish subset produces the same aggregate signal. This caps the upside of further pool-definition tweaks.

### 3.8 Ship choice: tier 2 (A1 pool)

Equivalent to tier 2b / tier 2c on performance; simpler pool definition (`default_training_slugs(n=20)`); fewer edge cases in deployment (no gap filter that can reject targets with too-wide gap distribution in the cohort).

---

## 4. Ship candidate specification (ridge_t2)

### 4.1 Features (17 total)

**Observation-window (10):**
- `observed_count`
- `first_review_dbc`
- `target_gap`
- `observed_rate` (= observed_count / obs_window_days)
- `rate_last_day` (reviews in last 24h of observation window)
- `rate_first_day` (reviews in first 24h post-first-review)
- `top_critic_frac`
- `pub_diversity`
- `pub_entropy`
- `low_activity_frac`

**Nonlinear transforms (4):**
- `log(1 + observed_count)`
- `log(1 + rate_last_day)`
- `sqrt(rate_last_day)`
- `rate_delta = rate_last_day − rate_first_day`

**Finite-pool (3):**
- `remaining_base_rate_sum` — Σ base_rate[c] for c ∉ observed, over A1 pool
- `pool_mass_consumed` — observed_base_rate_sum / total_base_rate_sum
- `observed_top_tier_frac` — |observed ∩ top_30_by_base_rate_in_A1| / 30

Where `base_rate[c] = movies_reviewed_in_A1[c] / 20`, and A1 pool = 20 most recent resolved movies before target close.

### 4.2 Model

```
Pipeline:
  StandardScaler  (training-set fit stats only)
  → Ridge(α = snap-specific)

α selection: 5-fold CV per snap over {0.01, 0.1, 1, 10, 100, 1000}

Validated α* under UTC-midnight convention (tier 2):
  T-5d: α = 10
  T-4d: α = 10
  T-3d: α = 10
  T-2d: α = 100
  T-1d: α = 100

Validated α* under ET-midnight convention (ship):
  T-5d: α = 100
  T-4d: α = 10
  T-3d: α = 10
  T-2d: α = 100
  T-1d: α = 100
```

### 4.2b Snap / phase convention (SHIP — ET-midnight anchored)

```
snap_time         = midnight ET on close−N days
phase_1 window    = (midnight ET on close day, snap_time]
phase_2 window    = (midnight ET on close day, close_ts]   — ~10h for 10am ET close; DST-dependent
phase_2 constant  = C = 1.0
predicted_total   = ridge_phase1_pred + 1.0
```

Verified 2026-04-19 in `notebooks/proposed_ship_stack_test.ipynb`. Cohort phase-1 MAE at T-3d = 9.96 under ET convention (vs. 9.87 under UTC convention used during tier-2 development). Equivalent for Ridge.

**C=1 empirical basis:** 5 h/m targets with full scraper coverage. Phase-2 counts: `the_drama=2, super_mario=1, forbidden_fruits=1, they_will_kill_you=1, you_me_and_tuscany=0`. Mean=1.0, median=1.

### 4.3 Performance summary

**Phase-1 only (Ridge output, what Ridge optimizes):**

```
snap     n    MAE    med|err|   p75|err|   p90|err|    mean_err
T-5d    73   32.14     23.40      48.11      66.57      −0.45
T-4d    94   21.10     15.17      29.56      47.19      −0.28
T-3d   130    9.96      7.47      12.82      21.53      −0.01
T-2d   137    3.42      2.73       4.73       7.59      −0.02
T-1d   139    2.22      1.61       3.07       4.71      −0.03
```

(ET-midnight convention; numbers from `notebooks/proposed_ship_stack_test.ipynb`.)

**Shipped composition on h/m subset (deployment-relevant, Ridge + C=1):**

```
snap     n    MAE    med|err|   max|err|   mean_err
T-5d     2   30.39     30.39      45.47     +15.08
T-4d     4   14.69     11.75      33.21      +1.92
T-3d     5   25.48     22.61      47.75      −5.15
T-2d     5    2.45      0.94       7.06      +0.67   ← calibrated
T-1d     5    1.48      1.63       2.64      −0.38   ← calibrated
```

T-3d / T-4d / T-5d dominated by long-gap architectural-ceiling movies (`they_will_kill_you`, `forbidden_fruits`). T-2d and T-1d are the sweet-spot — well-calibrated, low MAE on the real-timestamp deployment sample.

**Cohort composition caveat:** the cohort's phase_2_actual is inflated by the noon-shift preprocessing — day-level reviews dated to close day land at 12:00 UTC ≈ 8am ET, which puts them inside the phase-2 window (midnight ET → 10am ET) algorithmically but doesn't reflect actual close-day arrival times. Cohort composition MAE is ~10-14 at T-3d onward with systematic me ≈ -10, but this is a **preprocessing artifact**, not a model calibration issue. The phase-1-only MAE above is the model-quality measure; the h/m composition MAE above is the deployment-quality measure. Do not use cohort composition MAE for calibration monitoring.

### 4.4 Error distribution — unbiased ≠ consistently accurate

Signed error quantiles at T-3d:
```
p10      Q1     median    Q3      p90      mean    n_over   n_under
-16.3   -7.3    +2.3      +7.7    +12.8    -0.01   76       54
```

- Mean ≈ 0 (Ridge optimizes for this).
- Median slightly positive (+2.3) — typical prediction over-predicts by ~2 reviews.
- Distribution is asymmetric: long negative tail (p10=−16 vs p90=+13) indicates a few large under-predictions balanced against many small over-predictions.

**The long negative tail is the architectural-ceiling residual.** h/m critic-magnet targets like `the_drama` and `they_will_kill_you` live in that tail. Finite-pool features narrow it but don't eliminate it. Consumers should widen confidence intervals for predictions that come from the error-tail target profile (see §6).

---

## 5. Library-vs-ship improvement

Per-snap comparison (library baseline vs ridge_t2):

```
snap    library MAE   ridge_t2 MAE    Δ (units)    Δ (%)     library me   ridge_t2 me
T-5d      37.96         32.14           +5.82       +15.3%    −17.19      −0.45
T-4d      30.82         21.09           +9.73       +31.6%    −17.85      −0.29
T-3d      17.86          9.87           +7.99       +44.7%     −9.58      −0.01
T-2d       8.14          3.44           +4.70       +57.7%     +6.51      −0.03
T-1d       3.87          2.24           +1.62       +41.9%     +2.73      −0.03
```

Median absolute error improvement (typical-miss reduction):
```
snap    library med|err|   ridge_t2 med|err|    Δ%
T-5d       28.10              23.40            +16.7%
T-4d       24.07              14.54            +39.6%
T-3d       13.92               7.54            +45.8%
T-2d        7.74               2.76            +64.3%
T-1d        3.48               1.59            +54.3%
```

At T-3d, the typical miss drops from ~14 reviews to ~7.5. At T-2d, from ~8 to ~3. And mean error goes from ±6-18 reviews (systematic, snap-dependent direction) to essentially zero.

---

## 6. Deployment guidance for the orchestrator

Derived from the per-target-type error analysis (see `findings/trading_strategy_from_ridge_errors.md` for the orchestrator-facing doc):

1. **Snap gate:** T-3d / T-2d is the quality sweet spot. T-5d / T-4d has p90|err| of 47-67 reviews; too wide for reliable edge computation except in exploratory sizing.
2. **Target filter:** skip targets in observed-count Q1 (low-vol niche, over-predicted) or Q4 (high-vol magnet, under-predicted), OR `target_gap > 15d`, OR observation window has < 10 critics.
3. **Sizing bound:** scale position inversely to `p90|err|` at snap.
4. **Bias correction hook:** for targets that pass the filter but land in extreme quartiles post-prediction, the orchestrator can apply a known-bias correction (details in orchestrator doc).

---

## 7. What was ruled out or confirmed during the investigation

### Ruled out
- **E5a (sqrt-compressed base_rate in weighted KDE).** Regresses cohort MAE 31-63% across snaps (sig-neg). Sqrt is a level-inflating transform on [0,1]; inflates `Σ base_rate`; scaling clamp can't down-correct enough. Menu marked ruled out.
- **Tier 3 (stacking KDE pred as Ridge feature).** Not run. Given the pool-definition robustness finding (§3.7), unlikely to help — would bring the same aggregate signal with more deployment complexity (two models at inference).
- **A2 hard-filter selector (as C1 downgrade).** Tested vs ship (A3 α=0.5): A2 loses 16-20% at T-3d/T-2d because the C1 unweighted KDE downgrade eats the weighted-KDE gain. Confirmed the weighted-KDE mechanism is the load-bearing path_b_lite contribution, not the selector's Jaccard feature.

### Confirmed
- **Path_b_lite's architectural ceiling** is real at the per-critic KDE level but **doesn't block Ridge**. Ridge bypasses the `base_rate × KDE × exclusion` sum and regresses directly on observable features.
- **Weighted KDE (C2) is the ship-stack's real gain** over library, not Jaccard curation. A3 α=1 (no Jaccard) matches or beats A3 α=0.5 (ship) everywhere except T-5d where α=1 wins SIG.
- **h/m MAE under LOO is NOT a Ridge weakness**, contrary to path_b_lite §9 suggestion.
- **Pool-composition signals (`observed_top_tier_frac`) carry real information** even as observation-window features (rate_last_day, observed_count) get most of the weight.

---

## 8. What's changed in the codebase (this session)

**Notebooks added:**
- `notebooks/phase1_e5a_alpha_step.ipynb` — E5a ruled-out test
- `notebooks/phase1_a2_vs_a3alpha1.ipynb` — Jaccard-value check
- `notebooks/phase1_three_way.ipynb` — headline library/ship/ridge comparison
- `notebooks/phase1_ridge_tier1.ipynb` — standardization + features + α CV
- `notebooks/phase1_ridge_tier1_ablation.ipynb` — isolated std vs new-features
- `notebooks/phase1_ridge_tier2.ipynb` — finite-pool features (A1 pool)
- `notebooks/phase1_ridge_tier2b_a3gap.ipynb` — A3-gap pool variant
- `notebooks/phase1_ridge_tier2c_recent_gap.ipynb` — hybrid pool variant

**Docs added:**
- `brainstorm/brainstorm_ridge_optimization.md` — three-tier plan (pre-investigation)
- `findings/ridge_lambda_investigation.md` — this file
- `findings/trading_strategy_from_ridge_errors.md` — orchestrator-facing trading guidance
- `plans/plan_ridge_integration.md` — library integration plan

**Docs modified:**
- `brainstorm/brainstorm_phase1_kde_menu.md` — E5a marked ruled out
- Memory: `project_ridge_direction.md` updated; MEMORY.md index updated

**Library code:** unchanged this session. All integration pending per `plans/plan_ridge_integration.md`.

---

## 9. Stopping conditions for future Ridge extensions

Given §3.7's pool-definition robustness, further feature-engineering on pool-definition variants is unlikely to help. Legitimate future directions:

1. **More h/m cohort data** — when h/m-representative movies grow to ~30% of cohort, re-validate Ridge's h/m behavior under a stricter holdout. Residuals on critic-magnet movies may or may not reduce.
2. **TMDb metadata features** — path_b_lite's deferred direction. External movie features (distributor, genre, sequel status) may bring orthogonal signal. Requires API key setup; Jake deprioritized but it remains on the table if cohort-side gains plateau in live trading.
3. **Tier 3 stacking** — only worth revisiting if a different model class (GBM, quantile regression) shows structurally different errors from Ridge. Cohort-LOO comparison of Ridge vs GBM at T-3d per path_b_lite: Ridge wins on cohort, GBM slightly closer on h/m. Worth revisiting with the current 17-feature set.
4. **Uncertainty estimates.** Current `estimate_lambda` returns a point. Ridge can return a prediction interval (leveraging LOO residuals). Would let the orchestrator size by p90|err| natively.
