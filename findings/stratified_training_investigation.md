# Findings: Gap-Stratified Training Investigation

> **2026-04-19 CONVENTION WARNING:** this doc uses the older UTC-midnight snap convention, 14h phase-2 window, and `C=2`. These are **superseded** by the ship conventions in `CLAUDE.md` (midnight ET snap, 10h phase-2 window, `C=1`). Do not copy timestamp handling, snap definitions, or phase-2 logic from this doc into new code. See `findings/ridge_lambda_investigation.md` + `plans/plan_ridge_integration.md` for the current ship spec.

**Date:** 2026-04-17 (started); 2026-04-18 (extended with KDE quality diagnostics, bandwidth fix, and Phase A critic-overlap similarity).
**Status:** Active investigation. Stratified training validated. Close-day piecewise patch validated. KDE bandwidth cap identified as major upstream fix. **Phase A (critic-overlap as similarity dimension) validated at T-3d (+12.4% MAE, statistically significant)** but no improvement at T-1d. No library code committed yet — pending integration plan.
**Scope:** Originally an investigation of gap-stratified training as a cheaper alternative to embargo-anchor rebuild. Expanded to cover (a) stratified training A/B, (b) close-day midnight bias and piecewise patch validation, (c) KDE quality diagnostics revealing structural over-prediction, (d) shape vs scalar decomposition, (e) bandwidth selection fix, (f) critic-overlap as similarity dimension.
**Related:** `findings/embargo_anchor_investigation.md` (precursor); `brainstorm/brainstorm_close_day_lambda_bias.md` (close-day mechanism); `BACKLOG.md` §1.4 (bandwidth selection note).

---

## TL;DR

- **Gap-stratified training** wins +22% T-3d MAE and −8% T-1d MAE on the original (un-expanded actual) measurement. Q1 (short-gap) targets win biggest (+29%); Q4 (long-gap) only +8% because the cohort lacks long-gap training movies. Band sweep is flat — bottleneck is training-set scarcity, not band parameter.
- **Close-day piecewise patch** (add `F × mean_close_day_count_training` to predictions for the 14-hour close-day window) validated. T-1d MAE drops 28-39% across F ∈ [0.5, 1.0]. The win is robust to F choice. Best deployed at F=1.0 (matches h/m estimate from `the_drama` + `super_mario_galaxy` + `they_will_kill_you` + `forbidden_fruits`, n=21 reviews, all pre-market).
- **KDE quality investigation** (predict the (T-3d, T-1d] middle window, no close-day involvement) revealed structural over-prediction independent of close-day: median predicted/actual = **1.91 (baseline), 1.42 (stratified)**. Stratification reduces but doesn't eliminate the bias.
- **Shape vs scalar decomposition** showed bias is mostly scalar (~12% mild shape skew, dominated by 2x scalar over-prediction).
- **Relaxed scaling test falsified the obvious hypothesis.** Lower clamp barely binds (0-10%); upper clamp binds 14-38%. Scaling wants to go *up* (median factor 1.6), not down. Loosening clamps slightly worsens results.
- **Bandwidth cap identified as the actual fix.** Scott's rule produces effective bandwidths > 0.7d for 45% of critics, > 1.0d for 22%, max 2.6d — over-smoothing day-level data into multi-day humps. Capping at 0.7d cuts baseline MAE 10.99 → 6.90 (−37%), stratified 7.53 → 5.57 (−26%). Median over-prediction halved.
- **Best configuration found: gap-overlap-ranked + bandwidth cap (0.7d) + piecewise (F=1.0).** Combines four independent wins (stratified-via-gap-overlap-ranker, bandwidth cap, piecewise). Phase A's `gap_overlap_ranked` strictly improves on `stratified` at T-3d.
- **Phase A — critic-overlap similarity (§10)**: adding observed critic-set Jaccard as a similarity dimension on top of gap. **Initial middle-window result was misleadingly optimistic.** On the deployment-relevant full snap-to-midnight-UTC window (the convention from sections 3-5), `combined_score` (weighted gap + Jaccard, α=0.5) wins +6.9% T-3d MAE (CI95 [+0.04, +1.07], SIG) and **+11.6% T-5d MAE** (CI95 [+0.85, +6.55], SIG). Phase A's win **scales with horizon** — bigger improvement at earlier snaps where the model has more uncertainty to handle. T-1d still flat (close-day dynamics dominate, addressed by piecewise).
- **Important correction (§10.5):** the "structural over-prediction" we documented in §9 (median_ratio 1.42 for stratified, 1.91 for baseline on middle window) was partly an **artifact of the middle-window test convention**, which truncated the actual count one day before midnight UTC. On the full snap-to-midnight window, control_stratified's median_ratio = 1.006 (near-perfect calibration). Bandwidth cap is still a real fix; the urgency around base_rate alternatives is reduced.
- **Phase B (§11) — shape similarity: null result.** Early-arrival rate as a third feature on top of Phase A's gap+Jaccard fails the decision rule. Shape and Jaccard are partially redundant (both encode movie-type from observed early reviews).
- **Recency (§12) — mixed signal.** Adding recency as a third feature monotonically hurts T-3d (−1 to −5%) but helps T-5d (peak +5.9% at w=0.2 with worse calibration). The time-varying-coefficient pattern is real but small; deferred to Path B.
- **Feature-addition plateau reached.** Phase B null + recency mixed confirms we're at the local optimum given current feature set + n=143 cohort. Further improvements require either external movie metadata (genre, distributor, budget), or more h/m close-day data, or a different model class. See `plans/plan_learned_similarity_model.md` for the deferred Path B direction.
- **Decision:** Four changes worth library-committing once integration plan is written: (1) bandwidth cap to ~0.7d for current cohort, (2) `combined_score` training-set selector (α=0.5), (3) close-day piecewise patch with F as a config parameter, (4) skip-rule helper for snapshots without enough observed critics. Bandwidth cap is the highest-leverage and probably should land first.
- **Pre-ship tuning (§16, 2026-04-18): σ_gap=8 and n_training=20 re-validated.** Bootstrap-paired CI at T-3d/T-5d/T-7d — no alternative satisfies the ≥3% T-3d MAE + CI95_lo>0 decision rule. One SIG result (n=5 @ T-7d, +17% MAE with wide CI) is flagged as a future-work breadcrumb for time-varying `n(horizon)`.
- **G1 F-audit (§17, 2026-04-18): piecewise form REVISED.** Fresh reviews.csv pull + scraper-coverage audit on 5 h/m movies showed the prior F × mean_close_day_count form was wrong (F=0.7 over-predicts phase 2 by ~8 reviews on clean subset, MAE 8.34). Ship form is now `phase 1 → midnight UTC of close day` + `phase 2 = C = 2 (constant)`. Also fixed a correctness bug: the existing implementation used `dbc_to=0.0` for phase 1, which integrated the KDE into the pre-market window where it has no real training signal. Ship form splits the window explicitly.
- **Separate issue surfaced (§17.3, BACKLOG §1.8): high-volume movies are systematically under-predicted at T-3d.** Phase 1 mean_err = -14.78 on 5 h/m targets vs nearly zero at T-1d. Diagnosis: `combined_score` selector doesn't match on volume (gap + Jaccard don't directly encode expected activity level), and scaling clamp at 2.0 is too tight for outliers. Mitigation: loosen scaling upper clamp (cheap) or add volume feature to selector (Path B).

---

## 1. Origin

Followed the embargo-anchor investigation (`findings/embargo_anchor_investigation.md`). That investigation rejected embargo-anchor as a global rebuild because:

1. The implementation regressed empirically (−44% to −49% MAE).
2. Diagnosed root causes: scaling threshold mismatch (tuned for close-anchor), first-review-as-proxy noise, day-0 pinning artifact, training-set gap unrepresentativeness.
3. Even with hygiene fixes, the proxy noise structurally limits the upside without real embargo timestamps — a multi-week scraping project.

**Subsequent observation:** the current close-anchor model performs surprisingly well (T-1d MAE = 5.6 reviews on un-expanded actual). This can only be explained by `(close − embargo)` gaps being narrow due to industry timing structure — close dates and embargo dates are both anchored to theatrical release dates. Close-anchor is approximately embargo-anchor for typical movies, up to a small constant shift.

**The sharpened question:** if close-anchor's smearing is dominated by training-set gap variance — and if `(first_review → close)` is a usable noisy proxy for `(embargo → close)` — could we recover the conditioning benefit by **stratifying training data** instead of switching coordinate systems?

This is conceptually a strict subset of the embargo-anchor argument. Conditioning on the target's gap is more informative than marginalizing across the training pool's gap distribution, regardless of whether you do the conditioning via coordinate switch or training-set selection.

---

## 2. Theoretical case

### 2.1 The convolution identity

For any review *i* in the training set:

```
days_before_close_i = gap_i − days_after_embargo_i
```

Pooling close-anchor timings across training movies pools the distribution of a sum of two varying random quantities. The convolution identity gives:

```
close_anchor_density = embargo_anchor_density ⊛ gap_distribution_training
```

Variance decomposition (assuming gap and per-critic timing are independent):

```
Var(close-anchor timing) = Var(embargo-anchor timing) + Var(gap_training)
```

### 2.2 What stratification buys

Selecting training movies whose gap matches the target's shrinks `Var(gap_training)` toward zero (perfect matching) or toward `band²` (loose matching). The variance reduction at the per-review-timing level is exactly `Var(gap_training_unstratified) − Var(gap_training_stratified)`.

In our cohort:

- Unstratified (n=20 most recent): training σ_gap ≈ 2d (training IQR 2.74).
- Stratified ±3d: ideal σ_gap ≈ 1.5d, but heavily target-dependent (see §6.1).

Naive prediction: ~30% σ reduction → ~30% MAE reduction. T-3d came in at +22%, in the ballpark.

### 2.3 What this isn't

Stratified training does NOT replace the embargo-anchor argument fully. It captures the conditioning benefit *to the extent gaps can be matched.* If the cohort lacks training movies in a target's gap region (as for Q4 targets), stratification reduces to baseline.

---

## 3. What was implemented

`notebooks/stratified_training_validation.ipynb` (now 50 cells across 12 sections):

**Selector (§2):** `matched_training_slugs(target, target_gap, band, n=20)` returns the n most recent movies before `target.close` whose gap is within `±band` days of `target_gap`. Band expands in 0.5d steps until n found. Excludes target.

**LOO loop (§3):** For each of 143 resolved movies, build baseline + stratified profiles+KDE *once each per (target, method)*, snapshot at T-3d and T-1d, compute predicted vs actual remaining reviews. Cached to `notebooks/.cache/stratified_training_loo.pkl`.

**Piecewise patch (§8):** `run_loo_piecewise(band)` adds Phase 2 = F × mean_close_day_count over training. Uses inline `expand_actuals()` for symmetric ground-truth treatment.

**KDE quality test (§9):** `run_kde_quality_test()` predicts in the (T-3d, T-1d] middle window via inline `predict_window()` that integrates the KDE over arbitrary bounds. Cached to `kde_quality_test.pkl`.

**Shape vs scalar diagnostic (§10):** `run_window_split_test()` predicts (T-3d, T-2d] and (T-2d, T-1d] separately and compares ratios. Cached to `kde_window_split.pkl`.

**Relaxed scaling sweep (§11):** `run_relaxed_test(threshold, clamp)` with custom `compute_scaling_custom()` and `predict_window_custom()` allows arbitrary scaling parameters without library changes. Cached to `kde_relaxed_scaling.pkl`.

**Bandwidth cap test (§12):** `build_kde_lambda_model_capped(profiles, bandwidth_floor, bandwidth_ceiling)` fits KDEs with both a floor and a ceiling on effective bandwidth. Cached to `kde_bandwidth_cap.pkl`.

**Library code: unchanged.** All new logic lives in the notebook. Only doc changes: `BACKLOG.md` §1.4 added, `build_kde_lambda_model` docstring updated to flag bandwidth-floor/granularity coupling.

---

## 4. Validation methodology

Mirrors Step 4 from `findings/embargo_anchor_investigation.md` §4.2:

- **Leave-one-out training.** Two profile builds per target — baseline (`default_training_slugs`) and stratified (`matched_training_slugs(band=3.0)`).
- **Snapshot state from filtered reviews.** Observed state computed from reviews with `estimated_timestamp < snap_time` only.
- **`actual_remaining` matches the library's internal filter.** Both methods use `0 < dbc ≤ snap_dbc` — symmetric across methods.
- **Live adaptation unchanged.** Both methods call `estimate_lambda` with the same observed state.
- **Bands swept:** ±2d, ±3d, ±5d.
- **Snapshots:** T-3d (primary), T-1d (secondary).

**Decision rule (pre-registered):** stratified beats baseline by ≥10% MAE overall OR ≥20% on Q3/Q4 alone → green light.

---

## 5. Stratified training results

### Cohort gap diagnostics

```
Cohort size: 143 movies with valid gaps
Gap (days from first review to close):
  median = 7.58
  IQR    = 10.94  (Q1=5.62, Q3=16.56)
  max    = 883.62

Quantile cutoffs: Q1≤5.62d, Q2≤7.58d, Q3≤16.56d
```

### Overall A/B (n=138, both methods present, un-expanded actual)

```
                 baseline    stratified (band=3)
T-3d   MAE:        10.29              8.03           +22.0% BETTER
       median:     +4.49              +2.15
T-1d   MAE:         5.69              6.15            −8.1% WORSE
       median:     −3.49             −4.07
```

### By target gap quantile (T-3d)

```
Q1 (≤5.6d):   13.94 → 9.91   (+28.9%)
Q2 (≤7.6d):   10.39 → 8.33   (+19.8%)
Q3 (≤16.6d):   8.82 → 6.96   (+21.1%)
Q4 (>16.6d):   6.98 → 6.40    (+8.3%)
```

### By target gap quantile (T-1d): all quantiles regress 5-10%

### Band-expansion frequency at ±3d setting

```
Effective band: median=4.0d, p75=37.5d, p90=1000.5d
Fraction expanded beyond 3d: 53.1%
```

---

## 6. Diagnosis

### 6.1 Q1 wins biggest, not Q4 (vs prediction)

Cohort is concentrated short-gap. Q1 targets find dozens of close matches within ±3d → tight conditioning. Q4 targets need band expansion to 60-1000d → "stratified" reduces to baseline. The method is data-limited, not parameter-limited.

### 6.2 T-1d regression — close-day bias amplification

Stratified training produces sharper KDEs that drop more close-day-adjacent mass than baseline does. This compounds the existing close-day blind spot from the library's `dbc > 0` filter.

---

## 7. Connection to the close-day bias

The library drops close-day reviews because `Bet Close Date` parses to midnight UTC and ~98% of reviews have day-level timestamps that also resolve to midnight UTC. Real close is 10am EST (~14h after midnight UTC), so a 14-hour window of genuine review activity is invisible to the model. See `brainstorm/brainstorm_close_day_lambda_bias.md` for the full mechanism.

---

## 8. Close-day piecewise patch — design and validation

### 8.1 Design

Two-phase prediction:

```
Phase 1 (snap → midnight UTC of close day):
  use the existing model (with stratified training if applied)

Phase 2 (midnight UTC → 10am EST close, 14-hour window):
  add F × mean_close_day_count over training movies
```

`F` = fraction of close-day reviews that arrive in 12am UTC – 10am EST. Estimated from h/m-confidence reviews in close-day windows of live-tracked movies.

**Validation framework:** "Expanded actual" = existing `actual_remaining` + `F × close_day_count(target)`. Applied symmetrically to all four methods (baseline, stratified, baseline+pw, stratified+pw) so comparisons are apples-to-apples. The expansion is what unmasks the previously-hidden T-1d under-prediction.

### 8.2 F estimation

Audit of cohort-wide h/m close-day reviews:

```
Movies with at least one h/m close-day review: 4/143
  they_will_kill_you:           13
  forbidden_fruits_2026:         3
  the_super_mario_galaxy_movie:  3
  the_drama:                     2
Total: 21 reviews — ALL pre-market (zero post-market)
Aggregate F = 1.000
```

**Caveat:** all 4 movies are originally live-tracked, and reviews.csv was pulled around 11am EST on close day for the_drama and super_mario_galaxy (and likely similar timing for the others). So the F=1.0 measurement is consistent with two interpretations: (a) critics genuinely cluster pre-market, or (b) the pull captured only the pre-market window. Can't distinguish without later-pulled data on those same movies.

### 8.3 4-method 2x2 results (expanded actual, F=1.0, n=138)

```
T-3d  baseline                    MAE 13.30   median −5.75
T-3d  stratified                  MAE 13.72   median −7.70   −3.2% vs baseline
T-3d  baseline+piecewise          MAE 14.33   median +6.94   −7.7%
T-3d  stratified+piecewise        MAE 12.51   median +5.12   +6.0%

T-1d  baseline                    MAE 16.56   median −14.43
T-1d  stratified                  MAE 17.12   median −14.90  −3.4%
T-1d  baseline+piecewise          MAE 10.10   median  −1.68  +39.0%
T-1d  stratified+piecewise        MAE 10.21   median  −1.54  +38.3%
```

Key observations:

- **The previously-reported T-1d MAE = 5.69 was an artifact of symmetric filtering.** Both predicted and actual ignored close-day arrivals. Against expanded actual, baseline T-1d MAE is **16.56** — the under-prediction was always there but masked.
- **Piecewise dramatically closes that gap.** T-1d MAE 16.56 → 10.10 (−39%). Median_err goes from −14.43 to −1.68 (near-zero bias).
- **Stratified vs baseline (without piecewise) flips slightly negative under expanded actual** at both snapshots. The +22% T-3d win we saw before was real for Phase 1, but the un-expanded actual was hiding the close-day under-prediction symmetric to both methods.
- **stratified+piecewise wins at both snapshots** (+6% T-3d, +38% T-1d) — the only method that does. This is the case for promoting both interventions together.

### 8.4 F sensitivity (day-level-only subset, n=134)

Excluded the 4 movies with ambiguous `close_day_count` (their counts may already be ~pre-market due to pull timing). Sweep F ∈ [0.0, 1.0] in 0.1 steps:

```
T-1d MAE vs F:
  F=0.0: baseline=5.83  stratified=6.31  baseline+pw=5.83  strat+pw=6.31  (degenerate; pw=non-pw)
  F=0.5: baseline=11.19 stratified=11.75 baseline+pw=7.85  strat+pw=8.03  (+30% pw vs no-pw)
  F=0.7: baseline=13.46 stratified=14.03 baseline+pw=8.78  strat+pw=8.88  (+34%)
  F=1.0: baseline=16.91 stratified=17.49 baseline+pw=10.22 strat+pw=10.30 (+39%)
```

**Piecewise's T-1d win is robust across F**, growing from +30% at F=0.5 to +39% at F=1.0. The F=0 result is mathematically degenerate (piecewise = non-piecewise when F=0 and both predicted and actual collapse to original measurements). The right deployment F depends on real-world close-day arrival behavior, which we estimated as ~1.0 from h/m data (with the pull-timing caveat above).

### 8.5 Day-level-only ablation summary

Excluding the 4 ambiguous movies didn't materially change the stratification result (still strat slightly worse than baseline at T-1d on un-expanded; piecewise still wins big). The exclusion is a clean ablation, not a load-bearing methodological choice.

---

## 9. KDE quality investigation

### 9.1 The close-day-free middle window test

To assess KDE quality independent of close-day handling: snap at T-3d, predict reviews in the **(T-3d, T-1d] middle window** (no close day in scope). Both ends are far from close, so close-day mass isn't a factor for either predicted or actual.

```
                          n     MAE   median_err   mean_err   p90_abs_err
full       baseline      138  11.01     +9.41        +9.28       19.39
full       stratified    138   7.58     +5.09        +6.00       15.91
day-only   baseline      134  10.98     +9.85        +9.80       19.67
day-only   stratified    134   7.40     +5.31        +6.58       14.67
```

**The KDE over-predicts by 5-9 reviews median in the middle window.** This is independent of close-day — the (T-3d, T-1d] window doesn't touch close day. Stratified reduces the over-prediction (+9.41 → +5.09) but doesn't eliminate it.

**Implication:** the model has a structural over-prediction bias that's separate from close-day handling. The piecewise patch's T-1d win was real but partially amplified because the middle-window over-prediction was offsetting the close-day under-prediction in the original symmetric-filter measurement.

### 9.2 Shape vs scalar decomposition

Per-window calibration test: split (T-3d, T-1d] into half-windows (T-3d, T-2d] and (T-2d, T-1d]. Compare predicted/actual ratio in each:

```
            window  median_ratio   mean_ratio   MAE
baseline    early       1.96         2.68       6.69
            late        1.76         2.27       4.31
stratified  early       1.51         2.30       5.18
            late        1.36         1.71       2.99
```

**Shape verdict: mostly scalar bias.** Early/late ratio differs by 12% (baseline) and 11% (stratified) — mild skew, not dramatic. Most of the bias is multiplicative.

Per-target predicted/actual ratio distribution:

```
            median   mean    std    n
baseline     1.91    2.58   2.55   137
stratified   1.42    2.10   1.83   137
```

Implied global scalar correction: 0.52 (baseline) or 0.70 (stratified). High std (1.8-2.5) means a flat scalar correction can't fix per-target variance. Right-skewed (mean >> median) — a few targets have extreme over-prediction (5-50x), likely niche movies where critics in training don't review the target.

### 9.3 Relaxed scaling test — hypothesis falsified

Initial hypothesis: `_compute_scaling`'s threshold (40) and lower clamp (0.5) prevent aggressive enough corrections. Test with relaxed parameters:

```
                                 median_ratio   MAE   scaling_fire_rate   median_scaling
baseline   thr=40, clamp=[0.5,2]    1.91     10.99       83%               1.59
baseline   thr=10, clamp=[0.2,2]    2.07     11.64       99%               1.71
baseline   thr=5,  clamp=[0.1,2]    2.09     11.75      100%               1.71
stratified thr=40, clamp=[0.5,2]    1.42      7.53       98%               1.23
stratified thr=10, clamp=[0.2,2]    1.42      7.36      100%               1.23
stratified thr=5,  clamp=[0.1,2]    1.42      7.35      100%               1.23
```

Clamp pinning analysis (when scaling fired):

```
baseline   thr=40:  pinned at LOWER (0.5) =  7%   pinned at UPPER (2.0) = 32%
baseline   thr=10:  pinned at LOWER (0.2) =  0%   pinned at UPPER (2.0) = 38%
stratified thr=40:  pinned at LOWER (0.5) = 10%   pinned at UPPER (2.0) = 14%
```

**Hypothesis falsified.** The lower clamp barely binds (0-10%). The **upper clamp** binds 14-38% — scaling wants to go *higher*, not lower. Median scaling factor is 1.6 (baseline), 1.2 (stratified) — both *up-scaling*, not down-scaling.

Lowering the threshold makes baseline slightly worse (more low-volume targets get up-scaling applied). Stratified is barely affected.

**Mechanism (revealed):** The scaling correction assumes observed-rate persists into the future. At T-3d snap, the integration window for `expected_so_far` is `[snap_dbc=3, first_review_dbc≈7]` — capturing the spike region near embargo lift. Reviews arrive in the spike faster than the KDE expects → `observed > expected` → scaling = 1.6 → next 2 days get predicted at 1.6× elevated rate. Reality is the tapering region post-spike → over-prediction.

The under-prediction of the spike by the KDE is the upstream cause; scaling extrapolates the under-prediction forward. Loosening clamps doesn't fix this; the right fix is making the KDE fit the spike accurately.

### 9.4 Bandwidth cap — the actual fix

Bandwidth diagnostic (reference target: lilo_and_stitch_2025, gap=6.6d):

```
Scott's rule effective bandwidth distribution (137 critics with empirical KDEs):
  median = 0.62d
  p75    = 0.94d
  p90    = 1.36d
  p95    = 1.69d
  max    = 2.63d

Fraction of critics with bw > 0.7d: 45.1%
Fraction of critics with bw > 1.0d: 21.9%
Fraction of critics with bw > 1.5d:  7.6%
```

**About half the critics have kernels wider than 0.7d** — meaningful over-smoothing for day-level data, where each review only carries within-day timing uncertainty (~±0.5d).

Bandwidth cap test results on (T-3d, T-1d] window:

```
                    median_ratio   MAE     median_err
baseline   no_cap       1.91     10.99      +9.29
baseline   ceil=1.5     1.95     11.21      +10.58  (only 7.6% affected)
baseline   ceil=1.0     1.74      8.93       +8.26  (−19% MAE)
baseline   ceil=0.7     1.48      6.90       +5.85  (−37% MAE)

stratified no_cap       1.42      7.53       +5.07
stratified ceil=1.5     1.42      7.45       +5.13
stratified ceil=1.0     1.36      6.53       +4.42  (−13% MAE)
stratified ceil=0.7     1.28      5.57       +3.79  (−26% MAE)
```

**Tighter bandwidth materially fixes the over-prediction.** For baseline, MAE drops 10.99 → 6.90 (−37%); median ratio 1.91 → 1.48. About half the over-prediction was bandwidth-induced over-smoothing. For stratified, the effect is smaller (−26%) because gap-matching already produces tighter per-critic distributions.

**`stratified + ceil=0.7` is the new best configuration.** MAE 5.57, median ratio 1.28 — meaningfully closer to perfect calibration (1.0) than anything else we've tested.

**0.7d is appropriate for day-level data specifically.** As cohort granularity improves toward h/m, the ceiling needs to come down. `BACKLOG.md` §1.4 tracks this; `build_kde_lambda_model` docstring also flags the granularity coupling.

### 9.5 Residual over-prediction

Even with stratified + ceil=0.7, median ratio is still 1.28 (28% over-prediction). Likely candidates for what remains:

1. **base_rate over-counting on niche targets.** `base_rate = movies_reviewed / n_training` assumes per-critic review probability is invariant to movie. For niche targets, fewer critics actually review than the rate predicts. Over-prediction concentrates on low-volume targets (mean ratio >> median ratio confirms a heavy right tail).

2. **Scaling extrapolation residual.** Even with better-shaped KDEs, the `_compute_scaling` mechanism can still amplify mistakes in atypical targets where the spike timing differs from training.

3. **Live-adaptation incompleteness.** Critics dropped from the sum once observed, but base_rates of remaining critics aren't updated based on whether the target is high- or low-volume.

Next investigation: alternative base_rate formulations (top-critic-tier-conditional, empirical Bayes shrinkage on movie size, drop base_rate entirely and anchor on observed level, per-publication aggregation). See conversation history for the full riff; top-critic split is the cheapest first test.

---

## 10. Phase A: Critic-overlap similarity

Following the §9.5 recommendation to investigate base_rate alternatives, we explored a structural reframe (per conversation 2026-04-18): instead of correcting per-critic base_rates after the fact, dynamically re-curate the training set as features about the target accrue. Stratified training is a special case (k=20 nearest neighbors using gap-only similarity). Phase A adds **critic-overlap** as a second similarity dimension.

Plan: `plans/plan_critic_overlap_similarity.md`. Reviewer-revised before implementation.

### 10.1 Methods compared (T-3d snap, predict (T-3d, T-1d] middle window)

1. `control_stratified` + ceil=0.7 — current best from §9.4.
2. `gap_overlap_ranked` + ceil=0.7 — filter `|gap_diff| ≤ 5d`, rank by Jaccard against target's observed critics, top 20.
3. `combined_score` + ceil=0.7 — weighted score `α × exp(−|gap_diff|/8) + (1−α) × jaccard`, top 20. α swept ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, σ_gap=8d (matches cohort IQR).
4. `gap_overlap_no_cap` — bandwidth-cap ablation (same selection as method 2 but Scott's rule instead of cap).

**Time alignment** for cross-movie critic-set comparison: target's window length = `first_review_dbc_target − snap_dbc`. For each candidate training movie, take critics in the first `target_window_days` after *that movie's* first review.

**Skip rules:** target requires `first_review_dbc ≥ 4d` AND `≥3 observed critics`. At T-3d snap, only 2/143 targets skipped (low_first_review_dbc).

**Bootstrap CI:** paired per-target MAE deltas, 1000 resamples. Decision rule: point ≥10% AND lower CI bound > 0 → promote.

### 10.2 Results

```
T-3d snap, predict (T-3d, T-1d] | n_common = 106
            method   MAE   median_err   median_ratio
control_stratified   5.90    +3.93         1.28
gap_overlap_ranked   5.17    +2.55         1.20    +12.4%  CI95 [+0.21, +1.28]  SIG
    combined_score   5.34    +2.58         1.19    +9.6%   CI95 [+0.23, +0.92]  SIG
gap_overlap_no_cap   5.86    +3.84         1.30    +0.7%   CI95 [-0.55, +0.65]  ns
```

```
T-1d snap, predict (T-1d, midnight UTC of close]
control_stratified   6.40    -3.66         0.61
gap_overlap_ranked   6.39    -3.46         0.61    +0.1%   CI95 [-0.15, +0.18]  ns
    combined_score   6.49    -3.51         0.59    -1.3%   CI95 [-0.22, +0.04]  ns
gap_overlap_no_cap   6.62    -3.65         0.58    -3.4%   CI95 [-0.40, -0.02]  ns
```

**Alpha sweep (combined_score, T-3d):** plateaus at α=0.5-0.7 (MAE 5.23). Combined never beats simple gap-filter-then-rank-by-overlap.

**By gap quantile (T-3d):**

```
Q1 (n=40): stratified 7.16 → g+o_ranked 5.37  (+25%)
Q2 (n=27): 5.71 → 5.12   (+10%)
Q3 (n=30): 4.49 → 4.57   (~tied)
Q4 (n=9):  5.60 → 6.44   (worse, but n too small to trust)
```

Same pattern as gap-only stratified — niche/short-gap targets benefit most from better training selection.

### 10.3 Verdict

**`gap_overlap_ranked + ceil=0.7` becomes the new best.** Decision rule passes at T-3d. Adopt as the default training-set selector.

**Three notable findings:**

1. **Simpler beats fancier.** Filter-then-rank (gap as binary filter, Jaccard as ranker) outperforms the weighted combined score across all alpha values. The two signals don't combine cleanly when blended; sequential treatment is better.

2. **Bandwidth cap still pulls weight.** `gap_overlap_no_cap` is +0.7% (ns) — overlap-based curation alone doesn't replace the bandwidth fix. Both interventions are needed.

3. **No T-1d improvement.** Surprising — we'd expect overlap to help *more* at T-1d when more critics have arrived. Hypothesis: at T-1d the prediction window is the close-day-adjacent thin slice (T-1d to midnight UTC of close, ~1 day), where close-day dynamics and the existing under-prediction (median_err = −3.66, median_ratio = 0.61) dominate any improvement from training-set refinement. Phase A addresses *which movies to learn from* but not *what's happening in the close-day window itself*.

### 10.4 What §10.1-10.3 (middle-window test) did NOT fix

- **Residual over-prediction at T-3d** dropped from median_ratio 1.28 → 1.20 on middle window. Looked like 20% over.
- **T-1d under-prediction.** Piecewise patch is still required for the close-day window.
- **Q4 (long-gap) targets.** Sample is too small (n=9) and the +25% Q1 win contrasts with Q4 actually getting *worse*. Long-tail targets remain a structural weakness.
- **Compute cost.** Per-target re-curation is ~3-5x more expensive than gap-only. Manageable for backtest; for live deployment, would want triggered updates (rebuild only when observed critic set changes meaningfully — e.g., Jaccard < 0.9 with last build). Track in `BACKLOG.md` as a deployment optimization.

### 10.5 Full-window re-run (deployment-relevant convention)

The middle-window test in §10.2 deliberately stopped at T-1d (`window_dbc_to=1.0`) to "isolate KDE quality from close-day dynamics." Per Jake's clarification 2026-04-18: the deployment-relevant convention is the original sections 3-5 framework — predict from snap to **midnight UTC of close day** (`window_dbc_to=0.0`), with the library's natural `dbc > 0` filter on actual. This includes the day-before-close (which the middle-window test had cut off) but still excludes close-day-rounded reviews via the filter.

Re-run with this convention reveals **the middle-window over-prediction was partly an artifact**:

```
T-3d snap, predict (T-3d, midnight UTC of close] | n_common = 106
            method   MAE   median_err   median_ratio
control_stratified   7.92    +0.17         1.006     ← near-perfect calibration
gap_overlap_ranked   7.76    −1.04         0.966     +2.0%, CI95 [-0.53, +0.82]  ns
    combined_score   7.37    −0.68         0.959     +6.9%, CI95 [+0.04, +1.07]  SIG

T-1d snap, predict (T-1d, midnight UTC of close]
control_stratified   6.38    -3.62         0.61
gap_overlap_ranked   6.37    -3.43         0.61      +0.1%  ns
    combined_score   6.47    -3.47         0.59      -1.4%  ns
```

**What changed in the story:**

1. **Control_stratified is well-calibrated on the full window.** Median_ratio = 1.006 vs 1.28 on middle window. The big "structural over-prediction" we built much of §9 around was magnified by truncating actual one day before close.
2. **Combined_score wins instead of gap_overlap_ranked.** Reversal from middle-window result. The simpler filter-then-rank wins on the truncated window; the weighted score wins on the full window. Possibly because gap-matching contributes more to predicting the day-before-close than the middle-of-window arrivals.
3. **Phase A's deployment-relevant win is smaller** (+6.9% combined_score vs +12.4% gap_overlap_ranked on middle window) but still statistically significant.

### 10.6 Phase A at T-5d snap

To test whether the result generalizes to earlier decision points, ran the same methods at T-5d snap (predict 5-day window). 47/143 targets skipped (33%) due to `first_review_dbc < 6d`. Remaining sample skews toward Q3-Q4 long-gap targets.

```
T-5d snap, predict (T-5d, midnight UTC of close] | n_common = 65
            method   MAE   median_err   median_ratio
control_stratified  31.38    +9.14         1.12
gap_overlap_ranked  28.61    +3.49         1.03      +8.8%   CI95 [-0.64, +6.17]  ns
    combined_score  27.76    +4.46         1.04      +11.6%  CI95 [+0.85, +6.55]  SIG
```

**Phase A's win scales with horizon.** Combined_score: +6.9% at T-3d → +11.6% at T-5d. Longer prediction windows = more uncertainty = bigger payoff from better training-set curation. Calibration also improves more (median_ratio 1.12 → 1.04, ~70% bias reduction).

### 10.7 Alpha sweep on full window

To confirm α=0.5 is robust across the full-window evaluation:

```
T-3d full window (n=136 per α):
  α=0.1: MAE 7.07   α=0.3: MAE 7.05   α=0.5: MAE 6.95   α=0.7: MAE 6.99   α=0.9: MAE 7.03
T-5d full window (n=93 per α):
  α=0.1: MAE 30.38  α=0.3: MAE 30.02  α=0.5: MAE 30.02  α=0.7: MAE 30.39  α=0.9: MAE 30.47
```

Plateau is very flat (T-3d: 6.95-7.07; T-5d: 30.02-30.47). Anything in [0.3, 0.7] is essentially equivalent. **α=0.5 is a defensible default.**

### 10.8 Updated verdict

`combined_score (α=0.5) + ceil=0.7 + piecewise (F=1.0)` is the deployment-recommended stack:

- Strict improvement over current gap-stratified at T-3d (+6.9%, SIG) and T-5d (+11.6%, SIG)
- Better calibrated (median_ratio closer to 1.0) at both horizons
- Composes with bandwidth cap (validated) and piecewise patch (validated for T-1d/close-day window)
- Handles failure modes via skip rule (fall back to gap-only stratified when overlap signal is too thin)

The middle-window test was a useful diagnostic but should not be the basis for deployment decisions. Full-window numbers govern.

---

## 11. Phase B: shape similarity (null result)

Following §10.8's verdict, tested whether adding early-arrival rate as a third similarity feature on top of Phase A's gap + Jaccard improves predictions. Motivation: shape stats ("how fast did this movie attract reviews") might encode movie-type signal orthogonal to critic-overlap.

**Method:** `early_rate = critics_in_window_count / target_window_days`. Similarity = `exp(−|rate_target − rate_candidate| / 2.0)`. Combined score: `w_gap × exp(−|gap_diff|/8) + w_jaccard × jaccard + w_shape × exp(−|rate_diff|/2)`. Swept `w_shape ∈ {0.0, 0.1, 0.2, 0.33, 0.5}` at T-3d and T-5d full window. When `w_shape > 0`, remaining weight split 50/50 between gap and Jaccard.

**Results:**

```
T-3d (n=136):
  w_shape=0.00:  MAE 6.95   median_ratio 0.98   median_err -0.46   ← baseline (Phase A)
  w_shape=0.10:  MAE 6.99   median_ratio 0.99   median_err -0.23
  w_shape=0.20:  MAE 7.02   median_ratio 1.00   median_err -0.13
  w_shape=0.33:  MAE 7.05   median_ratio 1.00   median_err -0.13
  w_shape=0.50:  MAE 7.19   median_ratio 1.00   median_err -0.06

T-5d (n=93):
  w_shape=0.00:  MAE 30.03  median_ratio 1.07   median_err  6.33   ← baseline
  w_shape=0.20:  MAE 29.00  median_ratio 1.16   median_err  9.92   ← −3.4% MAE
  w_shape=0.50:  MAE 29.48  median_ratio 1.20   median_err 12.83
```

**Verdict: null / mixed.** At T-3d, adding shape monotonically worsens MAE (6.95 → 7.19 at w=0.5). At T-5d, w=0.2 gives a marginal −3.4% MAE improvement but calibration worsens (median_err 6.33 → 9.92, ~50% more over-prediction). Decision rule (≥3% on T-3d OR T-5d with no calibration regression) fails. **Shape and Jaccard are partially redundant** — both come from the same observed-early-reviews, both encode "what kind of movie is this." Adding shape doesn't bring orthogonal information.

---

## 12. Recency feature (marginal mixed signal)

Tested recency of candidate's close date relative to target's close as a third feature: `recency_score = exp(−|target_close − candidate_close| / 90d)`. Motivation: critic ecosystem may drift over time (new critics, retiring critics, publication policy changes).

**Results:**

```
T-3d (n=136):
  w_recency=0.00:  MAE 6.95   median_err -0.46   ← baseline
  w_recency=0.10:  MAE 7.05   −1.4% (hurts)
  w_recency=0.50:  MAE 7.33   −5.5%

T-5d (n=93):
  w_recency=0.00:  MAE 30.03  median_err  6.33   ← baseline
  w_recency=0.10:  MAE 28.93  +3.6%
  w_recency=0.20:  MAE 28.26  +5.9%   ← best; median_err 8.77 (worse calibration)
  w_recency=0.33:  MAE 28.61  +4.7%
  w_recency=0.50:  MAE 29.30  +2.4%
```

**Opposite signs at different horizons:** recency monotonically hurts T-3d (1-5% worse) but helps T-5d (peak +5.9% at w=0.2). Hypothesis: T-5d sample skews long-gap (skip rule), and long-gap movies' critic ecosystems have shifted more over time — or long-horizon predictions involve more late-stage critics, who turn over faster.

**Verdict: mixed.** Passes decision rule at T-5d but violates it at T-3d; the time-varying coefficient pattern (recency weight grows with horizon) is real but the gains are small and the calibration cost is real. Deferred to Path B (learned weights) where time-varying coefficients can be handled principally.

---

## 13. Recommendations

In rough order of expected impact per unit of work:

1. **Pre-ship tuning pass.** Per `brainstorm/brainstorm_pre_ship_tuning.md` and `PROMPTS.md` Prompt 4. Covers: anchor baseline, gap-distribution diagnostic, σ_gap sweep (includes `gap_overlap_ranked` as σ_gap→0 extreme), n_training sweep. Gated fresh-data follow-ups: F re-estimation, re-pick frequency curve on h/m targets. Expected headline: 0-5% additional MAE; this is polishing before library integration.

2. **Library integration (after pre-ship tuning).** Four interventions per `BACKLOG.md` §1.5: (a) `bandwidth_ceiling` parameter on `build_kde_lambda_model`, (b) similarity-based training-set selector (combined_score or T1-winner), (c) close-day piecewise helper with F as config, (d) snapshot-state helper. Bandwidth cap is the single highest-leverage change — lands first.

3. **Triggered re-curation for live deployment.** Per `BACKLOG.md` §1.6. Lives in orchestrator, not this library. Rebuild only when observed critic set changes by Jaccard < 0.9 since last build.

4. **Target-scope negotiation with trading layer.** Q4 (long-gap >16.6d) targets consistently underperform; even +5d band can't find 20 long-gap neighbors. Consider excluding from deployment scope or pricing more conservatively. Coordinate with orchestrator, not a library change.

5. **Path B deferred** — see `plans/plan_learned_similarity_model.md`. ~4-5 days focused work. Worth revisiting when (a) cohort grows meaningfully (n > 200), (b) Jake gathers external movie metadata, or (c) deployed performance plateaus at live trading.

6. **Don't pursue embargo data acquisition.** Per `findings/embargo_anchor_investigation.md` §7, real embargo timestamps are the highest-ceiling fix but most expensive. Combined_score + bandwidth cap + piecewise already captures most of what embargo data could offer.

---

## 14. Process notes

**What went right:**

- Pre-registered decision rules (e.g., +10% MAE / +20% Q3+Q4) caught the T-1d regression that the headline T-3d number was hiding.
- The KDE quality test (clean middle window) was the pivotal diagnostic that uncovered the structural over-prediction. Without it, we'd have committed to piecewise+stratified thinking we'd solved the problem, when there was a bigger upstream issue.
- Falsifying the relaxed scaling hypothesis was useful — it ruled out a plausible-seeming fix and pushed us to look at bandwidth, which was the actual issue.
- Disk-cached LOO results meant rapid iteration. Each new test reused profile/KDE builds where possible.

**What went wrong:**

- Pre-registered the Q4-wins prediction confidently; reality was Q1 wins biggest because of cohort gap distribution. Better caveat would have been: "depends on whether the cohort has training movies in the target's gap region."
- Initial hypothesis on over-prediction ("scaling clamps too tight") was wrong in direction. Should have looked at the actual scaling factor distribution before proposing the fix.
- Two notebook execution failures (tz handling, parquet engine) before the LOO ran cleanly. Lesson: smoke-test the LOO loop on 1-3 targets before kicking off the full 143-target run.
- Initial framing of "match-weighted base_rate" as a separate intervention from gap-stratified training was redundant — they're effectively the same when training is gap-matched. User correctly flagged this; useful pushback that improved the framing.

**Process meta:**

- The PROTOCOL.md "analysis notebook" tier was right for this. No formal plan doc needed; concise in-chat plans before each new section worked well.
- Notebook size grew large enough that subsequent NotebookEdit cells started failing (file-too-large). Workaround: direct JSON manipulation via temp scripts. Future notebooks of this scope might benefit from being split into multiple files earlier.

---

## 15. What was changed in the codebase

**Files added:**

- `notebooks/stratified_training_validation.ipynb` — 74-cell validation notebook covering all sections in this findings doc.
- `notebooks/.cache/` — LOO results caches (gitignored): `stratified_training_loo.pkl`, `kde_quality_test.pkl`, `kde_window_split.pkl`, `kde_relaxed_scaling.pkl`, `kde_bandwidth_cap.pkl`, `critic_overlap_test.pkl`, `critic_overlap_test_full_window.pkl`, `alpha_sweep_full.pkl`, `progression_t5.pkl`, `phase_b_shape.pkl`, `phase_b_recency.pkl`.
- `findings/stratified_training_investigation.md` — this file.
- `notebooks/pre_ship_tuning.ipynb` — pre-ship tuning successor notebook (18 cells: anchor baseline, gap diagnostic, σ_gap and n sweeps, ship decision).
- `notebooks/_helpers.py` — factored helpers from `stratified_training_validation.ipynb` (data loading, selectors, KDE-capped, predict_window, bootstrap). Importable by pre_ship_tuning and future notebooks.
- `notebooks/.cache/pre_ship_tuning.pkl` — T1 + T2 sweep LOO results (gitignored).
- `plans/plan_critic_overlap_similarity.md` — Phase A plan (gitignored).
- `plans/plan_learned_similarity_model.md` — Path B plan, deferred (gitignored).
- `brainstorm/brainstorm_pre_ship_tuning.md` — pre-ship tuning brainstorm (gitignored).
- `PROMPTS.md` Prompt 4 — handoff prompt for pre-ship tuning conversation.

**Files modified:**

- `.gitignore` — added `notebooks/.cache/`.
- `BACKLOG.md` — added §1.4 (bandwidth selection should adapt to timestamp granularity), §1.5 (recommended parameter values pending library integration), §1.6 (triggered re-curation for live deployment), §1.7 (long-term direction: richer training-set curation, retire the piecewise patch).
- `rotten_tomatoes_forecasting/critic_model.py` — `build_kde_lambda_model` docstring updated to flag bandwidth-floor / granularity coupling. No behavior changes.

**Library code: no behavior changes.** All experimental code lives in the notebook with inline `_fit_critic_kde_capped`, `build_kde_lambda_model_capped`, `predict_window`, `predict_window_custom`, `compute_scaling_custom`, `combined_score_selector`, `gap_overlap_ranked_selector`, etc., with the pure-function subset factored into `notebooks/_helpers.py` during the pre-ship tuning pass. Pending: integration plan for the four validated interventions (bandwidth ceiling parameter, similarity-based selector, piecewise helper, snapshot-state helper). Pre-ship tuning pass complete (see §16) — values re-validated, no parameter changes required.

---

## 16. Pre-ship tuning pass (2026-04-18)

Following §13 recommendation 1 and `brainstorm/brainstorm_pre_ship_tuning.md`, ran a re-validation pass to tighten or replace §1.5 parameter values before library integration. Notebook: `notebooks/pre_ship_tuning.ipynb`. Helpers factored out of `stratified_training_validation.ipynb` into `notebooks/_helpers.py` so the new notebook imports cleanly.

### 16.1 Anchor baseline

Deployment stack (`combined_score(α=0.5, σ_gap=8) + ceil=0.7 + piecewise(F=0.7)`) on full snap-to-midnight-UTC window with symmetric F-expansion on actual:

```
snap   n    MAE    median_err   median_abs_err
T-3d  136   9.99     +2.55          7.02
T-5d   93  31.86     +6.67         27.86
T-7d   61  61.14    +32.10         54.14
```

F shifted from the findings-time F=1.0 to ship-time F=0.7 per brainstorm G1 (conservative midpoint of the defensible [0.5, 1.0] range; pull-timing caveat on the 21-sample F=1.0 estimate).

### 16.2 Gap-distribution diagnostic

For each of 143 targets, count candidates (past-resolved movies excluding target) at `|gap_diff|` thresholds. Stratified by target gap quantile.

```
bucket  n_targets    ≤0.5d     ≤1d      ≤2d      ≤5d      (count; % of bucket with ≥20 candidates)
Q1         46      13 (37%)  22 (56%)  27 (70%)  31 (74%)
Q2         28       8 (7%)   28 (64%)  42 (79%)  48 (86%)
Q3         33       2 (0%)    6 (3%)    9 (18%)  22 (54%)
Q4         36       0 (0%)    0 (0%)    0 (0%)    1 (0%)
```

Cohort-wide fraction with ≥20 candidates:
- within 0.5d: **13.3%**
- within 1d: **31.5%**
- within 2d: **42.0%**
- within 5d: **53.1%**

**Implication:** tight-σ_gap (e.g. 2-4) is infeasible as a single cohort-wide setting. Q3 barely clears 20 candidates at 5d; Q4 never does. Any σ_gap<8 regresses to band-expansion for Q3-Q4 targets — which is identical to larger σ_gap behavior. This preempts T1's likely failure mode.

### 16.3 T1: σ_gap sweep

Sweep σ_gap ∈ {2, 4, 8, 16, ∞} at T-3d/T-5d/T-7d. Hold `n=20`, `α=0.5`, `ceil=0.7`, `F=0.7`. Bootstrap 1000 resamples, paired per target. Decision rule: ≥3% T-3d MAE improvement AND CI95_lo > 0 → replace σ_gap=8.

```
T-3d snap (n=136):
  sigma_gap    MAE    pct_vs_ctrl   CI95_lo   CI95_hi   sig
  2           10.02      -0.3%       -0.20     +0.14    ns
  4           10.13      -1.4%       -0.26     -0.02    ns
  8 (ctrl)     9.99       0.0%           —         —    ctrl
  16          10.02      -0.3%       -0.21     +0.13    ns
  inf         10.04      -0.5%       -0.42     +0.31    ns

T-5d snap (n=93):
  sigma_gap    MAE    pct_vs_ctrl   CI95_lo   CI95_hi   sig
  2           31.53      +1.0%       -0.59     +1.25    ns
  4           32.12      -0.8%       -0.90     +0.39    ns
  8 (ctrl)    31.86       0.0%           —         —    ctrl
  16          32.31      -1.4%       -1.10     +0.23    ns
  inf         33.39      -4.8%       -3.87     +1.03    ns

T-7d snap (n=61):
  sigma_gap    MAE    pct_vs_ctrl   CI95_lo   CI95_hi   sig
  2           60.99      +0.3%       -4.76     +5.36    ns
  4           61.81      -1.1%       -4.42     +3.32    ns
  8 (ctrl)    61.15       0.0%           —         —    ctrl
  16          60.05      +1.8%       -1.06     +3.32    ns
  inf         62.48      -2.2%       -9.17     +5.52    ns
```

**T1 verdict: σ_gap=8 stays.** No alternative satisfies the decision rule. The plateau at T-3d spans [9.99, 10.13] — 1.4% spread at measurement noise (136 targets, MAE ~10). σ_gap=8 happens to be the minimum; any value in {2, 16} is within noise. σ_gap=∞ (pure Jaccard) is meaningfully worse at T-5d (-4.8%) and T-7d (-2.2%), ruling out "gap is dead weight." Tight σ_gap (2-4) also loses slightly, confirming the diagnostic's prediction.

### 16.4 T2: n_training sweep

Using T1's σ_gap=8 winner. Sweep n ∈ {5, 10, 15, 20, 25, 30, 50}. Same decision rule.

```
T-3d snap (n=136):
  n_training   MAE    pct_vs_ctrl   CI95_lo   CI95_hi   sig
  5           10.46      -4.7%       -1.21     +0.19    ns
  10          10.27      -2.8%       -0.71     +0.16    ns
  15           9.97      +0.2%       -0.25     +0.29    ns
  20 (ctrl)    9.99       0.0%           —         —    ctrl
  25          10.08      -0.9%       -0.29     +0.09    ns
  30          10.02      -0.3%       -0.30     +0.23    ns
  50          10.33      -3.4%       -0.82     +0.08    ns

T-5d snap (n=93):
  n_training   MAE    pct_vs_ctrl   CI95_lo   CI95_hi   sig
  5           30.81      +3.3%       -2.47     +4.65    ns
  10          31.71      +0.4%       -2.05     +2.39    ns
  15          31.49      +1.2%       -1.09     +1.81    ns
  20 (ctrl)   31.86       0.0%           —         —    ctrl
  25          32.46      -1.9%       -1.55     +0.40    ns
  30          32.16      -0.9%       -1.79     +1.25    ns
  50          33.62      -5.6%       -4.83     +1.38    ns

T-7d snap (n=61):
  n_training   MAE    pct_vs_ctrl   CI95_lo   CI95_hi   sig
  5           50.70     +17.1%      +0.80    +22.03    SIG
  10          59.49      +2.7%      -4.55     +8.07    ns
  15          59.14      +3.3%      -0.75     +4.85    ns
  20 (ctrl)   61.15       0.0%          —         —    ctrl
  25          58.81      +3.8%      -4.29     +9.37    ns
  30          61.28      -0.2%      -7.94     +8.03    ns
  50          62.05      -1.5%     -12.22    +10.41    ns
```

**T2 verdict: n=20 stays.** No n satisfies the T-3d decision rule. Plateau at T-3d spans n=15-30 within ±1% of n=20. n=5 loses materially at T-3d (-4.7%) and n=50 at both T-3d (-3.4%) and T-5d (-5.6%).

**T-7d n=5 curiosity (noted, not shipped):** n=5 shows a statistically significant +17.1% MAE improvement at T-7d (CI95 [+0.80, +22.03], point +10.4 MAE units). Possible mechanism: at T-7d, 68/143 targets are skipped for having no observations yet — the surviving 61 skew toward pre-embargo festival-circuit / early-release movies, where a tight n=5 training set captures rare-regime signal that n=20 averages away. The CI95 is extremely wide (+0.80 to +22.03) — a handful of long-horizon targets are likely dominating. This is not enough to change the ship (fails T-3d rule), but is a breadcrumb for a future "n_training adapts to horizon" investigation — likely Path B territory (see `plans/plan_learned_similarity_model.md`, time-varying coefficients).

### 16.5 Ship decision

**Stack ships unchanged.**

```
  alpha:      0.5   (unchanged — α sweep at noise floor per §10.7)
  sigma_gap:  8     (re-validated — no replacement passes decision rule)
  n_training: 20    (re-validated — no replacement passes decision rule)
  floor:      0.5d  (unchanged)
  ceil:       0.7d  (unchanged)
  F:          0.7   (unchanged — G1 re-estimation pending fresh data)
```

No updates to `BACKLOG.md` §1.5 required except changing F's recommendation from 1.0 to 0.7 per the brainstorm's ship-conservative choice. (§1.5 currently reads F=1.0 citing the estimation; the ship F should be 0.7 until G1 lands.)

### 16.6 Re-validation value

The re-validation is informative even without a parameter change:

1. **σ_gap=8 is robust across horizons.** T-3d, T-5d, T-7d all show σ_gap=8 at or near the MAE minimum. No horizon-dependent σ_gap pattern emerged, which removes one motivation for Path B's time-varying-coefficient framework as applied to σ_gap.
2. **The gap-distribution diagnostic quantifies the cohort's structural limit.** Q4 (36 movies, >16.6d gaps) has zero targets with ≥20 candidates within 5d. Any "tight neighborhood" selector reduces to the general pool for Q4. This bounds the upside of any future gap-based refinement on the current cohort.
3. **n=5 @ T-7d is the only SIG result in the entire pass.** Worth keeping in mind as a signal for future investigation (learned `n(horizon)`), but the wide CI95 and fails-T-3d-rule outcome mean no immediate action.

The integration-ready stack is confirmed: `combined_score(α=0.5, σ_gap=8) + bandwidth_cap(ceil=0.7d) + piecewise(phase 1 to midnight UTC + phase 2 constant C=2)`. Piecewise form revised post-§16 per the G1 F-audit (§17).

---

## 17. G1 F-audit — phase 2 form revision (2026-04-18)

**Context:** Jake pulled a fresh `reviews.csv` from Neon (~23k rows) and added `you_me_and_tuscany` to `movies_index.csv` (close 2026-04-13T14:00Z). The G1 audit scoped in the brainstorm was then runnable: re-estimate F on the expanded h/m close-day cohort with proper post-market coverage.

**Notebook:** `notebooks/f_audit.ipynb`. Targets the 5 h/m movies (`the_drama`, `the_super_mario_galaxy_movie`, `forbidden_fruits_2026`, `they_will_kill_you`, `you_me_and_tuscany`). Uses the ship stack (combined_score α=0.5 σ_gap=8, ceil=0.7d, n=20) but **re-frames phase 1 and phase 2 to match the design spec** (not the implementation used in §16).

### 17.1 Framework correction

Earlier validation (§16 and stratified_training_validation.ipynb cell 58) ran `predict_window(dbc_to=0.0)`, which integrates the KDE all the way to market close (10am EDT) — despite the notebook comment claiming "full snap-to-midnight-UTC window." Piecewise then added `F × mean_close_day_count` ON TOP, double-counting the 12am-10am window.

Validation was symmetric (actual also got `F × close_day_count(target)` added), so MAE was relatively F-insensitive cohort-wide. But the MAE numbers didn't represent any clean quantity. For 98% of the cohort (day-level targets) they were proxy-vs-proxy; for the 2% h/m cohort, actual was inflated by post-market activity that the predicted side was trying to also account for via bloat.

**Corrected framework:**

- **Phase 1**: KDE integrates `(midnight_utc_dbc, snap_dbc]` — stops at midnight UTC of close day. Observable via day-level + h/m data.
- **Phase 2**: covers `(0, midnight_utc_dbc]` — the 14-hour pre-market window. Observable ONLY on h/m movies.
- **Predicted total** = phase1 + phase2.
- **Actual total** = `actual_remaining(snap)` for h/m targets, which naturally excludes post-market close-day reviews (their dbc < 0). No proxy needed.

### 17.2 Scraper coverage audit

Pre-check: does the fresh CSV capture post-market close-day reviews for each target?

```
Target                          Latest review vs close_ts   Scraper status
the_drama                       −1.1h                       OFF (disabled before close)
the_super_mario_galaxy_movie    −6.8h                       OFF
forbidden_fruits_2026           +102h                       OK
they_will_kill_you              +148h                       OK
you_me_and_tuscany              +1.8h                       OK
```

`the_drama` and `super_mario_galaxy` had their scraper configs disabled before close day ended. Their h/m pre-market counts are valid but their post-market counts are missing (which is fine — we only need pre-market for phase 2). However, the actual phase 2 count could be under-counted if a review arrived between scraper-shutoff and market close — mild bias.

`forbidden_fruits`, `they_will_kill_you`, `you_me_and_tuscany` form the **clean subset** (n=3): scraper stayed on past close, actual phase 2 count is complete.

### 17.3 Phase 1 quality (KDE on observable window)

```
Snap      n     MAE     mean_err
T-1d      5    0.89     −0.12       ← essentially unbiased
T-3d      5   14.78    −14.78       ← systematic under-prediction
```

**T-1d**: phase 1 predicts the (midnight_UTC, T-1d) window ≈ 0.42 days of pre-close activity. KDE calibration is excellent at this horizon: mean_err = -0.12 across 5 targets.

**T-3d**: phase 1 systematically under-predicts by ~15 reviews per target. Details:

| target | actual_phase1 | phase1_pred | phase1_err |
|---|---|---|---|
| the_drama | 52 | 18.28 | −33.72 |
| the_super_mario_galaxy_movie | 26 | 20.73 | −5.27 |
| forbidden_fruits_2026 | 17 | 13.78 | −3.22 |
| they_will_kill_you | 34 | 12.40 | −21.60 |
| you_me_and_tuscany | 26 | 15.93 | −10.07 |

These 5 h/m movies are **high-volume outliers** — they got live-tracked because they were expected to draw many reviews, and they did. At T-3d, the KDE systematically under-predicts this subset. Cohort-wide this averages out against over-predicted low-volume targets (findings §9.5). See `BACKLOG.md` §1.8 for diagnosis and mitigation plan.

### 17.4 Phase 2 form sweep (T-1d, clean subset n=3)

Ground truth actual_phase2: `forbidden_fruits=2, they_will_kill_you=2, you_me_and_tuscany=0`. Mean = 1.33.

```
phase2_form       MAE    mean_err
C=0              1.12    −1.01
C=1              0.79    −0.01     ← best MAE
C=2              1.03    +0.99
C=3              1.99    +1.99
F=0.2×mean_cd    1.66    +1.66
F=0.7×mean_cd    8.34    +8.34     ← catastrophic over-prediction
```

**Key findings:**

1. **Constant dominates proportional across all reasonable values.** F=0.7 × mean_cd (current ship) over-predicts phase 2 by ~8 reviews on average. F=0.2 × mean_cd is OK but still worse than simple constants in {1, 2}.
2. **Optimal C is in {1, 2}**, not sharply distinguishable at n=3. C=1 minimizes MAE and has zero mean bias; C=2 trades MAE slightly for a +0.99 safety margin against under-prediction.
3. **No positive correlation between close_day_count and pre-market count** across the 3 clean movies: `they_will_kill_you` has `close_day_count=13` but only 2 pre-market; `forbidden_fruits` has `close_day_count=3` and also 2 pre-market. The proportional form is structurally wrong for this dataset.

### 17.5 Ship decision for phase 2 form

**Ship `phase 2 = C = 2` (constant, no training-set aggregation).**

Rationale:
- C=1 and C=2 are within MAE noise at n=3. C=1 has slightly better MAE on our sample; C=2 errs on the safe side (under-predicting reviews is riskier than over-predicting for a "reviews cross threshold" market).
- Eliminates F parameter, training-set aggregation for close_day_count, and the proxy-vs-proxy validation framework that muddied §16.
- Round number, easy to reason about, easy to swap if more data says otherwise.

**What the library helper looks like now:**

```python
def compute_close_day_phase2(C: float = 2.0) -> float:
    """Phase 2 of piecewise close-day correction.

    Represents expected pre-market-close critic arrivals in the
    (midnight UTC of close day, market close] window. Empirically ~1-2
    across h/m-observed movies regardless of movie size.
    """
    return C
```

Much simpler than the prior `F × mean(close_day_count(s) for s in training_slugs)` design.

### 17.6 Revised phase 1 call site

Library integration should set `dbc_to = midnight_utc_dbc_for_target`, NOT `dbc_to = 0.0`:

```python
midnight_utc_dbc = (close_ts - close_ts.floor('D')).total_seconds() / 86400
phase1 = predict_window(model, dbc_from=snap_dbc, dbc_to=midnight_utc_dbc, ...)
phase2 = 2.0
predicted_total = phase1 + phase2
```

This is a correctness fix, not just a cleanup — the previous `dbc_to=0.0` made phase 1 integrate into a region (pre-market close day) where the KDE has essentially no training signal, producing bandwidth-bleed artifact mass that got added to the phase 2 constant on top.

### 17.7 Caveats

- **n=3 clean movies** is a tiny sample for committing to a phase 2 form. The C=2 choice is a "best defensible on current data," not a tightly-bounded estimate.
- **Cross-movie variance in pre-market counts is high** relative to sample size (0, 2, 2 → mean 1.33, range 2). We can't rule out a genuine movie-level factor we haven't identified. If per-movie variance stays this wide as more h/m movies accumulate, a future refinement might be to condition C on something observable (genre, distributor, release pattern) — Path B territory.
- **T-3d phase 1 under-prediction is a separate concern.** See BACKLOG §1.8. Not addressable via phase 2 changes; needs volume-signal curation feature or looser scaling clamp.

### 17.8 Updates triggered

- `BACKLOG.md` §1.5 — piecewise section rewritten for the constant-C form + corrected phase decomposition.
- `BACKLOG.md` §1.8 — NEW entry for high-volume under-prediction at longer horizons.
- `notebooks/f_audit.ipynb` — added to repo (gitignored caches live under `notebooks/.cache/`, but this notebook doesn't need one at n=5 targets).
- `movies_index.csv` — added `you_me_and_tuscany` close date entry.
- Memory file — updated with phase 2 ship form.

Integration conversation now has a cleaner spec to work from:
- Phase 1 helper: takes `dbc_to = midnight_utc_dbc` computed per-target.
- Phase 2 helper: returns a scalar constant (2.0 as default), no training-set arg needed.
