# Findings: Embargo-Anchor Investigation

**Date:** 2026-04-17
**Status:** Investigated, rejected in current form. Code reverted.
**Scope:** Consolidated record of the session that proposed switching the per-critic KDE model's timing coordinate from days-before-bet-close to days-after-embargo-lift.

---

## TL;DR

- **Motivating intuition:** Critics react to the embargo lift, not the bet close. Different Kalshi markets have different (embargo → close) gaps, which smears the close-anchor KDE shapes. Switching the anchor to embargo should remove that smearing.
- **Theoretical case is sound:** close-anchor KDE density is mathematically the convolution of embargo-anchor KDE density with the per-movie gap distribution. Conditioning on the target movie's known embargo is strictly more informative than implicitly marginalizing over gaps.
- **Implementation was correct per spec:** `anchor_mode="embargo"` added as a selectable option; 70 tests pass; all plan spec items verified by an independent review agent.
- **Empirical A/B validation on 143 resolved movies** (leave-one-out, scaling enabled): embargo-anchor **regressed** MAE at T-3d by −49.2% and T-1d by −44.7%. Embargo-anchor systematically under-predicted remaining reviews.
- **Three interacting failure modes diagnosed.** First-review-proxy day-0 pinning (8.99% of pooled mass vs 0.8% expected), training set unrepresentative of cohort (training gap IQR 2.74 vs cohort 10.94), and the scaling threshold disabled 2-20× more often under embargo — killing the corrective feedback that makes close-anchor accurate.
- **Decision:** Reject embargo-anchor as a default. Code reverted to the pre-session state. Follow-up directions noted below for anyone returning to this problem.

---

## 1. Motivating question

**User prompt (paraphrased):** How are we anchoring the time values for our individual critic KDE graphs? If a critic reviews at time t, is t relative to bet close? If so — critics likely react to the embargo lift rather than the bet close. Bet-close-to-embargo spacing varies across movies, so we might be conflating per-critic timing with per-movie gap variance. Can we add an option to anchor each review's time relative to embargo instead?

The current code (in `rotten_tomatoes_forecasting/critic_model.py` pre-session) computed each review's timing as `days_before_close = (bet_close - estimated_timestamp).days`, pooled across training movies into per-critic KDEs.

The proposed alternative: anchor each review to its movie's embargo lift (approximated by the first observed review per movie, since we don't have embargo timestamps as first-class data). Store timing as `days_after_embargo = (estimated_timestamp - anchor_ts).days`.

---

## 2. Theoretical case (why this should have worked)

### 2.1 The convolution identity

For any review *i* in the training set:

```
days_before_close_i = gap_i − days_after_embargo_i
```

where `gap_i = (close − embargo).days` for the review's movie. Pooling close-anchor timings across training movies therefore pools the distribution of a sum of two varying random quantities: the per-movie gap and the critic's intrinsic post-embargo timing.

The distribution of a sum of independent random variables is the convolution of their individual distributions. So:

```
close_anchor_density = embargo_anchor_density ⊛ gap_distribution
```

(with one density mirrored, which doesn't change the intuition).

**Consequence:** the close-anchored KDE is the embargo-anchored KDE with extra smearing baked in — that smearing coming from mixing movies with different (embargo → close) gaps in the training pool.

### 2.2 Worked toy example

Suppose a hypothetical critic always reviews on day 0 or day 1 after embargo (70/30 split). Suppose training-set gaps are 50/50 at 10 days vs 12 days.

**Embargo-anchor density** (what we want):

| Days after embargo | Density |
|--------------------|---------|
| 0                  | 70%     |
| 1                  | 30%     |

**Close-anchor density** (what pooled training data produces):

| Days before close | Density |
|-------------------|---------|
| 9                 | 15%     |
| 10                | 35%     |
| 11                | 15%     |
| 12                | 35%     |

Evaluating at 10 days before close:
- Close-anchor says 35%.
- Embargo-anchor conditioned on gap=10: **70%** (critic at day 0, peak mass).
- Embargo-anchor conditioned on gap=12: **0%** (critic would be at day 2, outside their support).

Close-anchor averages the two cases. Correct *if you don't know the gap*; wrong *if you do*.

### 2.3 Marginalize vs condition

Two standard Bayesian operations on a joint distribution `p(X, Y)`:
- **Marginalize over Y:** `p(X) = ∫ p(X, Y) dY`. Used when Y is unknown.
- **Condition on Y:** `p(X | Y=y)`. Used when Y=y is observed.

Conditioning is strictly more informative when Y is observed.

In our model, Y is the per-movie gap. Post-first-review we observe `gap = (close − first_review_ts).days` directly, so we can condition. Pre-first-review gap is unobserved, and the close-anchor density is exactly what marginalization would produce.

Verification for the toy example at 10 dbc:
```
close_anchor(10 dbc) = P(gap=10) · embargo(day 0) + P(gap=12) · embargo(day 2)
                     = 0.5 · 0.70 + 0.5 · 0.00
                     = 0.35
```
Which matches the pooled close-anchor value — confirming the identity by construction.

### 2.4 Library contract implication

`compute_edge` models how `fresh/total` evolves as remaining reviews arrive. Both counts must be ≥ 1 for the ratio to be defined. With zero observed reviews:
- Kalshi resolution `round(fresh/total × 100)` is undefined.
- There is no score to evolve; Poisson-binomial would be modeling the wrong object.

Pre-first-review pricing is structurally different (needs a prior on final scores conditioned on release/genre/cast features, not a review-arrival rate model). Not this library's scope. Contract: the library is post-first-review only. Orchestrator is responsible for only calling it after the first review has landed.

---

## 3. What was implemented

The code that existed at session end (before rewind) changed only `rotten_tomatoes_forecasting/critic_model.py` and `tests/test_critic_model.py`. All other library code was untouched. Summary of the API changes:

**Dataclasses**
- `CriticProfiles` gained `anchor_mode: Literal["close", "embargo"] = "close"` and `anchor_ts_map: dict[str, pd.Timestamp]` (resolved per-movie anchors; empty under close mode).
- `KDELambdaModel` gained `anchor_mode` mirroring `profiles.anchor_mode`.

**`build_critic_profiles`**
- New params: `anchor_mode` and `embargo_lift_ts_map: dict[str, pd.Timestamp] | None`.
- Under `"close"`: unchanged semantics.
- Under `"embargo"`: per-movie anchor resolved as `embargo_lift_ts_map[slug]` if provided else the movie's first observed review. `timing_data[i] = (review_ts − anchor_ts).total_seconds() / 86400` (fractional days, always ≥ 0). Reviews before anchor dropped. Reviews after close still dropped.
- Verbose print extended to include `(close − anchor) days` distribution (min/median/IQR/max) as a robust alternative to `σ`.

**`estimate_lambda` and `_compute_scaling`**
- New kwarg `anchor_lead_days: float | None`. Under embargo mode, required; raises `ValueError` with diagnostic quantities if missing.
- Integration window under embargo: `[current_dae, close_dae]` with `current_dae = anchor_lead_days − days_before_close` and `close_dae = anchor_lead_days`.
- Under embargo, `_compute_scaling` derives `first_review_dae = anchor_lead_days − first_review_dbc` internally and integrates `[first_review_dae, current_dae]`. Under first-review-proxy, `first_review_dae = 0` trivially; under explicit embargo_lift_ts, it can be positive.
- `current_dae < 0` (library called before the movie's first review) raises.

**Tests**
- 12 new tests covering mode-default preservation, anchor-map population, round-trip `anchor_mode` consistency between profiles and model, non-negative timing under embargo, missing-arg raise, close-mode backward compat, smoke tests, invalid-window raising, monotonicity under observed critics, explicit override precedence, verbose diagnostic output, and the `first_review_dae < 0` guard in `_compute_scaling`. All 70 tests (58 pre-session + 12 new) passed.

**Design decisions worth preserving**
- `anchor_mode` was a property of the model (on `CriticProfiles` and `KDELambdaModel`), not a call-site flag. Once built in a coord system, all calls had to match.
- First-iteration used first-review-ts uniformly, to avoid coordinate drift between movies anchored on true embargo and movies anchored on first-review-proxy (since the first critic typically reviews some latency after the true embargo).
- The library remained post-first-review only — pre-first-review was explicitly out of scope (library raises rather than falling back).

---

## 4. Validation methodology

### 4.1 Apples-to-apples conventions

- **Leave-one-out training.** For each target movie, build profiles via `default_training_slugs(exclude_slug=target, before_date=target.close, n=20)` to avoid lookahead. Two builds per target (one per anchor mode).
- **Snapshot state from filtered reviews.** At each snapshot, observed state (`observed_critics`, `observed_count`, `first_review_dbc`) was computed from reviews with `estimated_timestamp < snap_time` only — not from full-movie data.
- **`actual_remaining` matches the library's internal filter.** Both anchors drop reviews with `days_before_close ≤ 0` (close-day bias, `BACKLOG.md §1.3`). So `actual_remaining` was counted as reviews with `0 < dbc ≤ snap_dbc` — symmetric across anchors even though under-counting true close-day arrivals.
- **Scaling enabled on both sides.** Calls used `observed_count` and `first_review_dbc`. Under embargo, additionally `anchor_lead_days = (target.close − target.first_review_ts) / 86400`.

### 4.2 Eight-step validation plan (as executed)

1. **Cohort-wide first-review lead distribution.** Compute `(close − first_review_ts)` across all 143 resolved movies. Outputs: action-window bound, T-7d selection-bias diagnostic.
2. **Training-set eyeball.** Same stats for a generic 20-movie training set. Decision gate: if training spread is materially tighter than cohort, training is unrepresentative.
3. **Pooled density comparison.** Plot population-prior KDE density under both anchors. Check day-0 spike under embargo (flag if >5%).
4. **A/B predicted-vs-actual remaining reviews (primary gate).** LOO per movie per anchor at T-3d (primary) and T-1d (secondary); T-7d gated on Step 1 selection-bias diagnostic.
5. **Scaling-value distribution comparison.** Instrumented `_compute_scaling` that returns `expected_so_far` and threshold/clamp flags.
6. **First-review pinning spot-check.** Quantify contribution of sparse "frequent-first" critics (n ≤ 3, ≥ 66% at day 0) to `expected_so_far`.
7. **Per-snapshot data-availability sanity check.** Confirm A/B sample sizes match Step 1.
8. **Decision.** Apply plan rules (promote / experimental / reject).

### 4.3 Primary decision criterion

Promote embargo-anchor to default only if T-3d MAE improved by ≥ 10% relative, without materially worse T-1d MAE. Otherwise keep experimental or reject.

---

## 5. Results

### Step 1 — Cohort gaps

```
Cohort size: 143 resolved movies
(close − first_review) days:
  min:    3.58
  median: 7.58
  IQR:    10.94  (Q1 = 5.62, Q3 = 16.56)
  max:    883.62
```

Action-window upper bound: **T-3.6d** (latest snapshot at which embargo can evaluate every movie).
Coverage by snapshot: **T-7d: 52.4%** (excluded — selection bias too large), T-3d: 100%, T-1d: 100%.

### Step 2 — Training set unrepresentative (FLAG)

```
Training (20 most recent): min=3.62, median=5.71, IQR=2.74, max=134.62
Cohort-wide:               min=3.58, median=7.58, IQR=10.94, max=883.62
```

Training gap IQR is **2.74 vs cohort 10.94** — the 20 most recent movies are materially tighter-gapped than the cohort. KDEs fit on this training set will be calibrated for short-gap movies and under-represent the long-tail.

### Step 3 — Day-0 spike under embargo (FLAG)

Embargo-anchor mass within ±0.5d of day 0: **8.99%** (plan threshold 5%). First-review pinning is distorting the population prior meaningfully more than the back-of-envelope 0.8% estimate predicted. The entire [0, current_dae] integration window under embargo starts inside this spike.

### Step 4 — A/B (PRIMARY GATE)

```
T-3d (n=133)   <-- PRIMARY DECISION GATE
  close:   MAE=10.23  median_err=+4.38  p90|err|=22.40
  embargo: MAE=15.27  median_err=−11.53 p90|err|=29.35
  embargo vs close MAE: −49.2% (WORSE)

T-1d (n=133)
  close:   MAE=5.64   median_err=−3.37  p90|err|=15.12
  embargo: MAE=8.16   median_err=−6.97  p90|err|=17.99
  embargo vs close MAE: −44.7% (WORSE)
```

Embargo-anchor systematically under-predicts (all median errors negative, and much more negative than close-anchor's median errors at both snapshots).

### Step 5 — Scaling diagnostic (reveals the mechanism)

```
Threshold-fire rate (expected_so_far < 40 → scaling = 1.0):
  T-1d:  close = 2.3%   embargo = 37.6%
  T-3d:  close = 18.0%  embargo = 49.6%

Clamp-hit rate (scaling hit 0.5 or 2.0):
  T-1d:  close = 30.1%  embargo = 34.6%
  T-3d:  close = 32.3%  embargo = 24.1%

Expected_so_far distribution (median):
  T-1d:  close = 104.5  embargo = 51.5
  T-3d:  close = 84.1   embargo = 41.0
```

Under embargo, `expected_so_far` is roughly **half** what it is under close. The 40-review threshold (tuned for close-anchor) fires 2-20× more often under embargo, which **disables the scaling correction** in the majority of cases. The scaling feedback is what lets close-anchor correct for atypical movies; without it, embargo-anchor runs open-loop.

### Step 6 — First-review pinning spot-check

```
Frequent-first critics (n ≤ 3, ≥ 66% at day 0): 122/780 (15.6%)
At T-3d with typical anchor_lead_days = 7.58:
  Total expected_so_far: 55.72
  Frequent-first contribution: 3.47 (6.2%)
  Plan flag threshold: ≥ 10% → NOT flagged
```

Individual contribution from "frequent first" critics is modest. The day-0 mass is distributed across many critics, not concentrated in a few sparse ones.

### Step 7 — Data availability sanity

A/B sample sizes matched Step 1 diagnostics exactly. Spot-checks on 5 random movies confirmed `first_review_ts` matches `reviews.csv`. No bug in the A/B loop.

---

## 6. Diagnosis: why theory contradicted data

Four interacting factors, in rough order of importance:

### 6.1 Scaling threshold mismatch (mechanism)

The `_compute_scaling` 40-review threshold and [0.5, 2.0] clamp were empirically tuned under close-anchor coords. Under embargo coords, the `[first_review_dae, current_dae]` integration window covers a smaller fraction of the population prior's mass than the analogous close-anchor window `[days_before_close, first_review_dbc]` — because the embargo-anchor density is sharply peaked near day 0 and the window starts at day 0, while the close-anchor density is smeared across a wider range and the window sits in its meat.

Consequence: `expected_so_far` is consistently ~half under embargo, so the 40-threshold fires 2-20× more often, disabling scaling's corrective feedback. The A/B is effectively comparing "close-anchor with scaling" vs "embargo-anchor without scaling" in most cases.

**This is the primary mechanical cause of the MAE regression.** It was foreseen (plan flagged the threshold for revalidation, Step 5 was designed to check it), but the effect size was not predicted.

### 6.2 First-review-proxy noise

We used each movie's first observed review as the embargo proxy (we don't have true embargo timestamps). Two noise sources:
- **Day-level timestamp confidence.** ~98% of backfilled reviews have day-level timestamps (rounded to midnight UTC). The proxy anchor has ± half-day noise.
- **Selection bias in the proxy itself.** "First observed review" is a censored statistic — it's a minimum over critics, which biases earlier when more critics pile in early. A movie with 150 reviews has an earlier "first" than a movie with 80 reviews, even if true embargos are the same.

Both effects are small per-movie but systematic, and propagate to the KDE shape.

### 6.3 Day-0 pinning distortion

By construction, each training movie contributes exactly one data point at day 0 (the first review is the anchor). Step 3 measured 8.99% of pooled mass within ±0.5d of day 0 — an order of magnitude above the plan's 0.8% back-of-envelope estimate. This inflates the embargo-anchor density's left edge, exactly where the `_compute_scaling` window begins.

Compounding effect: under shrinkage toward the population prior, the day-0 spike propagates into sparse critics' blended KDEs, spreading the distortion.

### 6.4 Training set unrepresentative of cohort

Training gap IQR 2.74 vs cohort IQR 10.94. The 20 most recent movies are atypically tight-gap. Any model trained on this set will be calibrated for short gaps and mispredict on long-gap targets. This affects both anchors, but embargo-anchor more severely because its KDE shapes are sharper — less room to absorb gap mismatch.

Root cause not diagnosed. Hypothesis: Kalshi may have recently shortened its (embargo → close) market windows, or recent movies skew towards shorter-release-cycle genres. Worth investigating independently.

### 6.5 Why theory still holds in principle

The convolution identity is mathematically correct. Conditioning IS strictly more informative than marginalizing — *given a correctly calibrated model*. The failure modes above all boil down to calibration / data quality, not structural:
- 6.1 is a hyperparameter tuned for one coord system carrying into another.
- 6.2 and 6.3 are both consequences of using first-review-as-proxy; real embargo data would remove them.
- 6.4 is a training-set sampling issue affecting both anchors.

So "embargo-anchor as implemented" failed, but "embargo-anchor with correct calibration and real embargo data" remains a live hypothesis.

---

## 7. Recommendations if someone returns to this problem

In rough order of expected impact per unit of work:

1. **Recalibrate the scaling threshold and clamp under embargo coords.** The 40-threshold was tuned for close-anchor; the scaling distribution results from Step 5 suggest it needs to be lower under embargo (maybe 20-25). Pair with tighter clamp. Easy to test with the notebook's existing instrumented scaling function.

2. **Drop the first review per training movie.** Removes the day-0 pinning by construction. Costs ~143 data points out of ~23k (trivial). Re-run Step 3 to verify day-0 mass drops below 5%.

3. **Investigate training-set gap distribution.** Why is the 20-most-recent-movies gap IQR 4× tighter than the cohort's? Possible fixes: stratified sampling across gap quantiles; temporal weighting; or just using a larger training set (n=40+). Affects both anchors.

4. **Get real embargo data.** Removes 6.2 and 6.3 entirely. Requires external data source — possibly Rotten Tomatoes' own embargo posting page or Kalshi market descriptions. Not trivial but highest-ceiling fix.

5. **Only then re-run the A/B.** If (1)-(3) can recover parity with close-anchor under Step 4's MAE gate, (4) is what could make embargo-anchor clearly win. Without the hygiene fixes, real embargo data alone would probably not save it because the scaling-threshold mechanism (6.1) still fires.

---

## 8. Process notes

**What went right:**
- Two independent agent reviews caught design drift before code was written (scaling window under explicit embargo, snapshot state filtering, `actual_remaining` counting convention, LOO correctness).
- Plan doc → implementation → validation sequence (per `PROTOCOL.md`) meant all scaffolding was in place when the A/B failed; we had instrumented scaling and per-step diagnostics rather than just a raw MAE regression.
- Rejecting the feature based on empirical evidence rather than forcing it in is the right call. Keeping code-complete work but backing out is fine.

**What went wrong:**
- The plan's Step 5 (scaling diagnostic) should have run *before* the primary A/B, not after. The threshold mismatch was foreseeable. Running Step 5 first would have revealed the 40-threshold issue and motivated re-tuning before the Step 4 gate decided the fate of the whole proposal.
- Day-0 pinning estimate in the plan (0.8%) was ~10× too low. A sharper upfront sanity check (e.g., "what fraction of first-reviews happen on the training movie's first scraped day" — likely near 100%) would have flagged this.
- Training set unrepresentativeness (Step 2's FLAG) was noted but not acted on before Step 4. Should have either investigated the training selection or broadened it before running the A/B.

**Jupyter kernel sidenote:** during notebook execution, the PEP 660 editable install's `.pth`-based finder was not activated in the jupyter kernel subprocess (neither via `uv run jupyter` nor `uv run python -m jupyter`). Workaround used: prepend the repo root to `sys.path` in the notebook's first cell. Future notebooks may need the same workaround. Registered a `rt-forecasting` kernelspec at `~/Library/Jupyter/kernels/rt-forecasting/` which explicitly points to the venv's python3 — still subject to the same `.pth` issue but kept around as the working kernel.

---

## 9. What was changed in the codebase (reverted after this file was written)

Files modified:
- `rotten_tomatoes_forecasting/critic_model.py` — `AnchorMode` type alias added, `CriticProfiles` and `KDELambdaModel` gained `anchor_mode` (and profiles gained `anchor_ts_map`), `build_critic_profiles` accepted `anchor_mode` and `embargo_lift_ts_map` params, `estimate_lambda` and `_compute_scaling` branched on mode with `anchor_lead_days` kwarg.
- `tests/test_critic_model.py` — 12 new tests in a `TestEmbargoAnchor` class.

Files added:
- `findings/kde_anchor_choice.md` — standalone theory note (content consolidated into §2 above).
- `plans/plan_embargo_anchor.md` — implementation plan (gitignored, contents summarized in §3 and §4 above).
- `notebooks/embargo_anchor_validation.ipynb` — 21-cell validation notebook (contents documented in §4 and §5 above).
- This findings file.

All of the above (except this file, which you dragged out before rewinding) were removed when the session was rewound. Pre-session git state is the working, accurate-model state.
