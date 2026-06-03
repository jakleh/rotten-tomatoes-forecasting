# Prompts

Handoff prompts for starting new conversations on the rotten-tomatoes-forecasting forecasting library. Read CLAUDE.md and BACKLOG.md for full context before using any prompt.

---

## Handoff convention — READ THIS FIRST

Every prompt below is a task payload. Before acting on one, run the session-start ritual:
1. **Foundation:** read `CLAUDE.md` "Current Conventions" (authoritative) + the relevant `BACKLOG.md` section + any findings doc the task names.
2. **Plans:** read any referenced `plans/*.md` in full (gitignored, local).
3. **Sanity-check:** `uv sync`, then `.venv/bin/python -m pytest tests/ -q` (expect 98 green) before touching code.
4. **Then** start the task.

A good NEW handoff prompt gives: the task in one line; the foundation docs to read; a prerequisite/STOP check if the task is data-gated; concrete numbered steps; an explicit decision rule + out-of-scope list; and the deliverable.

## Context for all prompts

This repo is the focused RT-modeling workspace (v0.2.0, Ridge lambda model); the math is pure DataFrame-in / numbers-out, ending at `compute_edge()`. Per the scripts direction (CLAUDE.md "Project Overview"), eventual deployment is a script that imports the edge-calc + kalshi execution helpers — not the orchestrator.

0.2.0 public API (in `rotten_tomatoes_forecasting/`): `compute_edge`, `estimate_lambda`, `estimate_p_fresh`, `fit_lambda_regressor`, `load_default_regressor`, `extract_lambda_features`, `compute_close_day_phase2`, `LambdaRegressor`, `LambdaPrediction`, `naive_lambda`, `naive_p_fresh`. (The 0.1.x KDE API — `build_critic_profiles`, `KDELambdaModel`, `CriticProfiles`, `default_training_slugs`, etc. — was removed.)

---

### Prompt 1: Kalshi-independent lambda validation (Backlog 1.1)

```
Read CLAUDE.md and BACKLOG.md 1.1. Design and implement a validation notebook that tests estimate_lambda() accuracy using minute-level review data, without any reference to Kalshi market resolution.

The idea: for movies with minute-level timestamps, we know the exact review count at any point in time. At snapshot time T, we can call estimate_lambda() to predict remaining reviews, then compare to the actual count that arrived between T and close. This is a pure model accuracy test.

Currently only the_drama and the_super_mario_galaxy_movie have minute-level data. Check if that's enough for meaningful validation, or if we need to wait for more data. If the sample is too small, document what we'd need and defer.

Work in notebooks/lambda_validation.ipynb. Write findings to findings/lambda_validation.md if results are meaningful.
```

### Prompt 2: Close-day lambda bias patch (Backlog 1.3)

```
Read CLAUDE.md, BACKLOG.md 1.3, and brainstorm/brainstorm_close_day_lambda_bias.md. The KDE model drops reviews on the close day because of a UTC midnight vs actual close time mismatch. The partial fix (full UTC datetimes in movies_index.csv) was applied but 98% of reviews still have day-level timestamps.

Evaluate the patch approaches in the brainstorm doc and implement the best one. The fix should be in rotten_tomatoes_forecasting/critic_model.py (the build_critic_profiles or estimate_lambda path). Validate by checking whether lambda estimates change for movies near close.
```

### Prompt 3: p_fresh calibration audit (Backlog 1.2)

```
Read CLAUDE.md and BACKLOG.md 1.2. Break down the existing p_fresh calibration (from findings/critic_kde_model_validation.md) by movie characteristics. Are there systematic biases for certain types of movies?

Work in notebooks/p_fresh_calibration.ipynb.
```

### Prompt 4: Pre-ship tuning for validated deployment stack (COMPLETED 2026-04-18)

```
Read, in order: CLAUDE.md, BACKLOG.md (especially §1.5-§1.7), findings/stratified_training_investigation.md (especially TL;DR + §10-15), and brainstorm/brainstorm_pre_ship_tuning.md. The brainstorm has been reviewed by an independent agent and incorporates those fixes.

Context: A long investigation (summarized in the findings doc) validated a deployment-recommended stack: `combined_score (α=0.5) + bandwidth_cap (0.7d) + piecewise (F=1.0)`. Before integrating this into library code, a small tuning pass is warranted to re-validate or tighten a few parameters. The Path B (learned multi-feature model) alternative has been scoped and deferred — see plans/plan_learned_similarity_model.md.

Your task is to execute the pre-ship tuning pass as scoped in the brainstorm:

1. Start a successor notebook: `notebooks/pre_ship_tuning.ipynb`. Cell 1 imports helpers from `notebooks/stratified_training_validation.ipynb` (predict_window, combined_score_selector, gap_overlap_ranked_selector, build_kde_lambda_model_capped, passes_skip_rules_for_snap, critics_in_window, jaccard, actual_remaining, etc.). Do not rebuild these. Consider importing via `%run` or factoring into a small `.py` module under `notebooks/` if import hygiene matters.

2. Cell 2: compute the anchor baseline — current deployment stack MAE at T-3d, T-5d, T-7d full window (snap to midnight UTC of close). This is the comparison point for everything below. Use F=0.7 as the ship-time piecewise value (not F=1.0 — see brainstorm's G1 discussion).

3. Cell 3: gap-distribution diagnostic. For each of 143 targets, count candidates with |gap_diff| ≤ 0.5d, 1d, 2d, 5d. Print a summary table stratified by gap quantile (Q1-Q4). This decides whether tight-σ_gap is feasible.

4. Execute T1 (σ_gap sweep ∈ {2, 4, 8, 16, ∞}) per brainstorm. Use bootstrap CI (1000 resamples, paired per target) for decision rule: ≥3% T-3d MAE improvement with CI95 lower > 0 → replace σ_gap=8. Otherwise keep current.

5. Execute T2 (n_training sweep ∈ {5, 10, 15, 20, 25, 30, 50}) using T1's σ_gap winner. Same decision rule.

6. Report results. If neither T1 nor T2 produce a winner, the stack ships unchanged — but the re-validation itself is valuable.

7. Update `findings/stratified_training_investigation.md` with a new section summarizing pre-ship tuning. Update `BACKLOG.md` §1.5 if any parameter value changes. Update the memory pointer at `/Users/jakelehner/.claude/projects/-Users-jakelehner-Desktop-rotten-tomatoes-forecasting/memory/project_stratified_training_results.md` to reflect final ship values.

After this pass, the next conversation will cover library integration (a separate plan doc covering the four interventions from BACKLOG §1.5: bandwidth ceiling parameter, selector function, piecewise helper, snapshot-state helper).

Items explicitly OUT OF SCOPE for this pass (documented in brainstorm's "Out of scope" section): α tuning (at noise floor, keep 0.5), F varying by movie type (Path B), target-scope narrowing (strategy concern), new features, learned-weights model, retiring piecewise, library integration (next pass).

Gated on fresh reviews.csv from Neon (NOT blocking for this pass): G1 (F re-estimation), G2a (re-pick frequency curve on h/m targets). These can run as a follow-up after Jake pulls fresh data.

Follow PROTOCOL.md: the successor notebook is an analysis notebook tier, so a brief intent section at top suffices — no separate plan doc needed unless T1 or T2 reveals something that changes the approach. Do use a TodoWrite-style task breakdown for tracking.

Do NOT add new features during this pass. The user's stance: no more feature hunting, we're at the local optimum given current feature set + cohort size. If something looks like Path B territory, defer it.

Session-level goal: complete T1+T2, ship-decide, and hand off to the integration conversation.
```

### Prompt 5: Library integration for validated stack (SUPERSEDED by Prompt 8)

> **2026-04-19:** the Ridge investigation (`findings/ridge_lambda_investigation.md`) changed the ship candidate. Use **Prompt 8** below for library integration. This Prompt 5 is kept as historical context for the original KDE-vs-Ridge dual-model question that led to the Ridge investigation.

```
Read, in order: CLAUDE.md, BACKLOG.md §1.5 and §1.8, findings/stratified_training_investigation.md §16-17, and findings/path_b_lite_investigation.md (especially TL;DR, §9, §15-17).

Context: extensive investigation converged on two viable ship candidates, each with known tradeoffs (per path_b_lite_investigation.md §15):

  Weighted-KDE stack (per-target stability, known h/m bias):
    combined_score(α=0.5, σ_gap=8) top-20 selector
    + weighted KDE build (per-data-point weights = combined_score values) [new]
    + bandwidth (floor=0.5d, ceiling=0.7d)
    + midnight+noon snap convention: snap_time = midnight UTC on close-N, day-level reviews shifted to 12:00 UTC [new]
    + phase_1 = KDE integral over (midnight_utc_dbc, snap_dbc_effective]
    + phase_2 = constant C=2
    → cohort MAE 13.26, h/m MAE 18.23, h/m me −6.78

  Ridge(α=10) regression (cohort-calibrated, per-target variance):
    Features: observed_count, first_review_dbc, target_gap, observed_rate,
              rate_last_day, rate_first_day, top_critic_frac, pub_diversity,
              pub_entropy, low_activity_frac
    → cohort MAE 8.85, h/m MAE 26.91, h/m me +0.69

Neither dominates. Ridge wins cohort + calibration; KDE wins h/m MAE with systematic under-bias. Choice depends on deployment priorities.

Known architectural ceiling: 14+ interventions tested in Path B-lite investigation couldn't break through the h/m under-prediction limit. Root cause is that features observable at T-3d don't predict late-surge behavior on long-gap targets. See findings/path_b_lite_investigation.md §13 for full ruled-out list.

Task: write a library integration plan in plans/plan_library_integration.md covering:

1. API surface. Which new symbols to export from __init__.py, which live as internal helpers:
   - `build_kde_lambda_model(bandwidth_ceiling=0.7)` — add parameter with new default.
   - `build_kde_lambda_model_weighted(profiles, scores)` — NEW public function.
   - `combined_score_selector` + `combined_score_with_scores` — NEW public functions.
   - `compute_close_day_phase2(C=2.0)` — trivial, returns C.
   - `snapshot_state` helper — promote from notebook.
   - Consider: `fit_phase1_regressor(features, targets) -> regressor` + `predict_phase1_regression(regressor, features)` for the Ridge alternative. Ships alongside KDE.
   - Convention switch: rename defaults or document? Backward compat with existing consumers matters.

2. Dual-model policy: ship BOTH KDE and Ridge? Or just KDE with Ridge as experimental?
   Pros of both: deployment flexibility, orchestrator picks per strategy.
   Cons: doubles API surface, test matrix.

3. Default-value policy for new parameters: opt-in or on-by-default?
   Migration notes for the orchestrator consumer at ~/Desktop/kalshi-trading/.

4. Test additions for each new function + integration tests.

5. CHANGELOG entry and version bump (0.1.0 → 0.2.0? Breaking-ish given snap semantics change).

After the plan is signed off, the implementation pass is a separate conversation.

BLOCKERS / open questions for Jake before writing the plan:
- Ship single model (KDE only) or dual (KDE + Ridge)?
- Is midnight+noon snap convention a hard migration or does library support both?
- Does orchestrator currently expect the old snap convention? Check kalshi-trading/src/ before writing the plan.

OUT OF SCOPE (deferred):
- Finite-pool model (brainstorm exists; architectural rework).
- Hier-Bayes with different shape family (Gamma was null).
- TMDb metadata path (declined by Jake).
- More h/m cohort data (wait).
```

### Prompt 6: Wait-and-revisit pass (future, after h/m cohort grows)

```
Read findings/path_b_lite_investigation.md §13-14. Several interventions that were ruled out may become viable as the cohort's h/m fraction grows over time.

Trigger: when ≥30% of resolved cohort movies have h/m timestamps (currently ~5%).

Re-test with fresh cohort:
1. Weighted KDE + midnight+noon on h/m-representative cohort. Does the architectural ceiling still hold?
2. Hierarchical Bayes with non-parametric partial-pooling (per-movie empirical CDFs rather than Gamma parametric).
3. Shape-similarity selector with expanded cohort coverage (more shape-similar neighbors available).
4. Finite-pool model — if cohort shifts toward h/m, per-target P(review | movie) becomes tractable from more-granular data.

Evaluation: hold out last N h/m-resolved movies (not the original 5 — those are now permanent cohort members). Decision rule: h/m MAE materially below current ship candidates.
```

Optional follow-up (NOT in this plan, but worth logging as a BACKLOG item): time-series regression as alternative or blend. See findings §9 — Ridge beats KDE on cohort MAE (9.71 vs 13.45) and is near-calibrated on h/m (me ≈ 0), but higher h/m variance. Worth exploring as a blended predictor, but not prerequisite for initial integration.
```

### Prompt 7: Continue phase-1 KDE menu mix-and-match

```
Read, in order: CLAUDE.md, brainstorm/brainstorm_phase1_kde_menu.md (the menu reference — primary doc for this work), findings/path_b_lite_investigation.md (full investigation context), findings/stratified_training_investigation.md §16-17 (pre-ship tuning + G1 piecewise audit).

Context: Jake is running a mix-and-match iteration on the phase-1 KDE pipeline. Despite 14+ ruled-out interventions documented in path_b_lite_investigation.md (concluded 2026-04-18), he wants to keep exploring combinations of the tested pipeline components — particularly those not yet tested in combination. The component menu lives at brainstorm/brainstorm_phase1_kde_menu.md, organized by pipeline stage A-K plus alternative architectures.

Current ship candidate stack (per menu): A3 + B1 + C2 + D2 + D3 + E2 + F1 + G2 + H2 + I2 + J2 + K1.
  - cohort MAE 13.26, h/m MAE 18.23, h/m mean_err −6.78
  - Midnight+noon convention adopted per Jake 2026-04-18
  - Known architectural ceiling on h/m under-prediction (oracle test confirmed)

Primary untested options (per menu):
  - **E5 (transformed base_rate):** square-root / log / cap / rank compression to reduce workhorse-critic dominance without per-target tiering (which Option C / E3 mis-classified late-surge targets)
  - **C5 (hier-Bayes with log-normal or mixture-Gamma)**
  - **C6 (non-parametric empirical-CDF partial pooling)**
  - **D5 (custom bandwidth ceiling by target type)**
  - **E6 / K3 (per-critic base_rate conditioned on target features)** — requires metadata

Reference notebooks (in notebooks/):
  - _helpers.py — factored helpers (combined_score, weighted KDE build, etc.)
  - path_b_lite_weighted_kde.ipynb — C2 weighted KDE win
  - option_c_base_rate_adjustment.ipynb — E3 Option C (Q1 win, h/m regression)
  - scaling_clamp_sweep.ipynb + scaling_diagnostic.ipynb — F2/F4 ruled out
  - shape_scale_model.ipynb — C3 ruled out
  - hier_bayes_shape.ipynb — C4 ruled out
  - time_series_regression.ipynb — R1 Ridge alternative
  - critic_magnet.ipynb — K2 falsified

Workflow for each test:
  1. Jake proposes a combination: "A3 + B1 + C2 + D3 + E5a + F1 + G2 + H2 + I2 + J2 + K1"
  2. Build a notebook that runs the combo under midnight+noon convention (G2 + H2).
  3. Evaluate on cohort CV (no h/m) + h/m subset (5 movies: the_drama, super_mario_galaxy, forbidden_fruits, they_will_kill_you, you_me_and_tuscany).
  4. Compare to ship stack baseline.
  5. Decision rule: h/m MAE improvement AND cohort MAE non-regression (>−5%).

Per PROTOCOL.md, notebook tier: brief intent section at top, no formal plan doc required for each combo. If a combo produces a meaningful result (positive or negative), add it to brainstorm_phase1_kde_menu.md under the relevant category, updating "ruled out" or "adopted" status as appropriate.

Jake's stated preference: "in the face of all logic and reason, I still think the KDE approach can be fixed fully." Engage seriously — the architectural ceiling finding is real but not proven absolute across all untested combinations. A promising untested combo could still move things. The oracle test (findings §8.2) established the limit is in the architecture, not selection — but components like E5 (base_rate transform) haven't been tested in isolation.

Do NOT push for library integration without Jake's explicit go-ahead. Prior session included a course-correction where Jake pushed back on "eager to integrate" — the current model is good enough to document but not necessarily to ship.

Start by asking Jake what combo he wants to test first, or propose one based on the menu if he asks.
```

### Prompt 8: Ridge lambda integration (replaces Prompt 5)

```
BEFORE writing any timestamp / snap / phase logic, re-read the "Current Conventions"
section at the top of CLAUDE.md. It defines the ET-midnight snap convention, 10h
phase-2 window, C=1 constant, opt-in noon-shift, JSON artifact format, and snap_days
routing. These supersede ALL older conventions in findings/, notebooks/, brainstorm/,
and the current live library code (which is being replaced).

If you grep historical docs for "snap_time" or "phase_2" or "apply_noon_shift", you'll
land on notebooks that use the older UTC-midnight convention, C=2 phase-2, or silent
noon-shift — those are intentionally-preserved historical validation artifacts with
"CONVENTION WARNING" banners at the top. The banners say "do not copy into new code."
Heed them. Current conventions in CLAUDE.md win.

Read, in order: CLAUDE.md (conventions first, then overview), plans/plan_ridge_integration.md, findings/ridge_lambda_investigation.md (TL;DR + §2 + §4 for the spec), brainstorm/brainstorm_ridge_optimization.md (for the investigation context).

Context: the 2026-04-19 Ridge investigation established ridge_t2 as the ship lambda estimator, superseding the KDE-based architecture. Plan doc is signed-off and covers: module layout, API surface, fit artifact strategy, test additions, version bump, orchestrator migration, phased implementation.

Task: execute the plan's Phases A → E (library-side implementation). That's:

  Phase A (scaffolding):   create lambda_model.py, features.py, pool.py, p_fresh.py module stubs; define LambdaRegressor + LambdaPrediction dataclasses; migrate estimate_p_fresh unchanged.
  Phase B (features):      implement pool.py (A1 pool, base_rates, top-tier) + extract_lambda_features (17 features).
  Phase C (fit/predict):   implement fit_lambda_regressor + estimate_lambda returning LambdaPrediction.
  Phase D (artifact):      fit on current cohort, pickle to _artifacts/default_regressor.pkl, implement load_default_regressor.
  Phase E (cleanup):       delete critic_model.py contents, update __init__.py exports, update CLAUDE.md / BACKLOG.md §1.5a / PARAMETERS.md, version bump + changelog.

Per PROTOCOL.md: the plan doc exists; implementation can proceed without another planning pass. Test coverage target: 80%+ on new modules.

Decision points flagged in plan §13:
  - Default artifact naming: recommend `default_regressor.pkl` (simple).
  - p_fresh consolidation with pool.py: recommend YES (avoid duplicate base_rate paths).
  - snap_dbc routing: recommend explicit snap_days arg (no interpolation).
  - Out-of-range snaps: raise (don't interpolate or fallback).
  - Version sequencing: full replace at 0.2.0 (not a 0.2→0.3 deprecation).

If Jake has different preferences on any of these decision points, confirm before implementing.

Out of scope (per plan §12): tier 3 stacking, TMDb metadata, orchestrator-side changes (covered by findings/trading_strategy_from_ridge_errors.md separately). Phase F is an orchestrator-repo effort.

Deliverable: a PR-ready working library at version 0.2.0 with all Phase A-E work complete, tests passing, docs updated.
```

### Prompt 9: Multi-anchor instantaneous-rate Ridge features (notebook only, data-gated)

```
Read, in order: CLAUDE.md (especially "Current Conventions"), BACKLOG.md §2.4, findings/ridge_lambda_investigation.md (§4.1 for current 17-feature set, §4.3 for LOO baseline).

PREREQUISITE CHECK FIRST. Before any modeling, count cohort movies with m/h-confidence pre-close timestamps spanning enough of the first-review-to-close window to compute smoothed instantaneous rates at multiple anchor points. Memory notes only `the_drama` and `super_mario_galaxy` had useful pre-close minute-level data at the 0.2.0 ship (2026-04-19). If the current count is ≤5, STOP and tell Jake this is still data-gated — do not force the experiment on insufficient data.

Context: 2026-04-20 brainstorm with Jake. The committed proposal (BACKLOG §2.4) is rate-at-anchor features for Ridge, NOT a standalone MVT-style predictor — that framing was explicitly ruled out in the brainstorm because MVT doesn't guarantee a shared t* across movies.

Task (notebook tier only — NO library changes):

1. Work in notebooks/anchor_rate_features.ipynb. Brief intent section at top per PROTOCOL.md analysis-notebook tier.

2. Report the h/m-qualified cohort subset size. Apply the stop rule above.

3. Define fractional-time anchors (e.g., f ∈ {0.2, 0.4, 0.6, 0.8} of the first-review-to-close gap). For each anchor f, compute a smoothed instantaneous review rate using a small time window. The window width is a knob — experiment and document the choice. This is the regressor version of the KDE bandwidth problem; be explicit about it.

4. Add anchor-rate features to the existing 17-feature Ridge stack. Rerun cohort LOO using fit_lambda_regressor-compatible infrastructure. Compare MAE snap-by-snap against the shipped 0.2.0 baseline (numbers in findings/ridge_lambda_investigation.md §4 and BACKLOG.md §1.1).

5. Decision rule: materially better cohort LOO MAE at early snaps (especially T-5d / T-4d) AND no regression at late snaps. Not shipping regardless of outcome — this pass is notebook-only exploration.

6. Write findings summary in the notebook (or findings/anchor_rate_features.md if results are meaningful). Update BACKLOG.md §2.4 with the outcome.

Known risks to document in the notebook:
- Smoothing window choice reintroduces a bandwidth parameter in miniature.
- Local burst/lull near an anchor biases the rate estimate. Mitigate by averaging over several nearby anchors or short windows.
- h/m cohort may not be representative of the full 144-movie cohort; caveat any generalization claim.

OUT OF SCOPE: library integration (separate backlog item if this wins), standalone MVT-style predictor (ruled out in brainstorm), shipping to any 0.2.x.

Deliverable to Jake: sample-size check result, MAE comparison table if the experiment ran, short findings blurb.
```
