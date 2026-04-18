# Prompts

Handoff prompts for starting new conversations on the rotten-tomatoes-forecasting forecasting library. Read CLAUDE.md and BACKLOG.md for full context before using any prompt.

---

## Context for All Prompts

This repo is a pure forecasting library. Its contract ends at `compute_edge() -> (edge_cents, p_yes, p_no)`. Strategy, backtesting, and execution concerns live in the orchestrator repo (`~/Desktop/kalshi-trading/`).

The package is at `rotten_tomatoes_forecasting/` with 8 public API symbols: `compute_edge`, `build_critic_profiles`, `build_kde_lambda_model`, `estimate_lambda`, `estimate_p_fresh`, `default_training_slugs`, `CriticProfiles`, `KDELambdaModel`.

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

### Prompt 4: Pre-ship tuning for validated deployment stack

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
