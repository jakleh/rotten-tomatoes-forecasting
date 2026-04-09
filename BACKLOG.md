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
