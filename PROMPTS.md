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
