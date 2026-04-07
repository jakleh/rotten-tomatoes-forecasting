# Findings: Per-Critic KDE Model Validation

**Date:** 2026-04-07
**Notebook:** `notebooks/critic_model_validation.ipynb`
**Module:** `critic_model.py`

---

## Setup

- Training set: 20 most recent resolved movies
- 744 critics total: 400 empirical KDEs, 300 sparse fallback (0-1 reviews), 44 degenerate fallback (zero variance)
- Population prior peaks at ~4.1 days before close
- sum(base_rate) = mean reviews/movie = 128.4 (sanity check passes exactly)

---

## Lambda estimates (no observed reviews)

| Horizon | Expected remaining | Lambda (rev/hr) |
|---------|-------------------|-----------------|
| T-7d | 95.5 | 0.568 |
| T-3d | 31.0 | 0.431 |
| T-1d | 5.4 | 0.224 |
| T-12h | 2.3 | 0.191 |

These are higher than the critics_index.ipynb prototype (T-7d: 83.7, T-3d: 24.3, T-1d: 2.4) because the new model includes 344 sparse/degenerate critics via the population prior. This is by design — long-tail critics self-average into a smooth background rate.

---

## Predicted vs actual remaining reviews (141 resolved movies)

### Initial guard rails (min expected 5.0, clamp [0.3, 3.0])

| Snapshot | Scaled MAE | Unscaled MAE | Scaled median err | Unscaled median err |
|----------|-----------|-------------|-------------------|---------------------|
| T-7d | 74.7 | 49.8 | +38.8 | -12.3 |
| T-3d | 14.9 | 24.4 | -3.3 | -18.3 |
| T-1d | 2.6 | 3.3 | +1.3 | -0.6 |

**Problem:** Scaling massively overcorrects at T-7d. Most KDE mass is in the last 3-5 days, so at T-7d "expected so far" is based on tail mass — the denominator is small and the ratio explodes. Example: `a_complete_unknown` at T-7d — unscaled predicts 73.8 remaining (actual: 72, nearly perfect) but scaled blows it up to 221.

### After fix (min expected 40.0, clamp [0.5, 2.0])

| Snapshot | Scaled MAE | Unscaled MAE | Scaled median err | Unscaled median err |
|----------|-----------|-------------|-------------------|---------------------|
| T-7d | 49.8 | 49.8 | -12.3 | -12.3 |
| T-3d | 19.5 | 24.4 | -12.4 | -18.3 |
| T-1d | 2.7 | 3.3 | +0.9 | -0.6 |

- **T-7d:** Fixed. Scaling no longer engages (expected_so_far ~33 < threshold 40). Falls back to unscaled.
- **T-3d:** Regressed from 14.9 to 19.5 but still beats unscaled (24.4). Tighter 2x clamp limits aggressive correction. Systematic underprediction (median err -12.4) because the base model reflects the "average" 128-review movie.
- **T-1d:** Essentially unchanged. Excellent.

**Trade-off accepted.** T-3d underprediction is conservative for betting — overweights current score, so you'd underestimate edge rather than overestimate it.

---

## p_fresh estimates

- **Prior p_fresh (no observations):** 0.694 (in expected 0.6-0.8 range)
- **At T-1d vs final score:** MAE = 0.031, correlation = 0.990
- Blending works well — at T-1d with 90%+ reviews in, p_fresh closely tracks the final tomatometer score

---

## Key takeaways

1. **The model works.** Lambda estimates are reasonable, p_fresh is excellent, and the architecture (profiles → KDEs → aggregate lambda) is sound.
2. **Scaling is helpful but limited.** It improves T-3d and T-1d but can't engage safely at T-7d. The single-multiplier approach can't distinguish "popular movie" from "early-heavy movie."
3. **Systematic underprediction at T-3d.** The base model is calibrated to the average movie (128 reviews). High-volume movies are underestimated. This is conservative for betting.
4. **The action window is T-3d to T-1d.** At T-7d the model is informative but not precise enough for confident bets. By T-1d the model is tight (MAE=2.7 reviews).
5. **Next step: backtest.** Feed these estimates into `compute_edge()` against historical price data to measure whether the model actually identifies profitable bets.
