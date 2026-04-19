"""
p_fresh estimation: base-rate-weighted critic prior blended with the observed running rate.

Unchanged in behavior from 0.1.x. Moved out of `critic_model.py` as that module is
removed at 0.2.0. Uses the shared base-rate primitive from `pool.py` and adds
per-critic fresh_rate on top.
"""

import pandas as pd

from rotten_tomatoes_forecasting.pool import compute_critic_base_rates

_DEFAULT_FALLBACK_PRIOR = 0.65


def _compute_fresh_rates(
    reviews_df: pd.DataFrame, training_slugs: list[str]
) -> dict[str, float]:
    """Per-critic `positive / total` in the training set. Missing critics default to 0.5."""
    if not training_slugs:
        return {}
    sub = reviews_df[reviews_df["movie_slug"].isin(training_slugs)]
    if sub.empty:
        return {}
    is_fresh = sub["tomatometer_sentiment"] == "positive"
    totals = sub.groupby("reviewer_name").size()
    fresh = is_fresh.groupby(sub["reviewer_name"]).sum()
    fresh_rate = (fresh / totals).fillna(0.5)
    return fresh_rate.to_dict()


def estimate_p_fresh(
    reviews_df: pd.DataFrame,
    training_slugs: list[str],
    observed_critics: set[str],
    fresh_count: int,
    total_count: int,
    n_prior: float = 20.0,
) -> float:
    """Estimate p_fresh by blending the critic-weighted prior with the observed rate.

    Uses base_rate-weighted fresh_rate over unobserved critics as the prior, blended
    with the running fresh/total by a pseudo-count of `n_prior` observations.

    Args:
        reviews_df: Review rows. Needs columns `movie_slug`, `reviewer_name`,
            `tomatometer_sentiment`.
        training_slugs: Movies used to build per-critic base_rate and fresh_rate.
        observed_critics: Reviewers already seen for the target.
        fresh_count: Positive reviews observed so far.
        total_count: Total reviews observed so far.
        n_prior: Blend pseudo-count. At `total_count == n_prior`, weight is 50/50.

    Returns:
        Estimated probability that each future review is positive, in [0, 1].
    """
    base_rates = compute_critic_base_rates(reviews_df, training_slugs)
    fresh_rates = _compute_fresh_rates(reviews_df, training_slugs)

    remaining = [(c, br) for c, br in base_rates.items() if c not in observed_critics]
    weight_sum = sum(br for _, br in remaining)

    if weight_sum > 0:
        prior_p_fresh = sum(
            br * fresh_rates.get(c, 0.5) for c, br in remaining
        ) / weight_sum
    else:
        prior_p_fresh = _DEFAULT_FALLBACK_PRIOR

    if total_count == 0:
        return prior_p_fresh

    observed_p_fresh = fresh_count / total_count
    blend_weight = total_count / (total_count + n_prior)
    return blend_weight * observed_p_fresh + (1 - blend_weight) * prior_p_fresh
