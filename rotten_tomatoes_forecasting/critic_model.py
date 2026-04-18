"""
Per-critic KDE lambda model for estimating review arrival rates and fresh probabilities.

Architecture: three independent layers sharing a data foundation.
  Layer 1: CriticProfiles -- base_rate, fresh_rate, timing_data per critic
  Layer 2: KDELambdaModel -- fits KDEs to timing data, estimates remaining reviews
  Layer 3: estimate_p_fresh -- weighted average of fresh rates, no KDEs

See plans/plan_critic_kde_lambda.md for the full design.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# -- Data structures -----------------------------------------------------------


@dataclass
class CriticProfiles:
    """Per-critic base rates, fresh rates, and timing data. Shared data layer.

    Attributes:
        df: DataFrame with columns:
            - reviewer_name (str): Critic identifier.
            - base_rate (float): P(critic reviews a movie) = movies_reviewed / n_training_movies.
              Range [0, 1]. Sum across all critics ~ mean reviews per movie.
            - fresh_rate (float): P(positive | critic reviews) = fresh / total for this critic.
              Range [0, 1]. 0.5 default if no reviews (shouldn't happen in practice).
            - timing_data (list[float]): Days-before-close values for each review in training set.
              Used by KDELambdaModel to fit per-critic timing distributions.
            - n_reviews (int): Total reviews by this critic in the training set.
        training_slug_count: Number of movies in the training set.
    """

    df: pd.DataFrame
    training_slug_count: int


@dataclass
class KDELambdaModel:
    """KDE-based lambda estimator. Built once from CriticProfiles, evaluated many times.

    Live adaptation uses two mechanisms (neither is shape blending):
      1. Shape: observed critics are dropped from the sum, adapting the aggregate shape.
      2. Scale: observed/expected ratio multiplies the total (disabled when expected < 40).

    Per-critic KDE construction uses shrinkage toward a population prior to handle sparse
    critics -- this is a build-time concern, not a live update mechanism.

    Attributes:
        profiles: The CriticProfiles used to build this model.
        population_prior: Gaussian KDE fitted to ALL reviews' timing data across all critics.
            Used as fallback shape for sparse critics (0-1 reviews or degenerate variance).
        critic_kdes: Dict mapping reviewer_name to KDE entry dict:
            - 'empirical' (gaussian_kde | None): Critic's own KDE, or None if fallback.
            - 'n' (int): Number of timing data points for this critic.
            - 'k' (float): Build-time shrinkage parameter. When computing a critic's KDE
              integral, empirical gets n/(n+k) weight and population prior gets k/(n+k).
              This regularizes sparse critics; it does not interact with live observations.
        shrinkage_k: Global shrinkage parameter (default 3.0). Higher = more prior influence
            at build time for sparse critics.
        bandwidth_floor: Minimum KDE bandwidth in days (default 0.5). Prevents overfitting
            on critics with tightly clustered review times.
    """

    profiles: CriticProfiles
    population_prior: gaussian_kde
    critic_kdes: dict
    shrinkage_k: float
    bandwidth_floor: float


# -- Layer 1: Critic profiles -------------------------------------------------


def build_critic_profiles(
    reviews_df: pd.DataFrame,
    close_date_map: dict[str, pd.Timestamp],
    training_slugs: list[str],
    verbose: bool = True,
) -> CriticProfiles:
    """Build critic profiles from training set. No KDEs -- just base rates, fresh rates, timing data."""
    n_movies = len(training_slugs)

    # Filter reviews to training slugs
    train = reviews_df[reviews_df["movie_slug"].isin(training_slugs)].copy()

    # Join bet close date and compute days before close
    close_map = pd.Series(close_date_map)
    train["bet_close"] = train["movie_slug"].map(close_map)
    train["days_before_close"] = (
        train["bet_close"] - train["estimated_timestamp"]
    ).dt.total_seconds() / 86400

    # Keep only reviews before close (dbc > 0)
    train = train[train["days_before_close"] > 0]

    # Per-critic stats
    rows = []
    for name, group in train.groupby("reviewer_name"):
        movies_reviewed = group["movie_slug"].nunique()
        fresh = (group["tomatometer_sentiment"] == "positive").sum()
        total = len(group)
        timing = group["days_before_close"].values.tolist()

        rows.append(
            {
                "reviewer_name": name,
                "base_rate": movies_reviewed / n_movies,
                "fresh_rate": fresh / total if total > 0 else 0.5,
                "timing_data": timing,
                "n_reviews": total,
            }
        )

    df = pd.DataFrame(rows)

    if verbose:
        sum_base = df["base_rate"].sum()
        mean_reviews = train.groupby("movie_slug").size().mean()
        print(f"[profiles] {len(df)} critics, {n_movies} training movies")
        print(f"[profiles] sum(base_rate)={sum_base:.1f}, mean reviews/movie={mean_reviews:.1f}")
        print(
            f"[profiles] fresh_rate: median={df['fresh_rate'].median():.3f}, "
            f"mean={df['fresh_rate'].mean():.3f}"
        )

    return CriticProfiles(df=df, training_slug_count=n_movies)


# -- Layer 2: KDE lambda model ------------------------------------------------


def _fit_population_prior(all_timing_data: np.ndarray) -> gaussian_kde:
    """Fit a single KDE to ALL reviews across ALL critics. The fallback shape."""
    return gaussian_kde(all_timing_data)


def _fit_critic_kde(
    timing_data: np.ndarray,
    population_prior: gaussian_kde,
    shrinkage_k: float,
    bandwidth_floor: float,
) -> dict:
    """Fit one critic's KDE with shrinkage toward population prior.

    Returns dict: {'empirical': gaussian_kde | None, 'n': int, 'k': float}.
    - 0-1 reviews or zero variance: empirical=None (fallback to population prior)
    - 2+ reviews with variance: empirical KDE with bandwidth floor enforced
    """
    n = len(timing_data)
    result = {"empirical": None, "n": n, "k": shrinkage_k}

    if n < 2:
        return result

    if timing_data.std() == 0:
        return result

    try:
        kde = gaussian_kde(timing_data)
        # Enforce bandwidth floor
        effective_bw = kde.factor * timing_data.std()
        if effective_bw < bandwidth_floor:
            kde.set_bandwidth(bandwidth_floor / timing_data.std())
        result["empirical"] = kde
    except np.linalg.LinAlgError:
        pass  # Degenerate covariance -- fall back to population prior

    return result


def build_kde_lambda_model(
    profiles: CriticProfiles,
    shrinkage_k: float = 3.0,
    bandwidth_floor: float = 0.5,
    verbose: bool = True,
) -> KDELambdaModel:
    """Fit KDEs to timing data from critic profiles. Returns a model that can estimate lambda.

    Note on `bandwidth_floor`: the default 0.5d is calibrated for a cohort that is ~98%
    day-level timestamps (review estimated_timestamps round to midnight UTC). This bound
    reflects within-day measurement uncertainty. As cohort granularity improves (more
    h/m-confidence reviews from live-tracking), the floor should be lowered. There is
    currently no explicit upper cap — Scott's rule from scipy can produce effective
    bandwidths of 2-3 days for sparse, spread-out per-critic data, which over-smooths
    across days. A bandwidth ceiling is tracked in BACKLOG.md \u00a71.4.
    """
    # Pool all timing data for population prior
    all_timing = np.concatenate(profiles.df["timing_data"].values)
    population_prior = _fit_population_prior(all_timing)

    # Fit per-critic KDEs
    critic_kdes = {}
    n_empirical = 0
    n_fallback_sparse = 0
    n_fallback_degenerate = 0

    for _, row in profiles.df.iterrows():
        timing = np.array(row["timing_data"])
        entry = _fit_critic_kde(timing, population_prior, shrinkage_k, bandwidth_floor)
        critic_kdes[row["reviewer_name"]] = entry

        if entry["empirical"] is not None:
            n_empirical += 1
        elif len(timing) < 2:
            n_fallback_sparse += 1
        else:
            n_fallback_degenerate += 1

    if verbose:
        t_grid = np.linspace(0, 30, 300)
        peak_t = t_grid[np.argmax(population_prior(t_grid))]
        print(
            f"[kde] {n_empirical} empirical KDEs, "
            f"{n_fallback_sparse} sparse fallback, "
            f"{n_fallback_degenerate} degenerate fallback"
        )
        print(f"[kde] population prior peak ~{peak_t:.1f}d before close")

    return KDELambdaModel(
        profiles=profiles,
        population_prior=population_prior,
        critic_kdes=critic_kdes,
        shrinkage_k=shrinkage_k,
        bandwidth_floor=bandwidth_floor,
    )


# -- Lambda estimation ---------------------------------------------------------


def _blended_integral(
    critic_entry: dict,
    population_prior: gaussian_kde,
    a: float,
    b: float,
    pop_integral: float | None = None,
) -> float:
    """Compute blended KDE integral from a to b.

    For critics with empirical KDEs: (n/(n+k)) * empirical + (k/(n+k)) * population.
    For fallback critics: population prior only.

    If pop_integral is provided, skip recomputing it (optimization for batch calls).
    """
    if pop_integral is None:
        pop_integral = population_prior.integrate_box_1d(a, b)

    empirical = critic_entry["empirical"]
    if empirical is None:
        return pop_integral

    n = critic_entry["n"]
    k = critic_entry["k"]
    w_emp = n / (n + k)
    w_pop = k / (n + k)
    return w_emp * empirical.integrate_box_1d(a, b) + w_pop * pop_integral


def _compute_scaling(
    model: KDELambdaModel,
    days_before_close: float,
    observed_count: int,
    first_review_dbc: float,
) -> float:
    """Compute observed/expected scaling factor.

    expected_so_far = sum of w_i * integral(kde_i, days_before_close, first_review_dbc)
    for ALL critics (not just unreviewed -- we compare to what the model predicted).

    Guard rails: if expected < 40, return 1.0; clamp to [0.5, 2.0].
    Threshold raised from 5->40 and clamp tightened from [0.3,3.0]->[0.5,2.0]
    after validation showed scaling overcorrects at T-7d where expected_so_far
    is based on KDE tail mass and the ratio is unreliable.
    """
    pop_integral = model.population_prior.integrate_box_1d(
        days_before_close, first_review_dbc
    )

    expected_so_far = 0.0
    for _, row in model.profiles.df.iterrows():
        w = row["base_rate"]
        entry = model.critic_kdes.get(row["reviewer_name"])
        if entry is None:
            continue
        integral = _blended_integral(
            entry, model.population_prior, days_before_close, first_review_dbc,
            pop_integral=pop_integral,
        )
        expected_so_far += w * integral

    if expected_so_far < 40.0:
        return 1.0

    scaling = observed_count / expected_so_far
    return max(0.5, min(2.0, scaling))


def estimate_lambda(
    model: KDELambdaModel,
    days_before_close: float,
    hours_to_close: float,
    observed_critics: set[str],
    observed_count: int | None = None,
    first_review_dbc: float | None = None,
) -> float:
    """Estimate lambda_rate (reviews/hour) at the given time.

    1. Sum w_i * blended_integral(kde_i, 0, days_before_close) for unreviewed critics
       -> expected remaining reviews.
    2. If observed_count and first_review_dbc provided, scale by observed/expected ratio.
    3. Divide by hours_to_close -> reviews/hour for compute_edge().

    Note: first_review_dbc was added to the plan's signature to support _compute_scaling().
    The plan's _compute_scaling() requires it but the original estimate_lambda() signature
    omitted it.
    """
    if hours_to_close <= 0 or days_before_close <= 0:
        return 0.0

    # Precompute population prior integral for remaining window [0, dbc]
    pop_integral = model.population_prior.integrate_box_1d(0, days_before_close)

    expected_remaining = 0.0
    for _, row in model.profiles.df.iterrows():
        name = row["reviewer_name"]
        if name in observed_critics:
            continue

        w = row["base_rate"]
        entry = model.critic_kdes.get(name)
        if entry is None:
            continue

        integral = _blended_integral(
            entry, model.population_prior, 0, days_before_close,
            pop_integral=pop_integral,
        )
        expected_remaining += w * integral

    if expected_remaining <= 0:
        return 0.0

    # Scale by observed/expected ratio if live data provided
    if observed_count is not None and first_review_dbc is not None:
        scaling = _compute_scaling(
            model, days_before_close, observed_count, first_review_dbc
        )
        expected_remaining *= scaling

    return expected_remaining / hours_to_close


# -- Layer 3: p_fresh estimation -----------------------------------------------


def estimate_p_fresh(
    profiles: CriticProfiles,
    observed_critics: set[str],
    fresh_count: int,
    total_count: int,
    n_prior: float = 20.0,
) -> float:
    """Estimate p_fresh by blending critic-weighted prior with observed rate.

    No KDEs involved. Uses critic profiles only for base_rate and fresh_rate.
    Weighted by base_rate so critics more likely to review get more influence.
    """
    df = profiles.df
    remaining = df[~df["reviewer_name"].isin(observed_critics)]

    weight_sum = remaining["base_rate"].sum()
    if weight_sum > 0:
        prior_p_fresh = (
            (remaining["base_rate"] * remaining["fresh_rate"]).sum() / weight_sum
        )
    else:
        prior_p_fresh = 0.65  # reasonable default if no remaining critics known

    if total_count == 0:
        return prior_p_fresh

    observed_p_fresh = fresh_count / total_count
    blend_weight = total_count / (total_count + n_prior)
    return blend_weight * observed_p_fresh + (1 - blend_weight) * prior_p_fresh


# -- Convenience: default training slugs --------------------------------------


def default_training_slugs(
    movies_df: pd.DataFrame,
    exclude_slug: str | None = None,
    n: int = 20,
    before_date: pd.Timestamp | None = None,
) -> list[str]:
    """Select the most recent n resolved movies by Bet Close Date, excluding the target.

    Args:
        before_date: Only include movies with Bet Close Date before this timestamp.
                     Defaults to now (resolved only). Pass the test movie's close date
                     for backtesting to avoid lookahead bias.
    """
    cutoff = before_date if before_date is not None else pd.Timestamp.now(tz="UTC")
    candidates = movies_df.dropna(subset=["Bet Close Date"])
    candidates = candidates[candidates["Bet Close Date"] < cutoff]
    if exclude_slug is not None:
        candidates = candidates[candidates["Slug"] != exclude_slug]
    return candidates.nlargest(n, "Bet Close Date")["Slug"].tolist()
