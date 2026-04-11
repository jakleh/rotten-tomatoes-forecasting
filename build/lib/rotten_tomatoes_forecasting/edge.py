"""
Poisson-binomial betting function for Kalshi RT Tomatometer markets.

Computes the expected edge (in cents) for an "Above X" bet given current
review counts, estimated review arrival rate, and freshness probability.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import TypedDict

from scipy.stats import binom, poisson


class EdgeResult(TypedDict):
    """Return type for compute_edge()."""

    edge_cents: float
    p_yes: float
    p_no: float
    expected_reviews: float
    k_max: int


def compute_edge(
    threshold: int,
    market_price: float,
    fresh_count: int,
    total_count: int,
    hours_to_close: float,
    lambda_rate: float,
    p_fresh: float,
) -> EdgeResult:
    """Compute the expected edge for an "Above X" Kalshi RT bet.

    This is a pure math function -- it takes lambda_rate and p_fresh as given
    and computes from there. It makes no assumptions about how those
    parameters were estimated.

    Resolution rule: "Above X" resolves Yes when the displayed Tomatometer
    (round(fresh/total * 100), standard rounding) >= X + 1.  Equivalently,
    when fresh/total >= (X + 0.5) / 100.

    For each possible number of additional reviews k (Poisson-distributed
    with mu = lambda_rate * hours_to_close), we compute the minimum fresh
    reviews among those k needed for Yes resolution, then use the binomial
    CDF to get P(Yes|k).

    Args:
        threshold: Kalshi threshold integer (e.g. 75 for "Above 75").
        market_price: Current market price in cents (0-100).
        fresh_count: Current number of positive reviews.
        total_count: Current total number of reviews.
        hours_to_close: Hours remaining until bet close (>= 0).
        lambda_rate: Expected reviews per hour (Poisson rate parameter, >= 0).
        p_fresh: Probability each future review is positive (0-1).

    Returns:
        EdgeResult with keys: edge_cents, p_yes, p_no, expected_reviews, k_max.
        edge_cents = P(Yes) * 100 - market_price.
    """
    if not (0 <= market_price <= 100):
        raise ValueError(f"market_price must be 0-100, got {market_price}")
    if not (0 <= p_fresh <= 1):
        raise ValueError(f"p_fresh must be 0-1, got {p_fresh}")
    if lambda_rate < 0:
        raise ValueError(f"lambda_rate must be >= 0, got {lambda_rate}")
    if hours_to_close < 0:
        raise ValueError(f"hours_to_close must be >= 0, got {hours_to_close}")

    mu = lambda_rate * hours_to_close

    # Effective fractional threshold: score must reach this to resolve Yes
    effective_threshold = (threshold + 0.5) / 100

    if mu == 0:
        # No reviews expected -- outcome determined by current score
        k_max = 0
    else:
        k_max = int(poisson.ppf(1 - 1e-10, mu))

    p_yes = 0.0

    for k in range(0, k_max + 1):
        p_k = poisson.pmf(k, mu) if mu > 0 else (1.0 if k == 0 else 0.0)
        if p_k < 1e-15:
            continue

        final_total = total_count + k

        if final_total == 0:
            # No reviews at all -- score undefined, resolves No
            continue

        # Minimum fresh reviews (current + new) needed for Yes resolution
        fresh_needed = effective_threshold * final_total
        # j = additional fresh reviews needed beyond current fresh_count
        j_min = math.ceil(fresh_needed - fresh_count)

        if j_min <= 0:
            # Current fresh count already enough -- all outcomes resolve Yes
            p_yes_given_k = 1.0
        elif j_min > k:
            # Even all-fresh k reviews can't reach threshold
            p_yes_given_k = 0.0
        else:
            # P(J >= j_min) where J ~ Binomial(k, p_fresh)
            p_yes_given_k = binom.sf(j_min - 1, k, p_fresh)

        p_yes += p_k * p_yes_given_k

    p_no = 1.0 - p_yes
    edge_cents = p_yes * 100 - market_price

    return {
        "edge_cents": edge_cents,
        "p_yes": p_yes,
        "p_no": p_no,
        "expected_reviews": mu,
        "k_max": k_max,
    }


# -- Naive parameter estimates (defaults) -------------------------------------


def naive_lambda(recent_timestamps: list, hours: float = 6.0) -> float:
    """Estimate lambda as reviews in the last `hours` hours / hours.

    This is the v1 default. It's a placeholder -- see BACKLOG.md for
    refinement directions (cross-movie curves, KDE, etc).
    """
    if not recent_timestamps or hours <= 0:
        return 0.0
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    count = sum(1 for ts in recent_timestamps if ts >= cutoff)
    return count / hours


def naive_p_fresh(fresh_count: int, total_count: int) -> float:
    """Estimate p_fresh as the running average fresh/total.

    This is the v1 default. It's a placeholder -- see BACKLOG.md for
    refinement directions (top-critic correction, cross-movie shrinkage, etc).
    """
    if total_count == 0:
        return 0.5
    return fresh_count / total_count
