"""
Poisson-binomial betting function for Kalshi RT Tomatometer markets.

Computes the expected edge (in cents) for an "Above X" bet given current
review counts, estimated review arrival rate, and freshness probability.

Usage:
    uv run python edge.py <movie_slug> <threshold> <market_price> <hours_to_close>

Example:
    uv run python edge.py the_drama 75 42 24
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

from scipy.stats import binom, poisson
from sqlalchemy import create_engine, text


# ── Core edge calculation ───────────────────────────────────────────────────


def compute_edge(
    threshold: int,
    market_price: float,
    fresh_count: int,
    total_count: int,
    hours_to_close: float,
    lambda_rate: float,
    p_fresh: float,
) -> dict:
    """Compute the expected edge for an "Above X" Kalshi RT bet.

    Resolution rule: "Above X" resolves Yes when the displayed Tomatometer
    (round(fresh/total * 100), standard rounding) >= X + 1.  Equivalently,
    when fresh/total >= (X + 0.5) / 100.

    For each possible number of additional reviews k (Poisson-distributed),
    we compute the minimum fresh reviews among those k needed for Yes
    resolution, then use the binomial CDF to get P(Yes|k).

    Args:
        threshold: Kalshi threshold integer (e.g. 75 for "Above 75").
        market_price: Current market price in cents (0-100).
        fresh_count: Current number of positive reviews.
        total_count: Current total number of reviews.
        hours_to_close: Hours remaining until bet close.
        lambda_rate: Expected reviews per hour (Poisson rate parameter).
        p_fresh: Probability each future review is positive (0-1).

    Returns:
        Dict with keys: edge_cents, p_yes, p_no, expected_reviews, k_max.
    """
    mu = lambda_rate * hours_to_close

    # Effective fractional threshold: score must reach this to resolve Yes
    effective_threshold = (threshold + 0.5) / 100

    if mu == 0:
        # No reviews expected — outcome determined by current score
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
            # No reviews at all — score undefined, resolves No
            continue

        # Minimum fresh reviews (current + new) needed for Yes resolution
        fresh_needed = effective_threshold * final_total
        # j = additional fresh reviews needed beyond current fresh_count
        j_min = math.ceil(fresh_needed - fresh_count)

        if j_min <= 0:
            # Current fresh count already enough — all outcomes resolve Yes
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


# ── DB query helper ─────────────────────────────────────────────────────────


def get_movie_state(
    movie_slug: str,
    engine=None,
    lambda_hours: float = 6.0,
) -> dict:
    """Fetch current review state for a movie from the Neon PostgreSQL database.

    Args:
        movie_slug: Movie identifier (e.g. "the_drama").
        engine: SQLAlchemy engine. If None, creates one from DATABASE_URL env var.
        lambda_hours: Window (in hours) for estimating review arrival rate.

    Returns:
        Dict with keys: fresh_count, total_count, lambda_rate, p_fresh, movie_slug.
    """
    if engine is None:
        database_url = os.environ["DATABASE_URL"]
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        engine = create_engine(database_url)

    now = datetime.now(timezone.utc)

    with engine.connect() as conn:
        counts = conn.execute(
            text("""
                SELECT
                    COUNT(*) AS total_count,
                    COUNT(*) FILTER (WHERE tomatometer_sentiment = 'positive') AS fresh_count
                FROM reviews
                WHERE movie_slug = :slug
            """),
            {"slug": movie_slug},
        ).fetchone()

        total_count = counts.total_count
        fresh_count = counts.fresh_count

        if total_count == 0:
            raise ValueError(f"No reviews found for movie_slug='{movie_slug}'")

        recent = conn.execute(
            text("""
                SELECT COUNT(*) AS cnt
                FROM reviews
                WHERE movie_slug = :slug
                  AND estimated_timestamp >= :cutoff
            """),
            {"slug": movie_slug, "cutoff": now - timedelta(hours=lambda_hours)},
        ).fetchone()

        lambda_rate = recent.cnt / lambda_hours if lambda_hours > 0 else 0.0

    p_fresh = fresh_count / total_count

    return {
        "movie_slug": movie_slug,
        "fresh_count": fresh_count,
        "total_count": total_count,
        "lambda_rate": lambda_rate,
        "p_fresh": p_fresh,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) != 5:
        print("Usage: uv run python edge.py <movie_slug> <threshold> <market_price> <hours_to_close>")
        print("Example: uv run python edge.py the_drama 75 42 24")
        sys.exit(1)

    movie_slug = sys.argv[1]
    threshold = int(sys.argv[2])
    market_price = float(sys.argv[3])
    hours_to_close = float(sys.argv[4])

    state = get_movie_state(movie_slug)

    result = compute_edge(
        threshold=threshold,
        market_price=market_price,
        fresh_count=state["fresh_count"],
        total_count=state["total_count"],
        hours_to_close=hours_to_close,
        lambda_rate=state["lambda_rate"],
        p_fresh=state["p_fresh"],
    )

    # Display
    current_score_pct = state["fresh_count"] / state["total_count"] * 100
    displayed_score = round(current_score_pct)

    print(f"Movie           : {movie_slug}")
    print(f"Score           : {displayed_score}% ({state['fresh_count']}/{state['total_count']})")
    print(f"Threshold       : Above {threshold} (resolves Yes if displayed >= {threshold + 1})")
    print(f"Market price    : {market_price:.0f}c")
    print(f"Hours to close  : {hours_to_close:.1f}")
    print(f"Lambda          : {state['lambda_rate']:.2f} reviews/hr (last 6h)")
    print(f"p_fresh         : {state['p_fresh']:.4f}")
    print(f"Expected reviews: {result['expected_reviews']:.1f}")
    print(f"k_max           : {result['k_max']}")
    print("---")
    print(f"P(Yes)          : {result['p_yes']:.4f}")
    print(f"P(No)           : {result['p_no']:.4f}")

    edge = result["edge_cents"]
    direction = "Yes" if edge >= 0 else "No"
    print(f"Edge ({direction:3s})       : {'+' if edge >= 0 else ''}{edge:.1f}c")


if __name__ == "__main__":
    main()
