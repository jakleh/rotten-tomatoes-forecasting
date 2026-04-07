"""
Poisson-binomial betting function for Kalshi RT Tomatometer markets.

Computes the expected edge (in cents) for an "Above X" bet given current
review counts, estimated review arrival rate, and freshness probability.

Usage:
    uv run python edge.py <movie_slug> <threshold> <market_price> <hours_to_close> [--lambda RATE] [--p-fresh PROB]

Example:
    uv run python edge.py the_drama 75 42 24
    uv run python edge.py the_drama 75 42 24 --lambda 1.5 --p-fresh 0.72
"""

import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone

from scipy.stats import binom, poisson
from sqlalchemy import create_engine, text


# ── Core edge calculation ──────────────────────────��─────────────────────��──


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

    This is a pure math function — it takes lambda_rate and p_fresh as given
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
        Dict with keys: edge_cents, p_yes, p_no, expected_reviews, k_max.
        edge_cents = P(Yes) * 100 - market_price.
        Positive edge_cents => buy Yes is +EV, negative => buy No is +EV.
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
) -> dict:
    """Fetch current review state for a movie from the Neon PostgreSQL database.

    Returns raw counts only — no parameter estimation. The caller (or a
    separate estimator module) decides how to derive lambda_rate and p_fresh
    from these counts.

    Args:
        movie_slug: Movie identifier (e.g. "the_drama").
        engine: SQLAlchemy engine. If None, creates one from DATABASE_URL env var.

    Returns:
        Dict with keys:
            movie_slug: str
            fresh_count: int — total positive reviews
            total_count: int — total reviews
            top_fresh: int — positive reviews from top critics
            top_total: int — total reviews from top critics
            nontop_fresh: int — positive reviews from non-top critics
            nontop_total: int — total reviews from non-top critics
            recent_timestamps: list[datetime] — estimated_timestamps from last 24h (UTC)
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
                    COUNT(*) FILTER (WHERE tomatometer_sentiment = 'positive') AS fresh_count,
                    COUNT(*) FILTER (WHERE top_critic = true) AS top_total,
                    COUNT(*) FILTER (WHERE top_critic = true AND tomatometer_sentiment = 'positive') AS top_fresh
                FROM reviews
                WHERE movie_slug = :slug
            """),
            {"slug": movie_slug},
        ).fetchone()

        total_count = counts.total_count
        fresh_count = counts.fresh_count
        top_total = counts.top_total
        top_fresh = counts.top_fresh

        if total_count == 0:
            raise ValueError(f"No reviews found for movie_slug='{movie_slug}'")

        recent_rows = conn.execute(
            text("""
                SELECT estimated_timestamp
                FROM reviews
                WHERE movie_slug = :slug
                  AND estimated_timestamp >= :cutoff
                ORDER BY estimated_timestamp
            """),
            {"slug": movie_slug, "cutoff": now - timedelta(hours=24)},
        ).fetchall()

        recent_timestamps = [row.estimated_timestamp for row in recent_rows]

    return {
        "movie_slug": movie_slug,
        "fresh_count": fresh_count,
        "total_count": total_count,
        "top_fresh": top_fresh,
        "top_total": top_total,
        "nontop_fresh": fresh_count - top_fresh,
        "nontop_total": total_count - top_total,
        "recent_timestamps": recent_timestamps,
    }


# ── Naive parameter estimates (defaults) ────────────────────────────────���───


def naive_lambda(recent_timestamps: list, hours: float = 6.0) -> float:
    """Estimate lambda as reviews in the last `hours` hours / hours.

    This is the v1 default. It's a placeholder — see BACKLOG.md §3 for
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

    This is the v1 default. It's a placeholder — see BACKLOG.md §3 for
    refinement directions (top-critic correction, cross-movie shrinkage, etc).
    """
    if total_count == 0:
        return 0.5
    return fresh_count / total_count


# ── CLI ───────────────────────────────────────────��──────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compute edge for a Kalshi RT Tomatometer bet.",
        usage="uv run python edge.py <movie_slug> <threshold> <market_price> <hours_to_close> [--lambda RATE] [--p-fresh PROB]",
    )
    parser.add_argument("movie_slug", help="Movie identifier (e.g. the_drama)")
    parser.add_argument("threshold", type=int, help="Kalshi threshold (e.g. 75 for 'Above 75')")
    parser.add_argument("market_price", type=float, help="Current market price in cents (0-100)")
    parser.add_argument("hours_to_close", type=float, help="Hours remaining until bet close")
    parser.add_argument("--lambda", dest="lambda_rate", type=float, default=None,
                        help="Override lambda_rate (reviews/hr). If omitted, estimated from DB.")
    parser.add_argument("--p-fresh", dest="p_fresh", type=float, default=None,
                        help="Override p_fresh (0-1). If omitted, estimated from DB.")

    args = parser.parse_args()

    state = get_movie_state(args.movie_slug)

    lambda_rate = args.lambda_rate if args.lambda_rate is not None else naive_lambda(state["recent_timestamps"])
    p_fresh = args.p_fresh if args.p_fresh is not None else naive_p_fresh(state["fresh_count"], state["total_count"])

    result = compute_edge(
        threshold=args.threshold,
        market_price=args.market_price,
        fresh_count=state["fresh_count"],
        total_count=state["total_count"],
        hours_to_close=args.hours_to_close,
        lambda_rate=lambda_rate,
        p_fresh=p_fresh,
    )

    # Display
    current_score_pct = state["fresh_count"] / state["total_count"] * 100
    displayed_score = round(current_score_pct)

    lambda_source = "override" if args.lambda_rate is not None else "naive (last 6h)"
    p_fresh_source = "override" if args.p_fresh is not None else "naive (fresh/total)"

    print(f"Movie           : {args.movie_slug}")
    print(f"Score           : {displayed_score}% ({state['fresh_count']}/{state['total_count']})")
    print(f"  Top critics   : {state['top_fresh']}/{state['top_total']} fresh")
    print(f"  Non-top       : {state['nontop_fresh']}/{state['nontop_total']} fresh")
    print(f"Threshold       : Above {args.threshold} (resolves Yes if displayed >= {args.threshold + 1})")
    print(f"Market price    : {args.market_price:.0f}c")
    print(f"Hours to close  : {args.hours_to_close:.1f}")
    print(f"Lambda          : {lambda_rate:.2f} reviews/hr ({lambda_source})")
    print(f"p_fresh         : {p_fresh:.4f} ({p_fresh_source})")
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
