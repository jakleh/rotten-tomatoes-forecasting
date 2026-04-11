"""
CLI entry point for rotten-tomatoes-forecasting.

Usage:
    uv run python -m rotten_tomatoes_forecasting <movie_slug> <threshold> <market_price> <hours_to_close> [--lambda RATE] [--p-fresh PROB]
    uv run python -m rotten_tomatoes_forecasting <movie_slug> <threshold> <market_price> <hours_to_close> --kde [--shrinkage-k K] [--n-prior N]
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rotten_tomatoes_forecasting.edge import compute_edge, naive_lambda, naive_p_fresh
from rotten_tomatoes_forecasting._db import get_movie_state, get_observed_critics


def main():
    parser = argparse.ArgumentParser(
        description="Compute edge for a Kalshi RT Tomatometer bet.",
        usage=(
            "uv run python -m rotten_tomatoes_forecasting <movie_slug> <threshold> <market_price> <hours_to_close> "
            "[--lambda RATE] [--p-fresh PROB] [--kde]"
        ),
    )
    parser.add_argument("movie_slug", help="Movie identifier (e.g. the_drama)")
    parser.add_argument("threshold", type=int, help="Kalshi threshold (e.g. 75 for 'Above 75')")
    parser.add_argument("market_price", type=float, help="Current market price in cents (0-100)")
    parser.add_argument("hours_to_close", type=float, help="Hours remaining until bet close")
    parser.add_argument("--lambda", dest="lambda_rate", type=float, default=None,
                        help="Override lambda_rate (reviews/hr). Takes precedence over --kde.")
    parser.add_argument("--p-fresh", dest="p_fresh", type=float, default=None,
                        help="Override p_fresh (0-1). Takes precedence over --kde.")
    parser.add_argument("--kde", action="store_true",
                        help="Use per-critic KDE model for lambda/p_fresh estimation.")
    parser.add_argument("--shrinkage-k", type=float, default=3.0,
                        help="KDE shrinkage parameter (default 3.0). Only with --kde.")
    parser.add_argument("--n-prior", type=float, default=20.0,
                        help="p_fresh prior sample size (default 20.0). Only with --kde.")
    parser.add_argument("--training-slugs", nargs="+", default=None,
                        help="Override training movie slugs. Only with --kde.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing reviews.csv and movies_index.csv. "
                             "Defaults to repo root (parent of rotten_tomatoes_forecasting package).")

    args = parser.parse_args()

    if args.kde:
        import pandas as pd
        from rotten_tomatoes_forecasting.critic_model import (
            build_critic_profiles,
            build_kde_lambda_model,
            default_training_slugs,
            estimate_lambda,
            estimate_p_fresh,
        )

        if args.data_dir:
            root = Path(args.data_dir)
        else:
            root = Path(__file__).parent.parent

        # Load data
        reviews_df = pd.read_csv(root / "reviews.csv")
        reviews_df["estimated_timestamp"] = pd.to_datetime(
            reviews_df["estimated_timestamp"], format="ISO8601", utc=True
        )
        movies_df = pd.read_csv(root / "movies_index.csv")
        movies_df["Bet Close Date"] = pd.to_datetime(
            movies_df["Bet Close Date"], utc=True
        )

        # Training slugs
        if args.training_slugs:
            training_slugs = args.training_slugs
        else:
            training_slugs = default_training_slugs(movies_df, exclude_slug=args.movie_slug)

        # Build model
        close_date_map = movies_df.set_index("Slug")["Bet Close Date"].to_dict()
        profiles = build_critic_profiles(reviews_df, close_date_map, training_slugs)
        model = build_kde_lambda_model(profiles, shrinkage_k=args.shrinkage_k)

        # Get observed state from DB
        observed_critics, fresh_count, total_count, first_ts = get_observed_critics(
            args.movie_slug
        )

        if total_count == 0:
            print(f"No reviews found for '{args.movie_slug}' in DB.", file=sys.stderr)
            sys.exit(1)

        # Compute time parameters
        days_before_close = args.hours_to_close / 24
        first_review_dbc = None
        if first_ts is not None:
            close_time = datetime.now(timezone.utc) + timedelta(hours=args.hours_to_close)
            first_review_dbc = (close_time - first_ts).total_seconds() / 86400

        # Estimate parameters (manual overrides take precedence)
        kde_lambda = estimate_lambda(
            model, days_before_close, args.hours_to_close,
            observed_critics, observed_count=total_count, first_review_dbc=first_review_dbc,
        )
        kde_p_fresh = estimate_p_fresh(
            profiles, observed_critics, fresh_count, total_count, n_prior=args.n_prior,
        )

        lambda_rate = args.lambda_rate if args.lambda_rate is not None else kde_lambda
        p_fresh = args.p_fresh if args.p_fresh is not None else kde_p_fresh

        lambda_source = "override" if args.lambda_rate is not None else "KDE model"
        p_fresh_source = "override" if args.p_fresh is not None else "KDE model"

    else:
        state = get_movie_state(args.movie_slug)
        fresh_count = state["fresh_count"]
        total_count = state["total_count"]

        lambda_rate = args.lambda_rate if args.lambda_rate is not None else naive_lambda(state["recent_timestamps"])
        p_fresh = args.p_fresh if args.p_fresh is not None else naive_p_fresh(fresh_count, total_count)

        lambda_source = "override" if args.lambda_rate is not None else "naive (last 6h)"
        p_fresh_source = "override" if args.p_fresh is not None else "naive (fresh/total)"

    result = compute_edge(
        threshold=args.threshold,
        market_price=args.market_price,
        fresh_count=fresh_count,
        total_count=total_count,
        hours_to_close=args.hours_to_close,
        lambda_rate=lambda_rate,
        p_fresh=p_fresh,
    )

    # Display
    current_score_pct = fresh_count / total_count * 100
    displayed_score = round(current_score_pct)

    print(f"Movie           : {args.movie_slug}")
    print(f"Score           : {displayed_score}% ({fresh_count}/{total_count})")
    if not args.kde:
        print(f"  Top critics   : {state['top_fresh']}/{state['top_total']} fresh")
        print(f"  Non-top       : {state['nontop_fresh']}/{state['nontop_total']} fresh")
    else:
        print(f"  Critics seen  : {len(observed_critics)}")
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
