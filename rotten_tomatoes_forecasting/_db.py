"""
Database convenience functions for CLI use.

These functions query the Neon PostgreSQL database directly via DATABASE_URL.
The orchestrator owns its own DB connection and passes DataFrames to the
core API -- these functions are NOT part of the public API.
"""

import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, text


def _get_engine(engine=None):
    """Get or create a SQLAlchemy engine from DATABASE_URL."""
    if engine is not None:
        return engine
    database_url = os.environ["DATABASE_URL"]
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    return create_engine(database_url)


def get_movie_state(
    movie_slug: str,
    engine=None,
) -> dict:
    """Fetch current review state for a movie from the Neon PostgreSQL database.

    Returns raw counts only -- no parameter estimation. The caller (or a
    separate estimator module) decides how to derive lambda_rate and p_fresh
    from these counts.

    Args:
        movie_slug: Movie identifier (e.g. "the_drama").
        engine: SQLAlchemy engine. If None, creates one from DATABASE_URL env var.

    Returns:
        Dict with keys:
            movie_slug: str
            fresh_count: int -- total positive reviews
            total_count: int -- total reviews
            top_fresh: int -- positive reviews from top critics
            top_total: int -- total reviews from top critics
            nontop_fresh: int -- positive reviews from non-top critics
            nontop_total: int -- total reviews from non-top critics
            recent_timestamps: list[datetime] -- estimated_timestamps from last 24h (UTC)
    """
    engine = _get_engine(engine)
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


def get_observed_critics(
    movie_slug: str,
    engine=None,
) -> tuple[set[str], int, int, object]:
    """Query DB for the target movie's per-critic reviews.

    Returns (critic_names, fresh_count, total_count, first_review_timestamp).
    first_review_timestamp is the earliest estimated_timestamp (datetime or None).

    Extended from plan's 3-tuple to include first_review_timestamp for scaling.
    """
    engine = _get_engine(engine)

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT reviewer_name, tomatometer_sentiment, estimated_timestamp
                FROM reviews
                WHERE movie_slug = :slug
            """),
            {"slug": movie_slug},
        ).fetchall()

    if not rows:
        return set(), 0, 0, None

    critic_names = {row.reviewer_name for row in rows}
    total_count = len(rows)
    fresh_count = sum(1 for row in rows if row.tomatometer_sentiment == "positive")
    first_ts = min(row.estimated_timestamp for row in rows)

    return critic_names, fresh_count, total_count, first_ts
