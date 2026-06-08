"""Read-only ``db_facts`` for the reviews DB (Neon read replica).

Every DB claim in the Gate work should come from a function here, pinned by the serial
PK (``id <= as_of_id``) against the **insert-only** ``reviews`` table, and cited with
its ``as_of_id`` — the DB analog of ``/audit-numbers`` ("trace to a deterministic
helper"). See ``feedback_db_facts_verification`` + ``plans/plan_gate_1_2_calibration.md``.

Reproducibility note: pin by ``id <= N`` (immune to clock ties). The ``id`` is GAPPY
(``ON CONFLICT DO NOTHING`` burns the sequence on re-scraped duplicates) — valid for the
pin (monotonic + no deletes), but never infer row counts from id ranges.

Read-only: connects to the read replica; issues SELECT only. Never logs the connection
string / password.
"""
from __future__ import annotations

import os

import psycopg2


def _database_url() -> str:
    """Resolve DATABASE_URL from the environment, falling back to ``./.env``.

    Never returned to callers / printed elsewhere; used only to open the connection.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        with open(".env") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("DATABASE_URL") and "=" in s:
                    v = s.split("=", 1)[1].strip()
                    if v[:1] in ("\"", "'") and v[-1:] == v[:1]:
                        v = v[1:-1]
                    return v
    except FileNotFoundError:
        pass
    raise RuntimeError("DATABASE_URL not set and not found in ./.env")


def connect():
    """Open a read-only, autocommit connection to the reviews replica.

    Autocommit (each statement its own txn) is the clean mode for the pooled endpoint.
    The endpoint is a read replica (``pg_is_in_recovery() = true``) so writes are
    impossible at the infra level regardless.
    """
    conn = psycopg2.connect(_database_url(), connect_timeout=15)
    conn.autocommit = True
    return conn


def as_of_id(conn) -> int:
    """The reproducibility pin: current ``MAX(reviews.id)``. Cite alongside every claim."""
    with conn.cursor() as cur:
        cur.execute("SELECT max(id) FROM reviews;")
        return int(cur.fetchone()[0])


def confidence_counts(conn, as_of_id: int) -> dict[str, int]:
    """``timestamp_confidence`` -> count over reviews with ``id <= as_of_id``.

    Surfaces any ``'s'`` (sub-minute) rows — one of the open Phase-0 questions.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT timestamp_confidence, count(*) FROM reviews "
            "WHERE id <= %s GROUP BY 1 ORDER BY 1;",
            (as_of_id,),
        )
        return {row[0]: int(row[1]) for row in cur.fetchall()}


def movie_review_counts(conn, as_of_id: int) -> dict[str, int]:
    """``movie_slug`` -> review count (id <= as_of_id), descending."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT movie_slug, count(*) FROM reviews "
            "WHERE id <= %s GROUP BY 1 ORDER BY 2 DESC;",
            (as_of_id,),
        )
        return {row[0]: int(row[1]) for row in cur.fetchall()}


def movie_coverage(conn, slug: str, close_ts, as_of_id: int) -> dict:
    """Per-movie coverage relative to ``close_ts`` (tz-aware datetime), id <= as_of_id.

    Returns total/fresh counts, min/max ``estimated_timestamp`` (the scraper-timing
    check: does ``max_ts`` reach ``close_ts``?), and day-level counts in the final-day
    and post-close windows (the M2 close-day ``d``-review check, and a look-ahead sanity).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              count(*)                                                    AS total,
              count(*) FILTER (WHERE tomatometer_sentiment = 'positive')  AS fresh,
              min(estimated_timestamp)                                    AS min_ts,
              max(estimated_timestamp)                                    AS max_ts,
              count(*) FILTER (WHERE estimated_timestamp >  %(c)s - interval '1 day'
                                 AND estimated_timestamp <= %(c)s)        AS n_last_day,
              count(*) FILTER (WHERE estimated_timestamp >  %(c)s - interval '1 day'
                                 AND estimated_timestamp <= %(c)s
                                 AND timestamp_confidence = 'd')          AS n_last_day_d,
              count(*) FILTER (WHERE estimated_timestamp > %(c)s)         AS n_after_close
            FROM reviews
            WHERE movie_slug = %(slug)s AND id <= %(n)s;
            """,
            {"c": close_ts, "slug": slug, "n": as_of_id},
        )
        cols = ["total", "fresh", "min_ts", "max_ts",
                "n_last_day", "n_last_day_d", "n_after_close"]
        return dict(zip(cols, cur.fetchone()))


def observed_state(conn, slug, cutoff_ts, as_of_id):
    """(fresh, total) reviews for `slug` with estimated_timestamp <= cutoff_ts and
    id <= as_of_id — the publication-time observed state at a snap (cutoff_ts = close - snap).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FILTER (WHERE tomatometer_sentiment='positive'), count(*) "
            "FROM reviews WHERE movie_slug=%s AND id<=%s AND estimated_timestamp <= %s",
            (slug, as_of_id, cutoff_ts))
        fresh, total = cur.fetchone()
        return int(fresh), int(total)
