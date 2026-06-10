"""Driver: cache the cohort's review rows for the Gate-2 oracle (read-only, pinned).

Pulls per-review (estimated_timestamp, scrape_time, timestamp_confidence,
tomatometer_sentiment) for the 16 cohort movies at a FIXED ``as_of_id`` — the same pin
as the density run, so the oracle and the dense-cohort guard see the identical table
state (the table is insert-only, but late-scraped old reviews can still ADD rows with
pre-close ``estimated_timestamp`` under a later pin).

Output: gates/_cache/reviews_cohort.csv (~2.7k rows; gitignored).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from gates import db_facts as dbf

CACHE = os.path.join(os.path.dirname(__file__), "_cache")


def build(as_of: int | None = None) -> pd.DataFrame:
    mk = pd.read_csv(os.path.join(CACHE, "cohort_markets.csv"))
    slugs = sorted(mk["slug"].unique())
    conn = dbf.connect()
    try:
        n = as_of if as_of is not None else dbf.as_of_id(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT movie_slug, estimated_timestamp, scrape_time, "
                "       timestamp_confidence, tomatometer_sentiment "
                "FROM reviews WHERE movie_slug = ANY(%s) AND id <= %s "
                "ORDER BY movie_slug, estimated_timestamp;",
                (slugs, n))
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=["slug", "estimated_timestamp", "scrape_time",
                                     "timestamp_confidence", "tomatometer_sentiment"])
    df["as_of_id"] = n
    df.to_csv(os.path.join(CACHE, "reviews_cohort.csv"), index=False)
    print(f"as_of_id={n} | cached {len(df)} review rows for {df['slug'].nunique()} movies "
          f"-> _cache/reviews_cohort.csv")
    print(df.groupby("slug").size().to_string())
    return df


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build(as_of=int(sys.argv[1]) if len(sys.argv) > 1 else None)
