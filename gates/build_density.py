"""Driver: Gate-2 dense-cohort guard inputs (D1) — per-(movie, snap) oracle-input
density from the reviews DB, cached to ``gates/_cache/`` (gitignored).

Network I/O (DB) lives here; the density notebook reads only the cached CSVs. The
oracle-clean cell definition + dense-n floor are PRE-REGISTERED in
``plans/plan_gate_1_2_calibration.md`` ("Pre-registered dense-cohort floor") — written
before this driver was first run.

Outputs:
  gates/_cache/density.csv       — one row per (movie, snap in {1..5}d): remaining-window
                                   counts (total/fresh/mh/d), boundary d-count, scrape-lag
                                   quantiles, movie-level coverage + first_scrape, as_of_id.
  gates/_cache/density_meta.csv  — one row: as_of_id + cohort-wide timestamp_confidence
                                   counts (the 's'-row Phase-0 check).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd

from gates import db_facts as dbf

CACHE = os.path.join(os.path.dirname(__file__), "_cache")
SNAP_DAYS = [1, 2, 3, 4, 5]


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    mk = pd.read_csv(os.path.join(CACHE, "cohort_markets.csv"))
    movies = (mk.groupby("slug")["close_time"].first()).to_dict()
    print(f"cohort: {len(movies)} movies")

    conn = dbf.connect()
    try:
        n = dbf.as_of_id(conn)
        conf = dbf.confidence_counts(conn, n)
        rows = []
        for slug, close_iso in sorted(movies.items()):
            close = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
            cov = dbf.movie_coverage(conn, slug, close, n)
            fs = dbf.first_scrape(conn, slug, n)
            for sd in SNAP_DAYS:
                snap = close - timedelta(days=sd)
                d = dbf.snap_density(conn, slug, close, snap, n)
                rows.append({
                    "slug": slug, "close_time": close_iso, "snap_days": sd,
                    "snap_ts": snap.isoformat(), **d,
                    "first_scrape": fs.isoformat() if fs is not None else None,
                    "live_tracked_through_snap": (fs is not None and fs <= snap),
                    "movie_total": cov["total"], "movie_fresh": cov["fresh"],
                    "movie_max_est_ts": (cov["max_ts"].isoformat()
                                         if cov["max_ts"] is not None else None),
                    "n_last_day": cov["n_last_day"],
                    "n_last_day_d": cov["n_last_day_d"],
                    "n_after_close": cov["n_after_close"],
                    "as_of_id": n,
                })
            print(f"  {slug}: total {cov['total']}, first_scrape {fs}, "
                  f"last_day_d {cov['n_last_day_d']}, after_close {cov['n_after_close']}")
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CACHE, "density.csv"), index=False)
    meta = pd.DataFrame([{"as_of_id": n,
                          **{f"conf_{k}": v for k, v in sorted(conf.items())}}])
    meta.to_csv(os.path.join(CACHE, "density_meta.csv"), index=False)
    print(f"\nas_of_id={n} | confidence counts: {conf}")
    print(f"cached {len(df)} (movie, snap) rows -> _cache/density.csv")
    return df, meta


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build()
