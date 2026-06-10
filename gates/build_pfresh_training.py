"""Driver: the p_fresh-regression training pull + row scaffold (the one DB step).

Spec: ``plans/plan_p_fresh_regression.md`` v2 "Data build". Pulls the full
close-universe review history at ONE fresh pin, scaffolds (movie, snap) training rows
under the pre-registered eligibility rules, and caches everything the battery + bench
notebooks need (they are cache-only). The BENCH world stays on its locked 648979
caches — the two pins never mix.

Outputs (gates/_cache/, gitignored):
  pfresh_training_reviews.csv  pinned pull (lowercased sentiment; est_raw + noon-
                               shifted estimator view — gate3b cache conventions)
  pfresh_training_rows.csv     one row per ELIGIBLE (movie, snap): counts + close
  pfresh_meta.csv              pin + universe/eligibility bookkeeping
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "_cache")
STORE = os.path.join(os.path.dirname(__file__), "recorded")
SNAP_DAYS = [1, 2, 3, 4]
DATA_NOT_READY = {"animal_farm_2025", "power_ballad", "backrooms", "in_the_grey"}
MIN_TOTAL_AT_CLOSE = 20


def build() -> pd.DataFrame:
    from gates import db_facts as dbf
    from gates.build_gate3b import midnight_snap
    from rotten_tomatoes_forecasting.features import apply_noon_shift
    from rotten_tomatoes_forecasting.pool import _most_recent_resolved_slugs

    mi = pd.read_csv("movies_index.csv")
    mk = pd.read_csv(os.path.join(STORE, "markets.csv"))
    cdm = {**dict(zip(mi["Slug"], pd.to_datetime(mi["Bet Close Date"], utc=True))),
           **{s: pd.to_datetime(t, utc=True)
              for s, t in mk.groupby("slug")["close_time"].first().items()}}
    print(f"close-universe: {len(cdm)} slugs")

    # index-score consistency inputs (rule (d)): 'x/y' rows, measured n=2
    idx_score = {}
    for _, r in mi.iterrows():
        v = str(r.get("Tomatometer Score Bet Close", "")).strip()
        if "/" in v:
            try:
                x, y = v.split("/")
                idx_score[r["Slug"]] = round(100 * float(x) / float(y))
            except ValueError:
                pass
    print(f"index-score checkable movies: {len(idx_score)} (expected 2 — near-vacuous)")

    conn = dbf.connect()
    try:
        pin = dbf.as_of_id(conn)
        raw = dbf.fetch_reviews_full(conn, sorted(cdm), pin)
    finally:
        conn.close()
    rv = pd.DataFrame(raw, columns=[
        "movie_slug", "reviewer_name", "publication_name", "top_critic",
        "tomatometer_sentiment", "subjective_score", "estimated_timestamp",
        "scrape_time", "timestamp_confidence"])
    rv["estimated_timestamp"] = pd.to_datetime(rv["estimated_timestamp"], utc=True)
    rv["scrape_time"] = pd.to_datetime(rv["scrape_time"], utc=True)
    rv["tomatometer_sentiment"] = rv["tomatometer_sentiment"].str.lower()
    rv["est_raw"] = rv["estimated_timestamp"]
    rv = apply_noon_shift(rv)                 # ONCE at ingest (estimator view)
    rv["as_of_id"] = pin
    db_slugs = set(rv["movie_slug"].unique())
    print(f"pin {pin}: {len(rv)} rows / {len(db_slugs)} DB slugs of {len(cdm)}")

    rows, excl = [], {"no_full_pool": 0, "pool_member_rowless": 0,
                      "data_not_ready": 0, "thin_at_close": 0, "index_mismatch": 0,
                      "n_rem_zero": 0, "n_obs_zero": 0}
    eligible = 0
    for slug in sorted(db_slugs):
        close = cdm[slug]
        mr = rv[rv["movie_slug"] == slug]
        if slug in DATA_NOT_READY:
            excl["data_not_ready"] += 1
            continue
        pool = _most_recent_resolved_slugs(cdm, before=close, n=20, exclude_slug=slug)
        if len(pool) < 20:
            excl["no_full_pool"] += 1
            continue
        if any(p not in db_slugs for p in pool):
            excl["pool_member_rowless"] += 1
            continue
        at_close = mr[mr["estimated_timestamp"] <= close]
        if len(at_close) < MIN_TOTAL_AT_CLOSE:
            excl["thin_at_close"] += 1
            continue
        if slug in idx_score:
            self_label = round(
                100 * at_close["tomatometer_sentiment"].eq("positive").mean())
            if abs(self_label - idx_score[slug]) > 1:
                excl["index_mismatch"] += 1
                continue
        eligible += 1
        for n in SNAP_DAYS:
            snap_ts = midnight_snap(close, n)
            obs = mr[mr["estimated_timestamp"] < snap_ts]
            rem = mr[(mr["estimated_timestamp"] >= snap_ts)
                     & (mr["estimated_timestamp"] <= close)]
            if len(rem) == 0:
                excl["n_rem_zero"] += 1
                continue
            if len(obs) == 0:                  # brainstorm 3.1(iii) reinstated [F2]
                excl["n_obs_zero"] += 1
                continue
            rows.append({
                "slug": slug, "snap_days": n, "close": close.isoformat(),
                "snap_ts": snap_ts.isoformat(),
                "n_obs": len(obs),
                "fresh_obs": int(obs["tomatometer_sentiment"].eq("positive").sum()),
                "n_rem": len(rem),
                "fresh_rem": int(rem["tomatometer_sentiment"].eq("positive").sum()),
            })
    out = pd.DataFrame(rows)
    print(f"eligible movies: {eligible} | training rows: {len(out)}")
    print(f"exclusions: {excl}")
    print(out.groupby("snap_days").agg(rows=("slug", "size"),
                                       med_n_rem=("n_rem", "median")).to_string())

    rv.to_csv(os.path.join(CACHE, "pfresh_training_reviews.csv"), index=False)
    out.to_csv(os.path.join(CACHE, "pfresh_training_rows.csv"), index=False)
    meta = {"as_of_id": pin, "n_universe": len(cdm), "n_db_slugs": len(db_slugs),
            "n_eligible": eligible, "n_rows": len(out),
            **{f"excl_{k}": v for k, v in excl.items()},
            "n_index_checkable": len(idx_score),
            "built_at": pd.Timestamp.now(tz="UTC").isoformat()}
    pd.DataFrame([meta]).to_csv(os.path.join(CACHE, "pfresh_meta.csv"), index=False)
    print(f"cached: pfresh_training_reviews ({len(rv)}), pfresh_training_rows "
          f"({len(out)}), pfresh_meta -> gates/_cache/")
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build()
