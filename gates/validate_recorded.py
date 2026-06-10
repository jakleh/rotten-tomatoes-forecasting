"""Driver: validate the committed ``gates/recorded/`` store against the gitignored
``gates/_cache/`` build — the §1.7 recorder's Phase-3 cross-check, kept as a rerunnable
driver (the BACKLOG §1.7 "zero mismatches / zero candle diffs" claims trace here).

Local files only (no network/DB; sandbox-safe):

    python -m gates.validate_recorded

Checks: run ledger vs ``cohort_markets.csv`` on the five DB-join columns over the
overlapping tickers; per-ticker candle row-count parity vs ``candles.csv`` (closed-market
history is immutable — counts should match exactly for any window both pulls covered);
the event-set diff (newly-settled since the cache build); and echoes runs.csv +
the latest open-events snapshot. Report, don't assert — diffs are findings, not failures
(e.g. a later as_of pin CAN legitimately shift join values if backfill added pre-close
rows; see plan_recorder.md "Live validation").
"""
from __future__ import annotations

import os

import pandas as pd

from gates.recorder import STORE, implied_score_interval

CACHE = os.path.join(os.path.dirname(__file__), "_cache")
JOIN_COLS = ["slug", "score_self", "total_at_close", "fresh_at_close", "lastday_daylevel"]


def validate(store_dir: str = STORE, cache_dir: str = CACHE) -> dict:
    led = pd.read_csv(os.path.join(store_dir, "markets.csv"))
    runs = pd.read_csv(os.path.join(store_dir, "runs.csv"))
    old = pd.read_csv(os.path.join(cache_dir, "cohort_markets.csv"))

    print("=== runs.csv ===")
    print(runs.to_string(index=False))
    print(f"\n=== ledger === {len(led)} markets / {led['event_ticker'].nunique()} events / "
          f"{int(led['db_joined'].sum())} db_joined / {led['slug'].nunique()} slugs")

    new_tk = set(led["ticker"]) - set(old["ticker"])
    gone_tk = set(old["ticker"]) - set(led["ticker"])
    new_events = sorted(led.loc[led["ticker"].isin(new_tk), "event_ticker"].unique())
    print(f"tickers beyond the cache build: {len(new_tk)} across {new_events}")
    print(f"cache tickers missing from ledger: {sorted(gone_tk) if gone_tk else 'none'}")

    m = led.merge(old[["ticker"] + JOIN_COLS], on="ticker", suffixes=("", "_old"))
    mismatches = {}
    print(f"join-column comparison over {len(m)} overlapping rows:")
    for col in JOIN_COLS:
        bad = m[m[col].fillna(-9) != m[f"{col}_old"].fillna(-9)]
        mismatches[col] = len(bad)
        print(f"  {col}: {len(bad)} mismatches")
        if not bad.empty:
            print(bad[["ticker", col, f"{col}_old"]].drop_duplicates().to_string(index=False))

    oc = pd.read_csv(os.path.join(cache_dir, "candles.csv")).groupby("ticker").size()
    candle_diffs = []
    for et in led["event_ticker"].unique():
        nc = pd.read_csv(os.path.join(store_dir, "candles", f"{et}.csv.gz")
                         ).groupby("ticker").size()
        candle_diffs += [(tk, int(n), int(oc[tk]))
                         for tk, n in nc.items() if tk in oc.index and n != oc[tk]]
    print(f"per-ticker candle row-count diffs vs cache: {len(candle_diffs)}")
    for r in candle_diffs[:10]:
        print("  ", r)

    # The check that caught the 2026-06-02 sentiment-case switch: every self-label must
    # land inside the score interval implied by the event's OWN settlement results.
    print("\n=== settlement-implied label consistency ===")
    n_bad = 0
    for et, g in led.groupby("event_ticker"):
        iv = implied_score_interval(g.to_dict("records"))
        s = g["score_self"].iloc[0]
        if iv is None or pd.isna(s):
            continue
        if not (iv[0] <= int(s) <= iv[1]):
            n_bad += 1
            print(f"  MISMATCH {et} ({g['slug'].iloc[0]}): score_self={int(s)} "
                  f"vs implied [{iv[0]},{iv[1]}]")
    print(f"  {n_bad} inconsistent of {led['event_ticker'].nunique()} events")

    ev_path = os.path.join(store_dir, "events_open.csv")
    if os.path.exists(ev_path):
        ev = pd.read_csv(ev_path)
        latest = ev[ev["run_ts"] == ev["run_ts"].max()]
        print(f"\n=== open events (latest snapshot, {latest['run_ts'].iloc[0]}) ===")
        print(latest[["event_ticker", "movie_name", "slug", "n_reviews_db", "close_time"]]
              .to_string(index=False))

    return {"n_ledger": len(led), "n_overlap": len(m), "join_mismatches": mismatches,
            "n_candle_diffs": len(candle_diffs), "new_events": new_events,
            "n_label_inconsistent": n_bad}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    validate()
