"""Driver: assemble the Gate-1b input — for each (market, snap), the market mid (from
the candle cache) plus the publication-time observed review state at that snap (from the
reviews DB via db_facts). Caches gates/_cache/gate1b_input.csv.

Network I/O (DB) lives here — run sandbox-disabled. See plans/plan_gate_1_2_calibration.md.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd

from gates import db_facts as dbf

CACHE = os.path.join(os.path.dirname(__file__), "_cache")
SNAPS = {'5d':5*86400,'3d':3*86400,'2d':2*86400,'1d':86400,
         '12h':12*3600,'6h':6*3600,'3h':3*3600,'1h':3600}


def build():
    mk = pd.read_csv(os.path.join(CACHE, "cohort_markets.csv"))
    cd = pd.read_csv(os.path.join(CACHE, "candles.csv"))
    cd = cd[cd['mid'].notna()]

    # market mid as of each snap
    mids = {}
    for tk, g in cd.groupby('ticker'):
        for name, s in SNAPS.items():
            gg = g[g['secs_to_close'] >= s]
            if len(gg):
                mids[(tk, name)] = float(gg.loc[gg['secs_to_close'].idxmin(), 'mid'])

    # observed review state per (movie, snap) from the DB
    conn = dbf.connect()
    try:
        n = dbf.as_of_id(conn)
        close_by_slug = {}
        for _, r in mk.iterrows():
            close_by_slug.setdefault(r['slug'], r['close_time'])
        snap_state = {}
        for slug, close_iso in close_by_slug.items():
            close = datetime.fromisoformat(close_iso.replace('Z', '+00:00'))
            for name, s in SNAPS.items():
                snap_state[(slug, name)] = dbf.observed_state(
                    conn, slug, close - timedelta(seconds=s), n)
    finally:
        conn.close()

    rows = []
    for _, r in mk.iterrows():
        tk, slug, X = r['ticker'], r['slug'], r['floor_strike']
        y = int(r['result'] == 'yes')
        for name in SNAPS:
            if (tk, name) not in mids:
                continue
            fresh, total = snap_state.get((slug, name), (0, 0))
            obs_score = (fresh / total) if total else float('nan')
            rows.append({
                'ticker': tk, 'slug': slug, 'snap': name, 'mid': mids[(tk, name)], 'y': y,
                'floor_strike': X, 'obs_total': total, 'obs_score': obs_score,
                'obs_margin': (obs_score - X / 100.0) if total else 0.0,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CACHE, "gate1b_input.csv"), index=False)
    print(f"as_of_id={n}")
    print(f"gate1b_input: {len(df)} rows, {df['slug'].nunique()} movies")
    print(f"  obs_total==0 (no reviews yet at snap): {int((df['obs_total'] == 0).sum())}")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build()
