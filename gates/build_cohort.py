"""Driver: assemble the Gate cohort from Kalshi (settled KXRT) + the reviews DB,
and cache it to ``gates/_cache/`` (gitignored).

Network I/O lives here — run sandbox-disabled. The Gate-1 analysis notebook then reads
the cached CSVs (sandboxed, no network). See ``plans/plan_gate_1_2_calibration.md``.

Outputs:
  gates/_cache/cohort_markets.csv  — one row per settled market: ids, floor_strike,
                                     result, open/close/settlement times, the movie's
                                     self-labeled 10am score, last-day day-level count.
  gates/_cache/candles.csv         — per-market 1-min candles over open->close:
                                     ts, secs_to_close, yes_bid/ask, mid, last, volume.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from gates import db_facts as dbf
from gates import kalshi_data as kd
from gates.slug_map import NAME_RE, map_slug, norm

CACHE = os.path.join(os.path.dirname(__file__), "_cache")


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def build(status: str = "settled", chunk_minutes: int = 2880) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(CACHE, exist_ok=True)
    mkts = kd.list_markets(status=status)
    print(f"fetched {len(mkts)} {status} markets")

    conn = dbf.connect()
    try:
        n = dbf.as_of_id(conn)
        db_norm = {norm(s): s for s in dbf.movie_review_counts(conn, n)}
        # event -> movie name -> slug -> close_ts
        ev = {}
        for m in mkts:
            et = m["event_ticker"]
            if et not in ev:
                mt = NAME_RE.search(m.get("rules_primary") or "")
                ev[et] = {"name": mt.group(1).strip() if mt else None,
                          "close": _dt(m["close_time"])}
        for et, info in ev.items():
            info["slug"] = map_slug(info["name"], db_norm)
        # self-labeled 10am score per movie (one query per slug)
        score = {}
        for et, info in ev.items():
            slug = info["slug"]
            if slug is None or slug in score:
                continue
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FILTER (WHERE estimated_timestamp <= %(c)s), "
                    "count(*) FILTER (WHERE estimated_timestamp <= %(c)s "
                    "                 AND lower(tomatometer_sentiment)='positive'), "
                    "count(*) FILTER (WHERE estimated_timestamp > %(c)s - interval '1 day' "
                    "                 AND estimated_timestamp <= %(c)s "
                    "                 AND timestamp_confidence='d') "
                    "FROM reviews WHERE movie_slug=%(s)s AND id<=%(n)s",
                    {"c": info["close"], "s": slug, "n": n})
                score[slug] = cur.fetchone()  # (total, fresh, lastday_daylevel)
    finally:
        conn.close()

    mrows, crows = [], []
    for i, m in enumerate(mkts):
        et, tk = m["event_ticker"], m["ticker"]
        slug = ev[et]["slug"]
        tot, fresh, ld_d = score.get(slug, (None, None, None))
        mrows.append({
            "ticker": tk, "event": et, "slug": slug, "floor_strike": m.get("floor_strike"),
            "result": m.get("result"), "open_time": m.get("open_time"),
            "close_time": m.get("close_time"), "settlement_ts": m.get("settlement_ts"),
            "total_at_close": tot, "fresh_at_close": fresh,
            "score_self": (round(fresh / tot * 100) if tot else None),
            "lastday_daylevel": ld_d,
        })
        close_e = int(ev[et]["close"].timestamp())
        try:
            cs = kd.candles(tk, int(_dt(m["open_time"]).timestamp()), close_e + 3600,
                            chunk_minutes=chunk_minutes)
        except Exception as e:  # noqa: BLE001
            print(f"  candle pull failed for {tk}: {type(e).__name__}: {e}")
            cs = []
        for c in cs:
            crows.append({"ticker": tk, "ts": c.ts, "secs_to_close": close_e - c.ts,
                          "yes_bid": c.yes_bid, "yes_ask": c.yes_ask, "mid": c.mid,
                          "last": c.last, "volume": c.volume})
        if (i + 1) % 40 == 0:
            print(f"  ...{i + 1}/{len(mkts)} markets, {len(crows)} candle rows so far")

    mdf, cdf = pd.DataFrame(mrows), pd.DataFrame(crows)
    mdf.to_csv(os.path.join(CACHE, "cohort_markets.csv"), index=False)
    cdf.to_csv(os.path.join(CACHE, "candles.csv"), index=False)
    print(f"\nas_of_id={n}")
    print(f"cached {len(mdf)} markets ({mdf['slug'].nunique()} movies), {len(cdf)} candle rows")
    print(f"  markets with a self-labeled score: {int(mdf['score_self'].notna().sum())}")
    print(f"  candle rows with a real mid     : {int(cdf['mid'].notna().sum())} / {len(cdf)}")
    return mdf, cdf


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build()
