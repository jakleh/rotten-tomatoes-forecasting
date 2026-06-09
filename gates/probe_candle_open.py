"""Probe: do Kalshi order books persist through silent candle gaps?

The candlestick docs don't say whether a 1-min candle is emitted for EVERY minute or
only minutes with activity (the cache shows large gaps on thin markets). The arena map
needs to know whether "no candle at minute t" means "book unchanged" (then carrying the
last candle's bid/ask forward reconstructs the true book) or "book unknown".

Test: re-pull 1-min candles for a few representative markets keeping bid/ask **open**
AND **close** (the main cache kept only close). If `open` of the candle after a silent
gap equals `close` of the candle before it (book resumed exactly where it left off),
books persist through gaps and LOCF is valid.

Network I/O lives here (public Kalshi, no auth) — caches
``gates/_cache/candle_open_probe.csv``; the arena notebook computes the continuity
stats from the cache. See plans/plan_gate_1_2_calibration.md (arena-map section).
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd

from gates import kalshi_data as kd

CACHE = os.path.join(os.path.dirname(__file__), "_cache")
# candle-count quantiles to sample (dense -> sparse), de-duplicated
PICK_QUANTILES = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def candles_open_close(ticker: str, start_ts: int, end_ts: int,
                       chunk_minutes: int = 2880) -> list[dict]:
    """1-min candles keeping bid/ask open+close (the main fetcher keeps close only)."""
    step = chunk_minutes * 60
    by_ts: dict[int, dict] = {}
    lo = start_ts
    while lo < end_ts:
        hi = min(lo + step, end_ts)
        resp = kd._get(
            f"/series/{kd.RT_SERIES}/markets/{ticker}/candlesticks",
            {"start_ts": str(lo), "end_ts": str(hi), "period_interval": "1"},
        )
        for c in resp.get("candlesticks", []):
            ts = int(c["end_period_ts"])
            yb, ya = (c.get("yes_bid") or {}), (c.get("yes_ask") or {})
            by_ts[ts] = {
                "ticker": ticker, "ts": ts,
                "bid_open": kd._f(yb.get("open_dollars")),
                "bid_close": kd._f(yb.get("close_dollars")),
                "ask_open": kd._f(ya.get("open_dollars")),
                "ask_close": kd._f(ya.get("close_dollars")),
                "last": kd._f((c.get("price") or {}).get("previous_dollars")),
                "volume": float(c.get("volume_fp") or 0.0),
            }
        lo = hi
        time.sleep(kd._READ_PAUSE_S)
    return [by_ts[k] for k in sorted(by_ts)]


def build() -> pd.DataFrame:
    mk = pd.read_csv(os.path.join(CACHE, "cohort_markets.csv"))
    cd = pd.read_csv(os.path.join(CACHE, "candles.csv"))
    counts = cd.groupby("ticker").size().sort_values()  # ascending: sparse -> dense
    picks: list[str] = []
    for q in PICK_QUANTILES:
        idx = min(int(round(q * (len(counts) - 1))), len(counts) - 1)
        t = str(counts.index[idx])
        if t not in picks:
            picks.append(t)
    print(f"probing {len(picks)} markets (candle counts: "
          f"{[int(counts[t]) for t in picks]})")

    rows: list[dict] = []
    for tk in picks:
        r = mk[mk["ticker"] == tk].iloc[0]
        got = candles_open_close(tk, _ts(r["open_time"]), _ts(r["close_time"]))
        rows.extend(got)
        print(f"  {tk}: {len(got)} candles")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CACHE, "candle_open_probe.csv"), index=False)
    print(f"cached {len(df)} probe rows -> _cache/candle_open_probe.csv")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build()
