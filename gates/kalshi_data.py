"""Read-only public Kalshi market-data fetcher for the RT (KXRT) series.

Public endpoints only (markets / events / candlesticks) — **no credentials, no auth
header ever sent**. Uses stdlib ``urllib`` (this repo's venv has no httpx/requests).
Endpoint shapes cribbed from ``~/Desktop/kalshi-trading/src/kalshi/client.py`` but kept
self-contained per the repo-separation boundary.

Gate-calibration support code (not part of the shipped package). See
``plans/plan_gate_1_2_calibration.md``.

Verified 2026-06-07: series KXRT = "Rotten Tomatoes Scores"; settled markets are all
``strike_type='greater'`` (Above-X, strictly greater). Candles give the per-minute mid
+ last-trade + volume — NOT depth.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
RT_SERIES = "KXRT"
_READ_PAUSE_S = 0.05  # polite spacing; Basic tier allows ~20 reads/sec


def _get(path: str, params: dict | None = None, *, timeout: int = 30, retries: int = 3) -> dict:
    """GET a public endpoint, return parsed JSON. Retries transient failures. No auth."""
    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 — retry transient HTTP/timeout errors
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {path} failed after {retries} attempts: "
                       f"{type(last_err).__name__}: {last_err}")


def _paginate(path: str, params: dict, key: str, *, cap: int = 20000) -> list[dict]:
    """Follow Kalshi cursor pagination, accumulating ``resp[key]``."""
    out: list[dict] = []
    cursor: str | None = None
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        resp = _get(path, p)
        out.extend(resp.get(key, []))
        cursor = resp.get("cursor")
        if not cursor or len(out) >= cap:
            break
        time.sleep(_READ_PAUSE_S)
    return out


def list_markets(series: str = RT_SERIES, status: str | None = "settled") -> list[dict]:
    """All markets for a series (optionally filtered by status), as raw API dicts.

    The list response already carries ``close_time``, ``settlement_ts``, ``result``,
    ``floor_strike``, ``strike_type``, ``rules_primary``, and current
    ``yes_bid_dollars``/``yes_ask_dollars``/``last_price_dollars``/``volume_fp``/
    ``open_interest_fp``/``liquidity_dollars`` — enough for cohort assembly + a
    liquidity proxy without per-ticker ``get_market`` calls.
    """
    params: dict = {"series_ticker": series, "limit": "1000"}
    if status:
        params["status"] = status
    return _paginate("/markets", params, "markets")


def _f(x) -> float | None:
    return None if x is None else float(x)


@dataclass(frozen=True)
class Candle:
    ts: int                # end_period_ts (Unix seconds, end of the period)
    yes_bid: float | None  # best yes bid at period close (dollars), None if absent
    yes_ask: float | None  # best yes ask at period close (dollars), None if absent
    last: float | None     # last trade price (dollars), None if never traded
    volume: float          # contracts traded in the period
    open_interest: float

    @property
    def mid(self) -> float | None:
        """Order-book mid, only when a *real two-sided* quote exists.

        Returns None for no-quote / degenerate (0/1) minutes — those are not usable
        price observations. Gate 1 uses real-two-sided-quote minutes only (reported
        as calibration-conditional-on-tradeable).
        """
        b, a = self.yes_bid, self.yes_ask
        if b is None or a is None or not (0.0 < b <= a < 1.0):
            return None
        return (b + a) / 2.0


def candles(market_ticker: str, start_ts: int, end_ts: int, *, series: str = RT_SERIES,
            period_interval: int = 1, chunk_minutes: int = 1440) -> list[Candle]:
    """Candlesticks over ``[start_ts, end_ts]`` (Unix sec), default 1-min.

    Chunked under Kalshi's response cap and de-duplicated by timestamp. ``chunk_minutes``
    is kept conservatively below any plausible cap; raise it for fewer requests if the
    cap allows. Public, no auth.
    """
    step = chunk_minutes * 60 * period_interval
    by_ts: dict[int, Candle] = {}
    lo = start_ts
    while lo < end_ts:
        hi = min(lo + step, end_ts)
        resp = _get(
            f"/series/{series}/markets/{market_ticker}/candlesticks",
            {"start_ts": str(lo), "end_ts": str(hi), "period_interval": str(period_interval)},
        )
        for c in resp.get("candlesticks", []):
            ts = int(c["end_period_ts"])
            by_ts[ts] = Candle(
                ts=ts,
                yes_bid=_f((c.get("yes_bid") or {}).get("close_dollars")),
                yes_ask=_f((c.get("yes_ask") or {}).get("close_dollars")),
                last=_f((c.get("price") or {}).get("previous_dollars")),
                volume=float(c.get("volume_fp") or 0.0),
                open_interest=float(c.get("open_interest_fp") or 0.0),
            )
        lo = hi
        time.sleep(_READ_PAUSE_S)
    return [by_ts[k] for k in sorted(by_ts)]
