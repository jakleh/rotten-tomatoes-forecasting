"""Tests for gates/recorder.py — the §1.7 weekly settled-market recorder.

No network, no DB: Kalshi fetchers and the DB join are injected fakes (real
``kalshi_data.Candle`` objects; a duck-typed ``FakeDb``). Stores live in tmp_path.
Every ``rec.run`` call passes ``db_factory`` explicitly — the default would read the
repo's real ``.env``.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from gates import recorder as rec
from gates.kalshi_data import Candle

T0 = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
CLOSE = "2026-06-01T14:00:00Z"
CLOSE_E = int(datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc).timestamp())
OPEN_T = "2026-05-20T00:00:00Z"


def mkt(ticker: str, event: str, *, name: str | None = "Backrooms", close: str = CLOSE,
        strike: int = 80, result: str = "no") -> dict:
    return {
        "ticker": ticker, "event_ticker": event,
        "rules_primary": (f"If {name} has a Tomatometer score of {strike + 1} or above, "
                          f"then the market resolves Yes." if name else ""),
        "open_time": OPEN_T, "close_time": close,
        "settlement_ts": "2026-06-01T15:30:00Z", "result": result,
        "floor_strike": strike, "strike_type": "greater",
    }


class FakeKalshi:
    def __init__(self, settled=(), open_=(), fail_tickers=(), empty_tickers=()):
        self.settled = list(settled)
        self.open = list(open_)
        self.fail_tickers = set(fail_tickers)
        self.empty_tickers = set(empty_tickers)
        self.candle_calls: list[str] = []

    def fetch_markets(self, status=None, **_):
        return list(self.settled if status == "settled" else self.open)

    def fetch_candles(self, ticker, start_ts, end_ts, chunk_minutes=None):
        self.candle_calls.append(ticker)
        if ticker in self.fail_tickers:
            raise RuntimeError("boom")
        if ticker in self.empty_tickers:
            return []
        return [Candle(ts=CLOSE_E - 60 * k, yes_bid=0.40, yes_ask=0.50, last=0.45,
                       volume=1.0, open_interest=2.0) for k in (3, 2, 1)]


class FakeDb:
    def __init__(self, counts=None, state=(40, 50), lastday=2, as_of=999, states=None):
        self._counts = {"backrooms": 50, "pressure": 60} if counts is None else counts
        self._state, self._lastday = state, lastday
        self._states = states or {}  # per-slug (fresh, total) overrides
        self.as_of_id = as_of
        self.closed = False

    def review_counts(self):
        return dict(self._counts)

    def close_state(self, slug, close_ts):
        return self._states.get(slug, self._state)

    def lastday_d(self, slug, close_ts):
        return self._lastday

    def close(self):
        self.closed = True


def ledger(store) -> pd.DataFrame:
    return pd.read_csv(os.path.join(store, "markets.csv"))


def runs(store) -> pd.DataFrame:
    return pd.read_csv(os.path.join(store, "runs.csv"))


SETTLED3 = [mkt("KXRT-BAC-80", "KXRT-BAC"), mkt("KXRT-BAC-85", "KXRT-BAC", strike=85),
            mkt("KXRT-PRE-75", "KXRT-PRE", name="Pressure", strike=75)]


def test_fresh_capture(tmp_path):
    store = str(tmp_path)
    fk = FakeKalshi(SETTLED3)
    db = FakeDb()
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: db, now=T0)
    assert (res["n_new_markets"], res["n_new_events"]) == (3, 2)
    assert res["n_candle_rows_added"] == 9 and res["as_of_id"] == 999
    led = ledger(store)
    assert len(led) == 3 and led["ticker"].is_unique
    assert led["db_joined"].all() and set(led["slug"]) == {"backrooms", "pressure"}
    assert (led["score_self"] == 80).all() and (led["total_at_close"] == 50).all()
    assert (led["lastday_daylevel"] == 2).all() and (led["as_of_id"] == 999).all()
    bac = pd.read_csv(os.path.join(store, "candles", "KXRT-BAC.csv.gz"))
    assert list(bac.columns) == rec.CANDLE_COLS and len(bac) == 6
    assert (bac["mid"] == 0.45).all() and (bac["secs_to_close"] + bac["ts"] == CLOSE_E).all()
    assert len(runs(store)) == 1 and db.closed
    assert not os.path.exists(os.path.join(store, "events_open.csv"))  # no open events


def test_idempotent_rerun(tmp_path):
    store = str(tmp_path)
    fk = FakeKalshi(SETTLED3)
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0)
    n_calls = len(fk.candle_calls)
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(), now=T0 + timedelta(days=7))
    assert res["n_new_markets"] == 0 and res["n_topped_up"] == 0
    assert len(fk.candle_calls) == n_calls  # zero re-fetches
    assert len(ledger(store)) == 3 and len(runs(store)) == 2


def test_incremental_new_event(tmp_path):
    store = str(tmp_path)
    fk = FakeKalshi(SETTLED3)
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0)
    fk.settled.append(mkt("KXRT-ZOO-90", "KXRT-ZOO", name="Zootopia 3", strike=90))
    n_calls = len(fk.candle_calls)
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(counts={"backrooms": 50, "zootopia_3": 9}),
                  now=T0 + timedelta(days=7))
    assert res["n_new_markets"] == 1 and fk.candle_calls[n_calls:] == ["KXRT-ZOO-90"]
    led = ledger(store)
    assert len(led) == 4
    assert led.loc[led["ticker"] == "KXRT-ZOO-90", "slug"].item() == "zootopia_3"


def test_partial_event_merge_dedupes(tmp_path):
    store = str(tmp_path)
    fk = FakeKalshi([SETTLED3[0]])  # only BAC-80 settled "first week"
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0)
    fk.settled.append(SETTLED3[1])  # BAC-85 appears later, same event
    n_calls = len(fk.candle_calls)
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0 + timedelta(days=7))
    assert fk.candle_calls[n_calls:] == ["KXRT-BAC-85"]  # 80 not re-fetched
    bac = pd.read_csv(os.path.join(store, "candles", "KXRT-BAC.csv.gz"))
    assert len(bac) == 6  # 3 + 3, no duplicates from the merge
    assert not bac.duplicated(subset=["ticker", "ts"]).any()


def test_fetch_failure_skips_then_recovers(tmp_path):
    store = str(tmp_path)
    fk = FakeKalshi(SETTLED3, fail_tickers={"KXRT-BAC-85"}, empty_tickers={"KXRT-PRE-75"})
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(), now=T0)
    led = ledger(store)
    assert res["n_new_markets"] == 2 and "KXRT-BAC-85" not in set(led["ticker"])
    # the legitimately-empty market IS ledgered, with 0 candle rows + a warning
    assert led.loc[led["ticker"] == "KXRT-PRE-75", "candle_rows"].item() == 0
    assert res["n_warnings"] >= 2
    fk.fail_tickers.clear()
    res2 = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                   db_factory=lambda: FakeDb(), now=T0 + timedelta(days=7))
    assert res2["n_new_markets"] == 1 and len(ledger(store)) == 3
    bac = pd.read_csv(os.path.join(store, "candles", "KXRT-BAC.csv.gz"))
    assert len(bac) == 6 and not bac.duplicated(subset=["ticker", "ts"]).any()


def test_no_db_capture_then_topup(tmp_path):
    store = str(tmp_path)
    fk = FakeKalshi(SETTLED3)
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=None, now=T0)
    assert res["db_available"] is False and res["as_of_id"] == ""
    led = ledger(store)
    assert (~led["db_joined"]).all() and led["slug"].isna().all()
    n_calls = len(fk.candle_calls)
    res2 = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                   db_factory=lambda: FakeDb(), now=T0 + timedelta(days=1))
    assert res2["n_topped_up"] == 3 and res2["n_new_markets"] == 0
    assert len(fk.candle_calls) == n_calls  # top-up never touches candles
    led = ledger(store)
    assert led["db_joined"].all() and (led["score_self"] == 80).all()


def test_unmappable_name_defers_join(tmp_path):
    store = str(tmp_path)
    fk = FakeKalshi([mkt("KXRT-UNK-80", "KXRT-UNK", name="Totally Unknown Film")])
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(), now=T0)
    led = ledger(store)
    assert not led["db_joined"].item() and res["n_warnings"] >= 1
    # 0-reviews-at-close also defers (self-healing if the scraper backfills later)
    fk2 = FakeKalshi([mkt("KXRT-BAC-80", "KXRT-BAC")])
    res2 = rec.run(str(tmp_path / "s2"), fetch_markets=fk2.fetch_markets,
                   fetch_candles=fk2.fetch_candles,
                   db_factory=lambda: FakeDb(state=(0, 0)), now=T0)
    led2 = ledger(str(tmp_path / "s2"))
    assert not led2["db_joined"].item() and led2["slug"].item() == "backrooms"
    assert res2["n_warnings"] >= 1


def test_coverage_watch(tmp_path, capsys):
    store = str(tmp_path)
    fk = FakeKalshi([], open_=[mkt("KXRT-NEW-80", "KXRT-NEW", name="Mystery Film",
                                   close="2026-06-20T14:00:00Z"),
                               mkt("KXRT-BAC-80", "KXRT-BAC")])
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0)
    out = capsys.readouterr().out
    assert "[coverage]" in out and "Mystery Film" in out and "Backrooms" not in out.split(
        "[coverage]")[1]
    ev = pd.read_csv(os.path.join(store, "events_open.csv"))
    assert len(ev) == 2 and set(ev["event_ticker"]) == {"KXRT-NEW", "KXRT-BAC"}
    assert ev.loc[ev["event_ticker"] == "KXRT-BAC", "n_reviews_db"].item() == 50
    assert ev.loc[ev["event_ticker"] == "KXRT-NEW", "n_reviews_db"].item() == 0
    # without a DB: snapshot still recorded, no coverage warnings possible
    store2 = str(tmp_path / "s2")
    rec.run(store2, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=None, now=T0)
    assert "[coverage]" not in capsys.readouterr().out
    assert len(pd.read_csv(os.path.join(store2, "events_open.csv"))) == 2


def test_aged_out_counted_not_refetched(tmp_path):
    store = str(tmp_path)
    fk = FakeKalshi(SETTLED3)
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0)
    fk.settled = [SETTLED3[2]]  # BAC markets fell off the API window
    n_calls = len(fk.candle_calls)
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(), now=T0 + timedelta(days=7))
    assert res["n_aged_out"] == 2 and res["n_new_markets"] == 0
    assert len(fk.candle_calls) == n_calls
    assert len(ledger(store)) == 3  # nothing dropped from the store


def test_dry_run_writes_nothing(tmp_path):
    store = str(tmp_path / "fresh")
    fk = FakeKalshi(SETTLED3, open_=[mkt("KXRT-NEW-80", "KXRT-NEW", name="X")])
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: pytest.fail("dry run must not open the DB"),
                  dry_run=True, now=T0)
    assert res["dry_run"] and res["n_new_markets"] == 3 and res["n_open_events"] == 1
    assert fk.candle_calls == [] and not os.path.exists(store)


def test_check_staleness(tmp_path):
    store = str(tmp_path)
    code, msg = rec.check_staleness(store, now=T0)
    assert code == 1 and "NEVER" in msg
    fk = FakeKalshi(SETTLED3)
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0)
    code, msg = rec.check_staleness(store, now=T0 + timedelta(days=3))
    assert code == 0 and "fresh" in msg
    code, msg = rec.check_staleness(store, now=T0 + timedelta(days=11))
    assert code == 1 and "STALE" in msg


def test_csv_roundtrip_types(tmp_path):
    """Ledger survives a write->read->write cycle with bools and NaNs intact."""
    store = str(tmp_path)
    fk = FakeKalshi(SETTLED3)
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=None, now=T0)
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=None, now=T0 + timedelta(days=1))  # re-run over loaded CSV rows
    led = ledger(store)
    assert led["db_joined"].dtype == bool and len(led) == 3


# --- review-driven regression tests (2026-06-10 adversarial pass) ---


def test_duplicate_api_ticker_ignored(tmp_path):
    """A repeated ticker in one settled response (shifting cursor walk) must not
    produce duplicate ledger rows or wedge subsequent runs."""
    store = str(tmp_path)
    fk = FakeKalshi(SETTLED3 + [dict(SETTLED3[0])])
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0)
    led = ledger(store)
    assert len(led) == 3 and led["ticker"].is_unique
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(), now=T0 + timedelta(days=7))
    assert res["n_new_markets"] == 0 and len(ledger(store)) == 3


def test_partial_event_close_drift_no_wedge(tmp_path):
    """A late strike with a DIFFERENT close_time in an already-captured event must
    capture cleanly (per-market close), and later events in the same run too."""
    store = str(tmp_path)
    fk = FakeKalshi([SETTLED3[0]])
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0)
    late = "2026-06-08T14:00:00Z"
    late_e = int(datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc).timestamp())
    fk.settled += [mkt("KXRT-BAC-85", "KXRT-BAC", strike=85, close=late),
                   mkt("KXRT-ZOO-90", "KXRT-ZOO", name="Zootopia 3", strike=90)]
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(counts={"backrooms": 50, "zootopia_3": 9}),
                  now=T0 + timedelta(days=7))
    assert res["n_new_markets"] == 2  # the drifted strike AND the later event
    bac = pd.read_csv(os.path.join(store, "candles", "KXRT-BAC.csv.gz"))
    b85 = bac[bac["ticker"] == "KXRT-BAC-85"]
    b80 = bac[bac["ticker"] == "KXRT-BAC-80"]
    assert ((b85["ts"] + b85["secs_to_close"]) == late_e).all()
    assert ((b80["ts"] + b80["secs_to_close"]) == CLOSE_E).all()


def test_mixed_join_preserves_int_text(tmp_path):
    """A deferred join (NaN numerics) must not flip committed int columns to
    '999.0'-style float text on any later full-ledger rewrite."""
    store = str(tmp_path)
    fk = FakeKalshi(SETTLED3)
    mixed = lambda extra=None: FakeDb(  # noqa: E731
        counts={"backrooms": 50, "pressure": 60, **(extra or {})},
        states={"pressure": (0, 0)})
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=mixed, now=T0)
    led = ledger(store)
    assert int(led["db_joined"].sum()) == 2  # BAC joined, PRE deferred
    fk.settled.append(mkt("KXRT-ZOO-90", "KXRT-ZOO", name="Zootopia 3", strike=90))
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: mixed({"zootopia_3": 9}), now=T0 + timedelta(days=7))
    txt = open(os.path.join(store, "markets.csv")).read()
    assert "999.0" not in txt and "999" in txt  # as_of_id stays integer text
    assert ".0," not in txt.split("\n")[1]  # first data row carries no float-drift ints


def test_topup_after_zero_reviews_defer(tmp_path):
    """0-reviews-at-close defers the join (slug recorded); a later run with data
    self-heals."""
    store = str(tmp_path)
    fk = FakeKalshi([SETTLED3[2]])
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(state=(0, 0)), now=T0)
    led = ledger(store)
    assert not led["db_joined"].item() and led["slug"].item() == "pressure"
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(state=(30, 60)), now=T0 + timedelta(days=7))
    assert res["n_topped_up"] == 1
    led = ledger(store)
    assert led["db_joined"].item() and led["score_self"].item() == 50


def test_unparsed_name_topup_no_crash(tmp_path, capsys):
    """A nameless ledger row (movie_name NaN after round-trip) must aggregate-warn
    in top-up, not crash or warn per row."""
    store = str(tmp_path)
    fk = FakeKalshi([mkt("KXRT-UNK-80", "KXRT-UNK", name=None)])
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=None, now=T0)
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(), now=T0 + timedelta(days=1))
    assert res["n_topped_up"] == 0
    assert not ledger(store)["db_joined"].item()
    assert "no parsed movie_name" in capsys.readouterr().out


def test_check_staleness_malformed_run_ts(tmp_path):
    store = str(tmp_path)
    pd.DataFrame([{c: ("garbage" if c == "run_ts" else 0) for c in rec.RUNS_COLS}]).to_csv(
        os.path.join(store, "runs.csv"), index=False)
    code, msg = rec.check_staleness(store, now=T0)
    assert code == 1 and "unparseable" in msg


def test_check_staleness_empty_runs_csv(tmp_path):
    store = str(tmp_path)
    open(os.path.join(store, "runs.csv"), "w").close()  # zero-byte file
    code, msg = rec.check_staleness(store, now=T0)
    assert code == 1 and "NEVER" in msg


def test_label_consistency_rejoin(tmp_path, capsys):
    """A self-label contradicting the event's own settlement results warns at capture
    and gets re-joined on later runs (self-healing after processing-layer fixes, e.g.
    the 2026-06-02 sentiment-case switch)."""
    store = str(tmp_path)
    settled = [mkt("KXRT-BAC-75", "KXRT-BAC", strike=75, result="yes"),
               mkt("KXRT-BAC-85", "KXRT-BAC", strike=85, result="no")]  # implied [76, 85]
    fk = FakeKalshi(settled)
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(state=(10, 50)), now=T0)  # score 20: inconsistent
    assert "[label-consistency]" in capsys.readouterr().out
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(state=(10, 50)), now=T0 + timedelta(days=7))
    assert res["n_rejoined"] == 0  # DB still wrong: rejoin recomputes the same values
    assert "[label-consistency]" in capsys.readouterr().out  # persistent defect keeps warning
    res = rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
                  db_factory=lambda: FakeDb(state=(40, 50)), now=T0 + timedelta(days=14))
    assert res["n_rejoined"] == 2  # both strikes healed by the corrected join
    led = ledger(store)
    assert (led["score_self"] == 80).all()
    assert "[label-consistency]" not in capsys.readouterr().out
    # once healed, the stored consistent label is never re-touched (system of record
    # keeps the last settlement-consistent join even if the DB later regresses)
    rec.run(store, fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(state=(10, 50)), now=T0 + timedelta(days=21))
    assert (ledger(store)["score_self"] == 80).all()


def test_coverage_watch_missing_close_time(tmp_path, capsys):
    m = mkt("KXRT-NEW-80", "KXRT-NEW", name="Mystery Film")
    m["close_time"] = None
    fk = FakeKalshi([], open_=[m])
    rec.run(str(tmp_path), fetch_markets=fk.fetch_markets, fetch_candles=fk.fetch_candles,
            db_factory=lambda: FakeDb(), now=T0)
    out = capsys.readouterr().out
    assert "[coverage]" in out and "close time unknown" in out
