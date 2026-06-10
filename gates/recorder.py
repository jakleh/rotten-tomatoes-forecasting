"""Driver: the §1.7 weekly settled-market recorder — incremental, idempotent capture
of newly-settled KXRT markets into the COMMITTED store ``gates/recorded/``.

``gates/_cache/`` is the rebuildable working set (gitignored); ``gates/recorded/`` is
the system of record for what the Kalshi API eventually stops serving (settled
retention observed >= ~9 weeks) — so it IS committed. Design + sign-off:
``brainstorm/brainstorm_recorder_infra.md`` + ``plans/plan_recorder.md`` (2026-06-09).

    python -m gates.recorder            # capture newly-settled (+DB join when .env present)
    python -m gates.recorder --check    # staleness probe: exit 1 if never run / >10d old
    python -m gates.recorder --dry-run  # report what a run would do; writes nothing
    python -m gates.recorder --no-db    # skip the DB join (rows top up on a later run)

Kalshi reads are public/no-auth (sandbox-safe). The DB join wants DATABASE_URL (local
``.env``): self-label aggregates only — no raw review rows in the committed store —
pinned by ``as_of_id``. Rows captured without the DB carry ``db_joined=False`` and are
topped up by the next DB-enabled run. Candle files are written before their ledger rows
so a crash never yields a ledgered-but-uncaptured market (re-runs merge + dedupe).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time
from datetime import datetime, timezone

import pandas as pd

from gates import db_facts as dbf
from gates import kalshi_data as kd
from gates.slug_map import NAME_RE, map_slug, norm

STORE = os.path.join(os.path.dirname(__file__), "recorded")
STALE_DAYS = 10.0          # --check threshold (session ritual warns past this)
CANDLE_PAD_S = 3600        # capture window = open_time -> close_time + 1h (build_cohort parity)
CHUNK_MINUTES = 2880

LEDGER_COLS = ["ticker", "event_ticker", "movie_name", "floor_strike", "strike_type",
               "result", "open_time", "close_time", "settlement_ts", "captured_at",
               "candle_rows", "slug", "total_at_close", "fresh_at_close", "score_self",
               "lastday_daylevel", "as_of_id", "db_joined"]
CANDLE_COLS = ["ticker", "ts", "secs_to_close", "yes_bid", "yes_ask", "mid", "last",
               "volume", "open_interest"]
EVENTS_COLS = ["run_ts", "event_ticker", "movie_name", "close_time", "n_markets",
               "slug", "n_reviews_db"]
RUNS_COLS = ["run_ts", "duration_s", "n_settled_api", "n_ledger_before", "n_new_markets",
             "n_new_events", "n_candle_rows_added", "n_aged_out", "n_topped_up",
             "n_rejoined", "as_of_id", "db_available", "n_warnings"]

# Nullable-int columns per file: cast to pandas Int64 at write time so a deferred join's
# NaN never flips the committed CSVs to "999.0"-style float text (whole-file git churn).
LEDGER_INT_COLS = ["candle_rows", "total_at_close", "fresh_at_close", "score_self",
                   "lastday_daylevel", "as_of_id"]
EVENTS_INT_COLS = ["n_markets", "n_reviews_db"]
RUNS_INT_COLS = ["n_settled_api", "n_ledger_before", "n_new_markets", "n_new_events",
                 "n_candle_rows_added", "n_aged_out", "n_topped_up", "n_rejoined",
                 "as_of_id", "n_warnings"]
PAGINATE_CAP_GUARD = 19500  # kalshi_data._paginate caps at 20000, silently


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _s(x) -> str:
    """CSV-round-trip-safe string: None / NaN -> ''."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


def _read_rows(path: str) -> list[dict]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path).to_dict("records")
        except pd.errors.EmptyDataError:
            return []
    return []


def _write_csv(rows: list[dict], cols: list[str], path: str,
               int_cols: list[str] = ()) -> None:
    tmp = path + ".tmp"
    df = pd.DataFrame(rows, columns=cols)
    for c in int_cols:
        df[c] = pd.array(pd.to_numeric(df[c], errors="coerce"), dtype="Int64")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _append_csv(row: dict, cols: list[str], path: str, int_cols: list[str] = ()) -> None:
    rows = _read_rows(path)
    rows.append(row)
    _write_csv(rows, cols, path, int_cols=int_cols)


def _write_gz(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip")  # explicit: .tmp suffix would infer none
    os.replace(tmp, path)


class _Db:
    """Injectable wrapper over ``db_facts`` for the recorder's aggregate needs."""

    def __init__(self, conn, as_of: int):
        self._conn = conn
        self.as_of_id = as_of

    @classmethod
    def open(cls) -> "_Db":
        conn = dbf.connect()
        return cls(conn, dbf.as_of_id(conn))

    def review_counts(self) -> dict[str, int]:
        return dbf.movie_review_counts(self._conn, self.as_of_id)

    def close_state(self, slug: str, close_ts: datetime) -> tuple[int, int]:
        fresh, total = dbf.observed_state(self._conn, slug, close_ts, self.as_of_id)
        return fresh, total

    def lastday_d(self, slug: str, close_ts: datetime) -> int:
        return int(dbf.movie_coverage(self._conn, slug, close_ts, self.as_of_id)["n_last_day_d"])

    def close(self) -> None:
        self._conn.close()


def _default_db_factory() -> _Db | None:
    """A ``_Db`` when DATABASE_URL is resolvable, else None (loud skip in ``run``).

    A resolvable URL that then fails to CONNECT raises — never a silent no-db run.
    """
    try:
        dbf._database_url()
    except RuntimeError:
        return None
    return _Db.open()


def _event_name(markets: list[dict]) -> str:
    for m in markets:
        mt = NAME_RE.search(m.get("rules_primary") or "")
        if mt:
            return mt.group(1).strip()
    return ""


def implied_score_interval(rows: list[dict]) -> tuple[int, int] | None:
    """Settlement-implied bounds for an event's true score from its own strike results:
    'Above X' = yes ⇒ score ≥ X+1; = no ⇒ score ≤ X. None when no usable results."""
    yes, no = [], []
    for r in rows:
        fs = r.get("floor_strike")
        if fs is None or (isinstance(fs, float) and math.isnan(fs)):
            continue
        res = _s(r.get("result")).lower()
        if res == "yes":
            yes.append(int(fs))
        elif res == "no":
            no.append(int(fs))
    if not yes and not no:
        return None
    return ((max(yes) + 1) if yes else 0, (min(no) if no else 100))


def _join_row(row: dict, db: _Db, db_norm: dict[str, str], warn) -> bool:
    """Fill a ledger row's DB fields in place. False (and ``db_joined`` stays False)
    when the name is unmappable or the DB shows 0 reviews at close — both retried on
    later runs (self-healing if the scraper backfills the movie)."""
    name = _s(row.get("movie_name"))
    slug = map_slug(name or None, db_norm)
    if slug is None:
        warn(f"{row['event_ticker']}: cannot map '{name or '?'}' to a DB slug — join deferred")
        return False
    close = _dt(_s(row["close_time"]))
    fresh, total = db.close_state(slug, close)
    if total == 0:
        warn(f"{slug}: 0 reviews at close in DB (as_of_id={db.as_of_id}) — join deferred")
        row["slug"] = slug
        return False
    row.update({"slug": slug, "total_at_close": total, "fresh_at_close": fresh,
                "score_self": round(fresh / total * 100),  # build_cohort parity (banker's)
                "lastday_daylevel": db.lastday_d(slug, close),
                "as_of_id": db.as_of_id, "db_joined": True})
    return True


def run(store_dir: str = STORE, *,
        fetch_markets=kd.list_markets,
        fetch_candles=kd.candles,
        db_factory=_default_db_factory,
        dry_run: bool = False,
        now: datetime | None = None) -> dict:
    """One recorder pass. Returns the run-summary dict (also appended to runs.csv)."""
    t0 = time.monotonic()
    now = now or datetime.now(timezone.utc)
    run_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        print(f"  WARNING: {msg}")

    ledger_path = os.path.join(store_dir, "markets.csv")
    ledger = _read_rows(ledger_path)
    captured = {_s(r["ticker"]) for r in ledger}

    settled = fetch_markets(status="settled")
    if len(settled) >= PAGINATE_CAP_GUARD:
        warn(f"settled list ({len(settled)}) near the pagination cap — possible silent "
             f"truncation; raise kalshi_data._paginate cap")
    api_tickers = {m["ticker"] for m in settled}
    new, _seen_new = [], set()  # dedupe defensively: a shifting cursor walk can repeat
    for m in settled:
        if m["ticker"] not in captured and m["ticker"] not in _seen_new:
            new.append(m)
            _seen_new.add(m["ticker"])
    aged_out = sorted(captured - api_tickers)
    if aged_out:
        print(f"  retention canary: {len(aged_out)} captured tickers no longer on the API "
              f"(e.g. {aged_out[0]})")

    by_event: dict[str, list[dict]] = {}
    for m in new:
        by_event.setdefault(m["event_ticker"], []).append(m)

    open_markets = fetch_markets(status="open")
    open_by_event: dict[str, list[dict]] = {}
    for m in open_markets:
        open_by_event.setdefault(m["event_ticker"], []).append(m)

    print(f"settled on API: {len(settled)} | ledger: {len(ledger)} | "
          f"new: {len(new)} markets / {len(by_event)} events | "
          f"open events: {len(open_by_event)} | aged out: {len(aged_out)}")

    if dry_run:
        for et in sorted(by_event):
            print(f"  would capture {et} ({_event_name(by_event[et]) or '?'}): "
                  f"{len(by_event[et])} markets")
        return {"dry_run": True, "n_settled_api": len(settled),
                "n_ledger_before": len(ledger), "n_new_markets": len(new),
                "n_new_events": len(by_event), "n_aged_out": len(aged_out),
                "n_open_events": len(open_by_event)}

    os.makedirs(os.path.join(store_dir, "candles"), exist_ok=True)
    for p in glob.glob(os.path.join(store_dir, "**", "*.tmp"), recursive=True):
        os.remove(p)  # crashed-run leftovers; never commit cruft
    db = db_factory() if db_factory is not None else None
    if db is None:
        print("  DB join skipped (no DATABASE_URL, or --no-db) — rows top up on a later run")
    db_norm: dict[str, str] = {}
    counts_by_slug: dict[str, int] = {}
    if db is not None:
        counts_by_slug = db.review_counts()
        db_norm = {norm(s): s for s in counts_by_slug}

    # ---- capture new settled markets, one event at a time (ledger flushed per event) ----
    n_candle_rows = 0
    n_new_markets = 0
    n_new_events = 0
    n_fetch_failed = 0
    for et in sorted(by_event):
        ms = by_event[et]
        name = _event_name(ms)
        if not name:
            warn(f"{et}: no movie name parsed from rules_primary")

        # Per-MARKET close epoch: strikes added late can carry a different close_time;
        # one shared event close would truncate their window + skew secs_to_close.
        rows_per_ticker: dict[str, list[dict]] = {}
        for m in ms:
            tk = m["ticker"]
            close_e = int(_dt(m["close_time"]).timestamp())
            try:
                cs = fetch_candles(tk, int(_dt(m["open_time"]).timestamp()),
                                   close_e + CANDLE_PAD_S, chunk_minutes=CHUNK_MINUTES)
            except Exception as e:  # noqa: BLE001 — skip + retry next run, never half-record
                warn(f"candle pull failed for {tk}: {type(e).__name__}: {e}")
                n_fetch_failed += 1
                continue
            rows_per_ticker[tk] = [
                {"ticker": tk, "ts": c.ts, "secs_to_close": close_e - c.ts,
                 "yes_bid": c.yes_bid, "yes_ask": c.yes_ask, "mid": c.mid,
                 "last": c.last, "volume": c.volume, "open_interest": c.open_interest}
                for c in cs]
            if not rows_per_ticker[tk]:
                warn(f"{tk}: 0 candles over its listing window (barely-quoted strike?)")
        if not rows_per_ticker:
            continue  # every pull failed -> nothing ledgered, whole event retries next run

        cpath = os.path.join(store_dir, "candles", f"{et}.csv.gz")
        parts = ([pd.read_csv(cpath)] if os.path.exists(cpath) else [])
        n_existing = len(parts[0]) if parts else 0
        new_rows = [r for rs in rows_per_ticker.values() for r in rs]
        if new_rows:
            parts.append(pd.DataFrame(new_rows, columns=CANDLE_COLS))
        cdf = (pd.concat(parts, ignore_index=True)[CANDLE_COLS] if parts
               else pd.DataFrame(columns=CANDLE_COLS))
        cdf = (cdf.drop_duplicates(subset=["ticker", "ts"], keep="first")
                  .sort_values(["ticker", "ts"], kind="mergesort").reset_index(drop=True))
        assert not cdf[["ticker", "ts", "secs_to_close"]].isna().any().any(), \
            f"{et}: NaN in candle key columns"
        _write_gz(cdf, cpath)
        assert len(pd.read_csv(cpath)) == len(cdf), f"{et}: gz read-back row-count mismatch"
        n_candle_rows += len(cdf) - n_existing  # net rows added (merge dedupe-aware)

        assert os.path.exists(cpath), f"{et}: candle file missing before ledger append"
        for m in ms:
            tk = m["ticker"]
            if tk not in rows_per_ticker:
                continue
            row = {"ticker": tk, "event_ticker": et, "movie_name": name,
                   "floor_strike": m.get("floor_strike"),
                   "strike_type": _s(m.get("strike_type")), "result": _s(m.get("result")),
                   "open_time": _s(m.get("open_time")), "close_time": _s(m.get("close_time")),
                   "settlement_ts": _s(m.get("settlement_ts")), "captured_at": run_ts,
                   "candle_rows": len(rows_per_ticker[tk]), "slug": "",
                   "total_at_close": None, "fresh_at_close": None, "score_self": None,
                   "lastday_daylevel": None, "as_of_id": None, "db_joined": False}
            if db is not None and name:
                _join_row(row, db, db_norm, warn)
            ledger.append(row)
            n_new_markets += 1
        n_new_events += 1
        tickers = [_s(r["ticker"]) for r in ledger]
        assert len(tickers) == len(set(tickers)), \
            f"{et}: ledger ticker uniqueness violated — NOT flushing"
        _write_csv(ledger, LEDGER_COLS, ledger_path, int_cols=LEDGER_INT_COLS)
        print(f"  captured {et} ({name or '?'}): {len(rows_per_ticker)} markets, "
              f"{len(new_rows)} candle rows")

    # ---- top up ledger rows captured without the DB (or previously unmappable) ----
    n_topped = 0
    if db is not None:
        nameless = []
        for row in ledger:
            if not bool(row["db_joined"]) and _s(row["captured_at"]) != run_ts:
                if not _s(row.get("movie_name")):
                    nameless.append(_s(row["ticker"]))
                    continue  # no name to map; aggregate-warned below, not once per row
                if _join_row(row, db, db_norm, warn):
                    n_topped += 1
        if nameless:
            warn(f"{len(nameless)} ledger rows have no parsed movie_name (join never "
                 f"possible without manual fix), e.g. {nameless[0]}")
        if n_topped:
            _write_csv(ledger, LEDGER_COLS, ledger_path, int_cols=LEDGER_INT_COLS)
            print(f"  topped up DB join on {n_topped} previously-captured rows")

    # ---- settlement-consistency check: a self-label must land inside the score interval
    # implied by the event's OWN strike results. Inconsistent joined rows get a fresh
    # re-join (self-heals after processing-layer fixes — e.g. the 2026-06-02 sentiment-case
    # switch); still-inconsistent labels warn every run (coverage/sentiment defect). ----
    n_rejoined = 0
    if db is not None and ledger:
        by_ev: dict[str, list[dict]] = {}
        for r in ledger:
            by_ev.setdefault(_s(r["event_ticker"]), []).append(r)
        changed = False
        for et in sorted(by_ev):
            ev_rows = by_ev[et]
            iv = implied_score_interval(ev_rows)
            score = ev_rows[0].get("score_self")
            bad_score = score is None or (isinstance(score, float) and math.isnan(score))
            if iv is None or not bool(ev_rows[0]["db_joined"]) or bad_score:
                continue
            lo, hi = iv
            if lo <= int(score) <= hi:
                continue
            if _s(ev_rows[0]["captured_at"]) != run_ts:  # just-joined rows can't differ
                before = (ev_rows[0]["total_at_close"], ev_rows[0]["fresh_at_close"])
                for r in ev_rows:
                    _join_row(r, db, db_norm, warn)
                if (ev_rows[0]["total_at_close"], ev_rows[0]["fresh_at_close"]) != before:
                    n_rejoined += len(ev_rows)
                    changed = True
                score = ev_rows[0].get("score_self")
            if score is None or (isinstance(score, float) and math.isnan(score)) or \
                    not (lo <= int(score) <= hi):
                warn(f"[label-consistency] {et}: score_self={score} outside the "
                     f"settlement-implied [{lo}, {hi}] — DB coverage/sentiment defect?")
        if changed:
            _write_csv(ledger, LEDGER_COLS, ledger_path, int_cols=LEDGER_INT_COLS)
            print(f"  re-joined {n_rejoined} ledger rows whose labels contradicted settlement")

    tickers = [_s(r["ticker"]) for r in ledger]
    assert len(tickers) == len(set(tickers)), "ledger ticker uniqueness violated"

    # ---- open-events snapshot + coverage watch ----
    for et in sorted(open_by_event):
        ms = open_by_event[et]
        name = _event_name(ms)
        slug = map_slug(name or None, db_norm) if db is not None else None
        n_rev = counts_by_slug.get(slug, 0) if slug else 0
        close_iso = _s(ms[0].get("close_time"))
        _append_csv({"run_ts": run_ts, "event_ticker": et, "movie_name": name,
                     "close_time": close_iso, "n_markets": len(ms),
                     "slug": slug or "", "n_reviews_db": (n_rev if db is not None else "")},
                    EVENTS_COLS, os.path.join(store_dir, "events_open.csv"),
                    int_cols=EVENTS_INT_COLS)
        if db is not None and n_rev == 0:
            when = (f"closes in {(_dt(close_iso) - now).total_seconds() / 86400:.1f}d"
                    if close_iso else "close time unknown")
            warn(f"[coverage] Kalshi lists '{name or et}' ({et}, {when}): "
                 f"no DB reviews — untracked or pre-embargo?")

    if db is not None:
        db.close()

    summary = {"run_ts": run_ts, "duration_s": round(time.monotonic() - t0, 1),
               "n_settled_api": len(settled), "n_ledger_before": len(captured),
               "n_new_markets": n_new_markets, "n_new_events": n_new_events,
               "n_candle_rows_added": n_candle_rows, "n_aged_out": len(aged_out),
               "n_topped_up": n_topped, "n_rejoined": n_rejoined,
               "as_of_id": (db.as_of_id if db is not None else ""),
               "db_available": db is not None, "n_warnings": len(warnings)}
    _append_csv(summary, RUNS_COLS, os.path.join(store_dir, "runs.csv"),
                int_cols=RUNS_INT_COLS)  # strictly last
    print(f"run complete in {summary['duration_s']}s: +{n_new_markets} markets / "
          f"{n_new_events} events, {n_candle_rows} candle rows, topped up {n_topped}, "
          f"{n_fetch_failed} fetch failures, {len(warnings)} warnings")
    return summary


def check_staleness(store_dir: str = STORE, *, now: datetime | None = None,
                    stale_days: float = STALE_DAYS) -> tuple[int, str]:
    """(exit_code, message) — 1 if the recorder never ran or last ran > stale_days ago."""
    now = now or datetime.now(timezone.utc)
    runs = _read_rows(os.path.join(store_dir, "runs.csv"))
    real = [r for r in runs if _s(r["run_ts"])]
    if not real:
        return 1, "recorder has NEVER run — run `python -m gates.recorder`"
    try:
        last = _dt(_s(real[-1]["run_ts"]))
    except ValueError:
        return 1, f"runs.csv last run_ts unparseable ({_s(real[-1]['run_ts'])!r}) — inspect"
    age = (now - last).total_seconds() / 86400
    if age > stale_days:
        return 1, (f"recorder STALE: last run {age:.1f}d ago (> {stale_days:g}d) — "
                   f"run `python -m gates.recorder`")
    return 0, f"recorder fresh: last run {age:.1f}d ago"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="staleness probe only (no network)")
    ap.add_argument("--dry-run", action="store_true", help="report; write nothing")
    ap.add_argument("--no-db", action="store_true", help="skip the DB join")
    ap.add_argument("--store", default=STORE, help="store dir (default gates/recorded)")
    args = ap.parse_args(argv)
    if args.check:
        code, msg = check_staleness(args.store)
        print(msg)
        return code
    run(args.store, db_factory=(None if args.no_db else _default_db_factory),
        dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
