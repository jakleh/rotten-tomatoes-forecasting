"""Driver: live scorer — C2′ p_fresh + shipped λ → ``compute_edge`` vs the live book.

Spec: ``plans/plan_live_scorer.md`` (operator-commissioned 2026-06-10). READ-ONLY
decision support: it pulls fresh reviews (read-only role), reads the PUBLIC Kalshi
book, prints per-strike trade reads, and appends the score log that doubles as the
tripwire record. **It places no orders.**

    python -m gates.live_scorer --event KXRT-DIS --snap 3     # live scoring
    python -m gates.live_scorer --verify                      # settled-movie rehearsal

Two worlds by design (deploy parity, plan pin 1): the C2 fit comes from the FROZEN
training caches (pin 653572); the target's state from a FRESH pull at run time
(pin recorded in the log). The stack's own no-trade semantics carry over verbatim
(skip:features / skip:lambda / trimmed / prior_only at n_obs=0).

Network: Kalshi reads work sandboxed; the DB pull needs sandbox-off.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(__file__), "_cache")
STORE = os.path.join(os.path.dirname(__file__), "recorded")
LOG_PATH = os.path.join(CACHE, "live_scores.csv")
STALE_HOURS = 12.0
VALID_SNAPS = (1, 2, 3, 4)                    # T-5d excluded (Gate-2 floor); C2 trained on these
VERIFY_SLUG = "the_devil_wears_prada_2"      # the gate3b hand-walk movie
VERIFY_TOL_PF = 0.01                          # fresh-pin drift tolerance vs 648979
# C2' rehearsal anchors @ frozen training caches (pin 653572) + DEV target features;
# a rebuilt training cache SHOULD fail these loudly -> re-pin after re-verifying.
VERIFY_C2_ANCHOR = 0.7156
VERIFY_FIT_MOVIES = 126
VERIFY_FIT_C = 10.0

LOG_COLS = ["run_ts", "as_of_id", "slug", "event", "snap_days", "snap_ts", "close_ts",
            "status", "n_obs", "fresh_obs", "p_shipped", "p_C2", "fit_n_movies",
            "fit_C", "strike", "bid", "ask", "p_yes", "read", "ev_cents", "mode"]


def fee_cents(price_c: float) -> float:
    return 7.0 * (price_c / 100.0) * (1 - price_c / 100.0)


def trade_read(p_yes: float, bid: float, ask: float, buffer_c: float = 0.0
               ) -> tuple[str, float]:
    """(read, EV cents/contract) crossing the live book at probability p_yes.
    bid/ask in dollars; two-sided validity is the caller's gate."""
    pc, bid_c, ask_c = p_yes * 100, bid * 100, ask * 100
    if pc > ask_c + buffer_c:
        return "BUY YES", pc - ask_c - fee_cents(ask_c)
    if pc < bid_c - buffer_c:
        no_c = 100 - bid_c
        return "BUY NO", (100 - pc) - no_c - fee_cents(no_c)
    return "no trade", float("nan")


def snap_guard(now: datetime, snap_ts: pd.Timestamp,
               stale_hours: float = STALE_HOURS) -> str:
    """'early' (REFUSE: look-ahead — the snap hasn't happened), 'stale' (warn),
    or 'ok'."""
    if now < snap_ts:
        return "early"
    if (now - snap_ts).total_seconds() / 3600.0 > stale_hours:
        return "stale"
    return "ok"


def two_sided(bid, ask) -> bool:
    """Real two-sided quote per the Candle.mid rule (dollars)."""
    return (bid is not None and ask is not None and np.isfinite(bid)
            and np.isfinite(ask) and 0.0 < bid <= ask < 1.0)


def _append_log(rows: list[dict], path: str = LOG_PATH) -> None:
    df = pd.DataFrame(rows, columns=LOG_COLS)
    exists = os.path.exists(path)
    if exists:
        with open(path) as fh:
            head = fh.readline().strip().split(",")
        assert head == LOG_COLS, (
            f"live_scores.csv header drifted from LOG_COLS — migrate the log before "
            f"appending (found {head[:4]}…)")
    df.to_csv(path, mode="a", header=not exists, index=False)


def run(event: str | None, snap_days: int, *, verify: bool = False,
        buffer_c: float = 5.0) -> list[dict]:
    from gates import db_facts as dbf
    from gates import pfresh_lib as pl
    from gates.build_gate3b import cell_status, midnight_snap
    from gates.slug_map import NAME_RE, map_slug, norm
    from rotten_tomatoes_forecasting import (
        compute_edge,
        estimate_lambda,
        extract_lambda_features,
        load_default_regressor,
    )
    from rotten_tomatoes_forecasting.features import apply_noon_shift
    from rotten_tomatoes_forecasting.pool import (
        _most_recent_resolved_slugs,
        build_a1_pool_context,
    )

    assert snap_days in VALID_SNAPS, (
        f"snap_days={snap_days} outside {VALID_SNAPS} — C2 has no calibration there "
        f"(all-zero dummies would silently price off the T-1d intercept)")
    now = datetime.now(timezone.utc)
    run_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- market side: strikes + book ------------------------------------------------
    if verify:
        cells = pd.read_csv(os.path.join(CACHE, "gate3b_cells.csv"))
        cells = cells[(cells["slug"] == VERIFY_SLUG)
                      & (cells["snap_days"] == snap_days)]
        assert len(cells), f"--verify: no gate3b cells for {VERIFY_SLUG} @ T-{snap_days}d"
        close = pd.to_datetime(cells["close_time"].iloc[0], utc=True)
        name, event = "The Devil Wears Prada 2 (verify)", "KXRT-DEV"
        strikes = [{"floor_strike": int(r["X"]), "bid": float(r["bid"]),
                    "ask": float(r["ask"])} for _, r in cells.iterrows()]
        print(f"VERIFY mode: {VERIFY_SLUG} @ T-{snap_days}d, frozen gate3b book "
              f"({len(strikes)} strikes)")
    else:
        from gates import kalshi_data as kd
        mkts = [m for m in kd.list_markets(status="open")
                if m["event_ticker"] == event]
        assert mkts, f"no OPEN markets for event {event}"
        close = pd.to_datetime(mkts[0]["close_time"], utc=True)
        assert all(m["close_time"] == mkts[0]["close_time"] for m in mkts), \
            "per-event close_time not unique — inspect before trading"
        name = next((mt.group(1).strip() for m in mkts
                     if (mt := NAME_RE.search(m.get("rules_primary") or ""))), "")
        strikes = sorted(({"floor_strike": int(m["floor_strike"]),
                           "bid": (float(m["yes_bid_dollars"])
                                   if m.get("yes_bid_dollars") is not None else np.nan),
                           "ask": (float(m["yes_ask_dollars"])
                                   if m.get("yes_ask_dollars") is not None else np.nan)}
                          for m in ms_iter(mkts)), key=lambda s: s["floor_strike"])
        print(f"{event} '{name}': {len(strikes)} open strikes, close {close}")

    snap_ts = midnight_snap(close, snap_days)
    h = (close - snap_ts).total_seconds() / 3600.0
    guard = snap_guard(now, snap_ts)
    if not verify:
        if guard == "early":
            raise SystemExit(f"REFUSING: snap {snap_ts} is in the future "
                             f"(now {now:%Y-%m-%dT%H:%MZ}) — no look-ahead scoring")
        if guard == "stale":
            print(f"  WARNING: {((now - snap_ts).total_seconds() / 3600):.1f}h past "
                  f"the snap — decision is stale relative to the trained convention")

    # ---- data side: fresh pull for target + pool; frozen training fit ---------------
    conn = dbf.connect()
    try:
        pin = dbf.as_of_id(conn)
        db_norm = {norm(s): s for s in dbf.movie_review_counts(conn, pin)}
        slug = VERIFY_SLUG if verify else map_slug(name or None, db_norm)
        assert slug is not None, f"cannot map '{name}' to a DB slug"
        mi = pd.read_csv(os.path.join(ROOT, "movies_index.csv"))
        mk = pd.read_csv(os.path.join(STORE, "markets.csv"))
        cdm = {**dict(zip(mi["Slug"], pd.to_datetime(mi["Bet Close Date"], utc=True))),
               **{s: pd.to_datetime(t, utc=True)
                  for s, t in mk.groupby("slug")["close_time"].first().items()},
               slug: close}
        pool = _most_recent_resolved_slugs(cdm, before=close, n=20, exclude_slug=slug)
        assert len(pool) == 20
        assert max(cdm[p] for p in pool) <= snap_ts, (
            "pool member closes inside (snap, close] — unsettled at decision time "
            "(gate3b M5 leakage assert, ported)")
        raw = dbf.fetch_reviews_full(conn, sorted({slug, *pool}), pin)
        act = dbf.critic_activity(conn, pin)
    finally:
        conn.close()
    cache = pd.DataFrame(raw, columns=[
        "movie_slug", "reviewer_name", "publication_name", "top_critic",
        "tomatometer_sentiment", "subjective_score", "estimated_timestamp",
        "scrape_time", "timestamp_confidence"])
    cache["estimated_timestamp"] = pd.to_datetime(cache["estimated_timestamp"], utc=True)
    cache["scrape_time"] = pd.to_datetime(cache["scrape_time"], utc=True)
    cache["tomatometer_sentiment"] = cache["tomatometer_sentiment"].str.lower()
    cache = apply_noon_shift(cache)
    print(f"fresh pull @ pin {pin}: {len(cache)} rows "
          f"({int((cache['movie_slug'] == slug).sum())} for {slug} + pool of {len(pool)})")

    ft = pl.prepare_training_frame(
        pd.read_csv(os.path.join(CACHE, "pfresh_training_features.csv")))
    train_pin = int(pd.read_csv(os.path.join(CACHE, "pfresh_meta.csv"))
                    .iloc[0]["as_of_id"])
    sub = pl.temporal_rows(ft, slug, close, snap_ts, floor=60)
    model, fit_c, fit_dev = pl.fit_binomial_glm(sub, pl.F_C2)
    print(f"C2′ temporal fit: {sub['slug'].nunique()} movies / {len(sub)} rows "
          f"(C={fit_c}, OOS dev {fit_dev:.4f}; frozen training pin {train_pin})")

    # ---- target state + estimators ---------------------------------------------------
    row = pl.c2_feature_row(cache, slug, pool, snap_days, snap_ts)
    if row is None:
        p_shipped = p_c2 = pl.prior_remaining(cache, pool, set())
        n_obs = fresh_obs = 0
        status = "prior_only"
    else:
        n_obs, fresh_obs = row["n_obs"], row["fresh_obs"]
        p_shipped = row["p_shipped"]
        p_c2 = float(model.predict_proba(
            np.array([[row[f] for f in pl.F_C2]]))[0, 1])
        status = None
    assert 0.0 <= p_c2 <= 1.0

    ctx = build_a1_pool_context(slug, cdm, cache)
    feats = (extract_lambda_features(slug, snap_days=snap_days, close_ts=close,
                                     reviews_df=cache, close_date_map=cdm,
                                     a1_context=ctx, activity_lookup=act)
             if ctx is not None else None)
    rate = total_pred = None
    if feats is not None:
        pred = estimate_lambda(load_default_regressor(), feats, snap_days=snap_days,
                               close_ts=close, hours_to_close=h)
        rate, total_pred = pred.rate_per_hour, pred.total_pred
    if status is None:
        status = cell_status(feats, rate,
                             feats["target_gap"] if feats is not None else None)
    print(f"\nstate at snap {snap_ts} (T-{snap_days}d, h={h:.0f}): obs {fresh_obs}/"
          f"{n_obs} ({(fresh_obs / n_obs):.1%})" if n_obs else
          f"\nstate at snap {snap_ts}: 0 observed reviews")
    print(f"p_fresh: shipped {p_shipped:.4f} -> C2' {p_c2:.4f} | lambda: "
          f"{'total_pred %.1f (%.4f/h)' % (total_pred, rate) if rate is not None else 'N/A'}"
          f" | STATUS: {status}")
    if status != "ok":
        print(f">>> {status.upper()} — the stack does NOT price this target at this "
              f"snap; no trades. (Logged.)")

    # ---- per-strike table -------------------------------------------------------------
    log_rows, printed = [], False
    for s in strikes:
        X, bid, ask = s["floor_strike"], s["bid"], s["ask"]
        live = two_sided(bid, ask)
        p_yes = np.nan
        read, ev = "—", np.nan
        if status == "ok" and live:
            e = compute_edge(X, float((bid + ask) / 2 * 100), fresh_obs, n_obs,
                             h, rate, p_c2)
            p_yes = float(e["p_yes"])
            read, ev = trade_read(p_yes, bid, ask, buffer_c=0.0)
            read_b, _ = trade_read(p_yes, bid, ask, buffer_c=buffer_c)
            if read != "no trade" and read_b == "no trade":
                read += " (inside 5c buffer)"
        if not printed:
            print(f"\n{'strike':>7} {'bid':>6} {'ask':>6} {'P(Yes)':>8} "
                  f"{'read':>22} {'EV c':>7}")
            printed = True
        print(f"{X:>7} {bid if np.isfinite(bid) else '—':>6} "
              f"{ask if np.isfinite(ask) else '—':>6} "
              f"{p_yes:>8.3f} {read:>22} {ev:>7.1f}" if np.isfinite(p_yes) else
              f"{X:>7} {bid if np.isfinite(bid) else '—':>6} "
              f"{ask if np.isfinite(ask) else '—':>6} {'—':>8} "
              f"{'(no book)' if not live else read:>22} {'—':>7}")
        log_rows.append({
            "run_ts": run_ts, "as_of_id": pin, "slug": slug, "event": event,
            "snap_days": snap_days, "snap_ts": snap_ts.isoformat(),
            "close_ts": close.isoformat(), "status": status, "n_obs": n_obs,
            "fresh_obs": fresh_obs, "p_shipped": round(p_shipped, 6),
            "p_C2": round(p_c2, 6), "fit_n_movies": sub["slug"].nunique(),
            "fit_C": fit_c, "strike": X, "bid": bid, "ask": ask,
            "p_yes": round(p_yes, 6) if np.isfinite(p_yes) else "",
            "read": read, "ev_cents": round(ev, 2) if np.isfinite(ev) else "",
            "mode": "verify" if verify else "live"})

    # ---- verify-mode anchors ----------------------------------------------------------
    if verify:
        stored = pd.read_csv(os.path.join(CACHE, "gate3b_cells.csv"))
        stored = stored[(stored["slug"] == VERIFY_SLUG)
                        & (stored["snap_days"] == snap_days)].iloc[0]
        drift = abs(p_shipped - stored["p_fresh_hat"])
        print(f"\nVERIFY anchors (fresh pin {pin} vs bench pin 648979):")
        print(f"  shipped p̂  recomputed {p_shipped:.4f} vs bench "
              f"{stored['p_fresh_hat']:.4f} (|drift| {drift:.4f}, tol {VERIFY_TOL_PF})")
        assert drift < VERIFY_TOL_PF, "shipped-p̂ drift exceeds tolerance — inspect pull"
        print(f"  obs state  recomputed {fresh_obs}/{n_obs} vs bench "
              f"{int(stored['obs_fresh_est'])}/{int(stored['obs_total_est'])}")
        print(f"  C2' p̂     recomputed {p_c2:.4f} vs pinned rehearsal anchor "
              f"{VERIFY_C2_ANCHOR} (fit {sub['slug'].nunique()} movies, C={fit_c} vs "
              f"pinned {VERIFY_FIT_MOVIES}/{VERIFY_FIT_C})")
        assert abs(p_c2 - VERIFY_C2_ANCHOR) < VERIFY_TOL_PF, (
            "C2' anchor drifted — a C2-side regression or a rebuilt training cache; "
            "re-verify the chain and re-pin VERIFY_C2_ANCHOR before live use")
        assert sub["slug"].nunique() == VERIFY_FIT_MOVIES and fit_c == VERIFY_FIT_C, (
            "temporal-fit anchors drifted (rebuilt training cache?) — re-pin after "
            "re-verifying")
        print("VERIFY PASSED — full chain (shipped p̂ AND C2′) rehearsed + asserted "
              "on a settled movie.")

    _append_log(log_rows)
    print(f"\nlogged {len(log_rows)} rows -> {LOG_PATH} (the tripwire record)")
    return log_rows


def ms_iter(mkts):
    """Open-market dicts with usable strikes only."""
    for m in mkts:
        fs = m.get("floor_strike")
        if fs is not None and not (isinstance(fs, float) and np.isnan(fs)):
            yield m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--event", help="Kalshi event ticker, e.g. KXRT-DIS")
    ap.add_argument("--snap", type=int, default=3, choices=list(VALID_SNAPS),
                    help="snap_days (default 3)")
    ap.add_argument("--verify", action="store_true",
                    help="settled-movie rehearsal (frozen gate3b book; no live API)")
    args = ap.parse_args(argv)
    if not args.verify and not args.event:
        ap.error("--event required unless --verify")
    run(args.event, args.snap, verify=args.verify)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
