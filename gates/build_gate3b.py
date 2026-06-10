"""Driver: Gate 3b — the deployable-stack verdict inputs (real 0.2.0 estimator on a
re-derived ET-midnight cell grid). Spec: ``plans/plan_gate3b.md`` (twice adversarially
reviewed; every pre-registration locked there — this driver only EXECUTES it).

LOCK CHAIN (enforced by stage order; nothing downstream of the estimator can revise
anything upstream):
  STAGE 1  book-only LOCF measurement at the midnight-ET snaps, N in {1..4}, from the
           committed ``gates/recorded/`` store -> ct flags (contested AND spread<=10c).
  STAGE 2  provisional clean flags from the cached ``density.csv`` (original 16 movies;
           rule (a) exact at the midnight snap via cached first_scrape, rules (b)/(c)
           carried from the close-24N rows — checkpoint print only).
  STAGE 3  the one guard DB step — pin selection (data-readiness gate per movie at the
           candidate pins) + true guard via ``db_facts.snap_density`` at the midnight
           snaps -> final clean counts, >=8 floor per primary snap, T-3d retention
           check -> GRID LOCKED (Option-C fallback = STOP, never auto-built).
  STAGE 4  A1-pool review cache (sentiment lowercased + noon-shift applied ONCE at
           ingest; ``subjective_score`` + ``scrape_time`` columns) + full-universe
           ``activity_lookup`` + per-target pool compositions — same pin.
  STAGE 5  in-grid oracle (pure headline + lagged context) on the locked grid.
  STAGE 6  estimator pass: ``extract_lambda_features`` -> ``estimate_lambda`` (shipped
           artifact) + ``estimate_p_fresh`` -> ``compute_edge``; reason codes
           ``skip:features`` / ``skip:lambda``; 15d gap-cap as ``trimmed``;
           never clip-and-trade.

Network I/O (DB) lives here — run sandbox-disabled. The notebook
(``notebooks/gate3b_deployable.ipynb``) is cache-only and holds the citable numbers.

Outputs (gates/_cache/, gitignored):
  gate3b_grid.csv       every (ticker, snap) book row + flags (full audit trail)
  gate3b_cells.csv      locked cells x oracle (pure+lagged) x estimator outputs
  gate3b_a1_cache.csv   pinned union review cache (lowercased, noon-shifted = the
                        estimator view; oracle placement is invariant to the shift,
                        precondition asserted at ingest)
  gate3b_activity.csv   full-universe per-critic distinct-movie counts at the pin
  gate3b_pools.csv      per-target A1-pool composition (a deploy decision, disclosed)
  gate3b_readiness.csv  data-readiness table per movie per candidate pin
  gate3b_meta.csv       pin + lock-chain counts (floor, retention, coverage inputs)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "_cache")
STORE = os.path.join(os.path.dirname(__file__), "recorded")

SNAP_DAYS = [1, 2, 3, 4]            # T-5d excluded per the Gate-2 floor result
PRIMARY_SNAPS = [2, 3, 4]           # T-1d secondary (never pooled), per Gate-2 precedent
FLOOR_MOVIES = 8                    # >=8 ct AND clean movies per primary snap
RETENTION_MIN = 0.50                # T-3d must retain >=50% of Gate-2's ct AND clean markets
# Pin candidates, ascending: 2026-06-09 session pin / recorder seed pin / 2026-06-10 heal
# pin (the runs.csv row with n_rejoined=77). Rule: smallest at which every movie that
# passes the readiness gate at the heal pin also passes (one pin, never mixed).
PIN_CANDIDATES = [648979, 649484, 652074]
# Expected readiness-check failures at the heal pin (plan_gate3b "Empirical status",
# operator-verified 2026-06-10). A different fail set = a NEW data incident -> the
# cohort decision must go back to the operator, so the driver hard-asserts.
EXPECTED_CHECK_FAIL = {"animal_farm_2025", "backrooms", "in_the_grey"}
# Operator call 2026-06-10: power_ballad's post-fix label matches settlement but its
# history is known-incomplete (no rows before 2026-05-01; BACKLOG 1.9) -> data_not_ready.
OPERATOR_NOT_READY = {"power_ballad"}
GAP_CAP_DAYS = 15.0                 # CLAUDE.md orchestrator deployment rule
RIDGE_FIT_DATE = "2026-04-19"       # shipped artifact fit date (metadata.fit_date)


# ---------------------------------------------------------------------------------
# Pure helpers (no I/O) — unit-tested in tests/test_gate3b.py
# ---------------------------------------------------------------------------------

def midnight_snap(close_ts: pd.Timestamp, snap_days: int) -> pd.Timestamp:
    """The library's own snap: midnight ET on close-N days (CLAUDE.md convention).

    Computed through ``features.midnight_et_of_close`` so the grid and the estimator
    share one code path (bit-identical snap_time).
    """
    from rotten_tomatoes_forecasting.features import midnight_et_of_close

    return midnight_et_of_close(close_ts) - pd.Timedelta(days=snap_days)


def book_at(g: pd.DataFrame, snap_epoch: int) -> dict | None:
    """LOCF book state at ``snap_epoch`` from one ticker's candle rows.

    Last candle with ``ts <= snap_epoch`` carries the standing book (arena probe:
    P(state identical across silent gaps) = 1.00000); a null/one-sided book on that
    candle kills the carried quote (mid is NaN by construction in the store). None
    when the market has no candle at/before the snap (not yet listed/quoted).
    """
    sub = g[g["ts"] <= snap_epoch]
    if not len(sub):
        return None
    r = sub.loc[sub["ts"].idxmax()]
    return {
        "bid": float(r["yes_bid"]) if pd.notna(r["yes_bid"]) else np.nan,
        "ask": float(r["yes_ask"]) if pd.notna(r["yes_ask"]) else np.nan,
        "mid": float(r["mid"]) if pd.notna(r["mid"]) else np.nan,
        "stale_min": (snap_epoch - int(r["ts"])) / 60.0,
    }


def is_ct(bid: float, ask: float, mid: float) -> bool:
    """Contested AND tight: live two-sided mid in (0.2, 0.8), spread <= 10 cents.

    Spread quantized to integer cents before the comparison — Kalshi quotes are
    cent-quantized and float dust (0.76-0.66 = 0.10000000000000009) must not drop a
    true-10c book. (The Gate-2 grid used the raw float ``<=``; this grid is re-derived
    anyway, so the cleaner rule applies — disclosed here.)
    """
    if not np.isfinite(mid):
        return False
    spread_c = int(round((ask - bid) * 100))
    return (0.2 < mid < 0.8) and spread_c <= 10


def guard_clean(first_scrape, snap_ts, n_d_near_snap: int, n_remaining: int,
                n_last_day_d: int) -> bool:
    """The pre-registered oracle-clean cohort guard, re-evaluated at a midnight snap.

    (a) live-tracked-through-snap: first scrape_time <= snap_ts;
    (b) snap-boundary-clean: d-rows within +/-1d of the snap <= max(2, 10% of remaining);
    (c) M2 close-clean: last-day d-rows <= 2.
    """
    if first_scrape is None or pd.isna(first_scrape):
        return False
    return (
        bool(first_scrape <= snap_ts)
        and n_d_near_snap <= max(2, 0.10 * n_remaining)
        and n_last_day_d <= 2
    )


def readiness_pass(score_at_pin: int | None, interval: tuple[int, int] | None,
                   lastday_at_pin: int | None, ledger_lastday: int | None) -> bool:
    """Data-readiness gate for one movie at one pin (plan_gate3b, pre-registered):
    the recomputed case-insensitive self-label must land INSIDE the settlement-implied
    interval from the event's own strike results, with the pin's last-day d-count
    matching the ledger's (the build pin must reproduce the ledger's view)."""
    if score_at_pin is None or interval is None:
        return False
    lo, hi = interval
    if not (lo <= score_at_pin <= hi):
        return False
    if ledger_lastday is None or pd.isna(ledger_lastday):
        return False
    return lastday_at_pin == int(ledger_lastday)


def choose_pin(fail_sets: dict[int, set[str]], candidates: list[int]) -> int:
    """Smallest candidate pin whose readiness fail set equals the heal pin's.

    The heal pin (largest candidate) is the reference: a movie failing there is
    data_not_ready regardless; a movie passing there must also pass at the chosen pin
    (else the smaller pin would silently shrink the operator-confirmed cohort), and a
    movie passing ONLY at the smaller pin must not sneak in (fail sets must MATCH).
    """
    ref = fail_sets[candidates[-1]]
    for pin in candidates:
        if fail_sets[pin] == ref:
            return pin
    raise RuntimeError(f"no candidate pin matches the heal-pin fail set {sorted(ref)}")


def cell_status(features: dict | None, rate_per_hour: float | None,
                target_gap: float | None, gap_cap: float = GAP_CAP_DAYS) -> str:
    """Pre-registered estimator-cell classification (plan_gate3b wiring section).

    skip:features — the stack's own skip rules say no-trade (features None);
    skip:lambda   — the lambda path errors (negative Ridge total_pred would make
                    compute_edge raise per edge.py:66-67; never clip-and-trade);
    trimmed       — priceable but excluded by the 15d deployment gap-cap (headline
                    tables apply it; reported distinct from skip:*);
    ok            — priced and tradeable.
    """
    if features is None:
        return "skip:features"
    if rate_per_hour is None or rate_per_hour < 0:
        return "skip:lambda"
    if target_gap is not None and target_gap > gap_cap:
        return "trimmed"
    return "ok"


# ---------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------

def _load_store() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ledger + concatenated per-event candles from gates/recorded/."""
    mk = pd.read_csv(os.path.join(STORE, "markets.csv"))
    assert mk["ticker"].is_unique, "ledger ticker uniqueness violated"
    n_close = mk.groupby("event_ticker")["close_time"].nunique()
    assert (n_close == 1).all(), f"per-event close_time not unique: {n_close[n_close > 1]}"
    parts = []
    for et in sorted(mk["event_ticker"].unique()):
        parts.append(pd.read_csv(os.path.join(STORE, "candles", f"{et}.csv.gz")))
    cd = pd.concat(parts, ignore_index=True)
    return mk, cd


def build() -> pd.DataFrame:
    from gates import db_facts as dbf
    from gates.oracle import oracle_inputs
    from gates.recorder import implied_score_interval
    from rotten_tomatoes_forecasting import (
        compute_edge,
        estimate_lambda,
        estimate_p_fresh,
        extract_lambda_features,
        load_default_regressor,
    )
    from rotten_tomatoes_forecasting.features import apply_noon_shift
    from rotten_tomatoes_forecasting.pool import (
        _most_recent_resolved_slugs,
        build_a1_pool_context,
    )

    mk, cd = _load_store()
    mk["close_dt"] = pd.to_datetime(mk["close_time"], utc=True)
    slugs_all = sorted(mk["slug"].unique())
    assert len(slugs_all) == 19, f"expected the 19-movie cohort, got {len(slugs_all)}"
    close_by_slug = mk.groupby("slug")["close_dt"].first().to_dict()
    ext_slugs = set(
        mk.loc[mk["event_ticker"].isin(["KXRT-MAS", "KXRT-POW", "KXRT-SCA"]), "slug"]
    )
    g2_slugs = set(slugs_all) - ext_slugs
    assert len(g2_slugs) == 16 and len(ext_slugs) == 3

    # ---- STAGE 1: book-only LOCF at the midnight-ET snaps -> ct flags --------------
    print("=== STAGE 1: book measurement at midnight-ET snaps (recorded/ candles) ===")
    snap_ts_by = {
        (slug, n): midnight_snap(close_by_slug[slug], n)
        for slug in slugs_all for n in SNAP_DAYS
    }
    grid_rows = []
    for tk, g in cd.groupby("ticker"):
        row = mk.loc[mk["ticker"] == tk].iloc[0]
        slug = row["slug"]
        for n in SNAP_DAYS:
            snap_ts = snap_ts_by[(slug, n)]
            bk = book_at(g, int(snap_ts.timestamp()))
            if bk is None:
                bk = {"bid": np.nan, "ask": np.nan, "mid": np.nan, "stale_min": np.nan}
            grid_rows.append({
                "ticker": tk, "event_ticker": row["event_ticker"], "slug": slug,
                "snap_days": n, "snap_ts": snap_ts.isoformat(),
                "close_time": row["close_time"], "floor_strike": row["floor_strike"],
                "result": row["result"], **bk,
                "ct": is_ct(bk["bid"], bk["ask"], bk["mid"]),
            })
    grid = pd.DataFrame(grid_rows)
    grid["y"] = (grid["result"] == "yes").astype(int)
    print(grid[grid["ct"]].groupby("snap_days")
          .agg(ct_markets=("ticker", "size"), ct_movies=("slug", "nunique")).to_string())

    # ---- STAGE 2: provisional clean flags (cached density.csv; checkpoint only) ----
    print("\n=== STAGE 2: provisional guard from cached density.csv (16 movies; "
          "rules (b)/(c) carried from close-24N rows — checkpoint, not the lock) ===")
    de = pd.read_csv(os.path.join(CACHE, "density.csv"))
    de["first_scrape_dt"] = pd.to_datetime(de["first_scrape"], utc=True, format="ISO8601")
    prov = {}
    for (slug, n), r in de.set_index(["slug", "snap_days"]).iterrows():
        if n not in SNAP_DAYS:
            continue
        prov[(slug, n)] = guard_clean(
            r["first_scrape_dt"], snap_ts_by[(slug, n)],
            int(r["n_d_near_snap"]), int(r["n_remaining"]), int(r["n_last_day_d"]))
    gp = grid[grid["ct"]].copy()
    gp["prov_clean"] = [prov.get((s, n), False) for s, n in zip(gp["slug"], gp["snap_days"])]
    print(gp[gp["prov_clean"]].groupby("snap_days")
          .agg(markets=("ticker", "size"), movies=("slug", "nunique")).to_string())
    print("(plan's review-agent provisional: T-4d 21/9, T-3d 22/12, T-2d 11/9, T-1d 7/7)")

    # ---- STAGE 3: pin selection + true DB guard + floor -> GRID LOCK ---------------
    print("\n=== STAGE 3: data-readiness pin selection + true guard (DB) ===")
    iv_by_slug = {}
    for et, ev in mk.groupby("event_ticker"):
        iv_by_slug[ev["slug"].iloc[0]] = implied_score_interval(ev.to_dict("records"))
    ledger_lastday = mk.groupby("slug")["lastday_daylevel"].first().to_dict()

    conn = dbf.connect()
    try:
        ready_rows, fail_sets = [], {}
        for pin in PIN_CANDIDATES:
            fails = set()
            for slug in slugs_all:
                close = close_by_slug[slug].to_pydatetime()
                # self-label = case-insensitive observed state AT CLOSE (est <= close),
                # exactly the recorder's join (movie_coverage's fresh/total are
                # all-time counts — post-close rows would poison the label)
                fresh, total = dbf.observed_state(conn, slug, close, pin)
                cov = dbf.movie_coverage(conn, slug, close, pin)
                score = round(fresh / total * 100) if total else None
                ok = readiness_pass(score, iv_by_slug[slug],
                                    int(cov["n_last_day_d"]), ledger_lastday[slug])
                if not ok:
                    fails.add(slug)
                ready_rows.append({
                    "pin": pin, "slug": slug, "total": total,
                    "fresh": fresh, "score_at_pin": score,
                    "implied_lo": iv_by_slug[slug][0], "implied_hi": iv_by_slug[slug][1],
                    "lastday_d_at_pin": int(cov["n_last_day_d"]),
                    "ledger_lastday": ledger_lastday[slug], "pass": ok,
                })
            fail_sets[pin] = fails
            print(f"  pin {pin}: readiness FAIL = {sorted(fails) or 'none'}")
        assert fail_sets[PIN_CANDIDATES[-1]] == EXPECTED_CHECK_FAIL, (
            f"heal-pin fail set {sorted(fail_sets[PIN_CANDIDATES[-1]])} != the "
            f"operator-verified {sorted(EXPECTED_CHECK_FAIL)} — NEW data incident; "
            f"cohort decision goes back to the operator")
        pin = choose_pin(fail_sets, PIN_CANDIDATES)
        data_not_ready = fail_sets[pin] | OPERATOR_NOT_READY
        print(f"  BUILD PIN locked: {pin} | data_not_ready = {sorted(data_not_ready)} "
              f"(check-fail {sorted(fail_sets[pin])} + operator {sorted(OPERATOR_NOT_READY)})")

        guard_rows = []
        for slug in slugs_all:
            close = close_by_slug[slug].to_pydatetime()
            fs = dbf.first_scrape(conn, slug, pin)
            cov = dbf.movie_coverage(conn, slug, close, pin)
            for n in SNAP_DAYS:
                snap_ts = snap_ts_by[(slug, n)].to_pydatetime()
                d = dbf.snap_density(conn, slug, close, snap_ts, pin)
                guard_rows.append({
                    "slug": slug, "snap_days": n,
                    "oracle_clean": guard_clean(
                        fs, snap_ts, int(d["n_d_near_snap"]),
                        int(d["n_remaining"]), int(cov["n_last_day_d"])),
                    "n_remaining_db": int(d["n_remaining"]),
                    "n_d_near_snap": int(d["n_d_near_snap"]),
                    "n_last_day_d": int(cov["n_last_day_d"]),
                    "first_scrape": fs.isoformat() if fs is not None else None,
                })
        gu = pd.DataFrame(guard_rows)
        clean_map = gu.set_index(["slug", "snap_days"])["oracle_clean"].to_dict()

        grid["oracle_clean"] = [bool(clean_map.get((s, n), False))
                                for s, n in zip(grid["slug"], grid["snap_days"])]
        grid["data_ready"] = ~grid["slug"].isin(data_not_ready)
        grid["cell"] = grid["ct"] & grid["oracle_clean"] & grid["data_ready"]

        locked = grid[grid["cell"]]
        counts = locked.groupby("snap_days").agg(
            markets=("ticker", "size"), movies=("slug", "nunique"))
        print("\n  LOCKED grid (ct AND oracle-clean AND data-ready):")
        print(counts.to_string())
        print("  excluded ct cells:",
              "guard-dirty", grid[grid["ct"] & ~grid["oracle_clean"]]
              .groupby("snap_days").size().to_dict(),
              "| data_not_ready (guard-clean)",
              grid[grid["ct"] & grid["oracle_clean"] & ~grid["data_ready"]]
              .groupby("snap_days").size().to_dict())

        # floor per primary snap (ct AND clean movie count)
        floor_fail = [n for n in PRIMARY_SNAPS
                      if counts["movies"].get(n, 0) < FLOOR_MOVIES]
        # T-3d retention vs the Gate-2 grid (guard-clean basis, 16-movie numerator —
        # the review-I6 definition; readiness-applied variant disclosed alongside)
        g2p = pd.read_csv(os.path.join(CACHE, "gate2_cells_20260609_preexclusion.csv"))
        g2_t3 = g2p[(g2p["snap"] == "3d") & (g2p["mode"] == "pure")]
        t3 = grid[(grid["snap_days"] == 3) & grid["ct"] & grid["oracle_clean"]]
        n_num = int(t3["slug"].isin(g2_slugs).sum())
        n_num_ready = int((t3["slug"].isin(g2_slugs) & t3["data_ready"]).sum())
        retention = n_num / len(g2_t3)
        print(f"  T-3d retention vs Gate-2 ct-clean grid: {n_num}/{len(g2_t3)} = "
              f"{retention:.0%} (readiness-applied: {n_num_ready}/{len(g2_t3)})")
        if 3 in floor_fail or retention < RETENTION_MIN:
            raise RuntimeError(
                "PRE-REGISTERED FALLBACK TRIGGERED (T-3d floor fail or retention "
                f"<{RETENTION_MIN:.0%}) -> Option C (midnight features + close-24N "
                "book, shared 'deploy handicap'); STOP — operator decision, "
                "never auto-built")
        dropped = [n for n in floor_fail]
        if dropped:
            print(f"  snaps under the >={FLOOR_MOVIES}-movie floor, EXCLUDED: "
                  f"{['T-%dd' % n for n in dropped]}")
        primary_snaps = [n for n in PRIMARY_SNAPS if n not in dropped]
        print(f"  GRID LOCKED: primary snaps {['T-%dd' % n for n in primary_snaps]} "
              f"+ T-1d secondary")

        # ---- STAGE 4: A1-pool review cache + activity lookup (same pin) ------------
        print("\n=== STAGE 4: A1-pool review cache + activity_lookup (DB, pinned) ===")
        mi = pd.read_csv("movies_index.csv")
        mi_close = dict(zip(mi["Slug"],
                            pd.to_datetime(mi["Bet Close Date"], utc=True)))
        close_date_map = {**mi_close, **close_by_slug}   # ledger wins on conflict
        n_conflict = sum(1 for s in mi_close if s in close_by_slug
                         and mi_close[s] != close_by_slug[s])
        print(f"  close_date_map: {len(close_date_map)} slugs "
              f"({len(mi_close)} index + {len(close_by_slug)} ledger, "
              f"{n_conflict} conflicts -> ledger)")

        pool_rows = []
        for slug in slugs_all:
            pool = _most_recent_resolved_slugs(
                close_date_map, before=close_date_map[slug], n=20, exclude_slug=slug)
            assert len(pool) == 20, f"{slug}: A1 pool has {len(pool)} members"
            for rank, p in enumerate(pool):
                pool_rows.append({"target": slug, "rank": rank, "pool_slug": p,
                                  "pool_close": close_date_map[p].isoformat()})
        pools = pd.DataFrame(pool_rows)
        union_slugs = sorted(set(slugs_all) | set(pools["pool_slug"]))
        print(f"  union pull: {len(union_slugs)} slugs "
              f"(19 targets + {len(set(pools['pool_slug']) - set(slugs_all))} pool-only)")

        # pool-leakage assert: no pool member's close inside (snap, target_close]
        for slug in slugs_all:
            pool_closes = pools.loc[pools["target"] == slug, "pool_close"]
            max_pool_close = pd.to_datetime(pool_closes, utc=True, format="ISO8601").max()
            for n in SNAP_DAYS:
                assert max_pool_close <= snap_ts_by[(slug, n)], (
                    f"{slug} T-{n}d: pool member closes after the snap "
                    f"({max_pool_close} > {snap_ts_by[(slug, n)]}) — membership leakage")

        raw = dbf.fetch_reviews_full(conn, union_slugs, pin)
        cache = pd.DataFrame(raw, columns=[
            "movie_slug", "reviewer_name", "publication_name", "top_critic",
            "tomatometer_sentiment", "subjective_score", "estimated_timestamp",
            "scrape_time", "timestamp_confidence"])
        cache["estimated_timestamp"] = pd.to_datetime(
            cache["estimated_timestamp"], utc=True)
        cache["scrape_time"] = pd.to_datetime(cache["scrape_time"], utc=True)
        cache["tomatometer_sentiment"] = cache["tomatometer_sentiment"].str.lower()
        # Two views, one file (plan: noon-shift is ESTIMATOR-SIDE ONLY; "the oracle
        # keeps its own placement rules"): est_raw = the DB value (oracle view, same
        # convention as Gate-2's reviews_cohort.csv); estimated_timestamp = noon-
        # shifted (the estimator/library view, applied ONCE here at ingest, blind
        # +12h on d-rows exactly as the shipped fit's ingest did — relative-"Nd"
        # d-rows carry the scrape's time-of-day, so the shift is NOT midnight-only).
        cache["est_raw"] = cache["estimated_timestamp"]
        cache = apply_noon_shift(cache)
        d_raw = cache.loc[cache["timestamp_confidence"] == "d", "est_raw"]
        n_nonmid = int((d_raw.dt.time != datetime.min.time()).sum())
        cache["as_of_id"] = pin
        print(f"  cache: {len(cache)} rows / {cache['movie_slug'].nunique()} movies "
              f"(sentiment lowercased; d-rows noon-shifted in estimator view; "
              f"{n_nonmid}/{len(d_raw)} d-rows non-midnight in raw — relative-'Nd' "
              f"timestamps, oracle crowd-forward handles them)")

        act = dbf.critic_activity(conn, pin)
        print(f"  activity_lookup: {len(act)} critics (full universe at pin)")
    finally:
        conn.close()

    # ---- STAGE 5: in-grid oracle on the locked grid ---------------------------------
    print("\n=== STAGE 5: in-grid oracle at the midnight snaps ===")
    oracle_by = {}
    for slug in sorted(locked["slug"].unique()):
        # oracle view: RAW publication timestamps (its own placement rules)
        mrv = (cache[cache["movie_slug"] == slug]
               .assign(estimated_timestamp=lambda d: d["est_raw"]))
        for n in sorted(locked.loc[locked["slug"] == slug, "snap_days"].unique()):
            close, snap_ts = close_by_slug[slug], snap_ts_by[(slug, n)]
            o_pure = oracle_inputs(mrv, close, snap_ts, mode="pure")
            o_lag = oracle_inputs(mrv, close, snap_ts, mode="lagged")
            assert o_pure.total_obs + o_pure.n_remaining == o_pure.terminal_total
            oracle_by[(slug, n)] = (o_pure, o_lag)
    print(f"  oracle inputs for {len(oracle_by)} (movie, snap) pairs "
          f"(invariant observed+remaining==terminal holds)")

    # ---- STAGE 6: estimator pass -----------------------------------------------------
    print("\n=== STAGE 6: estimator pass (shipped 0.2.0 artifact) ===")
    reg = load_default_regressor()
    # in_ridge_fit: artifact stores no slug list — membership RE-DERIVED from
    # movies_index closes < fit date, cross-checked against metadata.cohort_size.
    n_prefit = int((pd.to_datetime(mi["Bet Close Date"], utc=True)
                    < pd.Timestamp(RIDGE_FIT_DATE, tz="UTC")).sum())
    assert n_prefit == reg.metadata.cohort_size == 144, (
        f"re-derived fit-cohort size {n_prefit} != artifact metadata "
        f"{reg.metadata.cohort_size}")
    in_fit = {s for s, c in mi_close.items()
              if c < pd.Timestamp(RIDGE_FIT_DATE, tz="UTC")}

    ctx_by = {}
    for slug in sorted(locked["slug"].unique()):
        ctx = build_a1_pool_context(slug, close_date_map, cache)
        assert ctx is not None, f"{slug}: A1 context unexpectedly None"
        ctx_by[slug] = ctx

    est_by = {}
    for (slug, n), (o_pure, o_lag) in oracle_by.items():
        close, snap_ts = close_by_slug[slug], snap_ts_by[(slug, n)]
        h = (close - snap_ts).total_seconds() / 3600.0
        ctx = ctx_by[slug]
        feats = extract_lambda_features(
            slug, snap_days=n, close_ts=close, reviews_df=cache,
            close_date_map=close_date_map, a1_context=ctx, activity_lookup=act)

        tr = cache[cache["movie_slug"] == slug]
        obs = tr[tr["estimated_timestamp"] < snap_ts]
        assert not len(obs) or obs["estimated_timestamp"].max() < snap_ts
        total, fresh = len(obs), int((obs["tomatometer_sentiment"] == "positive").sum())
        observed_critics = set(obs["reviewer_name"])

        p_hat = estimate_p_fresh(cache, ctx.training_slugs, observed_critics,
                                 fresh, total)
        assert 0.0 <= p_hat <= 1.0
        cache_le = cache[cache["estimated_timestamp"] <= snap_ts]
        p_hat_le = estimate_p_fresh(cache_le, ctx.training_slugs, observed_critics,
                                    fresh, total)

        pool_rows_t = cache[cache["movie_slug"].isin(ctx.training_slugs)]
        tail_frac = float((pool_rows_t["estimated_timestamp"] > snap_ts).mean())

        rate = total_pred = phase1 = None
        if feats is not None:
            pred = estimate_lambda(reg, feats, snap_days=n, close_ts=close,
                                   hours_to_close=h)
            rate, total_pred, phase1 = (pred.rate_per_hour, pred.total_pred,
                                        pred.phase1_pred)
        status = cell_status(feats, rate,
                             feats["target_gap"] if feats is not None else None)

        realized_conv = int(((tr["estimated_timestamp"] >= snap_ts)
                             & (tr["estimated_timestamp"] <= close)).sum())
        est_by[(slug, n)] = {
            "h": h, "obs_total_est": total, "obs_fresh_est": fresh,
            "p_fresh_hat": p_hat, "p_fresh_hat_le_snap": p_hat_le,
            "pool_tail_frac": tail_frac, "status": status,
            "target_gap": feats["target_gap"] if feats is not None else np.nan,
            "rate_hat": rate if rate is not None else np.nan,
            "total_pred": total_pred if total_pred is not None else np.nan,
            "phase1_pred": phase1 if phase1 is not None else np.nan,
            "realized_est_conv": realized_conv,
            "n_pool_rows": len(pool_rows_t),
        }

    cells = []
    for _, r in locked.iterrows():
        slug, n, tk = r["slug"], int(r["snap_days"]), r["ticker"]
        o_pure, o_lag = oracle_by[(slug, n)]
        e = est_by[(slug, n)]
        X, mid_c = int(r["floor_strike"]), float(r["mid"]) * 100
        eo = compute_edge(X, mid_c, o_pure.fresh_obs, o_pure.total_obs,
                          o_pure.t_rem_hours, o_pure.lambda_rate, o_pure.p_fresh)
        el = compute_edge(X, mid_c, o_lag.fresh_obs, o_lag.total_obs,
                          o_lag.t_rem_hours, o_lag.lambda_rate, o_lag.p_fresh)
        p_est = np.nan
        if e["status"] in ("ok", "trimmed"):
            # rate>=0 guaranteed by cell_status; compute_edge cannot raise here
            ee = compute_edge(X, mid_c, e["obs_fresh_est"], e["obs_total_est"],
                              e["h"], e["rate_hat"], e["p_fresh_hat"])
            assert abs(ee["expected_reviews"] - e["total_pred"]) <= 1e-9 * max(
                1.0, abs(e["total_pred"])), "mu != total_pred (hours_to_close mixup)"
            p_est = float(ee["p_yes"])
        cells.append({
            "ticker": tk, "slug": slug, "snap_days": n, "snap_ts": r["snap_ts"],
            "close_time": r["close_time"], "X": X, "y": int(r["y"]),
            "mid": float(r["mid"]), "bid": float(r["bid"]), "ask": float(r["ask"]),
            "spread": float(r["ask"]) - float(r["bid"]),
            "stale_min": float(r["stale_min"]),
            # oracle (pure = headline ceiling; lagged = context)
            "p_oracle": float(eo["p_yes"]), "p_oracle_lagged": float(el["p_yes"]),
            "lam_oracle": o_pure.lambda_rate, "p_fresh_oracle": o_pure.p_fresh,
            "n_rem_oracle": o_pure.n_remaining, "obs_total_oracle": o_pure.total_obs,
            "obs_fresh_oracle": o_pure.fresh_obs, "t_rem_h": o_pure.t_rem_hours,
            # estimator
            "status": e["status"], "p_est": p_est,
            "h": e["h"], "obs_total_est": e["obs_total_est"],
            "obs_fresh_est": e["obs_fresh_est"],
            "rate_hat": e["rate_hat"], "total_pred": e["total_pred"],
            "phase1_pred": e["phase1_pred"], "p_fresh_hat": e["p_fresh_hat"],
            "p_fresh_hat_le_snap": e["p_fresh_hat_le_snap"],
            "target_gap": e["target_gap"],
            "realized_est_conv": e["realized_est_conv"],
            "pool_tail_frac": e["pool_tail_frac"], "n_pool_rows": e["n_pool_rows"],
            # audit axes (3a band conventions: m on oracle n_remaining; delta excludes
            # the inert 0.5 sentinel at n_rem==0)
            "m_hat": (e["total_pred"] / o_pure.n_remaining
                      if e["total_pred"] is not None and not pd.isna(e["total_pred"])
                      and o_pure.n_remaining > 0 else np.nan),
            "delta_hat": (e["p_fresh_hat"] - o_pure.p_fresh
                          if o_pure.n_remaining > 0 else np.nan),
            # flags
            "oos_post_gate2": slug in ext_slugs, "in_ridge_fit": slug in in_fit,
            "primary": n in primary_snaps,   # floor-surviving primary snap (1d=False)
            "as_of_id": pin,
        })
    cdf = pd.DataFrame(cells)
    print(cdf.groupby(["snap_days", "status"]).size().unstack(fill_value=0).to_string())

    # ---- persist ---------------------------------------------------------------------
    grid.to_csv(os.path.join(CACHE, "gate3b_grid.csv"), index=False)
    cdf.to_csv(os.path.join(CACHE, "gate3b_cells.csv"), index=False)
    cache.to_csv(os.path.join(CACHE, "gate3b_a1_cache.csv"), index=False)
    pd.DataFrame(sorted(act.items()), columns=["reviewer_name", "n_movies"]).to_csv(
        os.path.join(CACHE, "gate3b_activity.csv"), index=False)
    pools.to_csv(os.path.join(CACHE, "gate3b_pools.csv"), index=False)
    pd.DataFrame(ready_rows).to_csv(os.path.join(CACHE, "gate3b_readiness.csv"),
                                    index=False)
    meta = {
        "as_of_id": pin, "n_cohort": len(slugs_all),
        "data_not_ready": ";".join(sorted(data_not_ready)),
        "check_fail_at_pin": ";".join(sorted(fail_sets[pin])),
        "primary_snaps": ";".join(str(n) for n in primary_snaps),
        "retention_t3_num": n_num, "retention_t3_num_ready": n_num_ready,
        "retention_t3_den": len(g2_t3), "floor_movies": FLOOR_MOVIES,
        "gap_cap_days": GAP_CAP_DAYS, "ridge_fit_date": RIDGE_FIT_DATE,
        "n_cache_rows": len(cache), "n_union_slugs": len(union_slugs),
        "n_activity_critics": len(act), "n_close_date_map": len(close_date_map),
        "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    pd.DataFrame([meta]).to_csv(os.path.join(CACHE, "gate3b_meta.csv"), index=False)
    print(f"\ncached: gate3b_grid ({len(grid)}), gate3b_cells ({len(cdf)}), "
          f"a1_cache ({len(cache)}), pools ({len(pools)}), readiness "
          f"({len(ready_rows)}), activity ({len(act)}), meta -> gates/_cache/")
    return cdf


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build()
