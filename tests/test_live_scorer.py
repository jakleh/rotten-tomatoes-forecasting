"""Tests for the live scorer's pure helpers + the shared C2′ feature-row builder.

No network/DB. The load-bearing checks: trade-read/EV math, the look-ahead snap
guard, two-sided book validity, F_C2 ordering parity, and c2_feature_row /
prepare_training_frame reproducing the bench notebook's formulas on fixtures.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from gates import pfresh_lib as pl
from gates.live_scorer import fee_cents, snap_guard, trade_read, two_sided


# ---- trade read ---------------------------------------------------------------------

def test_trade_read_sides_and_ev():
    read, ev = trade_read(0.90, 0.70, 0.75)           # p 90 > ask 75 -> YES
    assert read == "BUY YES"
    assert ev == pytest.approx(90 - 75 - fee_cents(75))
    read, ev = trade_read(0.50, 0.70, 0.75)           # p 50 < bid 70 -> NO
    assert read == "BUY NO"
    # NO EV: (100-p) - (100-bid) - fee(100-bid)
    assert ev == pytest.approx((100 - 50) - (100 - 70) - fee_cents(100 - 70))
    read, ev = trade_read(0.72, 0.70, 0.75)           # inside the spread
    assert read == "no trade" and np.isnan(ev)


def test_trade_read_buffer():
    read, _ = trade_read(0.78, 0.70, 0.75, buffer_c=5.0)   # 78 < 75+5 -> no trade
    assert read == "no trade"
    read, _ = trade_read(0.82, 0.70, 0.75, buffer_c=5.0)
    assert read == "BUY YES"


# ---- guards -------------------------------------------------------------------------

def test_snap_guard():
    snap = pd.Timestamp("2026-06-12T04:00:00Z")
    assert snap_guard(datetime(2026, 6, 11, 23, 0, tzinfo=timezone.utc), snap) == "early"
    assert snap_guard(datetime(2026, 6, 12, 4, 30, tzinfo=timezone.utc), snap) == "ok"
    assert snap_guard(datetime(2026, 6, 13, 4, 30, tzinfo=timezone.utc), snap) == "stale"


def test_two_sided():
    assert two_sided(0.40, 0.44)
    assert not two_sided(None, 0.44)
    assert not two_sided(np.nan, 0.44)
    assert not two_sided(0.0, 0.44)                   # degenerate zero bid
    assert not two_sided(0.40, 1.0)                   # degenerate one ask
    assert not two_sided(0.50, 0.44)                  # crossed


# ---- C2 feature parity ---------------------------------------------------------------

def _cache_fixture():
    rows = []
    for slug in [f"p{i}" for i in range(20)]:         # pool of 20
        for critic, sent in [("a", "positive"), ("b", "negative"), ("c", "positive")]:
            rows.append({"movie_slug": slug, "reviewer_name": critic,
                         "tomatometer_sentiment": sent,
                         "estimated_timestamp": pd.Timestamp("2026-01-01", tz="UTC")})
    for i, sent in enumerate(["positive"] * 30 + ["negative" if i % 3 else "positive"
                                                  for i in range(10)]):
        rows.append({"movie_slug": "target", "reviewer_name": f"t{i}",
                     "tomatometer_sentiment": sent,
                     "estimated_timestamp": pd.Timestamp("2026-06-01", tz="UTC")
                     + pd.Timedelta(hours=i)})
    return pd.DataFrame(rows)


def test_c2_feature_row_matches_formulas():
    cache = _cache_fixture()
    pool = [f"p{i}" for i in range(20)]
    snap = pd.Timestamp("2026-06-12T04:00:00Z")       # all 40 target rows observed
    row = pl.c2_feature_row(cache, "target", pool, 3, snap)
    assert row["n_obs"] == 40
    el = pl.emp_logit(row["fresh_obs"], 40)
    assert row["el_P1"] == pytest.approx(el)
    assert row["log1p_total"] == pytest.approx(np.log1p(40))
    assert row["el_x_log"] == pytest.approx(el * np.log1p(40))
    assert row["snap_3"] == 1.0 and row["snap_2"] == 0.0 and row["snap_4"] == 0.0
    assert row["el_x_snap3"] == pytest.approx(el) and row["el_x_snap2"] == 0.0
    # target critics are not in the pool -> zero consumed mass; full-prior blend
    assert row["mass_consumed"] == 0.0
    w = 40 / 60.0
    assert row["p_shipped"] == pytest.approx(
        w * (row["fresh_obs"] / 40) + (1 - w) * row["P2"])
    assert set(pl.F_C2) <= set(row)                   # every model feature present


def test_f_c2_order_pinned():
    """predict_proba is column-order-sensitive: the F_C2 order is load-bearing and
    must match the order the bench notebook fitted with."""
    assert pl.F_C2 == ["el_P1", "log1p_total", "el_x_log", "lp_P3", "snap_2",
                       "snap_3", "snap_4", "el_x_snap2", "el_x_snap3", "el_x_snap4",
                       "mass_consumed"]
    assert pl.F_C1 == ["lp_shipped", "snap_2", "snap_3", "snap_4"]


def test_append_log_roundtrip_and_header_guard(tmp_path):
    from gates.live_scorer import LOG_COLS, _append_log
    p = str(tmp_path / "log.csv")
    row = {c: "" for c in LOG_COLS}
    row.update({"run_ts": "2026-06-12T04:05:00Z", "slug": "x", "strike": 75,
                "bid": 0.7, "ask": 0.75, "read": "no trade"})
    _append_log([row], path=p)
    _append_log([row], path=p)                        # append keeps one header
    df = pd.read_csv(p)
    assert list(df.columns) == LOG_COLS and len(df) == 2
    with open(p, "w") as fh:                          # drifted header must refuse
        fh.write("wrong,header\n1,2\n")
    with pytest.raises(AssertionError):
        _append_log([row], path=p)


def test_c2_feature_row_zero_obs_is_prior_only():
    cache = _cache_fixture()
    pool = [f"p{i}" for i in range(20)]
    snap = pd.Timestamp("2026-05-01T00:00:00Z")       # before any target review
    assert pl.c2_feature_row(cache, "target", pool, 3, snap) is None


def test_prepare_training_frame_formulas():
    ft = pd.DataFrame({
        "slug": ["m1", "m1"], "snap_days": [3, 1], "n_obs": [40, 10],
        "fresh_obs": [30, 5], "n_rem": [20, 4], "fresh_rem": [10, 2],
        "P1": [0.75, 0.5], "P2": [0.6, 0.6], "P3": [0.62, 0.62],
        "mass_consumed": [0.3, 0.1],
        "close": ["2026-06-01T14:00:00Z"] * 2,
    })
    out = pl.prepare_training_frame(ft)
    w0 = 40 / 60.0
    assert out.loc[0, "p_shipped"] == pytest.approx(w0 * 0.75 + (1 - w0) * 0.6)
    assert out.loc[0, "snap_3"] == 1.0 and out.loc[1, "snap_3"] == 0.0
    assert out.loc[0, "el_x_snap3"] == pytest.approx(out.loc[0, "el_P1"])
    assert out.loc[1, "el_x_snap2"] == 0.0
    assert list(out["close_dt"].dt.year) == [2026, 2026]
    for col in pl.F_C2:
        assert col in out.columns
