"""Tests for gates/build_gate3b.py's pure helpers (no network, no DB).

The driver's I/O stages are exercised by the real build; these lock the pure logic
the lock chain depends on: the midnight-ET snap arithmetic, the LOCF book pick, the
ct/guard/readiness rules, the pin-selection set matching, and the pre-registered
estimator-cell status codes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gates.build_gate3b import (
    book_at,
    cell_status,
    choose_pin,
    guard_clean,
    is_ct,
    midnight_snap,
    readiness_pass,
)


# ---- midnight_snap ---------------------------------------------------------------

def test_midnight_snap_edt():
    close = pd.Timestamp("2026-06-08T14:00:00Z")          # 10am EDT Monday
    snap = midnight_snap(close, 3)
    assert snap == pd.Timestamp("2026-06-05T04:00:00Z")   # midnight ET = 04:00Z in EDT
    h = (close - snap).total_seconds() / 3600
    assert h == pytest.approx(82.0)                       # 3*24 + 10


def test_midnight_snap_est():
    close = pd.Timestamp("2026-01-19T15:00:00Z")          # 10am EST Monday
    snap = midnight_snap(close, 2)
    assert snap == pd.Timestamp("2026-01-17T05:00:00Z")   # midnight ET = 05:00Z in EST
    assert (close - snap).total_seconds() / 3600 == pytest.approx(58.0)


# ---- book_at ----------------------------------------------------------------------

def _candles(rows):
    return pd.DataFrame(rows, columns=["ts", "yes_bid", "yes_ask", "mid"])


def test_book_at_picks_last_at_or_before_snap():
    g = _candles([(100, 0.40, 0.44, 0.42), (200, 0.50, 0.54, 0.52)])
    bk = book_at(g, 150)
    assert bk["mid"] == 0.42 and bk["stale_min"] == pytest.approx(50 / 60)
    assert book_at(g, 200)["mid"] == 0.52                 # inclusive boundary
    assert book_at(g, 99) is None                         # not yet listed/quoted


def test_book_at_null_book_kills_carried_quote():
    g = _candles([(100, 0.40, 0.44, 0.42), (200, np.nan, 0.90, np.nan)])
    bk = book_at(g, 300)
    assert np.isnan(bk["mid"])                            # latest candle wins, even null


# ---- is_ct ------------------------------------------------------------------------

def test_is_ct_contested_and_tight():
    assert is_ct(0.45, 0.55, 0.50)
    assert not is_ct(0.10, 0.20, 0.15)                    # not contested (mid <= 0.2)
    assert not is_ct(0.78, 0.86, 0.82)                    # mid >= 0.8
    assert not is_ct(0.40, 0.51, 0.455)                   # spread 11c
    assert not is_ct(np.nan, np.nan, np.nan)              # no live book


def test_is_ct_spread_quantized_to_cents():
    # 0.76 - 0.66 = 0.10000000000000009 in floats — must still count as 10c
    assert is_ct(0.66, 0.76, 0.71)


# ---- guard_clean --------------------------------------------------------------------

SNAP = pd.Timestamp("2026-06-05T04:00:00Z")


def test_guard_clean_rules():
    early = pd.Timestamp("2026-06-01T00:00:00Z")
    late = pd.Timestamp("2026-06-06T00:00:00Z")
    assert guard_clean(early, SNAP, 0, 50, 0)
    assert not guard_clean(None, SNAP, 0, 50, 0)          # never scraped
    assert not guard_clean(pd.NaT, SNAP, 0, 50, 0)
    assert not guard_clean(late, SNAP, 0, 50, 0)          # rule (a): tracked after snap
    assert guard_clean(early, SNAP, 5, 50, 0)             # (b): 5 <= max(2, 5.0)
    assert not guard_clean(early, SNAP, 6, 50, 0)         # (b): 6 > 5.0
    assert guard_clean(early, SNAP, 2, 10, 0)             # (b): floor of 2 applies
    assert not guard_clean(early, SNAP, 3, 10, 0)
    assert guard_clean(early, SNAP, 0, 50, 2)             # (c): <= 2 last-day d-rows
    assert not guard_clean(early, SNAP, 0, 50, 3)


# ---- readiness_pass -----------------------------------------------------------------

def test_readiness_pass():
    assert readiness_pass(24, (23, 25), 1, 1)
    assert not readiness_pass(20, (23, 25), 1, 1)         # score outside interval
    assert not readiness_pass(24, (23, 25), 1, 2)        # last-day view drifted vs ledger
    assert not readiness_pass(None, (23, 25), 1, 1)       # 0 reviews at pin
    assert not readiness_pass(24, None, 1, 1)             # no usable strike results
    assert not readiness_pass(24, (23, 25), 1, None)
    assert not readiness_pass(24, (23, 25), 1, float("nan"))
    assert readiness_pass(23, (23, 25), 0, 0)             # interval bounds inclusive
    assert readiness_pass(25, (23, 25), 0, 0)


# ---- choose_pin ---------------------------------------------------------------------

def test_choose_pin_smallest_matching_heal_set():
    fails = {1: {"a"}, 2: {"a", "b"}, 3: {"a", "b"}}
    assert choose_pin(fails, [1, 2, 3]) == 2              # pin 1's set differs -> skip
    fails_eq = {1: {"a"}, 2: {"a"}, 3: {"a"}}
    assert choose_pin(fails_eq, [1, 2, 3]) == 1


def test_choose_pin_heal_pin_always_matches_itself():
    fails = {1: {"x"}, 2: {"y"}, 3: {"z"}}
    assert choose_pin(fails, [1, 2, 3]) == 3


# ---- cell_status --------------------------------------------------------------------

def test_cell_status_codes():
    feats = {"target_gap": 10.0}
    assert cell_status(None, None, None) == "skip:features"
    assert cell_status(feats, -0.01, 10.0) == "skip:lambda"   # negative Ridge total
    assert cell_status(feats, None, 10.0) == "skip:lambda"
    assert cell_status(feats, 0.5, 16.0) == "trimmed"         # deployment gap-cap
    assert cell_status(feats, 0.5, 15.0) == "ok"              # cap is strictly > 15
    assert cell_status(feats, 0.0, 10.0) == "ok"              # zero rate is priceable
