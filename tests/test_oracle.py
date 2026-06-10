"""Gate-2 oracle decomposition tests (gates/oracle.py) — placement rules, look-ahead
safety, the λ×t_rem invariant, and the two observation boundaries.

Synthetic fixtures only (no DB / network). See plans/plan_gate_1_2_calibration.md
"Math under test" + "Mixed-granularity placement".
"""
import numpy as np
import pandas as pd
import pytest

from gates.oracle import LAG, OracleInputs, effective_ts, oracle_inputs

CLOSE = pd.Timestamp("2026-06-01 14:00:00", tz="UTC")
SNAP = CLOSE - pd.Timedelta(days=3)          # 2026-05-29 14:00Z


def _df(rows):
    return pd.DataFrame(rows, columns=[
        "estimated_timestamp", "scrape_time", "timestamp_confidence",
        "tomatometer_sentiment"])


def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def _r(est, scrape, conf="m", sent="positive"):
    return {"estimated_timestamp": _ts(est), "scrape_time": _ts(scrape),
            "timestamp_confidence": conf, "tomatometer_sentiment": sent}


class TestPlacement:
    def test_m_review_at_snap_is_remaining(self):
        # est == snap exactly -> +1min lag pushes it to the remaining side (look-ahead-safe)
        df = _df([_r("2026-05-29 14:00:00", "2026-05-29 14:30:00")])
        o = oracle_inputs(df, CLOSE, SNAP, mode="pure")
        assert (o.total_obs, o.n_remaining) == (0, 1)

    def test_m_review_two_minutes_before_snap_is_observed(self):
        df = _df([_r("2026-05-29 13:58:00", "2026-05-29 14:30:00")])
        o = oracle_inputs(df, CLOSE, SNAP, mode="pure")
        assert (o.total_obs, o.n_remaining) == (1, 0)

    def test_d_review_same_day_as_snap_crowds_forward_to_remaining(self):
        # d row lands at 00:00Z of snap day; end-of-day edge = next midnight > 14:00Z snap
        df = _df([_r("2026-05-29 00:00:00", "2026-05-30 09:00:00", conf="d")])
        o = oracle_inputs(df, CLOSE, SNAP, mode="pure")
        assert (o.total_obs, o.n_remaining) == (0, 1)

    def test_d_review_day_before_snap_is_observed(self):
        # end-of-day edge = 2026-05-29 00:00Z <= 14:00Z snap
        df = _df([_r("2026-05-28 00:00:00", "2026-05-28 12:00:00", conf="d")])
        o = oracle_inputs(df, CLOSE, SNAP, mode="pure")
        assert (o.total_obs, o.n_remaining) == (1, 0)

    def test_m_review_in_last_minute_excluded_from_terminal(self):
        # eff = est + 1min > close -> outside the oracle's terminal (conservative edge)
        df = _df([_r("2026-06-01 13:59:30", "2026-06-01 14:30:00")])
        o = oracle_inputs(df, CLOSE, SNAP, mode="pure")
        assert o.terminal_total == 0

    def test_effective_ts_vectorizes_mixed_confidence(self):
        df = _df([_r("2026-05-28 10:00:00", "2026-05-28 11:00:00", conf="m"),
                  _r("2026-05-28 00:00:00", "2026-05-29 09:00:00", conf="d"),
                  _r("2026-05-28 10:00:00", "2026-05-28 11:00:00", conf="h")])
        eff = effective_ts(df)
        assert eff.iloc[0] == pd.Timestamp("2026-05-28 10:01:00", tz="UTC")
        assert eff.iloc[1] == pd.Timestamp("2026-05-29 00:00:00", tz="UTC")
        assert eff.iloc[2] == pd.Timestamp("2026-05-28 10:01:00", tz="UTC")


class TestInvariants:
    @pytest.fixture
    def random_cohort(self):
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(200):
            est = CLOSE - pd.Timedelta(hours=float(rng.uniform(0, 24 * 9)))
            lag_min = float(rng.uniform(1, 90))
            rows.append(_r(est, est + pd.Timedelta(minutes=lag_min),
                           conf=rng.choice(["m", "h", "d"], p=[0.5, 0.2, 0.3]),
                           sent=rng.choice(["positive", "negative"])))
        return _df(rows)

    @pytest.mark.parametrize("mode", ["pure", "lagged"])
    def test_observed_plus_remaining_equals_terminal(self, random_cohort, mode):
        o = oracle_inputs(random_cohort, CLOSE, SNAP, mode=mode)
        assert o.total_obs + o.n_remaining == o.terminal_total
        assert o.fresh_obs + o.fresh_remaining == o.terminal_fresh

    @pytest.mark.parametrize("mode", ["pure", "lagged"])
    def test_lambda_times_t_rem_is_realized_count(self, random_cohort, mode):
        o = oracle_inputs(random_cohort, CLOSE, SNAP, mode=mode)
        assert o.lambda_rate * o.t_rem_hours == pytest.approx(o.n_remaining, abs=1e-9)

    def test_t_rem_is_fractional_hours(self, random_cohort):
        snap = CLOSE - pd.Timedelta(hours=49.5)
        o = oracle_inputs(random_cohort, CLOSE, snap)
        assert o.t_rem_hours == pytest.approx(49.5)

    def test_pure_remaining_equals_window_count(self, random_cohort):
        # remaining (terminal - observed) == direct count of snap < eff <= close
        o = oracle_inputs(random_cohort, CLOSE, SNAP, mode="pure")
        eff = effective_ts(random_cohort)
        assert o.n_remaining == int(((eff > SNAP) & (eff <= CLOSE)).sum())


class TestLaggedMode:
    def test_scraped_after_snap_not_observed_even_if_published_before(self):
        df = _df([_r("2026-05-29 10:00:00", "2026-05-29 15:00:00")])  # est<snap<scrape
        pure = oracle_inputs(df, CLOSE, SNAP, mode="pure")
        lag = oracle_inputs(df, CLOSE, SNAP, mode="lagged")
        assert pure.total_obs == 1 and lag.total_obs == 0
        assert lag.n_remaining == 1            # still in the lagged oracle's remainder

    def test_published_preclose_scraped_postclose_is_remaining(self):
        df = _df([_r("2026-06-01 09:00:00", "2026-06-01 16:00:00")])
        lag = oracle_inputs(df, CLOSE, SNAP, mode="lagged")
        assert lag.terminal_total == 1 and lag.total_obs == 0 and lag.n_remaining == 1

    def test_lagged_observed_never_exceeds_pure_terminal(self):
        df = _df([_r("2026-05-28 10:00:00", "2026-05-28 10:30:00"),
                  _r("2026-05-30 10:00:00", "2026-05-30 10:40:00")])
        lag = oracle_inputs(df, CLOSE, SNAP, mode="lagged")
        assert lag.total_obs <= lag.terminal_total


class TestEdgeCases:
    def test_empty_remaining_gives_lambda_zero_and_sentinel_p_fresh(self):
        df = _df([_r("2026-05-20 10:00:00", "2026-05-20 11:00:00")])
        o = oracle_inputs(df, CLOSE, SNAP)
        assert o.lambda_rate == 0.0 and o.p_fresh == 0.5

    def test_snap_after_close_raises(self):
        df = _df([_r("2026-05-20 10:00:00", "2026-05-20 11:00:00")])
        with pytest.raises(ValueError):
            oracle_inputs(df, CLOSE, CLOSE + pd.Timedelta(hours=1))

    def test_bad_mode_raises(self):
        df = _df([_r("2026-05-20 10:00:00", "2026-05-20 11:00:00")])
        with pytest.raises(ValueError):
            oracle_inputs(df, CLOSE, SNAP, mode="clairvoyant")

    def test_p_fresh_is_remaining_fresh_fraction(self):
        df = _df([_r("2026-05-30 10:00:00", "2026-05-30 10:30:00", sent="positive"),
                  _r("2026-05-30 11:00:00", "2026-05-30 11:30:00", sent="negative"),
                  _r("2026-05-31 10:00:00", "2026-05-31 10:30:00", sent="positive"),
                  _r("2026-05-20 10:00:00", "2026-05-20 10:30:00", sent="negative")])
        o = oracle_inputs(df, CLOSE, SNAP)
        assert (o.n_remaining, o.fresh_remaining) == (3, 2)
        assert o.p_fresh == pytest.approx(2 / 3)
        assert isinstance(o, OracleInputs)
        assert LAG == pd.Timedelta(minutes=1)
