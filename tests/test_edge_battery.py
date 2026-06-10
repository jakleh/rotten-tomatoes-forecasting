"""compute_edge battery vs an independent reference implementation.

The reference enumerates the Poisson×Binomial sum directly and decides Yes-resolution
with EXACT INTEGERS — Yes iff 200·fresh ≥ (2X+1)·total (equivalent to
fresh/total ≥ (X+0.5)/100, T>0; T==0 resolves No) — avoiding any float disagreement at
exact X.5% scores. Battery cases per plans/plan_gate_1_2_calibration.md "Math under
test" (the keeper from the parked ridge golden-fixture).
"""
import math

import pytest
from scipy.stats import binom, poisson

from rotten_tomatoes_forecasting import compute_edge


def reference_p_yes(threshold: int, fresh: int, total: int, mu: float, p: float) -> float:
    """Independent enumeration. Yes iff 200*(fresh+j) >= (2X+1)*(total+k); total+k==0 -> No."""
    def yes(f: int, t: int) -> bool:
        return t > 0 and 200 * f >= (2 * threshold + 1) * t

    if mu == 0:
        return 1.0 if yes(fresh, total) else 0.0
    k_cap = int(math.ceil(mu + 12 * math.sqrt(mu) + 30))
    assert poisson.sf(k_cap, mu) < 1e-12, "enumeration cap too small"
    p_yes = 0.0
    for k in range(0, k_cap + 1):
        pk = poisson.pmf(k, mu)
        if pk < 1e-16:
            continue
        py_k = sum(binom.pmf(j, k, p) for j in range(0, k + 1) if yes(fresh + j, total + k))
        p_yes += pk * py_k
    return p_yes


def _mu_to_rate(mu: float, hours: float = 10.0) -> tuple[float, float]:
    """compute_edge takes (lambda_rate, hours); the unit cancels in mu = rate*hours."""
    return mu / hours, hours


class TestBatteryGrid:
    """Dense cross-product sweep — every branch of compute_edge against the reference."""

    @pytest.mark.parametrize("threshold", [30, 75, 89, 95])
    @pytest.mark.parametrize("fresh,total", [(0, 0), (5, 5), (10, 40), (60, 80), (81, 90)])
    @pytest.mark.parametrize("mu", [0.0, 0.3, 2.0, 15.0, 60.0])
    @pytest.mark.parametrize("p", [0.0, 0.13, 0.5, 0.87, 1.0])
    def test_matches_reference(self, threshold, fresh, total, mu, p):
        rate, hours = _mu_to_rate(mu)
        r = compute_edge(threshold, 50.0, fresh, total, hours, rate, p)
        ref = reference_p_yes(threshold, fresh, total, mu, p)
        assert r["p_yes"] == pytest.approx(ref, abs=1e-9)


class TestNamedCases:
    def test_mu_zero_already_yes(self):
        r = compute_edge(75, 50.0, 80, 100, 10.0, 0.0, 0.5)
        assert r["p_yes"] == 1.0 and r["k_max"] == 0

    def test_mu_zero_already_no(self):
        r = compute_edge(75, 50.0, 70, 100, 10.0, 0.0, 0.5)
        assert r["p_yes"] == 0.0

    def test_total_zero_mu_zero_resolves_no(self):
        r = compute_edge(75, 50.0, 0, 0, 10.0, 0.0, 0.5)
        assert r["p_yes"] == 0.0

    def test_total_zero_mu_positive_uses_k_ge_1_terms(self):
        # not auto-No: with reviews still coming, P(Yes) > 0 (all-fresh outcomes cross)
        mu, p = 2.0, 0.9
        rate, hours = _mu_to_rate(mu)
        r = compute_edge(75, 50.0, 0, 0, hours, rate, p)
        assert r["p_yes"] == pytest.approx(reference_p_yes(75, 0, 0, mu, p), abs=1e-9)
        assert r["p_yes"] > 0.3

    def test_exact_boundary_x_point_5_resolves_yes(self):
        # fresh/total == (X+0.5)/100 exactly: 151/200 = 0.755 at X=75 -> Yes (>=)
        assert reference_p_yes(75, 151, 200, 0.0, 0.5) == 1.0
        r = compute_edge(75, 50.0, 151, 200, 10.0, 0.0, 0.5)
        assert r["p_yes"] == 1.0

    def test_just_below_boundary_resolves_no(self):
        # 150/200 = 0.750 < 0.755 -> No
        r = compute_edge(75, 50.0, 150, 200, 10.0, 0.0, 0.5)
        assert r["p_yes"] == 0.0

    def test_p_fresh_one_monotone_path(self):
        # all future reviews fresh: needs ceil-from-below to cross; reference agrees
        mu, p = 5.0, 1.0
        rate, hours = _mu_to_rate(mu)
        r = compute_edge(89, 50.0, 85, 96, hours, rate, p)
        assert r["p_yes"] == pytest.approx(reference_p_yes(89, 85, 96, mu, p), abs=1e-9)

    def test_p_fresh_zero_cannot_cross_from_below(self):
        mu, p = 5.0, 0.0
        rate, hours = _mu_to_rate(mu)
        r = compute_edge(75, 50.0, 70, 100, hours, rate, p)
        assert r["p_yes"] == 0.0

    def test_tiny_mu_hand_checkable_sum(self):
        # X=75, state 7/10 (0.700), mu=0.1, p=0.5: first crossing needs k=3 all-fresh
        # (7+j)/(10+k) >= 0.755 -> k=3: j>=3. Hand sum ~ pois(3;.1)*0.125 ~= 1.885e-5
        # (+ k>=4 tail ~ 2.4e-7) -> ~1.91e-5.
        mu, p = 0.1, 0.5
        rate, hours = _mu_to_rate(mu)
        r = compute_edge(75, 50.0, 7, 10, hours, rate, p)
        ref = reference_p_yes(75, 7, 10, mu, p)
        # 1e-9 matches compute_edge's own truncation guarantee P(K > k_max) < 1e-10;
        # the reference enumerates further, so sub-1e-10 tail differences are by design
        assert r["p_yes"] == pytest.approx(ref, abs=1e-9)
        assert 1.8e-5 < r["p_yes"] < 2.0e-5

    def test_k_max_truncation_negligible(self):
        for mu in [0.3, 2.0, 15.0, 60.0]:
            rate, hours = _mu_to_rate(mu)
            r = compute_edge(75, 50.0, 10, 40, hours, rate, 0.5)
            assert poisson.sf(r["k_max"], mu) < 1e-10

    def test_edge_cents_sign_convention(self):
        rate, hours = _mu_to_rate(2.0)
        r = compute_edge(75, 30.0, 80, 100, hours, rate, 0.5)
        assert r["edge_cents"] == pytest.approx(r["p_yes"] * 100 - 30.0)
