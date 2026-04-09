"""Tests for rotten_tomatoes_forecasting.edge — compute_edge math and naive estimators."""

import pytest
from rotten_tomatoes_forecasting.edge import compute_edge, naive_p_fresh, EdgeResult


class TestComputeEdge:
    """Test the core Poisson-binomial edge calculation."""

    def test_returns_edge_result_keys(self):
        result = compute_edge(75, 50, 60, 80, 24, 1.0, 0.70)
        assert set(result.keys()) == {"edge_cents", "p_yes", "p_no", "expected_reviews", "k_max"}

    def test_p_yes_plus_p_no_equals_one(self):
        result = compute_edge(75, 50, 60, 80, 48, 2.0, 0.65)
        assert abs(result["p_yes"] + result["p_no"] - 1.0) < 1e-10

    def test_edge_equals_p_yes_times_100_minus_price(self):
        result = compute_edge(75, 42, 60, 80, 24, 1.0, 0.70)
        assert abs(result["edge_cents"] - (result["p_yes"] * 100 - 42)) < 1e-10

    def test_zero_lambda_locks_current_score(self):
        """With lambda=0, no future reviews — outcome determined by current score."""
        # 60/80 = 75%. Threshold 75 needs >= 75.5%. Score is below → resolves No.
        result = compute_edge(75, 50, 60, 80, 24, 0.0, 0.70)
        assert result["p_yes"] == 0.0
        assert result["p_no"] == 1.0
        assert result["expected_reviews"] == 0.0
        assert result["k_max"] == 0

    def test_zero_lambda_score_already_above(self):
        """With lambda=0 and score already above threshold, resolves Yes."""
        # 70/80 = 87.5%. Threshold 75 needs >= 75.5%. Score is above → resolves Yes.
        result = compute_edge(75, 50, 70, 80, 24, 0.0, 0.70)
        assert result["p_yes"] == 1.0
        assert result["p_no"] == 0.0

    def test_zero_hours_locks_current_score(self):
        """With hours_to_close=0, same as lambda=0."""
        result = compute_edge(75, 50, 60, 80, 0.0, 5.0, 0.70)
        assert result["expected_reviews"] == 0.0

    def test_high_lambda_high_p_fresh_pushes_p_yes_high(self):
        """Many expected positive reviews should push P(Yes) toward 1."""
        # Score is 60/80 = 75%. Threshold 75 needs 75.5%.
        # With many more positive reviews expected, should cross threshold.
        result = compute_edge(75, 50, 60, 80, 100, 10.0, 0.90)
        assert result["p_yes"] > 0.95

    def test_high_lambda_low_p_fresh_pushes_p_yes_low(self):
        """Many expected negative reviews dilute the score."""
        # Score is 70/80 = 87.5%. With many negative reviews, score drops.
        result = compute_edge(85, 50, 70, 80, 100, 10.0, 0.10)
        assert result["p_yes"] < 0.05

    def test_resolution_boundary_rounding(self):
        """Test the resolution rule: Above X resolves Yes when round(score) >= X+1.
        Equivalently: fresh/total >= (X + 0.5) / 100."""
        # 76/100 = 76%. Threshold 75: needs >= 75.5%. 76% >= 75.5% → Yes.
        result = compute_edge(75, 50, 76, 100, 0, 0.0, 0.5)
        assert result["p_yes"] == 1.0

        # 75/100 = 75%. Threshold 75: needs >= 75.5%. 75% < 75.5% → No.
        result = compute_edge(75, 50, 75, 100, 0, 0.0, 0.5)
        assert result["p_yes"] == 0.0

    def test_edge_symmetry_at_fair_price(self):
        """If market price equals model's P(Yes)*100, edge should be ~0."""
        result = compute_edge(75, 50, 60, 80, 24, 1.0, 0.70)
        fair_price = result["p_yes"] * 100
        result_at_fair = compute_edge(75, fair_price, 60, 80, 24, 1.0, 0.70)
        assert abs(result_at_fair["edge_cents"]) < 1e-10

    def test_edge_is_linear_in_market_price(self):
        """edge_cents = P(Yes)*100 - market_price, so it's linear in price."""
        r1 = compute_edge(75, 40, 60, 80, 24, 1.0, 0.70)
        r2 = compute_edge(75, 50, 60, 80, 24, 1.0, 0.70)
        # Difference in edge should equal difference in price (with sign flip)
        assert abs((r1["edge_cents"] - r2["edge_cents"]) - (50 - 40)) < 1e-10

    def test_expected_reviews_equals_lambda_times_hours(self):
        result = compute_edge(75, 50, 60, 80, 24, 2.5, 0.70)
        assert abs(result["expected_reviews"] - 60.0) < 1e-10

    # -- Input validation --

    def test_rejects_market_price_out_of_range(self):
        with pytest.raises(ValueError, match="market_price"):
            compute_edge(75, 101, 60, 80, 24, 1.0, 0.70)
        with pytest.raises(ValueError, match="market_price"):
            compute_edge(75, -1, 60, 80, 24, 1.0, 0.70)

    def test_rejects_p_fresh_out_of_range(self):
        with pytest.raises(ValueError, match="p_fresh"):
            compute_edge(75, 50, 60, 80, 24, 1.0, 1.1)
        with pytest.raises(ValueError, match="p_fresh"):
            compute_edge(75, 50, 60, 80, 24, 1.0, -0.1)

    def test_rejects_negative_lambda(self):
        with pytest.raises(ValueError, match="lambda_rate"):
            compute_edge(75, 50, 60, 80, 24, -1.0, 0.70)

    def test_rejects_negative_hours(self):
        with pytest.raises(ValueError, match="hours_to_close"):
            compute_edge(75, 50, 60, 80, -1, 1.0, 0.70)

    # -- Edge cases --

    def test_zero_total_count_with_reviews_coming(self):
        """No reviews yet, but lambda > 0 — model should still compute."""
        result = compute_edge(75, 50, 0, 0, 24, 2.0, 0.70)
        assert 0 <= result["p_yes"] <= 1

    def test_boundary_market_prices(self):
        """Price at 0 and 100 should work."""
        r0 = compute_edge(75, 0, 60, 80, 24, 1.0, 0.70)
        r100 = compute_edge(75, 100, 60, 80, 24, 1.0, 0.70)
        assert r0["edge_cents"] == r0["p_yes"] * 100
        assert abs(r100["edge_cents"] - (r100["p_yes"] * 100 - 100)) < 1e-10


class TestNaivePFresh:
    def test_returns_ratio(self):
        assert naive_p_fresh(60, 80) == 0.75

    def test_zero_total_returns_half(self):
        assert naive_p_fresh(0, 0) == 0.5
