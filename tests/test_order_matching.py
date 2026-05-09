"""Unit tests for OrderMatchingEngine.

Covers:
- Buy fills at ask (spot * (1 + hs + skew))
- Sell fills at bid (spot * (1 - hs + skew))
- Hold / zero quantity skipped
- net_order_flow sign and magnitude
- Skew lifts ask, lowers bid
- total_volume accumulation
- ask always > bid
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from multi_agent.order_matching import OrderMatchingEngine


def make_engine():
    return OrderMatchingEngine()


class TestOrderMatchingFillPrices:
    def test_buy_fills_at_ask(self):
        eng = make_engine()
        spot = 100.0
        hs = 0.05
        trades, _ = eng.match_orders(
            {"t0": {"direction": "buy", "quantity": 10, "size_bucket": "small"}},
            mm_half_spread=hs, mm_skew=0.0, spot_price=spot,
        )
        expected_ask = spot * (1 + hs)
        assert trades["t0"]["execution_price"] == pytest.approx(expected_ask)

    def test_sell_fills_at_bid(self):
        eng = make_engine()
        spot = 100.0
        hs = 0.05
        trades, _ = eng.match_orders(
            {"t0": {"direction": "sell", "quantity": 10, "size_bucket": "small"}},
            mm_half_spread=hs, mm_skew=0.0, spot_price=spot,
        )
        expected_bid = spot * (1 - hs)
        assert trades["t0"]["execution_price"] == pytest.approx(expected_bid)

    def test_skew_shifts_both_legs(self):
        eng = make_engine()
        spot = 100.0
        hs = 0.05
        skew = 0.02  # positive skew lifts both ask and bid
        trades_buy, _ = eng.match_orders(
            {"t0": {"direction": "buy", "quantity": 5, "size_bucket": "small"}},
            mm_half_spread=hs, mm_skew=skew, spot_price=spot,
        )
        trades_sell, _ = eng.match_orders(
            {"t0": {"direction": "sell", "quantity": 5, "size_bucket": "small"}},
            mm_half_spread=hs, mm_skew=skew, spot_price=spot,
        )
        assert trades_buy["t0"]["execution_price"] == pytest.approx(spot * (1 + hs + skew))
        assert trades_sell["t0"]["execution_price"] == pytest.approx(spot * (1 - hs + skew))

    def test_ask_always_greater_than_bid(self):
        eng = make_engine()
        for hs in (0.001, 0.05, 0.30):
            trades_b, _ = eng.match_orders(
                {"b": {"direction": "buy", "quantity": 1, "size_bucket": "small"}},
                mm_half_spread=hs, mm_skew=0.0, spot_price=100.0,
            )
            trades_s, _ = eng.match_orders(
                {"s": {"direction": "sell", "quantity": 1, "size_bucket": "small"}},
                mm_half_spread=hs, mm_skew=0.0, spot_price=100.0,
            )
            assert trades_b["b"]["execution_price"] > trades_s["s"]["execution_price"]


class TestOrderMatchingFlow:
    def test_net_order_flow_buy_positive(self):
        eng = make_engine()
        _, flow = eng.match_orders(
            {"t0": {"direction": "buy", "quantity": 20, "size_bucket": "medium"}},
            mm_half_spread=0.05, mm_skew=0.0, spot_price=100.0,
        )
        assert flow == pytest.approx(20.0)

    def test_net_order_flow_sell_negative(self):
        eng = make_engine()
        _, flow = eng.match_orders(
            {"t0": {"direction": "sell", "quantity": 15, "size_bucket": "small"}},
            mm_half_spread=0.05, mm_skew=0.0, spot_price=100.0,
        )
        assert flow == pytest.approx(-15.0)

    def test_net_order_flow_mixed(self):
        eng = make_engine()
        _, flow = eng.match_orders(
            {
                "t0": {"direction": "buy", "quantity": 30, "size_bucket": "medium"},
                "t1": {"direction": "sell", "quantity": 10, "size_bucket": "small"},
            },
            mm_half_spread=0.05, mm_skew=0.0, spot_price=100.0,
        )
        assert flow == pytest.approx(20.0)  # 30 - 10

    def test_hold_not_executed(self):
        eng = make_engine()
        trades, flow = eng.match_orders(
            {"t0": {"direction": "hold", "quantity": 10, "size_bucket": "small"}},
            mm_half_spread=0.05, mm_skew=0.0, spot_price=100.0,
        )
        assert "t0" not in trades
        assert flow == pytest.approx(0.0)

    def test_zero_quantity_not_executed(self):
        eng = make_engine()
        trades, flow = eng.match_orders(
            {"t0": {"direction": "buy", "quantity": 0, "size_bucket": "small"}},
            mm_half_spread=0.05, mm_skew=0.0, spot_price=100.0,
        )
        assert "t0" not in trades
        assert flow == pytest.approx(0.0)

    def test_cash_impact_correct_sign(self):
        eng = make_engine()
        trades, _ = eng.match_orders(
            {"t0": {"direction": "buy", "quantity": 10, "size_bucket": "small"}},
            mm_half_spread=0.05, mm_skew=0.0, spot_price=100.0,
        )
        # Buying costs cash: cash_impact should be negative
        assert trades["t0"]["cash_impact"] < 0

    def test_total_volume_accumulates(self):
        eng = make_engine()
        eng.match_orders(
            {"t0": {"direction": "buy", "quantity": 10, "size_bucket": "small"}},
            mm_half_spread=0.05, mm_skew=0.0, spot_price=100.0,
        )
        eng.match_orders(
            {"t0": {"direction": "sell", "quantity": 5, "size_bucket": "small"}},
            mm_half_spread=0.05, mm_skew=0.0, spot_price=100.0,
        )
        assert eng.total_volume == pytest.approx(15.0)

    def test_multiple_traders_all_filled(self):
        eng = make_engine()
        trades, flow = eng.match_orders(
            {
                "t0": {"direction": "buy", "quantity": 5, "size_bucket": "small"},
                "t1": {"direction": "buy", "quantity": 10, "size_bucket": "small"},
                "t2": {"direction": "sell", "quantity": 8, "size_bucket": "small"},
            },
            mm_half_spread=0.05, mm_skew=0.0, spot_price=100.0,
        )
        assert "t0" in trades and "t1" in trades and "t2" in trades
        assert flow == pytest.approx(5 + 10 - 8)
