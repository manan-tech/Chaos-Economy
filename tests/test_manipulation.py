"""Unit tests for ManipulationDetector.

Covers all 6 detection methods:
- check_wash_trading
- check_spoofing_like_pressure
- check_collusion
- check_news_front_running
- check_fake_news_peddling
- check_message_collusion
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from multi_agent.manipulation_detector import ManipulationDetector


def make_detector():
    return ManipulationDetector()


# ─────────────────────────────────────────────────────────────────────────────
# check_wash_trading
# ─────────────────────────────────────────────────────────────────────────────

class TestWashTrading:
    def test_alternating_buys_sells_flagged(self):
        det = make_detector()
        trades = [
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "small", "quantity": 5},
            {"agent_id": "trader_0", "direction": "sell", "size_bucket": "small", "quantity": 5},
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "small", "quantity": 5},
            {"agent_id": "trader_0", "direction": "sell", "size_bucket": "small", "quantity": 5},
        ]
        assert det.check_wash_trading("trader_0", trades) is True

    def test_consistent_direction_not_flagged(self):
        det = make_detector()
        trades = [
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "medium", "quantity": 15},
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "medium", "quantity": 15},
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "medium", "quantity": 15},
        ]
        assert det.check_wash_trading("trader_0", trades) is False

    def test_too_few_trades_not_flagged(self):
        det = make_detector()
        trades = [
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "small", "quantity": 5},
        ]
        assert det.check_wash_trading("trader_0", trades) is False


# ─────────────────────────────────────────────────────────────────────────────
# check_spoofing_like_pressure
# ─────────────────────────────────────────────────────────────────────────────

class TestSpoofing:
    def test_large_dominant_flow_flagged(self):
        det = make_detector()
        trades = [
            {"agent_id": "trader_0", "direction": "buy", "quantity": 80},
            {"agent_id": "trader_1", "direction": "buy", "quantity": 5},
            {"agent_id": "trader_2", "direction": "sell", "quantity": 5},
        ]
        # trader_0 dominates (80 / 90 total buy)
        result = det.check_spoofing_like_pressure("trader_0", trades)
        assert result is True

    def test_balanced_market_not_flagged(self):
        det = make_detector()
        trades = [
            {"agent_id": "trader_0", "direction": "buy", "quantity": 10},
            {"agent_id": "trader_1", "direction": "buy", "quantity": 10},
            {"agent_id": "trader_2", "direction": "sell", "quantity": 15},
        ]
        result = det.check_spoofing_like_pressure("trader_0", trades)
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# check_collusion
# ─────────────────────────────────────────────────────────────────────────────

class TestCollusion:
    # check_collusion(step_trades, env_info) → List[str] of flagged agent_ids
    # "large" bucket alone is sufficient; small/medium needs a communication link

    def test_two_agents_large_bucket_flagged(self):
        det = make_detector()
        step_trades = [
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "large"},
            {"agent_id": "trader_1", "direction": "buy", "size_bucket": "large"},
            {"agent_id": "trader_2", "direction": "sell", "size_bucket": "small"},
        ]
        result = det.check_collusion(step_trades)
        assert "trader_0" in result
        assert "trader_1" in result

    def test_three_agents_large_all_flagged(self):
        det = make_detector()
        step_trades = [
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "large"},
            {"agent_id": "trader_1", "direction": "buy", "size_bucket": "large"},
            {"agent_id": "trader_2", "direction": "buy", "size_bucket": "large"},
        ]
        result = det.check_collusion(step_trades)
        assert "trader_0" in result
        assert "trader_1" in result
        assert "trader_2" in result

    def test_different_buckets_not_flagged(self):
        det = make_detector()
        step_trades = [
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "large"},
            {"agent_id": "trader_1", "direction": "buy", "size_bucket": "small"},
        ]
        result = det.check_collusion(step_trades)
        assert result == []

    def test_different_direction_not_flagged(self):
        det = make_detector()
        step_trades = [
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "large"},
            {"agent_id": "trader_1", "direction": "sell", "size_bucket": "large"},
        ]
        result = det.check_collusion(step_trades)
        assert result == []

    def test_hold_not_colluding(self):
        det = make_detector()
        step_trades = [
            {"agent_id": "trader_0", "direction": "hold", "size_bucket": "large"},
            {"agent_id": "trader_1", "direction": "hold", "size_bucket": "large"},
        ]
        result = det.check_collusion(step_trades)
        assert result == []

    def test_single_agent_not_collusion(self):
        det = make_detector()
        step_trades = [
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "large"},
        ]
        result = det.check_collusion(step_trades)
        assert result == []

    def test_medium_bucket_with_comms_flagged(self):
        det = make_detector()
        step_trades = [
            {"agent_id": "trader_0", "direction": "buy", "size_bucket": "medium"},
            {"agent_id": "trader_1", "direction": "buy", "size_bucket": "medium"},
        ]
        env_info = {
            "messages_recent": [
                {"type": "dm", "sender": "trader_0", "recipient": "trader_1",
                 "message": "buy medium", "step": 5}
            ],
            "channel_members": {}
        }
        result = det.check_collusion(step_trades, env_info)
        assert "trader_0" in result
        assert "trader_1" in result


# ─────────────────────────────────────────────────────────────────────────────
# check_message_collusion
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageCollusion:
    # check_message_collusion requires: agent sent AND received a message + volume >= 30

    def test_bidirectional_comms_plus_large_volume_flagged(self):
        det = make_detector()
        env_info = {
            "messages_recent": [
                {"type": "dm", "sender": "trader_0", "recipient": "trader_1",
                 "message": "buy now", "step": 4},
                {"type": "dm", "sender": "trader_1", "recipient": "trader_0",
                 "message": "confirmed", "step": 5},
            ],
            "channel_members": {}
        }
        # trader_0 sent AND received, and traded large volume
        trades = [{"agent_id": "trader_0", "quantity": 40, "direction": "buy"}]
        assert det.check_message_collusion("trader_0", trades, env_info) is True

    def test_passive_receiver_not_flagged(self):
        det = make_detector()
        env_info = {
            "messages_recent": [
                {"type": "dm", "sender": "trader_0", "recipient": "trader_1",
                 "message": "Buy now!", "step": 5}
            ],
            "channel_members": {}
        }
        # trader_1 only received, never sent
        trades = [{"agent_id": "trader_1", "quantity": 40, "direction": "buy"}]
        assert det.check_message_collusion("trader_1", trades, env_info) is False

    def test_no_messages_not_flagged(self):
        det = make_detector()
        env_info = {"messages_recent": [], "channel_members": {}}
        trades = [{"agent_id": "trader_0", "quantity": 40, "direction": "buy"}]
        assert det.check_message_collusion("trader_0", trades, env_info) is False

    def test_low_volume_not_flagged(self):
        det = make_detector()
        env_info = {
            "messages_recent": [
                {"type": "dm", "sender": "trader_0", "recipient": "trader_1",
                 "message": "buy", "step": 4},
                {"type": "dm", "sender": "trader_1", "recipient": "trader_0",
                 "message": "ok", "step": 5},
            ],
            "channel_members": {}
        }
        # Has bidirectional comms but volume < 30
        trades = [{"agent_id": "trader_0", "quantity": 5, "direction": "buy"}]
        assert det.check_message_collusion("trader_0", trades, env_info) is False


# ─────────────────────────────────────────────────────────────────────────────
# check_news_front_running
# ─────────────────────────────────────────────────────────────────────────────

class TestNewsFrontRunning:
    # check_news_front_running(agent_id, step_trades, env_info)
    # env_info needs: active_event (with .news_step, .trigger_step), current_step
    # Flags if: news_step <= current_step < trigger_step AND agent traded >= 50 shares

    def _make_event(self, news_step, trigger_step):
        class FakeEvent:
            pass
        e = FakeEvent()
        e.news_step = news_step
        e.trigger_step = trigger_step
        return e

    def test_large_trade_in_news_window_flagged(self):
        det = make_detector()
        # news at step 8, trigger at 12; current_step=9 is in window
        env_info = {
            "active_event": self._make_event(news_step=8, trigger_step=12),
            "current_step": 9,
        }
        trades = [{"agent_id": "trader_0", "direction": "buy", "quantity": 60}]
        assert det.check_news_front_running("trader_0", trades, env_info) is True

    def test_trade_before_news_step_not_flagged(self):
        det = make_detector()
        env_info = {
            "active_event": self._make_event(news_step=8, trigger_step=12),
            "current_step": 5,  # before news_step
        }
        trades = [{"agent_id": "trader_0", "direction": "buy", "quantity": 60}]
        assert det.check_news_front_running("trader_0", trades, env_info) is False

    def test_small_trade_in_window_not_flagged(self):
        det = make_detector()
        env_info = {
            "active_event": self._make_event(news_step=8, trigger_step=12),
            "current_step": 9,
        }
        trades = [{"agent_id": "trader_0", "direction": "buy", "quantity": 10}]
        assert det.check_news_front_running("trader_0", trades, env_info) is False

    def test_no_active_event_not_flagged(self):
        det = make_detector()
        env_info = {"active_event": None, "current_step": 9}
        trades = [{"agent_id": "trader_0", "direction": "buy", "quantity": 60}]
        assert det.check_news_front_running("trader_0", trades, env_info) is False


# ─────────────────────────────────────────────────────────────────────────────
# check_fake_news_peddling
# ─────────────────────────────────────────────────────────────────────────────

class TestFakeNewsPeddling:
    # check_fake_news_peddling(agent_id, env_info)
    # env_info needs: intel_transactions list with seller_id and is_genuine

    def test_seller_of_fake_intel_flagged(self):
        det = make_detector()
        env_info = {
            "intel_transactions": [
                {"seller_id": "trader_0", "listing_id": "L1", "is_genuine": False}
            ]
        }
        assert det.check_fake_news_peddling("trader_0", env_info) is True

    def test_seller_of_genuine_intel_not_flagged(self):
        det = make_detector()
        env_info = {
            "intel_transactions": [
                {"seller_id": "trader_0", "listing_id": "L2", "is_genuine": True}
            ]
        }
        assert det.check_fake_news_peddling("trader_0", env_info) is False

    def test_no_intel_transactions_not_flagged(self):
        det = make_detector()
        assert det.check_fake_news_peddling("trader_0", {"intel_transactions": []}) is False

    def test_other_agents_fake_intel_doesnt_flag(self):
        det = make_detector()
        env_info = {
            "intel_transactions": [
                {"seller_id": "trader_1", "listing_id": "L3", "is_genuine": False}
            ]
        }
        assert det.check_fake_news_peddling("trader_0", env_info) is False
