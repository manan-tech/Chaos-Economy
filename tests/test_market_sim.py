"""Unit tests for market_sim, portfolio, and black swan subsystems.

Covers:
- advance_market: GBM step, news shock decay, regime transitions
- apply_black_swan: spot impact, regime set to black_swan, recovery
- apply_order_flow_impact: price impact formula
- update_position / compute_mtm_pnl
- BlackSwanGenerator: event ordering, episode_length safety
- NewsMarketplace: post_listing, buy_intel, double-purchase block,
  empty-content rejection, fake-intel clawback
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from vsr_env.engine.market_sim import advance_market, apply_black_swan, apply_order_flow_impact
from vsr_env.engine.portfolio import update_position, compute_mtm_pnl
from vsr_env.models import VSRState
import math
from multi_agent.black_swan import BlackSwanGenerator
from multi_agent.news_marketplace import NewsMarketplace
from multi_agent.models import AgentState, AgentRole


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_state(spot=100.0, variance=0.04, regime="normal"):
    s = VSRState(episode_id="test", spot_price=spot, variance=variance, step_count=0)
    s.regime = regime
    return s


def make_agent_state(cash=10_000.0, shares=0):
    s = AgentState(agent_id="trader_0", role=AgentRole.TRADER, cash_balance=cash)
    s.shares = shares
    return s


# ─────────────────────────────────────────────────────────────────────────────
# advance_market
# ─────────────────────────────────────────────────────────────────────────────

class TestAdvanceMarket:
    def test_spot_changes_after_step(self):
        rng = np.random.RandomState(42)
        state = make_state(spot=100.0)
        advance_market(state, rng)
        # GBM should move spot
        assert state.spot_price != 100.0

    def test_spot_positive_after_many_steps(self):
        rng = np.random.RandomState(0)
        state = make_state(spot=100.0)
        for _ in range(100):
            advance_market(state, rng)
        assert state.spot_price > 0

    def test_news_shock_decays(self):
        rng = np.random.RandomState(42)
        state = make_state()
        state.news_shock_remaining = 0.10
        advance_market(state, rng)
        assert state.news_shock_remaining < 0.10

    def test_news_shock_cannot_go_negative(self):
        rng = np.random.RandomState(42)
        state = make_state()
        state.news_shock_remaining = 0.001  # almost zero
        advance_market(state, rng)
        assert state.news_shock_remaining >= 0.0

    def test_black_swan_regime_recovers(self):
        rng = np.random.RandomState(42)
        state = make_state(regime="black_swan")
        advance_market(state, rng)
        assert state.regime == "high_vol"  # should transition out of black_swan

    def test_variance_stays_positive(self):
        rng = np.random.RandomState(42)
        state = make_state()
        for _ in range(50):
            advance_market(state, rng)
        assert state.variance > 0


# ─────────────────────────────────────────────────────────────────────────────
# apply_black_swan
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyBlackSwan:
    def test_spot_impact_applied(self):
        state = make_state(spot=100.0)
        apply_black_swan(state, spot_impact=0.70, variance_impact=3.0)
        assert state.spot_price == pytest.approx(70.0)

    def test_variance_increased(self):
        state = make_state(spot=100.0, variance=0.04)
        apply_black_swan(state, spot_impact=1.0, variance_impact=3.0)
        assert state.variance > 0.04

    def test_regime_set_to_black_swan(self):
        state = make_state()
        apply_black_swan(state, spot_impact=0.80, variance_impact=2.0)
        assert state.regime == "black_swan"

    def test_spot_clipped_above_zero(self):
        state = make_state(spot=5.0)
        apply_black_swan(state, spot_impact=0.01, variance_impact=1.0)
        assert state.spot_price > 0


# ─────────────────────────────────────────────────────────────────────────────
# apply_order_flow_impact
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderFlowImpact:
    # apply_order_flow_impact(state, net_shares, lam)

    def test_positive_flow_increases_spot(self):
        state = make_state(spot=100.0)
        original = state.spot_price
        apply_order_flow_impact(state, net_shares=50.0, lam=1e-4)
        assert state.spot_price > original

    def test_negative_flow_decreases_spot(self):
        state = make_state(spot=100.0)
        original = state.spot_price
        apply_order_flow_impact(state, net_shares=-50.0, lam=1e-4)
        assert state.spot_price < original

    def test_zero_flow_no_change(self):
        state = make_state(spot=100.0)
        apply_order_flow_impact(state, net_shares=0.0, lam=1e-4)
        assert state.spot_price == pytest.approx(100.0)

    def test_impact_formula(self):
        state = make_state(spot=100.0)
        apply_order_flow_impact(state, net_shares=100.0, lam=1e-4)
        expected = min(300.0, 100.0 * math.exp(1e-4 * 100.0))
        assert state.spot_price == pytest.approx(expected)

    def test_stored_on_state(self):
        state = make_state(spot=100.0)
        apply_order_flow_impact(state, net_shares=25.0, lam=1e-4)
        assert state.last_net_order_flow == pytest.approx(25.0)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolio:
    # update_position(position: dict, shares_delta, fill_price) → modifies position in place
    # compute_mtm_pnl(shares, entry_price, spot) → float

    def test_buy_increases_shares(self):
        pos = {"shares": 0, "entry_price": 0.0}
        update_position(pos, shares_delta=10, fill_price=100.0)
        assert pos["shares"] == 10

    def test_sell_decreases_shares(self):
        pos = {"shares": 20, "entry_price": 100.0}
        update_position(pos, shares_delta=-5, fill_price=100.0)
        assert pos["shares"] == 15

    def test_buy_updates_entry_price(self):
        pos = {"shares": 0, "entry_price": 0.0}
        update_position(pos, shares_delta=10, fill_price=105.0)
        assert pos["entry_price"] == pytest.approx(105.0)

    def test_sell_from_long_position(self):
        pos = {"shares": 10, "entry_price": 100.0}
        update_position(pos, shares_delta=-10, fill_price=110.0)
        assert pos["shares"] == 0

    def test_mtm_pnl_long_position(self):
        pnl = compute_mtm_pnl(shares=10, entry_price=100.0, spot=110.0)
        # Long 10 shares, spot rose from 100 to 110 → +100 unrealised
        assert pnl == pytest.approx(10 * (110.0 - 100.0))

    def test_mtm_pnl_short_position(self):
        pnl = compute_mtm_pnl(shares=-10, entry_price=100.0, spot=90.0)
        # Short 10 shares, spot fell from 100 to 90 → +100 unrealised
        assert pnl == pytest.approx(-10 * (90.0 - 100.0))

    def test_mtm_pnl_flat_position(self):
        pnl = compute_mtm_pnl(shares=0, entry_price=100.0, spot=150.0)
        assert pnl == pytest.approx(0.0)

    def test_mtm_pnl_loss(self):
        pnl = compute_mtm_pnl(shares=10, entry_price=100.0, spot=90.0)
        assert pnl == pytest.approx(-100.0)


# ─────────────────────────────────────────────────────────────────────────────
# BlackSwanGenerator
# ─────────────────────────────────────────────────────────────────────────────

class TestBlackSwanGenerator:
    def test_events_generated(self):
        rng = np.random.RandomState(42)
        gen = BlackSwanGenerator(rng, episode_length=100)
        assert len(gen.events) > 0

    def test_news_precedes_trigger(self):
        for seed in range(20):
            rng = np.random.RandomState(seed)
            gen = BlackSwanGenerator(rng, episode_length=100)
            for e in gen.events:
                assert e.news_step <= e.trigger_step, (
                    f"News at {e.news_step} after trigger at {e.trigger_step}"
                )

    def test_trigger_within_episode(self):
        for seed in range(20):
            rng = np.random.RandomState(seed)
            gen = BlackSwanGenerator(rng, episode_length=100)
            for e in gen.events:
                assert e.trigger_step < 100

    def test_safe_at_short_episode(self):
        for seed in range(20):
            rng = np.random.RandomState(seed)
            gen = BlackSwanGenerator(rng, episode_length=30)
            for e in gen.events:
                assert e.trigger_step < 30

    def test_news_step_non_negative(self):
        rng = np.random.RandomState(7)
        gen = BlackSwanGenerator(rng, episode_length=100)
        for e in gen.events:
            assert e.news_step >= 0


# ─────────────────────────────────────────────────────────────────────────────
# NewsMarketplace
# ─────────────────────────────────────────────────────────────────────────────

def make_trader_state(agent_id, cash=1000.0):
    s = AgentState(agent_id=agent_id, role=AgentRole.TRADER, cash_balance=cash)
    return s


class TestNewsMarketplace:
    def test_valid_listing_created(self):
        rng = np.random.RandomState(42)
        mp = NewsMarketplace(rng)
        listing = mp.post_listing("trader_0", 50.0,
                                   "Tech sector collapse imminent due to regulation",
                                   "all", current_step=1)
        assert listing is not None
        assert listing.seller_id == "trader_0"
        assert listing.price == 50.0

    def test_empty_content_rejected(self):
        rng = np.random.RandomState(42)
        mp = NewsMarketplace(rng)
        assert mp.post_listing("trader_0", 50.0, "", "all", current_step=1) is None

    def test_trivially_short_content_rejected(self):
        rng = np.random.RandomState(42)
        mp = NewsMarketplace(rng)
        assert mp.post_listing("trader_0", 50.0, "short", "all", current_step=1) is None

    def test_buy_intel_transfers_cash(self):
        rng = np.random.RandomState(42)
        mp = NewsMarketplace(rng)
        states = {
            "seller": make_trader_state("seller", cash=1000.0),
            "buyer": make_trader_state("buyer", cash=1000.0),
        }
        listing = mp.post_listing("seller", 100.0, "Major earnings beat expected this quarter",
                                   "all", current_step=1)
        assert listing is not None
        mp.buy_intel("buyer", listing.listing_id, states, step=2)
        assert states["buyer"].cash_balance == pytest.approx(900.0)
        assert states["seller"].cash_balance >= 1000.0  # seller received payment

    def test_no_double_purchase(self):
        rng = np.random.RandomState(42)
        mp = NewsMarketplace(rng)
        states = {
            "seller": make_trader_state("seller", cash=1000.0),
            "buyer": make_trader_state("buyer", cash=1000.0),
        }
        listing = mp.post_listing("seller", 50.0, "Spot will crash due to bank failure",
                                   "all", current_step=1)
        mp.buy_intel("buyer", listing.listing_id, states, step=2)
        result = mp.buy_intel("buyer", listing.listing_id, states, step=3)
        assert result is None

    def test_buyer_receives_content(self):
        rng = np.random.RandomState(42)
        mp = NewsMarketplace(rng)
        states = {
            "seller": make_trader_state("seller"),
            "buyer": make_trader_state("buyer"),
        }
        content = "Central bank will cut rates at Thursday meeting confirmed"
        listing = mp.post_listing("seller", 50.0, content, "all", current_step=1)
        result = mp.buy_intel("buyer", listing.listing_id, states, step=2)
        assert result is not None

    def test_fake_intel_seller_clawback(self):
        # Find a seed where is_genuine=False
        for seed in range(200):
            rng = np.random.RandomState(seed)
            mp = NewsMarketplace(rng)
            listing = mp.post_listing("seller", 100.0,
                                       "Fake intelligence signal for testing purposes only",
                                       "all", current_step=1)
            if listing and not listing.is_genuine:
                states = {
                    "seller": make_trader_state("seller", cash=1000.0),
                    "buyer": make_trader_state("buyer", cash=1000.0),
                }
                mp.buy_intel("buyer", listing.listing_id, states, step=2)
                # Seller should NOT get full price if fake — clawback applied
                assert states["seller"].cash_balance < 1100.0, (
                    f"seed={seed}: seller got full price for fake intel"
                )
                return
        pytest.skip("Could not find fake intel seed in 200 tries")

    def test_unknown_listing_returns_none(self):
        rng = np.random.RandomState(42)
        mp = NewsMarketplace(rng)
        states = {"buyer": make_trader_state("buyer")}
        result = mp.buy_intel("buyer", "nonexistent_id", states, step=1)
        assert result is None
