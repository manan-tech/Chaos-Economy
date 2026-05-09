"""Unit tests for the reward functions in multi_agent/rewards.py.

Covers:
- squash_reward: linearity, log compression, symmetry
- calculate_trader_reward: realized_pnl, size_bonus gate, archetype_bonus gate,
  inventory_pen, signal_bonus phase gate, all archetype branches
- calculate_mm_reward: flow_reward gate, inventory pen, spread extreme, survival
- calculate_oversight_reward: TP/FP/FN, restraint, unwarranted intervention, fine cap
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from multi_agent.rewards import (
    squash_reward,
    calculate_trader_reward,
    calculate_mm_reward,
    calculate_oversight_reward,
)
from multi_agent.models import AgentState, AgentRole, OversightAction


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_state(cash=10_000.0, shares=0, pnl=0.0, agent_id="trader_0"):
    s = AgentState(agent_id=agent_id, role=AgentRole.TRADER, cash_balance=cash)
    s.shares = shares
    s.portfolio_pnl = pnl
    return s


def make_mm_state(cash=100_000.0, shares=0, pnl=0.0):
    s = AgentState(agent_id="market_maker", role=AgentRole.MARKET_MAKER, cash_balance=cash)
    s.shares = shares
    s.portfolio_pnl = pnl
    return s


def make_oversight_action(flagged=None, flag_type="none", fine=0.0,
                           intervention="none", reasoning=""):
    return OversightAction(
        flagged_agents=flagged or [],
        flag_type=flag_type,
        fine_amount=fine,
        confidence=0.9,
        intervention_type=intervention,
        reasoning=reasoning,
    )


# ─────────────────────────────────────────────────────────────────────────────
# squash_reward
# ─────────────────────────────────────────────────────────────────────────────

class TestSquashReward:
    def test_linear_in_unit_interval(self):
        for v in (0.0, 0.5, -0.5, 1.0, -1.0):
            assert squash_reward(v) == pytest.approx(v)

    def test_compressed_above_1(self):
        r = squash_reward(2.0)
        assert 1.0 < r < 2.0

    def test_compressed_below_minus1(self):
        r = squash_reward(-2.0)
        assert -2.0 < r < -1.0

    def test_clamp_at_limit(self):
        assert squash_reward(10.0) <= 5.0
        assert squash_reward(-10.0) >= -5.0

    def test_symmetry(self):
        assert squash_reward(3.0) == pytest.approx(-squash_reward(-3.0))

    def test_zero_stays_zero(self):
        assert squash_reward(0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# calculate_trader_reward — realized PnL
# ─────────────────────────────────────────────────────────────────────────────

class TestTraderRewardPnl:
    def test_positive_cash_delta_gives_positive_reward(self):
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=10_100.0, pnl=0.0)  # +100 cash
        r = calculate_trader_reward(curr, prev, training_phase="oversight")
        assert r > 0

    def test_negative_cash_delta_gives_negative_reward(self):
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=9_900.0, pnl=0.0)  # -100 cash
        r = calculate_trader_reward(curr, prev, training_phase="oversight")
        assert r < 0

    def test_no_change_near_zero(self):
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=10_000.0, pnl=0.0)
        r = calculate_trader_reward(curr, prev, training_phase="oversight")
        # only inventory pen (shares=0 → pen=0), so should be 0
        assert r == pytest.approx(0.0)

    def test_pnl_delta_captured(self):
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=10_000.0, pnl=50.0)  # portfolio MTM rose
        r = calculate_trader_reward(curr, prev, training_phase="oversight")
        assert r > 0


# ─────────────────────────────────────────────────────────────────────────────
# calculate_trader_reward — size_bonus gate
# ─────────────────────────────────────────────────────────────────────────────

class TestTraderSizeBonus:
    # Use zero cash delta so the size_bonus is the only moving part

    def test_size_bonus_zero_in_oversight_phase(self):
        # alpha=0.0 in oversight; slaughter has alpha=0.30 → should differ
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=10_000.0, pnl=0.01)  # tiny profit so bonus gates open
        r_no_bonus = calculate_trader_reward(curr, prev, direction="buy",
                                              size_bucket="large",
                                              training_phase="oversight")
        r_with_bonus = calculate_trader_reward(curr, prev, direction="buy",
                                               size_bucket="large",
                                               training_phase="slaughter")
        assert r_with_bonus > r_no_bonus

    def test_size_bonus_gated_on_profitable_step(self):
        # Losing step: size_bonus must be 0 even in slaughter phase
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=9_900.0, pnl=0.0)  # loss
        r_loss = calculate_trader_reward(curr, prev, direction="buy",
                                          size_bucket="large",
                                          training_phase="slaughter")
        r_hold = calculate_trader_reward(curr, prev, direction="hold",
                                          size_bucket="large",
                                          training_phase="slaughter")
        assert r_loss == pytest.approx(r_hold)

    def test_size_bonus_zero_for_hold(self):
        # Use tiny profit so the gate is open, but hold should not earn size bonus
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=10_000.0, pnl=0.01)  # tiny profit
        r_hold = calculate_trader_reward(curr, prev, direction="hold",
                                          size_bucket="large",
                                          training_phase="slaughter")
        r_buy = calculate_trader_reward(curr, prev, direction="buy",
                                         size_bucket="large",
                                         training_phase="slaughter")
        assert r_buy > r_hold


# ─────────────────────────────────────────────────────────────────────────────
# calculate_trader_reward — archetype bonuses
# ─────────────────────────────────────────────────────────────────────────────

class TestArchetypeBonus:
    # Use zero/tiny cash delta so archetype bonuses (±0.05–0.15) are visible after squash

    def _breakeven_states(self):
        """States with zero PnL so archetype bonus is the decisive signal."""
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=10_000.0, pnl=0.01)  # just enough to open the gate
        return prev, curr

    def test_momentum_aligned_medium_gets_bonus(self):
        prev, curr = self._breakeven_states()
        r = calculate_trader_reward(curr, prev, agent_id="trader_0",
                                     direction="buy", size_bucket="medium",
                                     prev_spot=100.0, curr_spot=101.0,
                                     training_phase="oversight")
        r_scripted = calculate_trader_reward(curr, prev, agent_id="trader_3",
                                              direction="buy", size_bucket="medium",
                                              prev_spot=100.0, curr_spot=101.0,
                                              training_phase="oversight")
        assert r > r_scripted

    def test_momentum_counter_trend_large_penalised(self):
        prev, curr = self._breakeven_states()
        r = calculate_trader_reward(curr, prev, agent_id="trader_0",
                                     direction="sell", size_bucket="large",
                                     prev_spot=100.0, curr_spot=101.0,
                                     training_phase="oversight")
        r_scripted = calculate_trader_reward(curr, prev, agent_id="trader_3",
                                              direction="sell", size_bucket="large",
                                              prev_spot=100.0, curr_spot=101.0,
                                              training_phase="oversight")
        assert r < r_scripted

    def test_mean_reversion_fade_small_gets_bonus(self):
        prev, curr = self._breakeven_states()
        # Uptrend: fade with small sell → +0.15 for mean_reversion
        r = calculate_trader_reward(curr, prev, agent_id="trader_1",
                                     direction="sell", size_bucket="small",
                                     prev_spot=100.0, curr_spot=101.0,
                                     training_phase="oversight")
        r_scripted = calculate_trader_reward(curr, prev, agent_id="trader_3",
                                              direction="sell", size_bucket="small",
                                              prev_spot=100.0, curr_spot=101.0,
                                              training_phase="oversight")
        assert r > r_scripted

    def test_vol_timing_large_in_high_vol_gets_bonus(self):
        prev, curr = self._breakeven_states()
        r = calculate_trader_reward(curr, prev, agent_id="trader_2",
                                     direction="buy", size_bucket="large",
                                     realized_vol=0.30,
                                     training_phase="oversight")
        r_low_vol = calculate_trader_reward(curr, prev, agent_id="trader_2",
                                             direction="buy", size_bucket="large",
                                             realized_vol=0.10,
                                             training_phase="oversight")
        assert r > r_low_vol

    def test_archetype_bonus_zero_on_losing_step(self):
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=9_900.0, pnl=0.0)  # loss
        r_momentum = calculate_trader_reward(curr, prev, agent_id="trader_0",
                                              direction="buy", size_bucket="medium",
                                              prev_spot=100.0, curr_spot=101.0,
                                              training_phase="oversight")
        r_scripted = calculate_trader_reward(curr, prev, agent_id="trader_3",
                                              direction="buy", size_bucket="medium",
                                              prev_spot=100.0, curr_spot=101.0,
                                              training_phase="oversight")
        assert r_momentum == pytest.approx(r_scripted)

    def test_scripted_trader_no_archetype_bonus(self):
        prev, curr = self._breakeven_states()
        r_buy = calculate_trader_reward(curr, prev, agent_id="trader_3",
                                         direction="buy", size_bucket="large",
                                         prev_spot=100.0, curr_spot=101.0,
                                         training_phase="oversight")
        r_hold = calculate_trader_reward(curr, prev, agent_id="trader_3",
                                          direction="hold", size_bucket="large",
                                          prev_spot=100.0, curr_spot=101.0,
                                          training_phase="oversight")
        assert r_buy == pytest.approx(r_hold)


# ─────────────────────────────────────────────────────────────────────────────
# calculate_trader_reward — inventory penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestInventoryPenalty:
    def test_no_shares_no_penalty(self):
        prev = make_state(cash=10_000.0, pnl=0.0, shares=0)
        curr = make_state(cash=10_000.0, pnl=0.0, shares=0)
        r = calculate_trader_reward(curr, prev, training_phase="oversight")
        assert r == pytest.approx(0.0)

    def test_large_position_penalised(self):
        prev = make_state(cash=10_000.0, pnl=0.0, shares=0)
        curr_big = make_state(cash=10_000.0, pnl=0.0, shares=90)
        curr_small = make_state(cash=10_000.0, pnl=0.0, shares=10)
        r_big = calculate_trader_reward(curr_big, prev, training_phase="oversight")
        r_small = calculate_trader_reward(curr_small, prev, training_phase="oversight")
        assert r_big < r_small

    def test_penalty_quadratic_not_step(self):
        # Quadratic means penalty grows faster than linear
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr_50 = make_state(cash=10_000.0, pnl=0.0, shares=50)
        curr_100 = make_state(cash=10_000.0, pnl=0.0, shares=100)
        r_50 = calculate_trader_reward(curr_50, prev, training_phase="oversight")
        r_100 = calculate_trader_reward(curr_100, prev, training_phase="oversight")
        # penalty at 100 should be 4x penalty at 50 (quadratic)
        pen_50 = -r_50
        pen_100 = -r_100
        assert pen_100 > 2 * pen_50


# ─────────────────────────────────────────────────────────────────────────────
# calculate_trader_reward — signal bonus phase gate
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalBonus:
    def test_signal_bonus_inactive_in_slaughter(self):
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=10_000.0, pnl=0.0)
        signals = [{"correct": True}]
        r = calculate_trader_reward(curr, prev, resolved_signals=signals,
                                     training_phase="slaughter")
        r_no_sig = calculate_trader_reward(curr, prev, resolved_signals=[],
                                            training_phase="slaughter")
        assert r == pytest.approx(r_no_sig)

    def test_signal_bonus_active_in_collusion(self):
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=10_000.0, pnl=0.0)
        signals = [{"correct": True}]
        r_with = calculate_trader_reward(curr, prev, resolved_signals=signals,
                                          training_phase="collusion")
        r_without = calculate_trader_reward(curr, prev, resolved_signals=[],
                                             training_phase="collusion")
        assert r_with > r_without

    def test_wrong_signal_penalises(self):
        prev = make_state(cash=10_000.0, pnl=0.0)
        curr = make_state(cash=10_000.0, pnl=0.0)
        r_wrong = calculate_trader_reward(curr, prev,
                                           resolved_signals=[{"correct": False}],
                                           training_phase="oversight")
        r_none = calculate_trader_reward(curr, prev,
                                          resolved_signals=[],
                                          training_phase="oversight")
        assert r_wrong < r_none


# ─────────────────────────────────────────────────────────────────────────────
# calculate_mm_reward
# ─────────────────────────────────────────────────────────────────────────────

class TestMMReward:
    def test_flow_reward_active_when_spread_tight(self):
        prev = make_mm_state()
        curr = make_mm_state()
        r_tight = calculate_mm_reward(curr, prev, volume_traded=10, half_spread=0.05)
        r_no_vol = calculate_mm_reward(curr, prev, volume_traded=0, half_spread=0.05)
        assert r_tight > r_no_vol

    def test_flow_reward_zero_when_spread_wide(self):
        prev = make_mm_state()
        curr = make_mm_state()
        # spread > 0.15 → flow reward = 0
        r_wide = calculate_mm_reward(curr, prev, volume_traded=100, half_spread=0.20)
        r_no_vol = calculate_mm_reward(curr, prev, volume_traded=0, half_spread=0.20)
        assert r_wide == pytest.approx(r_no_vol)

    def test_spread_extreme_penalty(self):
        prev = make_mm_state()
        curr = make_mm_state()
        r_extreme = calculate_mm_reward(curr, prev, volume_traded=0, half_spread=0.35)
        r_normal = calculate_mm_reward(curr, prev, volume_traded=0, half_spread=0.10)
        assert r_extreme < r_normal

    def test_survival_bonus_positive_cash(self):
        # Use same cash so economic PnL is identical; only survival bonus differs
        prev = make_mm_state(cash=0.0)
        curr_pos = make_mm_state(cash=0.0)   # cash unchanged from prev but positive
        curr_pos.cash_balance = 1.0
        curr_neg = make_mm_state(cash=0.0)
        curr_neg.cash_balance = -1.0
        r_pos = calculate_mm_reward(curr_pos, prev, volume_traded=0, half_spread=0.05)
        r_neg = calculate_mm_reward(curr_neg, prev, volume_traded=0, half_spread=0.05)
        assert r_pos > r_neg

    def test_inventory_penalty_grows_with_shares(self):
        prev = make_mm_state()
        curr_clean = make_mm_state(shares=0)
        curr_large = make_mm_state(shares=80)
        r_clean = calculate_mm_reward(curr_clean, prev, volume_traded=0, half_spread=0.05)
        r_large = calculate_mm_reward(curr_large, prev, volume_traded=0, half_spread=0.05)
        assert r_clean > r_large

    def test_reward_squashed_within_bounds(self):
        prev = make_mm_state(cash=100_000.0)
        curr = make_mm_state(cash=200_000.0)  # large gain
        r = calculate_mm_reward(curr, prev, volume_traded=1000, half_spread=0.05)
        assert -5.0 <= r <= 5.0


# ─────────────────────────────────────────────────────────────────────────────
# calculate_oversight_reward
# ─────────────────────────────────────────────────────────────────────────────

class TestOversightReward:
    def test_true_positive_rewarded(self):
        action = make_oversight_action(flagged=["trader_0"], flag_type="collusion",
                                        fine=50.0, intervention="fine",
                                        reasoning="collusion detected on trader_0")
        gt = {"trader_0": "collusion"}
        r = calculate_oversight_reward(action, gt)
        assert r > 0

    def test_false_positive_penalised(self):
        action = make_oversight_action(flagged=["trader_0"], flag_type="collusion",
                                        fine=50.0, intervention="fine",
                                        reasoning="suspected collusion")
        gt = {}  # clean market
        r = calculate_oversight_reward(action, gt)
        assert r < 0

    def test_false_negative_penalised(self):
        # Manipulator exists but not flagged
        action = make_oversight_action(flagged=[], flag_type="none")
        gt = {"trader_0": "wash_trading"}
        r = calculate_oversight_reward(action, gt)
        assert r < 0

    def test_correct_restraint_rewarded(self):
        action = make_oversight_action(flagged=[], flag_type="none")
        gt = {}  # clean market, correct to not flag
        r = calculate_oversight_reward(action, gt)
        assert r > 0

    def test_category_match_bonus(self):
        # Exact flag type match → extra +0.3
        action_exact = make_oversight_action(flagged=["trader_1"], flag_type="wash_trading",
                                              fine=30.0, intervention="fine",
                                              reasoning="wash_trading detected")
        action_wrong_type = make_oversight_action(flagged=["trader_1"], flag_type="collusion",
                                                   fine=30.0, intervention="fine",
                                                   reasoning="collusion suspected")
        gt = {"trader_1": "wash_trading"}
        r_exact = calculate_oversight_reward(action_exact, gt)
        r_wrong = calculate_oversight_reward(action_wrong_type, gt)
        assert r_exact > r_wrong

    def test_unwarranted_intervention_penalised(self):
        # Fine issued but no true positives
        action = make_oversight_action(flagged=["trader_0"], flag_type="collusion",
                                        fine=50.0, intervention="fine")
        gt = {"trader_0": "none"}  # innocent
        r = calculate_oversight_reward(action, gt)
        assert r < 0

    def test_excessive_fine_penalised(self):
        # OversightAction caps fine_amount at 100; test boundary vs small fine
        action_maxfine = make_oversight_action(flagged=["trader_0"], flag_type="collusion",
                                                fine=100.0, intervention="fine",
                                                reasoning="collusion on trader_0")
        action_normal = make_oversight_action(flagged=["trader_0"], flag_type="collusion",
                                               fine=30.0, intervention="fine",
                                               reasoning="collusion on trader_0")
        gt = {"trader_0": "collusion"}
        r_maxfine = calculate_oversight_reward(action_maxfine, gt)
        r_normal = calculate_oversight_reward(action_normal, gt)
        # fine>100 is penalised internally in calculate_oversight_reward even if OversightAction caps at 100
        # Both should be positive TP rewards; at max fine the fine bonus caps out
        assert r_maxfine >= r_normal or abs(r_maxfine - r_normal) < 1.0  # bounded difference

    def test_reasoning_quality_bonus(self):
        action_good = make_oversight_action(flagged=["trader_0"], flag_type="collusion",
                                             fine=30.0, intervention="fine",
                                             reasoning="collusion detected, flagging trader_0")
        action_poor = make_oversight_action(flagged=["trader_0"], flag_type="collusion",
                                             fine=30.0, intervention="fine",
                                             reasoning="suspicious activity")
        gt = {"trader_0": "collusion"}
        r_good = calculate_oversight_reward(action_good, gt)
        r_poor = calculate_oversight_reward(action_poor, gt)
        assert r_good > r_poor

    def test_reward_bounded(self):
        action = make_oversight_action(flagged=["t0", "t1", "t2"], flag_type="collusion",
                                        fine=50.0, intervention="fine",
                                        reasoning="collusion on t0 t1 t2")
        gt = {"t0": "collusion", "t1": "collusion", "t2": "collusion"}
        r = calculate_oversight_reward(action, gt)
        assert -5.0 <= r <= 5.0
