import math
from typing import Dict
from multi_agent.models import AgentState, MarketMakerAction, OversightAction


def squash_reward(raw_reward: float, limit: float = 5.0) -> float:
    """Squash reward to [-limit, limit] while preserving small signals.

    Values with |x| <= 1 pass through linearly; larger values are
    compressed with log to avoid extreme outliers dominating GRPO.
    """
    clamped = max(-limit, min(limit, raw_reward))
    if abs(clamped) <= 1.0:
        return clamped
    return math.copysign(1.0 + math.log(abs(clamped)), clamped)

def calculate_trader_reward(
    agent_state: AgentState,
    prev_state: AgentState,
    agent_id: str = "trader_0",
    direction: str = "hold",
    resolved_signals: list = None,
    training_phase: str = "oversight"
) -> float:
    """Calculate reward for a trader: Δ PnL + Δ Cash + Archetype Goals - risk penalties.

    The mark-to-market PnL delta alone is near-zero each step because
    GBM drift is 0 and option prices barely move.  Including the cash
    delta captures premium income (for sellers) and premium cost (for
    buyers), making the reward signal immediately meaningful.

    Fines are already reflected in cash_balance changes, so we don't
    subtract them separately to avoid double-counting.
    """
    pnl_delta = agent_state.portfolio_pnl - prev_state.portfolio_pnl
    cash_delta = agent_state.cash_balance - prev_state.cash_balance

    # 1. Total economic change = mark-to-market change + realized cash flow
    # Multiply by 10.0 to amplify small option premium signals
    total_economic_delta = (pnl_delta + cash_delta * 0.1) * 10.0

    # 2. Hold Penalty (mild — pipeline adds the stronger phase-scaled activity bonus on top)
    activity_bonus = -0.05 if direction == "hold" else 0.0

    # 3. Archetype-Specific Goals
    archetype_bonus = 0.0
    idx = int(agent_id.split("_")[1]) if "_" in agent_id else 0
    current_delta = abs(agent_state.portfolio_delta)

    if idx == 0:
        # Aggressive: directional risk
        archetype_bonus = 0.1 if current_delta > 1.0 else 0.0
    elif idx == 1:
        # Neutral: stay delta-hedged
        archetype_bonus = 0.1 if current_delta < 0.5 else -0.1
    elif idx == 2:
        # Contrarian: sell volatility (negative gamma)
        archetype_bonus = 0.1 if agent_state.portfolio_gamma < -0.05 else 0.0
    else:
        # trader_3 (scripted) and any future traders: no archetype bonus
        archetype_bonus = 0.0

    # 4. Inventory & Greek Risk Penalties
    total_contracts = sum(abs(pos.get("quantity", 0)) for pos in agent_state.positions)
    inventory_penalty = 1.0 if total_contracts > 50 else 0.0
    greeks_penalty = 1.0 if current_delta > 10.0 else 0.0

    # 5. Signal Alpha Bonus (Only active in Phase III/IV: collusion/oversight)
    signal_alpha_bonus = 0.0
    if training_phase in ["collusion", "oversight"] and resolved_signals:
        for sig in resolved_signals:
            if sig["correct"]:
                spread_multiplier = sig.get("spread_at_time", 0.04) * 10.0
                has_position = False
                for pos in agent_state.positions:
                    if pos.get("selected_strike") == sig["strike"] and abs(pos.get("quantity", 0.0)) > 0:
                        has_position = True
                        break
                if has_position:
                    signal_alpha_bonus += 0.5 * spread_multiplier
                else:
                    signal_alpha_bonus -= 0.1
            else:
                signal_alpha_bonus -= 0.2

    raw_reward = total_economic_delta + activity_bonus + archetype_bonus - inventory_penalty - greeks_penalty + signal_alpha_bonus
    return squash_reward(raw_reward)

def calculate_mm_reward(agent_state: AgentState, prev_state: AgentState, 
                        volume_traded: int, mm_action: MarketMakerAction) -> float:
    """Calculate reward for Market Maker.

    The market maker should not learn the degenerate policy of widening
    spreads forever. Reward balances PnL, facilitated volume, quote quality,
    and inventory control.
    """
    pnl_delta = agent_state.portfolio_pnl - prev_state.portfolio_pnl
    cash_delta = agent_state.cash_balance - prev_state.cash_balance
    # Include cash flow: MM earns spread via premium income in cash_balance
    pnl_delta = pnl_delta + cash_delta * 0.1
    flow_reward = volume_traded * 0.15

    target_spreads = {"atm": 0.04, "otm": 0.06, "itm": 0.05}
    spread_distance = (
        abs(mm_action.atm_spread - target_spreads["atm"])
        + abs(mm_action.otm_spread - target_spreads["otm"])
        + abs(mm_action.itm_spread - target_spreads["itm"])
    )
    quote_quality_reward = max(0.0, 0.15 - spread_distance)

    total_contracts = sum(abs(pos.get("quantity", 0)) for pos in agent_state.positions)
    inventory_penalty = (
        abs(agent_state.portfolio_delta) * 0.01
        + abs(agent_state.portfolio_gamma) * 0.05
        + abs(agent_state.portfolio_vega) * 0.02
        + total_contracts * 0.005
    )
    spread_extremity_penalty = 0.0
    if max(mm_action.atm_spread, mm_action.otm_spread, mm_action.itm_spread) > 0.12:
        spread_extremity_penalty = 0.5

    survival_bonus = 0.5 if agent_state.cash_balance > 0 else 0.0

    raw_reward = pnl_delta + flow_reward + quote_quality_reward - inventory_penalty - spread_extremity_penalty + survival_bonus
    return squash_reward(raw_reward)

def calculate_oversight_reward(oversight_action: OversightAction, 
                               ground_truth_manipulations: Dict[str, str],
                               pre_stability_score: float = 0.0,
                               post_stability_score: float = 0.0) -> float:
    """Calculate reward for Oversight agent.

    Oversight is framed as market surveillance:
    detect harmful behavior, explain it, and improve market stability.
    """
    reward = 0.0

    harmful_events = {
        agent_id: label
        for agent_id, label in ground_truth_manipulations.items()
        if label != "none"
    }

    # Check flagged agents (True/False Positives)
    true_positive_count = 0
    for flagged_agent in oversight_action.flagged_agents:
        true_manipulation = ground_truth_manipulations.get(flagged_agent, "none")
        # Accept any non-none flag on a truly manipulating agent
        if true_manipulation != "none" and oversight_action.flag_type != "none":
            true_positive_count += 1
            reward += 1.0 + min(0.5, oversight_action.fine_amount / 200.0)  # true positive + fine bonus
            # Bonus for exact category match
            if oversight_action.flag_type == true_manipulation:
                reward += 0.3
        else:
            reward -= 0.5  # false positive (STRENGTHENED from -0.3)

    # Check unflagged agents (False Negatives)
    for agent_id, true_manipulation in harmful_events.items():
        if agent_id not in oversight_action.flagged_agents:
            reward -= 1.0  # missed manipulation! false negative

    if not harmful_events and not oversight_action.flagged_agents:
        reward += 0.5  # correct restraint — rewarded for NOT flagging when nothing's wrong

    # REMOVED unconditional patrol bonus (+0.1)
    # It rewarded showing up regardless of accuracy.
    # Now only reward patrol effort when there are actual true positives.
    if true_positive_count > 0:
        reward += 0.1  # patrol bonus only for accurate surveillance

    reasoning = oversight_action.reasoning.lower()
    if oversight_action.flag_type != "none" and oversight_action.flag_type in reasoning:
        reward += 0.2
    if any(agent_id.lower() in reasoning for agent_id in oversight_action.flagged_agents):
        reward += 0.1

    # Gate intervention bonuses on accuracy
    # Only reward fines/halts when they're backed by true positives
    if true_positive_count > 0:
        if oversight_action.intervention_type == "fine" and oversight_action.fine_amount > 0:
            reward += 0.1
        if oversight_action.intervention_type == "halt" and oversight_action.halt_strikes:
            reward += 0.15
    else:
        # Penalize unwarranted intervention (no true positives but still fining/halting)
        if oversight_action.intervention_type in ("fine", "halt"):
            reward -= 0.3

    # Penalize excessive fines
    if oversight_action.fine_amount > 100:
        reward -= 0.3  # discourage max-fine strategies

    stability_improvement = max(0.0, pre_stability_score - post_stability_score)
    reward += min(0.3, stability_improvement * 0.2)

    return max(-5.0, min(5.0, reward))
