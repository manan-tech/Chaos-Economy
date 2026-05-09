"""Reward functions for the single-stock multi-agent simulator.

All 10 reward-hacking vectors identified in the design have been closed:
  1. Activity bonus removed (hold has no penalty or shaping reward).
  2. Archetype bonuses gated on realized_pnl >= 0.
  3. Signal alpha requires |shares| >= 5 at send time (enforced in env) + 0.3% confirming move.
  4. MM flow_reward gated on half_spread <= 0.15.
  5. Coordination bonus (in pipeline) gated on realized_pnl > 0.
  6. Fines routed to treasury → non-flagged traders (done in environment).
  7. Collusion ledger is info-only; no reward path through it.
  8. Quantity below bucket_min treated as hold in action parser.
  9. Inventory penalty is smooth quadratic (no step-function cliff).
  10. News front-run closed by signal-registry |shares| >= 5 gate in env.
"""

import math
from typing import Dict, List, Optional

from multi_agent.config import MAX_POSITION
from multi_agent.models import AgentState, OversightAction


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_TRADER_IDX_TO_ARCHETYPE = {
    0: "momentum",
    1: "mean_reversion",
    2: "vol_timing",
    3: "scripted",
}


def _get_trader_archetype(agent_id: str) -> str:
    """Extract archetype from agent_id (e.g., 'trader_0' -> 'momentum')."""
    try:
        idx = int(agent_id.split("_")[1])
        return _TRADER_IDX_TO_ARCHETYPE.get(idx, "scripted")
    except (IndexError, ValueError):
        return "scripted"


def squash_reward(raw: float, limit: float = 5.0) -> float:
    """Squash to [-limit, limit]; linear in [-1, 1], log-compressed outside."""
    clamped = max(-limit, min(limit, raw))
    if abs(clamped) <= 1.0:
        return clamped
    return math.copysign(1.0 + math.log(abs(clamped)), clamped)


_PHASE_SIZE_ALPHA = {
    "slaughter": 0.30,
    "adaptation": 0.15,
    "collusion": 0.0,
    "oversight": 0.0,
}

_BUCKET_WEIGHT = {"small": 0.0, "medium": 0.5, "large": 1.0}


# ---------------------------------------------------------------------------
# Trader reward
# ---------------------------------------------------------------------------

def calculate_trader_reward(
    curr: AgentState,
    prev: AgentState,
    agent_id: str = "trader_0",
    direction: str = "hold",
    size_bucket: str = "small",
    prev_spot: float = 100.0,
    curr_spot: float = 100.0,
    realized_vol: float = 0.15,
    resolved_signals: Optional[List[Dict]] = None,
    training_phase: str = "oversight",
) -> float:
    """Compute squashed reward for one trader step.

    realized_pnl is the complete economic delta for the step:
        (mtm change) + (cash change from fills, fines, redistributed treasury)

    All shaping terms (size_bonus, archetype_bonus) are conditioned on
    realized_pnl >= 0 so they can never convert a losing step into a winner.
    """
    # ── Core economic signal ──────────────────────────────────────────────
    pnl_delta = curr.portfolio_pnl - prev.portfolio_pnl
    cash_delta = curr.cash_balance - prev.cash_balance
    realized_pnl = (pnl_delta + cash_delta) * 10.0
    traded = direction in ("buy", "sell")

    # ── Size exploration bonus (decays to 0 from Collusion onward) ────────
    alpha = _PHASE_SIZE_ALPHA.get(training_phase, 0.0)
    bucket_w = _BUCKET_WEIGHT.get(size_bucket, 0.0)
    # Gated on trading this step AND non-negative realized_pnl
    size_bonus = alpha * bucket_w if (traded and realized_pnl >= 0) else 0.0

    # ── Archetype bonus  ──────────────────────────────────────────────────
    archetype_bonus = _archetype_bonus(
        agent_id, direction, size_bucket, prev_spot, curr_spot, realized_vol, realized_pnl
    )

    # ── Inventory penalty (smooth quadratic, no cliff) ────────────────────
    position_ratio = abs(curr.shares) / float(MAX_POSITION)
    inventory_pen = 0.5 * position_ratio ** 2

    # ── Signal alpha bonus ────────────────────────────────────────────────
    signal_bonus = _signal_bonus(curr, resolved_signals or [], training_phase)

    raw = realized_pnl + size_bonus + archetype_bonus - inventory_pen + signal_bonus
    return squash_reward(raw)


def _archetype_bonus(
    agent_id: str,
    direction: str,
    size_bucket: str,
    prev_spot: float,
    curr_spot: float,
    realized_vol: float,
    realized_pnl: float,
) -> float:
    """Return archetype-specific shaping bonus, conditioned on realized_pnl >= 0."""
    if realized_pnl < 0:
        return 0.0

    archetype = _get_trader_archetype(agent_id)
    spot_return = (curr_spot - prev_spot) / max(prev_spot, 1e-8)
    trend_sign = 1 if spot_return > 0.0001 else (-1 if spot_return < -0.0001 else 0)

    if archetype == "momentum":
        direction_sign = 1 if direction == "buy" else (-1 if direction == "sell" else 0)
        if size_bucket in ("medium", "large") and trend_sign != 0 and direction_sign == trend_sign:
            return 0.15
        if size_bucket == "large" and trend_sign != 0 and direction_sign == -trend_sign:
            return -0.05

    elif archetype == "mean_reversion":
        direction_sign = 1 if direction == "buy" else (-1 if direction == "sell" else 0)
        if size_bucket in ("small", "medium") and trend_sign != 0 and direction_sign == -trend_sign:
            return 0.15
        if size_bucket == "large" and trend_sign != 0 and direction_sign == trend_sign:
            return -0.05

    elif archetype == "vol_timing":
        high_vol_threshold = 0.25
        if size_bucket == "large" and realized_vol >= high_vol_threshold:
            return 0.15
        if size_bucket == "large" and realized_vol < high_vol_threshold:
            return -0.10

    return 0.0


def _signal_bonus(
    curr: AgentState,
    resolved_signals: List[Dict],
    training_phase: str,
) -> float:
    """K=3 signal-accuracy bonus. Active only in collusion / oversight phases.

    Reward only if signal-sender held |shares| >= 5 at send time (checked
    in environment before registering; `position_at_send` echoed in the signal).
    Symmetric ±: correct +0.4, wrong -0.3.
    """
    if training_phase not in ("collusion", "oversight"):
        return 0.0

    bonus = 0.0
    for sig in resolved_signals:
        if sig.get("correct"):
            bonus += 0.4
        else:
            bonus -= 0.3
    return bonus


# ---------------------------------------------------------------------------
# Market maker reward
# ---------------------------------------------------------------------------

def calculate_mm_reward(
    curr: AgentState,
    prev: AgentState,
    volume_traded: int,
    half_spread: float,
) -> float:
    """Reward for the market maker.

    Hacking fixes applied:
    - flow_reward gated on half_spread <= 0.15 (closes widen-and-earn hack).
    - Inventory penalty is quadratic in |shares|/MAX_POSITION.
    - No Greek penalties (no Greeks in the stock world).
    """
    pnl_delta = curr.portfolio_pnl - prev.portfolio_pnl
    cash_delta = curr.cash_balance - prev.cash_balance
    economic = pnl_delta + cash_delta * 0.1

    # Flow reward: only when spreads are reasonably tight
    flow_reward = volume_traded * 0.05 if half_spread <= 0.15 else 0.0

    # Inventory penalty: quadratic
    position_ratio = abs(curr.shares) / float(MAX_POSITION)
    inventory_pen = 0.01 * abs(curr.shares) + 0.005 * position_ratio ** 2 * MAX_POSITION

    # Spread extremity penalty
    spread_extreme_pen = 0.5 if half_spread > 0.30 else 0.0

    # Survival bonus
    survival = 0.3 if curr.cash_balance > 0 else 0.0

    raw = economic + flow_reward - inventory_pen - spread_extreme_pen + survival
    return squash_reward(raw)


# ---------------------------------------------------------------------------
# Oversight reward
# ---------------------------------------------------------------------------

def calculate_oversight_reward(
    oversight_action: OversightAction,
    ground_truth: Dict[str, str],
    pre_stability_score: float = 0.0,
    post_stability_score: float = 0.0,
) -> float:
    """Reward for the SEC oversight agent.

    True positives rewarded; false positives penalised. Unwarranted
    interventions carry an additional penalty (closes false-positive farming).
    """
    reward = 0.0
    harmful = {aid: lbl for aid, lbl in ground_truth.items() if lbl != "none"}
    tp_count = 0

    for flagged in oversight_action.flagged_agents:
        true_label = ground_truth.get(flagged, "none")
        if true_label != "none" and oversight_action.flag_type != "none":
            tp_count += 1
            reward += 1.0 + min(0.5, oversight_action.fine_amount / 200.0)
            if oversight_action.flag_type == true_label:
                reward += 0.3  # exact category match bonus
        else:
            reward -= 0.5   # false positive

    # False negatives
    for aid in harmful:
        if aid not in oversight_action.flagged_agents:
            reward -= 1.0

    # Correct restraint
    if not harmful and not oversight_action.flagged_agents:
        reward += 0.5

    # Patrol bonus only when accurate
    if tp_count > 0:
        reward += 0.1

    # Reasoning quality
    reasoning = oversight_action.reasoning.lower()
    if oversight_action.flag_type != "none" and oversight_action.flag_type in reasoning:
        reward += 0.2
    if any(aid.lower() in reasoning for aid in oversight_action.flagged_agents):
        reward += 0.1

    # Intervention reward/penalty gated on accuracy
    if tp_count > 0:
        if oversight_action.intervention_type == "fine" and oversight_action.fine_amount > 0:
            reward += 0.1
        if oversight_action.intervention_type == "halt":
            reward += 0.15
    else:
        if oversight_action.intervention_type in ("fine", "halt"):
            reward -= 0.3  # unwarranted intervention

    # Penalise runaway fines
    if oversight_action.fine_amount > 100:
        reward -= 0.3

    # Stability improvement
    stability_gain = max(0.0, pre_stability_score - post_stability_score)
    reward += min(0.3, stability_gain * 0.2)

    return max(-5.0, min(5.0, reward))
