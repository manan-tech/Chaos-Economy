import copy
import re
import numpy as np
from typing import Dict, Tuple, Any
from collections import defaultdict

from multi_agent.models import AgentRole, MultiAgentObservation, AgentState, MarketMakerAction, OversightAction
from multi_agent.rewards import calculate_trader_reward, calculate_mm_reward, calculate_oversight_reward
from multi_agent.manipulation_detector import ManipulationDetector
from multi_agent.order_matching import OrderMatchingEngine
from multi_agent.config import NUM_TRADERS, EPISODE_LENGTH, INITIAL_CASH

from vsr_env.engine.market_sim import advance_market, apply_black_swan
from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.models import VSRState
from vsr_env.engine.portfolio import add_position, update_positions_on_market_move
from multi_agent.black_swan import BlackSwanGenerator
from multi_agent.news_marketplace import NewsMarketplace
from multi_agent.messaging import MessageChannel

class MultiAgentVSREnvironment:
    AGENT_IDS = [f"trader_{i}" for i in range(NUM_TRADERS)] + ["market_maker", "oversight"]

    def __init__(self, episode_length: int = None):
        self.rng = None
        self._episode_length = episode_length
        self.option_engine = OptionChainEngine()
        self.manipulation_detector = ManipulationDetector()
        self.matching_engine = OrderMatchingEngine()
        self.current_step = 0
        self.vsr_state = None
        self.agent_states = {}
        self._agent_vsr_states = {}
        self.mm_last_spreads = {"atm": 0.02, "otm": 0.04, "itm": 0.03}
        self.trade_log = []
        self.intervention_log = []
        # Phase tracking for narrative arc
        self.training_phase = "oversight"  # Options: slaughter, adaptation, collusion, oversight
        self.total_fines_redistributed = 0.0
        self.black_swan_gen = None
        self.marketplace = None
        self.messaging = None
        self.signal_registry = []
        self.reputation_scores = {}

    def reset(self, seed: int = 42) -> Dict[str, MultiAgentObservation]:
        """Reset the environment."""
        self.current_step = 0
        self.rng = np.random.RandomState(seed)
        self.trade_log = []
        self.intervention_log = []
        self.signal_registry = []
        self.reputation_scores = {f"trader_{i}": {"correct_predictions": 0, "total_predictions": 0} for i in range(4)}
        
        # Base simulation state
        variance = 0.04
        spot_price = 100.0
        self.vsr_state = VSRState(
            episode_id=f"ep_{seed}",
            spot_price=spot_price,
            variance=variance,
            step_count=0
        )
        
        # Initialize agents
        self.agent_states = {}
        self._agent_vsr_states = {}
        for idx in range(NUM_TRADERS):
            agent_id = f"trader_{idx}"
            self.agent_states[agent_id] = AgentState(agent_id=agent_id, role=AgentRole.TRADER, cash_balance=INITIAL_CASH)
            self._agent_vsr_states[agent_id] = VSRState(episode_id=f"ep_{seed}", spot_price=spot_price, variance=variance, step_count=0)
            
        self.agent_states["market_maker"] = AgentState(agent_id="market_maker", role=AgentRole.MARKET_MAKER, cash_balance=INITIAL_CASH * 10)
        self._agent_vsr_states["market_maker"] = VSRState(episode_id=f"ep_{seed}", spot_price=spot_price, variance=variance, step_count=0)
        
        self.agent_states["oversight"] = AgentState(agent_id="oversight", role=AgentRole.OVERSIGHT, cash_balance=0.0)

        self.mm_last_spreads = {"atm": 0.02, "otm": 0.04, "itm": 0.03}
        
        # Initialize sub-systems
        env_len = self._episode_length if self._episode_length is not None else EPISODE_LENGTH
        self.black_swan_gen = BlackSwanGenerator(self.rng, env_len)
        self.marketplace = NewsMarketplace(self.rng)
        self.messaging = MessageChannel()

        return self._get_observations()

    def _normalize_trader_actions(self, actions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        normalized = {}
        for agent_id, action in actions.items():
            if not agent_id.startswith("trader"):
                continue
            if hasattr(action, "model_dump"):
                normalized[agent_id] = action.model_dump()
            elif isinstance(action, dict):
                normalized[agent_id] = dict(action)
        return normalized

    def _get_observations(self) -> Dict[str, MultiAgentObservation]:
        """Generate observations for all agents."""
        obs = {}
        S = self.vsr_state.spot_price
        sigma = np.sqrt(self.vsr_state.variance)
        
        # Simple IV surface
        iv_surface = [[sigma] * len(self.option_engine.MATURITIES) for _ in self.option_engine.STRIKES]
        
        risk_summary = self._build_agent_risk_summary()
        market_summary = self._build_market_state_summary()

        for agent_id, state in self.agent_states.items():
            obs[agent_id] = MultiAgentObservation(
                agent_id=agent_id,
                role=state.role,
                iv_surface=iv_surface,
                spot_price=S,
                mm_spreads=self.mm_last_spreads,
                own_greeks={"delta": state.portfolio_delta, "gamma": state.portfolio_gamma, "vega": state.portfolio_vega},
                own_pnl=state.portfolio_pnl,
                own_positions=state.positions,
                own_cash=state.cash_balance,
                step_number=self.current_step,
                steps_remaining=EPISODE_LENGTH - self.current_step,
                all_agent_pnls=None,
                trade_log=None,
                agent_risk_summary=None,
                market_state_summary=None,
                recent_interventions=None,
            )
            
        # Add oversight-specific data
        obs["oversight"].all_agent_pnls = {aid: s.portfolio_pnl for aid, s in self.agent_states.items() if s.role != AgentRole.OVERSIGHT}
        obs["oversight"].trade_log = self.trade_log[-50:] # keep recent
        obs["oversight"].agent_risk_summary = risk_summary
        obs["oversight"].market_state_summary = market_summary
        obs["oversight"].recent_interventions = self.intervention_log[-20:]

        # Handle News, Messaging, and Intel
        active_headline = None
        for event in self.black_swan_gen.events:
            if event.news_step <= self.current_step < event.trigger_step:
                active_headline = event.headline
                break
                
        private_intel_dict = defaultdict(list)
        for t in self.marketplace.transaction_log:
            if t["step"] == self.current_step:
                private_intel_dict[t["buyer_id"]].append(t)
                
        available_listings = {}
        for agent_id in obs:
            if agent_id.startswith("trader"):
                available_listings[agent_id] = self.marketplace.get_available_listings(agent_id, current_step=self.current_step)

        for agent_id, ob in obs.items():
            if agent_id.startswith("trader") or agent_id == "oversight":
                ob.news_headline = active_headline
            if agent_id.startswith("trader"):
                ob.private_intel = private_intel_dict[agent_id]
                ob.inbox = self.messaging.get_inbox(agent_id, self.current_step)
                
                # Add listings to market_stats for convenience
                # We do this later in the market_stats block.

        # Add enhanced market stats to ALL trader observations
        strike_volume = defaultdict(float)
        for t in self.trade_log[-25:]:
            strike_volume[t.get("selected_strike", -1)] += abs(t.get("quantity", 0))

        for agent_id in obs:
            if agent_id.startswith("trader"):
                obs[agent_id].market_stats = {
                    "strike_volume": dict(sorted(strike_volume.items())),
                    "total_fines_issued": self.total_fines_redistributed,
                    "training_phase": self.training_phase,
                    "available_intel_listings": available_listings.get(agent_id, [])
                }

        return obs

    def _build_agent_risk_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for agent_id, state in self.agent_states.items():
            if state.role == AgentRole.OVERSIGHT:
                continue
            total_contracts = float(sum(abs(pos.get("quantity", 0.0)) for pos in state.positions))
            risk_score = (
                abs(state.portfolio_delta) * 0.1
                + abs(state.portfolio_gamma) * 0.4
                + abs(state.portfolio_vega) * 0.1
                + total_contracts * 0.02
            )
            summary[agent_id] = {
                "pnl": float(state.portfolio_pnl),
                "delta": float(state.portfolio_delta),
                "gamma": float(state.portfolio_gamma),
                "vega": float(state.portfolio_vega),
                "cash": float(state.cash_balance),
                "contracts": total_contracts,
                "risk_score": float(risk_score),
            }
        return summary

    def _build_market_state_summary(self) -> Dict[str, float]:
        trade_window = self.trade_log[-25:]
        total_volume = float(sum(t.get("quantity", 0.0) for t in trade_window))
        mm_state = self.agent_states.get("market_maker", AgentState(agent_id="tmp", role=AgentRole.MARKET_MAKER))
        inventory_stress = (
            abs(mm_state.portfolio_delta) * 0.1
            + abs(mm_state.portfolio_gamma) * 0.6
            + abs(mm_state.portfolio_vega) * 0.08
        )
        avg_spread = (
            self.mm_last_spreads["atm"] + self.mm_last_spreads["otm"] + self.mm_last_spreads["itm"]
        ) / 3.0
        market_stability_score = float(inventory_stress + avg_spread + (total_volume * 0.01))
        return {
            "inventory_stress": float(inventory_stress),
            "avg_spread": float(avg_spread),
            "recent_volume": total_volume,
            "market_stability_score": market_stability_score,
        }

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, MultiAgentObservation], Dict[str, float], bool, Dict]:
        """Execute one step of the environment."""
        self.current_step += 1
        done = self.current_step >= EPISODE_LENGTH
        rewards = {agent_id: 0.0 for agent_id in self.AGENT_IDS}
        info = {}
        
        # Deep-copy previous state for reward deltas
        prev_states = copy.deepcopy(self.agent_states)
        
        # 1. Parse actions
        trader_actions = self._normalize_trader_actions(actions)
        
        mm_raw = actions.get("market_maker")
        if isinstance(mm_raw, dict):
            # Clamp all constrained fields to prevent Pydantic validation crashes
            mm_raw = dict(mm_raw)  # shallow copy
            mm_raw["atm_spread"] = max(0.001, min(0.20, float(mm_raw.get("atm_spread", 0.02))))
            mm_raw["otm_spread"] = max(0.001, min(0.30, float(mm_raw.get("otm_spread", 0.04))))
            mm_raw["itm_spread"] = max(0.001, min(0.25, float(mm_raw.get("itm_spread", 0.03))))
            mm_raw["skew_adjustment"] = max(-0.05, min(0.05, float(mm_raw.get("skew_adjustment", 0.0))))
            try:
                mm_action = MarketMakerAction(**mm_raw)
            except Exception:
                mm_action = MarketMakerAction()
        elif isinstance(mm_raw, MarketMakerAction):
            mm_action = mm_raw
        else:
            mm_action = MarketMakerAction()
            
        self.mm_last_spreads = {
            "atm": mm_action.atm_spread,
            "otm": mm_action.otm_spread,
            "itm": mm_action.itm_spread
        }

        oversight_raw = actions.get("oversight")
        if isinstance(oversight_raw, dict):
            # Clamp all constrained fields to prevent Pydantic validation crashes
            oversight_raw = dict(oversight_raw)  # shallow copy
            oversight_raw["confidence"] = max(0.0, min(1.0, float(oversight_raw.get("confidence", 0.0))))
            oversight_raw["fine_amount"] = max(0.0, min(100.0, float(oversight_raw.get("fine_amount", 0.0))))
            if not isinstance(oversight_raw.get("flagged_agents"), list):
                oversight_raw["flagged_agents"] = []
            if not isinstance(oversight_raw.get("halt_strikes"), list):
                oversight_raw["halt_strikes"] = []
            try:
                oversight_action = OversightAction(**oversight_raw)
            except Exception:
                oversight_action = OversightAction()
        elif isinstance(oversight_raw, OversightAction):
            oversight_action = oversight_raw
        else:
            oversight_action = OversightAction()
            
        # Handle messaging and intel actions first
        for agent_id, action in trader_actions.items():
            if action.get("sell_intel"):
                intel = action["sell_intel"]
                self.marketplace.post_listing(
                    seller_id=agent_id, 
                    price=intel.get("price", 50.0), 
                    content=intel.get("content", ""), 
                    target=intel.get("target", "all"),
                    current_step=self.current_step
                )
            if action.get("buy_intel"):
                self.marketplace.buy_intel(
                    buyer_id=agent_id,
                    listing_id=action["buy_intel"],
                    agent_states=self.agent_states,
                    step=self.current_step
                )
            if action.get("send_message"):
                msg = action["send_message"]
                target = msg.get("to", "all")
                text = msg.get("message", "")
                if target == "all":
                    self.messaging.broadcast(agent_id, text, self.current_step)
                elif target.startswith("group"):
                    self.messaging.send_group(agent_id, target, text, self.current_step)
                elif target.startswith("trader"):
                    self.messaging.send_dm(agent_id, target, text, self.current_step)
            # Create group? Trader action doesn't have it explicitly right now,
            # but maybe we can add it later if needed.
            
        # 2. Trader orders + Matching
        executed_trades = self.matching_engine.match_orders(
            trader_actions, mm_action, self.vsr_state.spot_price, self.option_engine, self.vsr_state.variance
        )

        step_trades = []

        # APPLY TRADES to agent portfolios
        for agent_id, trade in executed_trades.items():
            if isinstance(trade, dict) and trade.get("direction") in ["buy", "sell"]:
                execution_price = float(trade.get("execution_price", trade.get("theo_price", 0.0)))
                quantity = float(trade["quantity"])
                premium = execution_price * quantity

                # Log trade
                step_trade = {
                    "step": self.current_step,
                    "agent_id": agent_id,
                    **trade,
                }
                step_trades.append(step_trade)
                self.trade_log.append(step_trade)

                # Cash flow: buyers pay premium, sellers receive premium
                # Add market maker fee: $0.02 per contract executed
                mm_fee = quantity * 0.02
                
                if trade["direction"] == "buy":
                    self.agent_states[agent_id].cash_balance -= (premium + mm_fee)
                    self.agent_states["market_maker"].cash_balance += (premium + mm_fee)
                else:  # sell
                    self.agent_states[agent_id].cash_balance += (premium - mm_fee)
                    self.agent_states["market_maker"].cash_balance -= (premium - mm_fee)

                # Trader position
                add_position(
                    state=self._agent_vsr_states[agent_id],
                    strike_idx=trade["selected_strike"],
                    maturity_idx=trade["selected_maturity"],
                    direction=trade["direction"],
                    quantity=trade["quantity"],
                    engine=self.option_engine,
                    option_type=trade.get("option_type", "call"),
                    entry_price_override=execution_price,
                )
                
                # Market maker position (zero sum)
                mm_dir = "sell" if trade["direction"] == "buy" else "buy"
                add_position(
                    state=self._agent_vsr_states["market_maker"],
                    strike_idx=trade["selected_strike"],
                    maturity_idx=trade["selected_maturity"],
                    direction=mm_dir,
                    quantity=trade["quantity"],
                    engine=self.option_engine,
                    option_type=trade.get("option_type", "call"),
                    entry_price_override=execution_price,
                )
        
        # 3. Market advance
        advance_market(self.vsr_state, self.rng)
        
        for event in self.black_swan_gen.events:
            if event.trigger_step == self.current_step:
                apply_black_swan(self.vsr_state, event.spot_impact, event.variance_impact)
        
        # 5. Greeks/PnL Update 
        # Update Greeks/PnL for ALL agents after market advance
        for agent_id, state in self.agent_states.items():
            if state.role != AgentRole.OVERSIGHT:
                vsr = self._agent_vsr_states[agent_id]
                # Sync market conditions
                vsr.spot_price = self.vsr_state.spot_price
                vsr.variance = self.vsr_state.variance
                update_positions_on_market_move(vsr, self.option_engine)
                
                state.positions = vsr.positions
                state.portfolio_pnl = vsr.portfolio_pnl
                state.portfolio_delta = vsr.portfolio_delta
                state.portfolio_gamma = vsr.portfolio_gamma
                state.portfolio_vega = vsr.portfolio_vega
        
        pre_stability_score = self._build_market_state_summary()["market_stability_score"]

        # 4. Oversight labels and enforcement
        active_event = None
        for event in self.black_swan_gen.events:
            if event.news_step <= self.current_step <= event.trigger_step:
                active_event = event
                break
                
        env_info = {
            "current_step": self.current_step,
            "active_event": active_event,
            "intel_transactions": [t for t in self.marketplace.transaction_log if t["step"] == self.current_step],
            "messages_recent": [m for m in self.messaging.message_log if m["step"] >= self.current_step - 2],
            "channel_members": self.messaging.channels
        }

        ground_truth = {
            aid: self.manipulation_detector.detect_manipulation(self.agent_states[aid], step_trades, env_info)
            for aid in trader_actions
        }

        for flagged in oversight_action.flagged_agents:
            gt_label = ground_truth.get(flagged, "none")
            if (
                flagged in trader_actions
                and gt_label != "none"  # agent IS actually manipulating
                and oversight_action.flag_type != "none"  # SEC IS flagging something
            ):
                # Cap actual applied fine to match PnL scale
                applied_fine = min(oversight_action.fine_amount, 50.0)
                self.agent_states[flagged].fines_received += applied_fine
                self.agent_states[flagged].cash_balance -= applied_fine

                # REDISTRIBUTE fines: 80% to market maker, 20% to oversight
                mm_share = applied_fine * 0.8
                oversight_share = applied_fine * 0.2
                self.agent_states["market_maker"].cash_balance += mm_share
                self.agent_states["oversight"].cash_balance += oversight_share
                self.total_fines_redistributed += applied_fine

                intervention_record = {
                    "step": self.current_step,
                    "agent_id": flagged,
                    "flag_type": oversight_action.flag_type,
                    "fine_amount": oversight_action.fine_amount,
                    "intervention_type": oversight_action.intervention_type,
                }
                if oversight_action.halt_strikes or oversight_action.intervention_type == "halt":
                    self.agent_states[flagged].is_halted = True
                    intervention_record["halt_strikes"] = list(oversight_action.halt_strikes)
                self.intervention_log.append(intervention_record)

        post_stability_score = self._build_market_state_summary()["market_stability_score"]

        # Resolve pending signals (K=3 steps) — must happen before reward calculation
        resolved_this_step = []
        for sig in self.signal_registry:
            if not sig["resolved"] and self.current_step - sig["step_sent"] >= 3:
                sig["resolved"] = True
                price_moved_up = self.vsr_state.spot_price > sig["base_spot"]
                price_moved_down = self.vsr_state.spot_price < sig["base_spot"]
                if (sig["direction"] == "bullish" and price_moved_up) or \
                   (sig["direction"] == "bearish" and price_moved_down):
                    sig["correct"] = True
                    if sig["agent_id"] in self.reputation_scores:
                        self.reputation_scores[sig["agent_id"]]["correct_predictions"] += 1
                resolved_this_step.append(sig)

        # 5. Rewards
        for aid in trader_actions:
            direction = trader_actions[aid].get("direction", "hold") if aid in trader_actions else "hold"
            agent_resolved = [sig for sig in resolved_this_step if sig["agent_id"] == aid]
            rewards[aid] = calculate_trader_reward(
                self.agent_states[aid], prev_states[aid], aid, direction,
                resolved_signals=agent_resolved,
                training_phase=self.training_phase,
            )
        rewards["market_maker"] = calculate_mm_reward(
            self.agent_states["market_maker"],
            prev_states["market_maker"],
            len(executed_trades),
            mm_action,
        )
                                                      
        rewards["oversight"] = calculate_oversight_reward(
            oversight_action,
            ground_truth,
            pre_stability_score=pre_stability_score,
            post_stability_score=post_stability_score,
        )

        observations = self._get_observations()

        risk_summary = observations["oversight"].agent_risk_summary or {}
        market_summary = observations["oversight"].market_state_summary or {}
        info = {
            "trade_count": len(step_trades),
            "total_volume": float(sum(t.get("quantity", 0.0) for t in step_trades)),
            "market_maker_spreads": dict(self.mm_last_spreads),
            "detected_manipulations": ground_truth,
            "agent_risk_summary": risk_summary,
            "market_state_summary": market_summary,
            "recent_interventions": observations["oversight"].recent_interventions or [],
            "messages_this_step": [m for m in self.messaging.message_log if m["step"] == self.current_step],
            "intel_transactions": [t for t in self.marketplace.transaction_log if t["step"] == self.current_step]
        }

        return observations, rewards, done, info
