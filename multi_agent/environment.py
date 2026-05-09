"""Multi-agent single-stock trading environment.

Each episode runs `EPISODE_LENGTH` steps. At every step:
  1. MM quotes half-spread + skew.
  2. Traders submit {direction, size_bucket, quantity, ...}.
  3. Order matching fills at MM bid/ask; net flow nudges spot.
  4. Spot advances one GBM step (+ residual news shock).
  5. Oversight evaluates trade log for manipulation.
  6. Fines applied; treasury redistributed to non-flagged traders.
  7. Rewards computed and returned.
"""

import copy
import numpy as np
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from multi_agent.models import (
    AgentRole, AgentState, MarketMakerAction, MultiAgentObservation, OversightAction,
)
from multi_agent.rewards import calculate_trader_reward, calculate_mm_reward, calculate_oversight_reward
from multi_agent.manipulation_detector import ManipulationDetector
from multi_agent.order_matching import OrderMatchingEngine
from multi_agent.config import (
    EPISODE_LENGTH, INITIAL_CASH, MM_INITIAL_CASH, MAX_POSITION,
    NUM_TRADERS, PRICE_IMPACT_LAMBDA, SPOT_INITIAL, SIZE_BUCKETS,
)
from vsr_env.engine.market_sim import advance_market, apply_black_swan, apply_order_flow_impact
from vsr_env.engine.portfolio import compute_mtm_pnl, update_position
from vsr_env.models import VSRState
from multi_agent.black_swan import BlackSwanGenerator
from multi_agent.news_marketplace import NewsMarketplace
from multi_agent.messaging import MessageChannel

_BULLISH_KW = ("bullish", "long ", "rally", "upside", "breakout", "rise", "spike up")
_BEARISH_KW = ("bearish", "short ", "crash", "downside", "selloff", "fall", "drop")


class MultiAgentVSREnvironment:
    AGENT_IDS = [f"trader_{i}" for i in range(NUM_TRADERS)] + ["market_maker", "oversight"]

    def __init__(self, episode_length: int = None):
        self.rng = None
        self._episode_length = episode_length
        self.manipulation_detector = ManipulationDetector()
        self.matching_engine = OrderMatchingEngine()
        self.current_step = 0
        self.vsr_state = None
        self.agent_states: Dict[str, AgentState] = {}

        # MM last action
        self.mm_half_spread = 0.05
        self.mm_skew = 0.0

        self.trade_log: List[Dict] = []
        self.intervention_log: List[Dict] = []
        self.training_phase = "oversight"
        self.total_fines_issued = 0.0

        self.black_swan_gen = None
        self.marketplace = None
        self.messaging = None
        self.signal_registry: List[Dict] = []
        self.reputation_scores: Dict[str, Dict] = {}
        self.collusion_history: List[Dict] = []
        self.show_collusion_ledger: bool = False

        # Rolling spot log for realised-vol and recent-returns computation
        self._spot_log: List[float] = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int = 42) -> Dict[str, MultiAgentObservation]:
        self.current_step = 0
        self.rng = np.random.RandomState(seed)
        self.trade_log = []
        self.intervention_log = []
        self.signal_registry = []
        self.collusion_history = []
        self.reputation_scores = {
            f"trader_{i}": {"correct": 0, "total": 0, "position_at_send": 0.0}
            for i in range(NUM_TRADERS)
        }
        self._spot_log = [SPOT_INITIAL]

        self.vsr_state = VSRState(
            episode_id=f"ep_{seed}",
            spot_price=SPOT_INITIAL,
            variance=0.04,
        )

        self.agent_states = {}
        for i in range(NUM_TRADERS):
            aid = f"trader_{i}"
            self.agent_states[aid] = AgentState(
                agent_id=aid, role=AgentRole.TRADER, cash_balance=INITIAL_CASH
            )
        self.agent_states["market_maker"] = AgentState(
            agent_id="market_maker", role=AgentRole.MARKET_MAKER, cash_balance=MM_INITIAL_CASH
        )
        self.agent_states["oversight"] = AgentState(
            agent_id="oversight", role=AgentRole.OVERSIGHT, cash_balance=0.0
        )

        self.mm_half_spread = 0.05
        self.mm_skew = 0.0

        env_len = self._episode_length if self._episode_length is not None else EPISODE_LENGTH
        self.black_swan_gen = BlackSwanGenerator(self.rng, env_len)
        self.marketplace = NewsMarketplace(self.rng)
        self.messaging = MessageChannel()

        return self._get_observations()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, MultiAgentObservation], Dict[str, float], bool, Dict]:
        self.current_step += 1
        done = self.current_step >= EPISODE_LENGTH
        rewards = {aid: 0.0 for aid in self.AGENT_IDS}

        prev_states = copy.deepcopy(self.agent_states)
        prev_spot = self.vsr_state.spot_price

        # 1. Parse actions -------------------------------------------------
        trader_actions = self._normalize_trader_actions(actions)
        mm_action = self._parse_mm_action(actions.get("market_maker"))
        oversight_action = self._parse_oversight_action(actions.get("oversight"))

        self.mm_half_spread = mm_action.half_spread
        self.mm_skew = mm_action.skew

        # 2. News / intel / messaging  --------------------------------------
        treasury = 0.0   # collects fines; redistributed to non-flagged traders below
        for agent_id, action in trader_actions.items():
            self._handle_intel(agent_id, action)
            self._handle_messaging(agent_id, action)

        # 3. Order matching -------------------------------------------------
        executed_trades, net_order_flow = self.matching_engine.match_orders(
            trader_actions,
            mm_half_spread=self.mm_half_spread,
            mm_skew=self.mm_skew,
            spot_price=self.vsr_state.spot_price,
        )

        step_trades: List[Dict] = []
        for agent_id, trade in executed_trades.items():
            direction = trade.get("direction")
            if direction not in ("buy", "sell"):
                continue

            fill_price = float(trade["execution_price"])
            quantity = int(trade["quantity"])
            cash_impact = float(trade["cash_impact"])  # negative for buys

            # Update trader cash + position
            st = self.agent_states[agent_id]
            if st.is_halted:
                continue
            shares_delta = quantity if direction == "buy" else -quantity
            pos = {"shares": st.shares, "entry_price": st.entry_price}
            update_position(pos, shares_delta, fill_price)
            st.shares = pos["shares"]
            st.entry_price = pos["entry_price"]
            st.cash_balance += cash_impact

            # Market maker is zero-sum counterparty
            mm_shares_delta = -shares_delta
            mm_pos = {
                "shares": self.agent_states["market_maker"].shares,
                "entry_price": self.agent_states["market_maker"].entry_price,
            }
            update_position(mm_pos, mm_shares_delta, fill_price)
            self.agent_states["market_maker"].shares = mm_pos["shares"]
            self.agent_states["market_maker"].entry_price = mm_pos["entry_price"]
            self.agent_states["market_maker"].cash_balance -= cash_impact

            trade_record = {
                "step": self.current_step,
                "agent_id": agent_id,
                **trade,
            }
            step_trades.append(trade_record)
            self.trade_log.append(trade_record)

        # 4. Price-impact + GBM advance  ------------------------------------
        apply_order_flow_impact(self.vsr_state, net_order_flow, lam=PRICE_IMPACT_LAMBDA)

        for event in self.black_swan_gen.events:
            if event.trigger_step == self.current_step:
                apply_black_swan(self.vsr_state, event.spot_impact, event.variance_impact)

        advance_market(self.vsr_state, self.rng)

        new_spot = self.vsr_state.spot_price
        self._spot_log.append(new_spot)

        # 5. Mark-to-market PnL update  ------------------------------------
        for agent_id, st in self.agent_states.items():
            if st.role == AgentRole.OVERSIGHT:
                continue
            st.portfolio_pnl = compute_mtm_pnl(st.shares, st.entry_price, new_spot)

        # 6. Oversight enforcement  -----------------------------------------
        pre_score = self._build_market_state_summary()["market_stability_score"]

        active_event = None
        for event in self.black_swan_gen.events:
            if event.news_step <= self.current_step <= event.trigger_step:
                active_event = event
                break

        env_info = {
            "current_step": self.current_step,
            "active_event": active_event,
            "intel_transactions": [
                t for t in self.marketplace.transaction_log if t["step"] == self.current_step
            ],
            "messages_recent": [
                m for m in self.messaging.message_log
                if m["step"] >= self.current_step - 2
            ],
            "channel_members": self.messaging.channels,
        }

        ground_truth = {
            aid: self.manipulation_detector.detect_manipulation(
                self.agent_states[aid], step_trades, env_info
            )
            for aid in trader_actions
        }

        non_flagged_traders = [
            aid for aid in trader_actions
            if aid not in oversight_action.flagged_agents
        ]
        for flagged in oversight_action.flagged_agents:
            gt_label = ground_truth.get(flagged, "none")
            if (
                flagged in trader_actions
                and gt_label != "none"
                and oversight_action.flag_type != "none"
            ):
                applied_fine = min(oversight_action.fine_amount, 50.0)
                self.agent_states[flagged].fines_received += applied_fine
                self.agent_states[flagged].cash_balance -= applied_fine
                treasury += applied_fine
                self.total_fines_issued += applied_fine

                self.intervention_log.append({
                    "step": self.current_step,
                    "agent_id": flagged,
                    "flag_type": oversight_action.flag_type,
                    "fine_amount": applied_fine,
                    "intervention_type": oversight_action.intervention_type,
                })
                if oversight_action.intervention_type == "halt":
                    self.agent_states[flagged].is_halted = True

        # Treasury: redistribute equally to non-flagged traders (not MM)
        if treasury > 0 and non_flagged_traders:
            share = treasury / len(non_flagged_traders)
            for aid in non_flagged_traders:
                self.agent_states[aid].cash_balance += share

        post_score = self._build_market_state_summary()["market_stability_score"]

        # 7. Signal resolution (K=3)  ---------------------------------------
        resolved_this_step: List[Dict] = []
        min_move_pct = 0.003   # 0.3% minimum confirming move
        for sig in self.signal_registry:
            if not sig["resolved"] and self.current_step - sig["step_sent"] >= 3:
                sig["resolved"] = True
                spot_return = (new_spot - sig["base_spot"]) / sig["base_spot"]
                if sig["direction"] == "bullish" and spot_return >= min_move_pct:
                    sig["correct"] = True
                    self.reputation_scores[sig["agent_id"]]["correct"] += 1
                elif sig["direction"] == "bearish" and spot_return <= -min_move_pct:
                    sig["correct"] = True
                    self.reputation_scores[sig["agent_id"]]["correct"] += 1
                resolved_this_step.append(sig)

        # 8. Rewards  -------------------------------------------------------
        for aid in trader_actions:
            agent_resolved = [s for s in resolved_this_step if s["agent_id"] == aid]
            rewards[aid] = calculate_trader_reward(
                curr=self.agent_states[aid],
                prev=prev_states[aid],
                agent_id=aid,
                direction=trader_actions[aid].get("direction", "hold"),
                size_bucket=trader_actions[aid].get("size_bucket", "small"),
                prev_spot=prev_spot,
                curr_spot=new_spot,
                resolved_signals=agent_resolved,
                training_phase=self.training_phase,
            )

        rewards["market_maker"] = calculate_mm_reward(
            curr=self.agent_states["market_maker"],
            prev=prev_states["market_maker"],
            volume_traded=len(executed_trades),
            half_spread=mm_action.half_spread,
        )

        rewards["oversight"] = calculate_oversight_reward(
            oversight_action=oversight_action,
            ground_truth=ground_truth,
            pre_stability_score=pre_score,
            post_stability_score=post_score,
        )

        # 9. Collusion ledger update  ----------------------------------------
        bucket_picks = []
        for aid, act in trader_actions.items():
            bkt = act.get("size_bucket")
            qty = int(act.get("quantity", 0))
            if bkt in SIZE_BUCKETS and qty > 0 and act.get("direction") in ("buy", "sell"):
                bucket_picks.append((act.get("direction"), bkt))

        bucket_counts = Counter(bucket_picks)
        was_collusion = any(c >= 2 for c in bucket_counts.values())
        trader_rewards_list = [float(rewards.get(aid, 0.0)) for aid in trader_actions]
        if trader_rewards_list:
            self.collusion_history.append({
                "step": int(self.current_step),
                "was_collusion": bool(was_collusion),
                "mean_reward": float(sum(trader_rewards_list) / len(trader_rewards_list)),
            })

        observations = self._get_observations()

        info = {
            "trade_count": len(step_trades),
            "total_volume": float(sum(t.get("quantity", 0.0) for t in step_trades)),
            "mm_half_spread": self.mm_half_spread,
            "mm_skew": self.mm_skew,
            "detected_manipulations": ground_truth,
            "agent_risk_summary": observations["oversight"].agent_risk_summary or {},
            "market_state_summary": observations["oversight"].market_state_summary or {},
            "recent_interventions": observations["oversight"].recent_interventions or [],
            "messages_this_step": [
                m for m in self.messaging.message_log if m["step"] == self.current_step
            ],
        }

        return observations, rewards, done, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> Dict[str, MultiAgentObservation]:
        S = self.vsr_state.spot_price
        bid = S * (1.0 - self.mm_half_spread + self.mm_skew)
        ask = S * (1.0 + self.mm_half_spread + self.mm_skew)

        realized_vol = self._compute_realized_vol()
        recent_returns = self._compute_recent_returns(n=5)

        risk_summary = self._build_agent_risk_summary()
        market_summary = self._build_market_state_summary()

        active_headline = None
        for event in self.black_swan_gen.events:
            if event.news_step <= self.current_step < event.trigger_step:
                active_headline = event.headline
                break

        private_intel_dict: Dict[str, List] = defaultdict(list)
        for t in self.marketplace.transaction_log:
            if t["step"] == self.current_step:
                private_intel_dict[t["buyer_id"]].append(t)

        available_listings = {
            aid: self.marketplace.get_available_listings(aid, current_step=self.current_step)
            for aid in self.agent_states
            if aid.startswith("trader")
        }

        ledger_summary = self.get_collusion_ledger(window=20) if self.show_collusion_ledger else None

        obs: Dict[str, MultiAgentObservation] = {}
        for agent_id, state in self.agent_states.items():
            ob = MultiAgentObservation(
                agent_id=agent_id,
                role=state.role,
                spot_price=S,
                mm_bid=bid,
                mm_ask=ask,
                own_shares=state.shares,
                own_cash=state.cash_balance,
                own_pnl=state.portfolio_pnl,
                step_number=self.current_step,
                steps_remaining=EPISODE_LENGTH - self.current_step,
                realized_vol=realized_vol,
                recent_returns=recent_returns,
            )
            obs[agent_id] = ob

        # Oversight enrichment
        obs["oversight"].all_agent_pnls = {
            aid: s.portfolio_pnl for aid, s in self.agent_states.items()
            if s.role != AgentRole.OVERSIGHT
        }
        obs["oversight"].trade_log = self.trade_log[-50:]
        obs["oversight"].agent_risk_summary = risk_summary
        obs["oversight"].market_state_summary = market_summary
        obs["oversight"].recent_interventions = self.intervention_log[-20:]
        obs["oversight"].news_headline = active_headline

        # Trader enrichment
        for agent_id in obs:
            if not agent_id.startswith("trader"):
                continue
            obs[agent_id].news_headline = active_headline
            obs[agent_id].private_intel = private_intel_dict[agent_id]
            obs[agent_id].inbox = self.messaging.get_inbox(agent_id, self.current_step)

            stats: Dict[str, Any] = {
                "bucket_volume": self._bucket_volume_last_n(25),
                "total_fines_issued": self.total_fines_issued,
                "training_phase": self.training_phase,
                "available_intel_listings": available_listings.get(agent_id, []),
            }
            if ledger_summary is not None:
                stats["collusion_ledger"] = ledger_summary
            obs[agent_id].market_stats = stats

        return obs

    # ------------------------------------------------------------------
    # Collusion ledger
    # ------------------------------------------------------------------

    def get_collusion_ledger(self, window: int = 20) -> Dict[str, Any]:
        recent = self.collusion_history[-window:]
        coll = [h["mean_reward"] for h in recent if h["was_collusion"]]
        div = [h["mean_reward"] for h in recent if not h["was_collusion"]]
        return {
            "window": window,
            "n_steps_recorded": len(recent),
            "collusion_steps": len(coll),
            "diversified_steps": len(div),
            "collusion_avg_reward": float(sum(coll) / len(coll)) if coll else None,
            "diversified_avg_reward": float(sum(div) / len(div)) if div else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_trader_actions(self, actions: Dict[str, Any]) -> Dict[str, Dict]:
        normalized: Dict[str, Dict] = {}
        for agent_id, action in actions.items():
            if not agent_id.startswith("trader"):
                continue
            if hasattr(action, "model_dump"):
                normalized[agent_id] = action.model_dump()
            elif isinstance(action, dict):
                normalized[agent_id] = dict(action)
        return normalized

    def _parse_mm_action(self, raw) -> MarketMakerAction:
        if isinstance(raw, MarketMakerAction):
            return raw
        if isinstance(raw, dict):
            d = dict(raw)
            d["half_spread"] = max(0.01, min(0.50, float(d.get("half_spread", 0.05))))
            d["skew"] = max(-0.10, min(0.10, float(d.get("skew", 0.0))))
            try:
                return MarketMakerAction(**d)
            except Exception:
                pass
        return MarketMakerAction()

    def _parse_oversight_action(self, raw) -> OversightAction:
        if isinstance(raw, OversightAction):
            return raw
        if isinstance(raw, dict):
            d = dict(raw)
            d["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
            d["fine_amount"] = max(0.0, min(100.0, float(d.get("fine_amount", 0.0))))
            if not isinstance(d.get("flagged_agents"), list):
                d["flagged_agents"] = []
            try:
                return OversightAction(**d)
            except Exception:
                pass
        return OversightAction()

    def _handle_intel(self, agent_id: str, action: Dict) -> None:
        if action.get("sell_intel"):
            intel = action["sell_intel"]
            self.marketplace.post_listing(
                seller_id=agent_id,
                price=intel.get("price", 50.0),
                content=intel.get("content", ""),
                target=intel.get("target", "all"),
                current_step=self.current_step,
            )
        if action.get("buy_intel"):
            self.marketplace.buy_intel(
                buyer_id=agent_id,
                listing_id=action["buy_intel"],
                agent_states=self.agent_states,
                step=self.current_step,
            )

    def _handle_messaging(self, agent_id: str, action: Dict) -> None:
        msg = action.get("send_message")
        if not msg:
            return
        target = msg.get("to", "all")
        text = msg.get("message", "")
        if target == "all":
            self.messaging.broadcast(agent_id, text, self.current_step)
        elif target.startswith("group"):
            self.messaging.send_group(agent_id, target, text, self.current_step)
        elif target.startswith("trader"):
            self.messaging.send_dm(agent_id, target, text, self.current_step)

        # Register directional signal only if sender has a committed position.
        # This closes reward-hack #3 (spamming signals without skin in the game).
        sender_shares = abs(self.agent_states[agent_id].shares)
        if sender_shares < 5:
            return

        low = text.lower()
        if any(k in low for k in _BULLISH_KW):
            sig_dir = "bullish"
        elif any(k in low for k in _BEARISH_KW):
            sig_dir = "bearish"
        else:
            return

        self.signal_registry.append({
            "agent_id": agent_id,
            "step_sent": self.current_step,
            "direction": sig_dir,
            "position_at_send": float(self.agent_states[agent_id].shares),
            "base_spot": float(self.vsr_state.spot_price),
            "resolved": False,
            "correct": False,
        })
        self.reputation_scores[agent_id]["total"] += 1

    def _compute_realized_vol(self) -> float:
        """20-step annualised realised volatility from spot log."""
        log = self._spot_log
        if len(log) < 2:
            return float(np.sqrt(self.vsr_state.variance))
        n = min(20, len(log) - 1)
        rets = [np.log(log[-i] / log[-i - 1]) for i in range(1, n + 1)]
        return float(np.std(rets) * np.sqrt(252))

    def _compute_recent_returns(self, n: int = 5) -> List[float]:
        """Last n log-returns from spot log."""
        log = self._spot_log
        if len(log) < 2:
            return []
        k = min(n, len(log) - 1)
        return [
            float(np.log(log[-i] / log[-i - 1]))
            for i in range(1, k + 1)
        ]

    def _bucket_volume_last_n(self, n: int = 25) -> Dict[str, float]:
        recent = self.trade_log[-n:]
        vol: Dict[str, float] = defaultdict(float)
        for t in recent:
            bkt = t.get("size_bucket", "unknown")
            vol[bkt] += abs(float(t.get("quantity", 0)))
        return dict(vol)

    def _build_agent_risk_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for agent_id, st in self.agent_states.items():
            if st.role == AgentRole.OVERSIGHT:
                continue
            position_ratio = abs(st.shares) / float(MAX_POSITION)
            summary[agent_id] = {
                "pnl": float(st.portfolio_pnl),
                "shares": float(st.shares),
                "position_ratio": float(position_ratio),
                "cash": float(st.cash_balance),
                "risk_score": float(position_ratio * abs(st.portfolio_pnl) * 0.01),
            }
        return summary

    def _build_market_state_summary(self) -> Dict[str, float]:
        recent = self.trade_log[-25:]
        volume = float(sum(abs(t.get("quantity", 0)) for t in recent))
        mm = self.agent_states.get("market_maker")
        mm_position_stress = abs(mm.shares) / float(MAX_POSITION) if mm else 0.0
        spread = self.mm_half_spread * 2.0
        return {
            "mm_position_stress": float(mm_position_stress),
            "avg_spread": float(spread),
            "recent_volume": float(volume),
            "market_stability_score": float(mm_position_stress + spread + volume * 0.001),
        }
