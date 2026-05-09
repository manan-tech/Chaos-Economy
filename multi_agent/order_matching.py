"""Single-asset order matching engine for the stock trading simulator.

The market maker quotes a single half-spread around spot. Traders fill at:
  - buy  → execution_price = spot * (1 + half_spread + skew)
  - sell → execution_price = spot * (1 - half_spread + skew)

`net_order_flow` (signed shares) is returned so the environment can apply
`apply_order_flow_impact` to nudge the spot price.
"""

from typing import Dict, Tuple


class OrderMatchingEngine:
    def __init__(self):
        self.total_volume = 0

    def match_orders(
        self,
        trader_actions: Dict[str, dict],
        mm_half_spread: float,
        mm_skew: float,
        spot_price: float,
    ) -> Tuple[Dict[str, dict], float]:
        """Fill trader orders against the MM's single bid/ask.

        Args:
            trader_actions: {agent_id: action_dict} with keys direction, size_bucket, quantity.
            mm_half_spread: Half the bid-ask spread (positive float).
            mm_skew: Signed skew applied to both legs (+skew lifts ask, lowers bid).
            spot_price: Current stock spot.

        Returns:
            (executed_trades, net_order_flow)
            executed_trades: dict of filled trade details per agent.
            net_order_flow: signed sum of filled shares (buys positive, sells negative).
        """
        executed_trades: Dict[str, dict] = {}
        net_order_flow = 0.0

        hs = max(0.001, float(mm_half_spread))
        skew = float(mm_skew)

        ask = spot_price * (1.0 + hs + skew)
        bid = spot_price * (1.0 - hs + skew)
        ask = max(bid + 0.01, ask)  # sanity: ask > bid

        for agent_id, action_dict in trader_actions.items():
            direction = action_dict.get("direction")
            if direction not in ("buy", "sell"):
                continue

            quantity = int(action_dict.get("quantity", 0))
            if quantity <= 0:
                continue

            if direction == "buy":
                fill_price = ask
                signed_qty = quantity
            else:
                fill_price = bid
                signed_qty = -quantity

            executed = dict(action_dict)
            executed["execution_price"] = float(fill_price)
            executed["spot_at_fill"] = float(spot_price)
            executed["half_spread_applied"] = float(hs)
            executed["volume"] = float(quantity)
            executed["notional"] = float(fill_price * quantity)
            executed["cash_impact"] = -float(fill_price * signed_qty)

            executed_trades[agent_id] = executed
            net_order_flow += signed_qty

        self.total_volume += sum(
            abs(t["volume"]) for t in executed_trades.values()
        )
        return executed_trades, float(net_order_flow)
