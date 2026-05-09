"""Share-based portfolio helpers for the single-stock simulator.

Each agent maintains a position dict:
    {"shares": int, "entry_price": float}

The position lives on `AgentState.position` (in multi_agent/environment.py),
not on VSRState. These helpers compute mark-to-market PnL and provide
position update utilities used by the environment's order matching step.
"""


def compute_mtm_pnl(shares: float, entry_price: float, spot: float) -> float:
    """Mark-to-market PnL for a share position.

    Args:
        shares: Signed share count (positive = long, negative = short).
        entry_price: Average entry cost per share.
        spot: Current spot price.

    Returns:
        Unrealised PnL = shares * (spot - entry_price).
    """
    return float(shares) * (float(spot) - float(entry_price))


def update_position(
    position: dict,
    shares_delta: float,
    fill_price: float,
) -> None:
    """Update a position dict after a fill.

    Computes the new average entry price using a weighted average.
    For a position flip (long → short or vice versa), resets entry_price
    to the fill_price of the flipping leg.

    Args:
        position: Dict with {"shares": float, "entry_price": float} — modified in place.
        shares_delta: Signed shares added/removed (buy = +, sell = -).
        fill_price: Price at which the fill occurred.
    """
    old_shares = float(position["shares"])
    old_entry = float(position["entry_price"])
    new_shares = old_shares + float(shares_delta)

    if new_shares == 0.0:
        position["shares"] = 0.0
        position["entry_price"] = 0.0
        return

    # Same-direction add: weighted-average entry.
    if old_shares == 0.0 or (old_shares * shares_delta > 0):
        if old_shares == 0.0:
            position["entry_price"] = float(fill_price)
        else:
            total_cost = old_shares * old_entry + float(shares_delta) * float(fill_price)
            position["entry_price"] = total_cost / new_shares
    else:
        # Partial close or full flip: entry price becomes fill_price for the residual.
        # (We reset on flip; existing PnL has already been realised by cash update.)
        if (old_shares > 0 and new_shares < 0) or (old_shares < 0 and new_shares > 0):
            position["entry_price"] = float(fill_price)
        # else: partial close keeps old entry_price unchanged (residual still at same cost basis)

    position["shares"] = float(new_shares)
