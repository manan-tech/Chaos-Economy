"""Pydantic state model for the single-stock multi-agent simulator."""

from typing import List, Optional

from pydantic import BaseModel, Field


class VSRState(BaseModel):
    """Internal market state for the single-stock simulator.

    Tracks stock spot price, return-volatility used by the GBM step,
    market regime (normal/high_vol/low_vol/crash/black_swan), and any
    transient news shock still decaying.

    Per-agent positions (shares + entry_price) live on the agent state
    objects in `multi_agent.environment`, not here. This object is the
    *market*, not the *book*.
    """

    episode_id: str = Field(default="", description="Unique episode identifier")
    step_count: int = Field(0, description="Steps taken in current episode")

    regime: str = Field("normal", description="Current market regime")
    spot_price: float = Field(100.0, description="Current stock spot price")
    variance: float = Field(0.04, description="Annualized return variance for the GBM step")

    # Transient news-shock multiplier still decaying into the spot path.
    # `news_shock_remaining` is the residual multiplicative shock per step,
    # decayed each `advance_market` call by `news_shock_decay`.
    news_shock_remaining: float = Field(0.0, description="Residual news shock (multiplicative, per step)")
    news_shock_decay: float = Field(0.5, description="Per-step decay factor for residual news shock")

    # Last realised one-step net order flow (signed shares, traders' net). Used
    # by `apply_order_flow_impact` to nudge spot before/after the matching step.
    last_net_order_flow: float = Field(0.0, description="Signed net trader order flow at last step (shares)")

    # Episode-level event scheduling (kept for back-compat with shock helpers).
    regime_shift_step: int = Field(5, description="Step at which a scripted regime shift occurs (if used)")
