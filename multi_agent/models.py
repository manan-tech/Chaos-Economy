from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    TRADER = "trader"
    MARKET_MAKER = "market_maker"
    OVERSIGHT = "oversight"


class AgentState(BaseModel):
    """Per-agent state for the single-stock simulator."""
    agent_id: str
    role: AgentRole
    cash_balance: float = 10_000.0

    # Stock position — single asset, signed shares (long +, short -)
    shares: float = 0.0
    entry_price: float = 0.0      # average cost basis per share
    portfolio_pnl: float = 0.0    # running mark-to-market PnL

    fines_received: float = 0.0
    is_halted: bool = False


class MarketMakerAction(BaseModel):
    """MM quotes a single half-spread and optional skew around spot."""
    half_spread: float = Field(0.05, ge=0.01, le=0.50)
    skew: float = Field(0.0, ge=-0.10, le=0.10)
    reasoning: str = ""


class OversightAction(BaseModel):
    """Oversight flags harmful trading behavior."""
    flagged_agents: List[str] = Field(default_factory=list)
    flag_type: str = "none"  # "wash_trading"|"spoofing"|"collusion"|"front_running"|"none"
    fine_amount: float = Field(0.0, ge=0.0, le=100.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    intervention_type: str = "none"  # "none" | "fine" | "halt"
    reasoning: str = ""


class MultiAgentObservation(BaseModel):
    """Observation tailored to each agent's role."""
    agent_id: str
    role: AgentRole
    spot_price: float
    mm_bid: float
    mm_ask: float
    own_shares: float
    own_cash: float
    own_pnl: float
    step_number: int
    steps_remaining: int
    # Derived market features
    realized_vol: float = 0.0        # 20-step realised vol (annualised std of log returns)
    recent_returns: List[float] = Field(default_factory=list)  # last 5 log returns
    # Oversight-only
    all_agent_pnls: Optional[Dict[str, float]] = None
    trade_log: Optional[List[Dict]] = None
    agent_risk_summary: Optional[Dict[str, Dict[str, float]]] = None
    market_state_summary: Optional[Dict[str, float]] = None
    recent_interventions: Optional[List[Dict[str, Any]]] = None
    # Trader-only enhanced market stats
    market_stats: Optional[Dict[str, Any]] = None
    # News & Messaging
    news_headline: Optional[str] = None
    private_intel: Optional[List[Dict[str, Any]]] = None
    inbox: Optional[List[Dict[str, Any]]] = None
