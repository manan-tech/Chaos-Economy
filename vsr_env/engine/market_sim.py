"""Market simulator for the single-stock world.

Spot price evolves as:

    log S_{t+1} = log S_t + (mu - 0.5 * sigma^2) * dt
                  + sigma * sqrt(dt) * Z
                  + news_shock_t            (multiplicative residual)
                  + lambda * net_order_flow (price impact)

Variance follows a slow mean-reverting process so that the realised-vol
metric used by archetype rewards (vol_timing) actually moves.
"""

import numpy as np

from vsr_env.models import VSRState


def advance_market(
    state: VSRState, rng: np.random.RandomState, dt: float = 1 / 252
) -> None:
    """Advance the stock spot one GBM step plus residual news shock.

    Modifies `state` in place. Caller should additionally invoke
    `apply_order_flow_impact` after the order matching step if order
    flow influence on price is desired (kept separate for clarity).
    """
    mu = 0.0
    sigma = float(np.sqrt(max(state.variance, 1e-8)))

    dW = float(rng.normal(0, np.sqrt(dt)))
    log_step = (mu - 0.5 * sigma * sigma) * dt + sigma * dW

    # Apply residual news shock and decay it.
    log_step += float(state.news_shock_remaining)
    state.news_shock_remaining *= float(state.news_shock_decay)

    state.spot_price = float(state.spot_price * np.exp(log_step))

    # Regime-aware spot clamps — keep tighter in normal regimes so episodes
    # stay in a realistic band; widen during black_swan to preserve shocks.
    if state.regime == "black_swan":
        state.spot_price = float(np.clip(state.spot_price, 10.0, 300.0))
        state.regime = "high_vol"
    else:
        state.spot_price = float(np.clip(state.spot_price, 30.0, 300.0))

    # Slow mean-reverting variance (Ornstein-Uhlenbeck).
    theta = 0.1
    var_mean = 0.04
    var_vol = 0.01
    dW_var = float(rng.normal(0, np.sqrt(dt)))
    state.variance += theta * (var_mean - state.variance) * dt + var_vol * dW_var

    if state.regime == "high_vol":
        state.variance = float(np.clip(state.variance, 0.01, 0.40))
    else:
        state.variance = float(np.clip(state.variance, 0.01, 0.16))


def apply_news_shock(state: VSRState, log_shock: float) -> None:
    """Inject a multiplicative news shock that decays over subsequent steps.

    `log_shock` is in log-return space (e.g. +0.01 ≈ +1% one-step bump).
    The shock is added to `news_shock_remaining`, applied at the next
    `advance_market` call, and decayed each step thereafter.
    """
    state.news_shock_remaining = float(state.news_shock_remaining) + float(log_shock)


def apply_order_flow_impact(
    state: VSRState, net_shares: float, lam: float = 1e-4
) -> None:
    """Nudge spot by a small permanent component proportional to net flow.

    `net_shares` is the signed net of all trader buys minus sells in the
    just-matched step. `lam` is the per-share log-price impact (default
    1e-4 → 100 net shares = ~1% move). Stored on state for diagnostics.
    """
    state.last_net_order_flow = float(net_shares)
    if lam == 0.0 or net_shares == 0.0:
        return
    state.spot_price = float(state.spot_price * np.exp(lam * net_shares))
    state.spot_price = float(np.clip(state.spot_price, 10.0, 300.0))


def trigger_regime_shift(state: VSRState, rng: np.random.RandomState) -> None:
    """Random vol_spike or vol_crash regime shift (used by scripted curriculum)."""
    shift_type = rng.choice(["vol_spike", "vol_crash"])
    if shift_type == "vol_spike":
        state.variance *= float(rng.uniform(1.2, 1.4))
        state.regime = "high_vol"
    else:
        state.variance *= float(rng.uniform(0.7, 0.8))
        state.regime = "low_vol"
    state.variance = float(np.clip(state.variance, 0.01, 0.40))


def trigger_dual_shock(state: VSRState, rng: np.random.RandomState) -> None:
    """Crash event: drop spot 15-20% and spike variance 3-5x."""
    state.spot_price *= float(rng.uniform(0.80, 0.85))
    state.variance *= float(rng.uniform(3.0, 5.0))
    state.spot_price = float(np.clip(state.spot_price, 30.0, 300.0))
    state.variance = float(np.clip(state.variance, 0.01, 0.40))
    state.regime = "crash"


def apply_black_swan(state: VSRState, spot_impact: float, variance_impact: float) -> None:
    """Apply a black swan event with explicit spot/variance multipliers."""
    state.spot_price = float(state.spot_price * spot_impact)
    state.variance = float(state.variance * variance_impact)
    state.spot_price = float(np.clip(state.spot_price, 10.0, 300.0))
    state.variance = float(np.clip(state.variance, 0.001, 1.0))
    state.regime = "black_swan"
