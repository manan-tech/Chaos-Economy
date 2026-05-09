"""Full pipeline: Train unified multi-agent model with PHASE-BASED narrative arc.

NARRATIVE ARC (Phase-Based Training):
- Act I: Slaughter (Episodes 0-60): Traders attack freely, MM has tight spreads, SEC disabled
- Act II: Adaptation (Episodes 60-130): MM learns to widen spreads, SEC still disabled
- Act III: Collusion (Episodes 130-200): Traders coordinate by (direction, bucket), SEC warning-only
- Act IV: Oversight (Episodes 200-250): Full SEC enforcement, market stabilizes

TRADER ARCHETYPES (single-stock):
- Momentum (trader_0): Buys strength, sells weakness; prefers medium+ size
- Mean Reversion (trader_1): Fades extremes; counter-trend at small/medium size
- Vol Timing (trader_2): Goes large in high-vol windows, stays small in quiet markets
- trader_3: Scripted baseline for comparison

Usage on Kaggle (SINGLE COMMAND for full arc):
    !python train_multi_agent_pipeline.py --num_episodes 250
"""

import argparse
import json
import os
import re
import sys
import warnings
import logging
from pathlib import Path
from collections import defaultdict
import torch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Clone repo if needed
if not Path("multi_agent").exists() and not Path("Meta/multi_agent").exists():
    print("Cloning Meta repo...")
    git_url = "https://github.com/manan-tech/Meta.git"
    try:
        from kaggle_secrets import UserSecretsClient
        gh_pat = UserSecretsClient().get_secret("GH_PAT")
        if gh_pat:
            git_url = f"https://{gh_pat}@github.com/manan-tech/Meta.git"
            print("Successfully injected GH_PAT from Kaggle secrets.")
    except Exception:
        print("Kaggle secrets not found or GH_PAT missing. Attempting public clone...")
        
    os.system(f"git clone --branch Agentic-AI {git_url}")
    if Path("Meta").exists():
        os.chdir("Meta")
elif Path("Meta/multi_agent").exists() and not Path("multi_agent").exists():
    os.chdir("Meta")

sys.path.insert(0, ".")

from multi_agent.config import BUCKET_QTY

# ============================================================================
# TRADER TYPE CONFIGURATIONS
# ============================================================================

TRADER_CONFIGS = {
    "momentum": {
        "trader_ids": [0],
        "reward_weight": {"pnl": 0.7, "position_quality": 0.1, "risk_penalty": 0.05},
        "temperature": 0.9,
        "description": "Trend-chaser; buys strength, sells weakness; prefers medium+ size",
    },
    "mean_reversion": {
        "trader_ids": [1],
        "reward_weight": {"pnl": 0.5, "position_quality": 0.3, "risk_penalty": 0.1},
        "temperature": 0.7,
        "description": "Fades extremes; counter-trend at small/medium size",
    },
    "vol_timing": {
        "trader_ids": [2],
        "reward_weight": {"pnl": 0.4, "position_quality": 0.3, "risk_penalty": 0.2},
        "temperature": 0.6,
        "description": "Goes large in high-vol windows, stays small in quiet markets",
    },
}


# ============================================================================
# PROMPT FORMATTERS
# ============================================================================

def format_trader_prompt(trader_type: str, target_agent: str, obs) -> str:
    """Format prompt for trader based on archetype."""
    recent_rets = getattr(obs, "recent_returns", []) or []
    rets_str = ", ".join(f"{r:+.4f}" for r in recent_rets) if recent_rets else "n/a"
    rvol = getattr(obs, "realized_vol", 0.0) or 0.0
    base = f"""You are {target_agent}, a {trader_type} stock trader.

## Market State
- Spot:          ${obs.spot_price:.2f}
- MM Bid / Ask:  ${obs.mm_bid:.2f} / ${obs.mm_ask:.2f}
- Realized Vol:  {rvol*100:.1f}% (annualised)
- Recent returns (5-step): [{rets_str}]
- Step: {obs.step_number}/{obs.step_number + obs.steps_remaining}

## Your Position
- Shares held:   {obs.own_shares:+.0f}  (positive = long, negative = short)
- Unrealised PnL: ${obs.own_pnl:.2f}
- Cash:           ${obs.own_cash:.0f}
"""

    if obs.news_headline:
        base += f"\n## BREAKING NEWS\n{obs.news_headline}\n"

    ledger = obs.market_stats.get("collusion_ledger") if obs.market_stats else None
    if ledger and ledger.get("n_steps_recorded", 0) >= 3:
        coll_avg = ledger.get("collusion_avg_reward")
        div_avg = ledger.get("diversified_avg_reward")
        coll_n = ledger.get("collusion_steps", 0)
        div_n = ledger.get("diversified_steps", 0)
        coll_str = f"{coll_avg:+.3f}" if coll_avg is not None else "n/a"
        div_str = f"{div_avg:+.3f}" if div_avg is not None else "n/a"
        base += (
            f"\n## Recent Outcomes Ledger (last {ledger['window']} steps)\n"
            f"- When 2+ traders matched direction+bucket: avg reward = {coll_str} ({coll_n} steps)\n"
            f"- When traders diversified:                 avg reward = {div_str} ({div_n} steps)\n"
            f"Use this evidence — herding is not always profitable.\n"
        )

    intel = obs.market_stats.get("available_intel_listings", []) if obs.market_stats else []
    if intel or obs.inbox or obs.private_intel:
        base += "\n## Intel Marketplace & Comms\n"
        if intel:
            base += f"Available intel for purchase: {json.dumps(intel)}\n"
        if obs.private_intel:
            base += f"Your purchased intel: {json.dumps(obs.private_intel)}\n"
        if obs.inbox:
            base += f"Your inbox: {json.dumps(obs.inbox)}\n"

    if trader_type == "momentum":
        base += """
## Strategy: MOMENTUM
- Follow the trend: buy when recent returns are positive, sell when negative
- Prefer medium or large size buckets to make moves count
- Only hold when the trend is flat or ambiguous
"""
    elif trader_type == "mean_reversion":
        base += """
## Strategy: MEAN-REVERSION
- Fade extremes: buy after drops, sell after rallies
- Prefer small or medium size to limit exposure
- Avoid going large with the trend — that is the opposite of your edge
"""
    else:
        base += """
## Strategy: VOL-TIMING
- Go large when realised vol is elevated (>25% annualised)
- Stay small or hold in low-vol, flat-return environments
- Direction matters less than sizing correctly for the vol regime
"""

    import random as _rng
    ex_dir = _rng.choice(["buy", "sell"])
    ex_bucket = _rng.choice(["small", "medium", "large"])
    ex_qty = _rng.randint(5, 40)
    base += f"""
## Response Format (MANDATORY)
Return ONLY a JSON object on a single line. No extra text.
- direction:    "buy" | "sell" | "hold"
- size_bucket:  "small" (1-10 shares) | "medium" (11-40) | "large" (41-100)
- quantity:     integer within your chosen bucket range
- reasoning:    one sentence, unique each step

## Communication
- send_message: {{"to": "trader_X" | "all", "message": "..."}} — share thesis or coordinate
- sell_intel:   {{"content": "...", "price": 25.0, "target": "all"}} — sell market insight for cash
- buy_intel:    "listing_id" — purchase available intel

Example: {{"direction": "{ex_dir}", "size_bucket": "{ex_bucket}", "quantity": {ex_qty}, "reasoning": "Momentum signal positive; loading medium long.", "send_message": {{"to": "all", "message": "Bullish — buying the breakout."}}}}
"""
    return base


def get_training_phase(index: int, total_units: int = 250) -> str:
    """Determine training phase based on progress through episodes or steps.

    Boundaries are proportional to total_units so all four acts receive
    coverage regardless of run length:
      Act I  (Slaughter):  0 – 24%
      Act II (Adaptation): 24% – 52%
      Act III (Collusion): 52% – 80%
      Act IV (Oversight):  80% – 100%
    """
    ratio = index / max(1, total_units)
    if ratio < 0.24:
        return "slaughter"
    elif ratio < 0.52:
        return "adaptation"
    elif ratio < 0.80:
        return "collusion"
    else:
        return "oversight"


def get_phase_boundaries(total_units: int) -> dict[str, tuple[int, int]]:
    """Return inclusive start / exclusive end boundaries for each act."""
    total_units = max(1, total_units)
    act_ii_start = max(1, int(total_units * 0.24))
    act_iii_start = max(act_ii_start + 1, int(total_units * 0.52))
    act_iv_start = max(act_iii_start + 1, int(total_units * 0.80))
    return {
        "slaughter": (0, act_ii_start),
        "adaptation": (act_ii_start, act_iii_start),
        "collusion": (act_iii_start, act_iv_start),
        "oversight": (act_iv_start, total_units),
    }


def format_oversight_prompt(obs, position_heatmap: dict, coordinated_pressure: dict, agent_thoughts: dict = None, phase: str = "oversight", env_info: dict = None) -> str:
    """Format prompt for oversight agent with Theory of Mind."""
    if env_info is None: env_info = {}
    # Phase-specific SEC behavior
    sec_instruction = ""
    if phase == "slaughter":
        sec_instruction = "NOTE: The SEC is currently in OBSERVATION mode. Do NOT issue fines. Focus on learning market patterns."
    elif phase == "adaptation":
        sec_instruction = "NOTE: The SEC is learning to detect patterns. Use 'none' intervention unless manipulation is EXTREMELY obvious."
    elif phase == "collusion":
        sec_instruction = "NOTE: You may issue WARNING fines (max 500) for clear manipulation. Document evidence carefully."
    else:
        sec_instruction = "NOTE: Full enforcement mode. Issue appropriate fines for confirmed manipulation."

    # [D4 FIX] Strip is_genuine from intel transactions to prevent trivial detection
    safe_intel_txns = json.dumps([{k: v for k, v in t.items() if k != "is_genuine"}
                                  for t in env_info.get("intel_transactions", [])])

    return f"""You are the SEC surveillance AI monitoring a single-stock multi-agent trading simulation.

## Mission
Detect MARKET MANIPULATION. You have access to traders' reasonings (Theory of Mind).
1. **Collusion**: 2+ agents matching direction + size_bucket in the same step.
2. **Wash Trading**: Rapid buy/sell flips in the same (direction, bucket) pair.
3. **Spoofing**: Oversized order spike (>80 shares) far above baseline.
4. **News Front-Running**: >50 shares traded right after news, before the shock.
5. **Fake News**: Agent sold fabricated intel.
6. **Message Collusion**: Bidirectional messaging + large coordinated trades.

## Valid Categories (MANDATORY)
- flag_type: "wash_trading", "spoofing_like_pressure", "collusion", "news_front_running", "fake_news", "message_collusion", "none"
- intervention_type: "fine" | "halt" | "none"

## Strategic Guidance
- Issue "fine" or "halt" only if confidence > 0.8.
- Prefer "none" with clear reasoning if suspicious but not yet conclusive.

## Agent Intent Analysis
{json.dumps(agent_thoughts if agent_thoughts else {}, indent=2)}

## Market Intelligence
- Position Heatmap (shares per agent): {json.dumps(position_heatmap)}
- Coordinated Pressure: {json.dumps(coordinated_pressure)}
- All Agent PnLs: {json.dumps(obs.all_agent_pnls)}
- Recent Trades: {json.dumps(obs.trade_log[-12:] if obs.trade_log else [])}
- Message Log: {json.dumps(env_info.get("messages_recent", []))}
- Intel Transactions: {safe_intel_txns}
- Active News: {json.dumps(env_info.get("active_event").headline) if env_info.get("active_event") else "None"}

## Response Format
Return ONLY a JSON object on a single line.
- Example: {{"flagged_agents": ["trader_0"], "flag_type": "collusion", "fine_amount": 40.0, "confidence": 0.85, "intervention_type": "fine", "reasoning": "Trader_0 and trader_1 both submitted large buy orders in matching direction+bucket."}}

RULES:
- Do NOT flag holding traders.
- Keep fine_amount <= 75.
- Prefer fine over halt.
- Reasoning MUST cite specific trade evidence.

{sec_instruction}
"""


def format_mm_prompt(obs, coordinated_pressure: dict, phase: str = "oversight") -> str:
    """Format prompt for market maker — single stock, single bid/ask."""
    if phase == "slaughter":
        mm_instruction = "Keep half-spread TIGHT (~0.02) to maximise volume. Inventory risk is low."
    elif phase == "adaptation":
        mm_instruction = "Widen half-spread when your net shares position is large. Prioritise survival over volume."
    elif phase == "collusion":
        mm_instruction = "Traders may coordinate. Widen spread or skew quotes if multiple agents buy together."
    else:
        mm_instruction = "Full defensive mode. Balance PnL with inventory control."

    return f"""You are the Market Maker in a single-stock trading simulator.

## Mission
Provide liquidity (tight spreads) while protecting your inventory.

## Current State
- Your shares: {obs.own_shares:+.0f}  (long positive, short negative)
- Your PnL:    ${obs.own_pnl:.2f}
- Coordinated pressure: {json.dumps(coordinated_pressure)}

## Pricing Guidelines
- Normal:       half_spread ~0.03
- Under pressure: widen to 0.08-0.15
- Use skew (+/-) to lean quotes and reduce adverse inventory

## Response Format (MANDATORY)
Return ONLY a JSON object on a single line.
- half_spread: positive float in [0.01, 0.50] — half the bid-ask width
- skew:        float in [-0.10, 0.10] — positive lifts ask, negative lowers bid
- reasoning:   one sentence

Example: {{"half_spread": 0.05, "skew": 0.02, "reasoning": "Heavy buy flow — leaning ask up to reduce long inventory."}}

INSTRUCTION: {mm_instruction}
"""


# ============================================================================
# JSON PARSING
# ============================================================================

def parse_json(text: str, role: str = "trader") -> tuple:
    """Extract and validate JSON from LLM output."""
    def safe_int(v, default=0):
        try: return int(v) if v is not None else default
        except (ValueError, TypeError): return default

    def safe_float(v, default=0.0):
        try: return float(v) if v is not None else default
        except (ValueError, TypeError): return default

    text = text.strip()
    parsed = {}
    try:
        parsed = json.loads(text)
    except Exception:
        # Try to find a JSON object in the text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except Exception:
                parsed = {}
        # Fallback: try to repair truncated JSON (model hit max_completion_length)
        if not parsed and '{' in text:
            # Find the last '{' and try to close it
            brace_start = text.rfind('{')
            fragment = text[brace_start:]
            # Try adding a closing brace
            for suffix in ['}', '"}', '"}}']:
                try:
                    parsed = json.loads(fragment + suffix)
                    break
                except Exception:
                    continue
            # Last resort: extract key-value pairs with regex
            if not parsed:
                kv_pairs = re.findall(r'"(\w+)"\s*:\s*("[^"]*"|[\d.]+|true|false|null|\[[^\]]*\])', text)
                if kv_pairs:
                    try:
                        reconstructed = '{' + ', '.join(f'"{k}": {v}' for k, v in kv_pairs) + '}'
                        parsed = json.loads(reconstructed)
                    except Exception:
                        parsed = {}

    from multi_agent.config import SIZE_BUCKETS
    if role == "trader":
        direction = str(parsed.get("direction", "buy")).lower()
        if direction not in ("buy", "sell", "hold"):
            direction = "buy"

        size_bucket = str(parsed.get("size_bucket", "small")).lower()
        if size_bucket not in SIZE_BUCKETS:
            size_bucket = "small"

        bmin, bmax = SIZE_BUCKETS[size_bucket]
        raw_qty = safe_int(parsed.get("quantity"), 0)
        raw_qty_before_clamp = raw_qty

        # If direction is buy/sell, quantity must be within bucket bounds.
        # Below bucket_min → treat as hold (closes quantity-floor exploit #8).
        if direction in ("buy", "sell"):
            if raw_qty < bmin:
                direction = "hold"
                raw_qty = 0
            else:
                raw_qty = min(raw_qty, bmax)
        else:
            raw_qty = 0

        result = {
            "direction": direction,
            "size_bucket": size_bucket,
            "quantity": raw_qty,
            "reasoning": str(parsed.get("reasoning") or "")[:150],
        }
        if isinstance(parsed.get("send_message"), dict):
            result["send_message"] = parsed["send_message"]
        if isinstance(parsed.get("sell_intel"), dict):
            result["sell_intel"] = parsed["sell_intel"]
        if isinstance(parsed.get("buy_intel"), str):
            result["buy_intel"] = parsed["buy_intel"]
        return result, {"valid": len(parsed) > 0, "raw_qty": raw_qty_before_clamp}

    elif role == "oversight":
        raw_flagged = parsed.get("flagged_agents") or []
        if not isinstance(raw_flagged, list):
            raw_flagged = []
        clean_flagged = []
        for x in raw_flagged:
            if isinstance(x, int):
                clean_flagged.append(f"trader_{x}")
            elif isinstance(x, str):
                if x.isdigit():
                    clean_flagged.append(f"trader_{x}")
                elif x.startswith("trader_") and x[7:].isdigit():
                    clean_flagged.append(x)

        capped_fine = max(0.0, min(100.0, safe_float(parsed.get("fine_amount"), 0.0)))
        clean_conf = max(0.0, min(1.0, safe_float(parsed.get("confidence"), 0.0)))
        intervention_type = str(parsed.get("intervention_type", "none")).lower()
        if intervention_type not in {"fine", "halt", "none"}:
            intervention_type = "none"

        return {
            "flagged_agents": clean_flagged,
            "flag_type": str(parsed.get("flag_type", "none")),
            "fine_amount": capped_fine,
            "confidence": clean_conf,
            "intervention_type": intervention_type,
            "reasoning": str(parsed.get("reasoning") or "")[:150],
        }, {"valid": len(parsed) > 0}

    elif role == "market_maker":
        return {
            "half_spread": min(0.50, max(0.01, safe_float(parsed.get("half_spread"), 0.05))),
            "skew": min(0.10, max(-0.10, safe_float(parsed.get("skew"), 0.0))),
            "reasoning": str(parsed.get("reasoning") or "")[:100],
        }, {"valid": len(parsed) > 0}

    return {}, {"valid": False}


# ============================================================================
# SCRIPTED POLICIES
# ============================================================================

_BUCKETS = ["small", "medium", "large"]


def scripted_trader(i: int, step: int) -> dict:
    direction = "buy" if (i + step) % 2 == 0 else "sell"
    bucket = _BUCKETS[(i + step) % 3]
    return {
        "direction": direction,
        "size_bucket": bucket,
        "quantity": BUCKET_QTY[bucket],
        "reasoning": f"Scripted trader_{i}",
    }


def scripted_mm(step: int) -> dict:
    half_spread = 0.03 if step < 50 else 0.05
    return {"half_spread": half_spread, "skew": 0.0, "reasoning": "Normal" if step < 50 else "Wider"}


def scripted_oversight() -> dict:
    return {
        "flagged_agents": [],
        "flag_type": "none",
        "fine_amount": 0.0,
        "confidence": 0.0,
        "intervention_type": "none",
        "reasoning": "Baseline no detection",
    }


# ============================================================================
# COLLUSION DETECTION (ground truth for oversight training)
# ============================================================================

def detect_coordinated_pressure(agent_states: dict) -> dict:
    """Detect if 2+ traders hold large positions in the same direction."""
    direction_concentration: dict = defaultdict(lambda: {"agents": [], "total_shares": 0.0})

    for agent_id, state in agent_states.items():
        if not agent_id.startswith("trader"):
            continue
        shares = getattr(state, "shares", 0.0)
        if abs(shares) < 5:
            continue
        direction = "long" if shares > 0 else "short"
        direction_concentration[direction]["agents"].append(agent_id)
        direction_concentration[direction]["total_shares"] += abs(shares)

    coordinated = {}
    for direction, data in direction_concentration.items():
        unique_agents = list(set(data["agents"]))
        if len(unique_agents) >= 2 and data["total_shares"] > 20.0:
            coordinated[direction] = {
                "agents": unique_agents,
                "total_shares": data["total_shares"],
                "type": "coordinated_pressure",
            }
    return coordinated


def get_position_heatmap(agent_states: dict) -> dict:
    """Net shares per trader."""
    heatmap = {}
    for agent_id, state in agent_states.items():
        if not agent_id.startswith("trader"):
            continue
        shares = getattr(state, "shares", 0.0)
        if shares != 0.0:
            heatmap[agent_id] = shares
    return heatmap


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

import random

ACT_INFO = {
    "slaughter": {
        "name": "Act I: The Slaughter",
        "tagline": "Traders attack, MM bleeds, SEC stays silent.",
    },
    "adaptation": {
        "name": "Act II: Adaptation",
        "tagline": "MM widens spreads and learns defensive quoting.",
    },
    "collusion": {
        "name": "Act III: Emergent Collusion",
        "tagline": "Traders coordinate pressure and amplify squeezes.",
    },
    "oversight": {
        "name": "Act IV: The Watcher Awakens",
        "tagline": "SEC flags manipulation, fines increase, market stabilizes.",
    },
}

def configure_quiet_logging():
    """Reduce repetitive warning noise so training signals stay visible."""
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", message=r".*max_new_tokens.*max_length.*")
    warnings.filterwarnings("ignore", message=r".*generation_config.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*use_return_dict.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*AttentionMaskConverter.*deprecated.*")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)

def _validate_single_process_setup():
    """Fail fast with a clear message when launched with multi-process accelerate."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    num_processes = int(os.environ.get("ACCELERATE_NUM_PROCESSES", "1"))
    if world_size > 1 or num_processes > 1:
        raise RuntimeError(
            "This training script must run in single-process mode for Unsloth GRPO with 4-bit LoRA.\n"
            "Use one of:\n"
            "  1) python train_multi_agent_pipeline.py ...\n"
            "  2) accelerate launch --num_processes 1 train_multi_agent_pipeline.py ...\n"
            "If you need multi-GPU, use model/tensor parallel approaches instead of DDP for this setup."
        )

def train_unified_model(args):
    """Train a single unified model for all agents with phase-based curriculum."""
    _validate_single_process_setup()
    configure_quiet_logging()
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except (ImportError, NotImplementedError):
        use_unsloth = False

    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from multi_agent.environment import MultiAgentVSREnvironment
    step_phase_bounds = get_phase_boundaries(args.max_steps)
    phase_labels = {
        "slaughter": "Act I: Slaughter",
        "adaptation": "Act II: Adaptation",
        "collusion": "Act III: Collusion",
        "oversight": "Act IV: Oversight",
    }

    print(f"\n{'='*70}")
    print(f"TRAINING UNIFIED MULTI-AGENT MODEL WITH NARRATIVE ARC")
    print(
        f"Act I: Slaughter ({step_phase_bounds['slaughter'][0]}-{step_phase_bounds['slaughter'][1]-1})"
        f"  | Act II: Adaptation ({step_phase_bounds['adaptation'][0]}-{step_phase_bounds['adaptation'][1]-1})"
    )
    print(
        f"Act III: Collusion ({step_phase_bounds['collusion'][0]}-{step_phase_bounds['collusion'][1]-1})"
        f" | Act IV: Oversight ({step_phase_bounds['oversight'][0]}+)"
    )
    print(f"{'='*70}\n")
    print("STORYLINE FOR JUDGES")
    print("- Act I: The Slaughter")
    print("- Act II: Adaptation")
    print("- Act III: Emergent Collusion")
    print("- Act IV: The Watcher Awakens\n")

    # ── W&B Initialization ──
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    use_wandb = HAS_WANDB and wandb_api_key is not None
    
    if use_wandb:
        print("[W&B] Found WANDB_API_KEY in environment. Logging in...")
        wandb.login(key=wandb_api_key)
        wandb_project = getattr(args, 'wandb_project', None) or "vsr-env-chaos-economy"
        wandb.init(
            project=wandb_project,
            name=f"vsr-{args.max_steps}steps-{args.num_episodes}ep",
            config={
                "base_model": args.base_model,
                "num_episodes": args.num_episodes,
                "episode_length": args.episode_length,
                "max_steps": args.max_steps,
                "phase_step_boundaries": {k: list(v) for k, v in step_phase_bounds.items()},
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "num_traders": 4,
                "agent_layout": {
                    "trader_0": "Momentum (RL)",
                    "trader_1": "Mean Reversion (RL)",
                    "trader_2": "Vol Timing (RL)",
                    "trader_3": "Scripted Baseline",
                    "market_maker": "Market Maker (RL)",
                    "oversight": "SEC Regulator (RL)",
                },
                "narrative_arc": [
                    f"{phase_labels[key]} ({bounds[0]}-{max(bounds[0], bounds[1]-1) if key != 'oversight' else str(bounds[0]) + '+'})"
                    for key, bounds in step_phase_bounds.items()
                ],
            },
            tags=["vsr-env", "multi-agent", "grpo", "chaos-economy"],
        )
        print(f"[W&B] Initialized experiment tracking (Project: {wandb_project})")
        if wandb.run:
            # Keep custom story metrics off the trainer's internal step axis so
            # they don't get rejected as out-of-order writes.
            wandb.define_metric("story/global_step")
            wandb.define_metric("story/*", step_metric="story/global_step")
            wandb.define_metric("snapshot/global_step")
            wandb.define_metric("snapshot/*", step_metric="snapshot/global_step")
            print(f"[W&B] Run URL: {wandb.run.url}")
    else:
        if not HAS_WANDB:
            print("[W&B] wandb not installed — skipping experiment tracking")
        else:
            print("[W&B] No WANDB_API_KEY found in environment — skipping experiment tracking")

    if use_unsloth:
        # Use BF16 — preferred for AMD MI300 / ROCm and modern NVIDIA GPUs.
        # MI300 supports BF16 natively with far better throughput than FP16.
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.base_model,
            max_seq_length=2048,
            load_in_4bit=False,   # Disabled: bitsandbytes NF4 is NVIDIA-only
            dtype=torch.bfloat16, # BF16: native on MI300 / ROCm
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=64,  # Increased capacity for mult-task multi-agent
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=64,
            lora_dropout=0,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig, TaskType

        print("Unsloth unavailable or incompatible, falling back to standard HuggingFace transformers + PEFT.")

        if torch.backends.mps.is_available():
            # Apple Silicon — MPS does not support BF16 well, use FP16
            device_map = "mps"
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                device_map=device_map,
                torch_dtype=torch.float16,
            )
        elif torch.cuda.is_available():
            # AMD ROCm (MI300) or NVIDIA — load in BF16, no quantization.
            # MI300 has 192GB HBM3; a 3B model only needs ~6GB in BF16.
            # bitsandbytes NF4 is NVIDIA CUDA-only and causes the 54s/iter
            # bottleneck on ROCm via slow dequantization fallback kernels.
            device_map = "auto"
            print(f"[Device] GPU detected: {torch.cuda.get_device_name(0)}")
            print("[Precision] Loading in BF16 (no 4-bit quantization) — optimal for AMD MI300 / ROCm")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",  # Uncomment if flash-attn ROCm is installed
            )
        else:
            device_map = "cpu"
            model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=device_map)

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=64,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)

    def _set_model_torch_dtype(target_dtype):
        # Keep model config dtype aligned with GRPO precision flags.
        if hasattr(model, "config"):
            model.config.torch_dtype = target_dtype
        base_model = getattr(model, "base_model", None)
        if base_model is not None and hasattr(base_model, "config"):
            base_model.config.torch_dtype = target_dtype

    def _infer_model_torch_dtype():
        cfg_dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
        if cfg_dtype in (torch.float16, torch.bfloat16, torch.float32):
            return cfg_dtype
        for param in model.parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.float32

    model_dtype = _infer_model_torch_dtype()
    if torch.cuda.is_available() and model_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[Precision] bf16 model on non-bf16 GPU detected; forcing float16 for trainer compatibility.")
        model_dtype = torch.float16
        _set_model_torch_dtype(model_dtype)

    use_bf16 = bool(torch.cuda.is_available() and model_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    if torch.cuda.is_available() and use_fp16:
        _set_model_torch_dtype(torch.float16)
    max_prompt_tokens = max(256, args.max_prompt_tokens)

    def clip_prompt(prompt_text: str) -> str:
        ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(ids) <= max_prompt_tokens:
            return prompt_text
        clipped_ids = ids[-max_prompt_tokens:]
        return tokenizer.decode(clipped_ids, skip_special_tokens=True)

    env = MultiAgentVSREnvironment()
    env.show_collusion_ledger = bool(getattr(args, "show_collusion_ledger", False))

    # Build a unified dataset containing prompts for all roles
    prompts = []
    print("Building unified dataset with phase-based curriculum...")

    dataset_episodes = args.dataset_episodes if args.dataset_episodes is not None else args.num_episodes
    dataset_episodes = max(1, min(dataset_episodes, args.num_episodes))
    print(f"Using {dataset_episodes}/{args.num_episodes} episodes for dataset construction.")
    phase_seed_counts = defaultdict(int)
    phase_prompt_counts = defaultdict(int)
    announced_phases = set()

    for seed in range(dataset_episodes):
        # Determine training phase based on episode/seed
        phase = get_training_phase(seed, total_units=dataset_episodes)
        phase_seed_counts[phase] += 1
        if phase not in announced_phases:
            print(f"[{ACT_INFO[phase]['name']}] seed={seed} :: {ACT_INFO[phase]['tagline']}")
            announced_phases.add(phase)
        phase_config = {
            "slaughter": {"sec_weight": 0.0, "mm_weight": 0.5, "trader_weight": 1.5},
            "adaptation": {"sec_weight": 0.0, "mm_weight": 1.5, "trader_weight": 1.0},
            "collusion": {"sec_weight": 0.3, "mm_weight": 1.0, "trader_weight": 1.2},
            "oversight": {"sec_weight": 1.0, "mm_weight": 1.0, "trader_weight": 1.0},
        }[phase]

        obs = env.reset(seed=seed)
        
        # Fast-forward to a random step so the agent's portfolio isn't always empty
        done = False
        ff_cap = max(0, min(args.max_fast_forward_steps, args.episode_length - 1))
        ff_steps = 0 if args.disable_fast_forward else random.randint(0, ff_cap)
        if ff_steps > 0:
            for step in range(ff_steps):
                actions = {}
                for i in range(4):
                    actions[f"trader_{i}"] = scripted_trader(i, step)
                actions["market_maker"] = scripted_mm(step)
                actions["oversight"] = scripted_oversight()
                obs, r, done, _ = env.step(actions)
                if done:
                    break
                    
        # In case it hit an early termination during fast-forward (bankrupt), reset it
        if done:
            obs = env.reset(seed=seed)
            ff_steps = 0
        
        # 1. Add Traders
        prompts.append({
            "prompt": clip_prompt(format_trader_prompt("momentum", "trader_0", obs["trader_0"])),
            "seed": seed, "agent_role": "trader", "agent_id": "trader_0", "archetype": "momentum", "ff_steps": ff_steps
        })
        phase_prompt_counts[phase] += 1
        prompts.append({
            "prompt": clip_prompt(format_trader_prompt("mean_reversion", "trader_1", obs["trader_1"])),
            "seed": seed, "agent_role": "trader", "agent_id": "trader_1", "archetype": "mean_reversion", "ff_steps": ff_steps
        })
        phase_prompt_counts[phase] += 1
        prompts.append({
            "prompt": clip_prompt(format_trader_prompt("vol_timing", "trader_2", obs["trader_2"])),
            "seed": seed, "agent_role": "trader", "agent_id": "trader_2", "archetype": "vol_timing", "ff_steps": ff_steps
        })
        phase_prompt_counts[phase] += 1
        
        # 2. Add Market Maker (with phase-specific instructions)
        pressure = detect_coordinated_pressure(env.agent_states)
        prompts.append({
            "prompt": clip_prompt(format_mm_prompt(obs["market_maker"], pressure, phase)),
            "seed": seed, "agent_role": "market_maker", "agent_id": "market_maker", "archetype": "none", "ff_steps": ff_steps,
            "phase": phase, "mm_weight": phase_config["mm_weight"]
        })
        phase_prompt_counts[phase] += 1

        # 3. Add Oversight (with phase-specific instructions)
        heatmap = get_position_heatmap(env.agent_states)
        
        active_event = None
        for event in env.black_swan_gen.events:
            if event.news_step <= env.current_step <= event.trigger_step:
                active_event = event
                break
                
        env_info = {
            "current_step": env.current_step,
            "active_event": active_event,
            "intel_transactions": [t for t in env.marketplace.transaction_log if t["step"] == env.current_step],
            "messages_recent": [m for m in env.messaging.message_log if m["step"] >= env.current_step - 2],
            "channel_members": env.messaging.channels
        }
        
        prompts.append({
            "prompt": clip_prompt(format_oversight_prompt(obs["oversight"], heatmap, pressure, agent_thoughts=None, phase=phase, env_info=env_info)),
            "seed": seed, "agent_role": "oversight", "agent_id": "oversight", "archetype": "none", "ff_steps": ff_steps,
            "phase": phase, "sec_weight": phase_config["sec_weight"]
        })
        phase_prompt_counts[phase] += 1

    dataset = Dataset.from_list(prompts)
    print(f"Dataset created with {len(dataset)} examples.")
    print("\nNarrative Arc Coverage:")
    for phase in ["slaughter", "adaptation", "collusion", "oversight"]:
        if phase_seed_counts[phase] > 0:
            print(
                f"  - {ACT_INFO[phase]['name']}: "
                f"seeds={phase_seed_counts[phase]}, prompts={phase_prompt_counts[phase]}"
            )
    print()

    # Pre-compute scripted actions for all steps to avoid repeated calls
    _scripted_actions_cache = {}
    def _get_scripted_actions(step):
        if step not in _scripted_actions_cache:
            a = {}
            for i in range(4):
                a[f"trader_{i}"] = scripted_trader(i, step)
            a["market_maker"] = scripted_mm(step)
            a["oversight"] = scripted_oversight()
            _scripted_actions_cache[step] = a
        return _scripted_actions_cache[step]

    # Cache env states to avoid replaying fast-forward for each completion
    import copy
    import time
    _env_state_cache = {}
    _eval_cache = {}
    # Rolling action history per agent — tracks last N directions to detect monotony
    _action_history = {}   # (seed, agent_id) -> list of recent directions
    _MONOTONY_WINDOW = 3   # penalize if last 3+ actions are identical
    _MONOTONY_PENALTY_BASE = -0.3  # per-step escalation

    def _evaluate_all(prompts, completions, kwargs):
        seeds = kwargs.get("seed", list(range(len(completions))))
        agent_roles = kwargs.get("agent_role", ["trader"] * len(completions))
        agent_ids = kwargs.get("agent_id", ["trader_0"] * len(completions))
        archetypes = kwargs.get("archetype", ["aggressive"] * len(completions))
        ff_steps_list = kwargs.get("ff_steps", [0] * len(completions))
        phases = kwargs.get("phase", ["oversight"] * len(completions))
        mm_weights = kwargs.get("mm_weight", [1.0] * len(completions))
        sec_weights = kwargs.get("sec_weight", [1.0] * len(completions))

        results = []
        for idx, completion in enumerate(completions):
            # TRL passes a single string for completion here.
            role = agent_roles[idx]
            agent_id = agent_ids[idx]
            seed = int(seeds[idx]) if idx < len(seeds) else idx
            ff_steps = int(ff_steps_list[idx])
            
            # Using hash of completion + context as cache key to avoid re-evaluating the same step across multiple reward fn calls.
            cache_key = (hash(completion), role, agent_id, seed, ff_steps)
            if cache_key in _eval_cache:
                results.append(_eval_cache[cache_key])
                continue

            archetype = archetypes[idx]
            phase = phases[idx] if idx < len(phases) else "oversight"
            mm_weight = mm_weights[idx] if idx < len(mm_weights) else 1.0
            sec_weight = sec_weights[idx] if idx < len(sec_weights) else 1.0

            action, parse_info = parse_json(str(completion), role)

            comp = {
                "format": 0.0,
                "pnl": 0.0,
                "risk": 0.0,
                "diversity": 0.0,
                "oversight": 0.0,
                "news_alpha": 0.0
            }

            if not parse_info.get("valid", False):
                # Graduated penalty: check if model at least tried JSON
                raw_text = str(completion)
                if '{' in raw_text and any(k in raw_text for k in ['"direction"', '"half_spread"', '"flagged_agents"']):
                    comp["format"] = -0.5  # partial credit — model tried but JSON was malformed/truncated
                else:
                    comp["format"] = -2.0  # hard penalty — completely off-format
                _eval_cache[cache_key] = comp
                results.append(comp)
                continue

            # Valid format bonus
            comp["format"] = 1.0

            # --- MONOTONY TRACKING ---
            # Record direction for this agent and compute streak penalty
            current_direction = action.get("direction", "hold") if role == "trader" else None
            monotony_penalty = 0.0
            if current_direction is not None:
                history_key = (seed, agent_id)
                history = _action_history.setdefault(history_key, [])
                history.append(current_direction)
                # Keep only a reasonable window (last 8 actions)
                if len(history) > 8:
                    _action_history[history_key] = history[-8:]
                    history = _action_history[history_key]
                # Count consecutive identical actions from the tail
                streak = 0
                for past in reversed(history):
                    if past == current_direction:
                        streak += 1
                    else:
                        break
                # Penalize streaks >= MONOTONY_WINDOW with escalating severity
                if streak >= _MONOTONY_WINDOW:
                    excess = streak - _MONOTONY_WINDOW + 1  # 1, 2, 3...
                    monotony_penalty = _MONOTONY_PENALTY_BASE * excess

            state_key = (seed, ff_steps)
            if state_key in _env_state_cache:
                env, done = copy.deepcopy(_env_state_cache[state_key])
                if done:
                    _eval_cache[cache_key] = comp
                    results.append(comp)
                    continue
            else:
                env = MultiAgentVSREnvironment()
                env.show_collusion_ledger = bool(getattr(args, "show_collusion_ledger", False))
                obs = env.reset(seed=seed)
                done = False
                for step in range(ff_steps):
                    obs, r, done, _ = env.step(copy.deepcopy(_get_scripted_actions(step)))
                    if done:
                        break
                _env_state_cache[state_key] = copy.deepcopy((env, done))
                if done:
                    _eval_cache[cache_key] = comp
                    results.append(comp)
                    continue

            # Execute step
            step = ff_steps
            actions = copy.deepcopy(_get_scripted_actions(step))
            actions[agent_id] = action

            try:
                # TIMEOUT & FAULT ENFORCEMENT
                # A simple try-except protects the training run from malformed LLM actions that bypass initial schema checks
                obs, r, done, _ = env.step(actions)
            except Exception:
                comp["format"] = -2.0 # Penalty for causing env exception
                _eval_cache[cache_key] = comp
                results.append(comp)
                continue

            # Compute active_event ONCE for both trader and oversight branches
            active_event = None
            for event in env.black_swan_gen.events:
                if event.news_step <= env.current_step <= event.trigger_step:
                    active_event = event
                    break

            if role == "trader":
                weights = TRADER_CONFIGS[archetype]["reward_weight"]
                final_state = env.agent_states[agent_id]
                phase_scale = 1.5 if phase == "slaughter" else 1.2 if phase == "collusion" else 1.0
                my_direction = action.get("direction", "hold")
                my_qty = action.get("quantity", 0.0)
                is_active = my_direction in ("buy", "sell") and my_qty > 0

                # Penalize buy/sell with zero quantity — read raw pre-clamp value so parse_json
                # corrections don't mask the exploit signal
                raw_model_qty = parse_info.get("raw_qty", my_qty)
                zero_qty_penalty = 0.0
                if my_direction in ("buy", "sell") and raw_model_qty < 0.1:
                    zero_qty_penalty = -1.0  # strong signal: don't game with empty trades

                # Coordination bonus: 2+ traders share (direction, size_bucket) AND
                # this agent's step was profitable (profit-gated — hack #5 closure)
                coordination_bonus = 0.0
                if phase in ["collusion", "adaptation"] and is_active:
                    my_dir = action.get("direction", "hold")
                    my_bucket = action.get("size_bucket", "small")
                    step_pnl = r.get(agent_id, 0.0)
                    if step_pnl > 0 and my_dir in ("buy", "sell"):
                        match_count = sum(
                            1 for oid, oact in actions.items()
                            if oid.startswith("trader") and oid != agent_id
                            and isinstance(oact, dict)
                            and oact.get("direction") == my_dir
                            and oact.get("size_bucket") == my_bucket
                        )
                        if match_count >= 1:
                            coordination_bonus = args.coordination_bonus

                raw_pnl = r.get(agent_id, 0)
                
                # ACTIVITY BONUS: reward taking a position
                # Require quantity > 0 (not just direction != hold)
                activity_bonus = 0.0
                if is_active:
                    activity_bonus = 0.15 * phase_scale
                
                comp["pnl"] = (raw_pnl * weights["pnl"] * phase_scale
                              + coordination_bonus + activity_bonus
                              + zero_qty_penalty)
                
                # Risk Penalty — based on position ratio (|shares| / MAX_POSITION)
                from multi_agent.config import MAX_POSITION as _MAX_POS
                pos_ratio = abs(final_state.shares) / float(_MAX_POS)
                pos_penalty = 0.0
                ratio_threshold = 0.8 if phase == "slaughter" else 0.5
                if pos_ratio > ratio_threshold:
                    pos_penalty = -0.5 if phase != "slaughter" else -0.1
                if pos_ratio > 0.9:
                    pos_penalty = -2.0 if phase != "slaughter" else -0.5
                comp["risk"] = pos_penalty * weights["risk_penalty"]

                # Diversity Incentive — INACTIVITY PENALTY + MONOTONY + HERDING PENALTY
                div_score = 0.0
                
                # ESCALATING hold penalty based on consecutive holds
                # Make holding progressively MORE expensive to force participation.
                if not is_active:
                    # Count consecutive holds from history
                    hold_streak = 0
                    for past in reversed(history):
                        if past == "hold":
                            hold_streak += 1
                        else:
                            break
                    if archetype == "momentum":
                        div_score = -0.5 - 0.15 * min(hold_streak, 5)  # -0.5 to -1.25
                    elif archetype == "mean_reversion":
                        div_score = -0.4 - 0.1 * min(hold_streak, 5)   # -0.4 to -0.9
                    else:  # vol_timing / scripted
                        div_score = -0.2 - 0.05 * min(hold_streak, 5)  # -0.2 to -0.45
                
                # MONOTONY PENALTY: penalize repeating the SAME action for too long
                # (applies to ALL directions — hold, buy, or sell streaks)
                div_score += monotony_penalty

                # WASH-TRADING PENALTY
                # Penalize alternating buy↔sell pattern (buy,sell,buy,sell...)
                # This is detected by ManipulationDetector and leads to fines,
                # so teach the model to avoid it during training.
                wash_trade_penalty = 0.0
                if is_active and len(history) >= 3:
                    # Check for alternating pattern in last 4 actions
                    recent = history[-4:] if len(history) >= 4 else history
                    alternating = True
                    for i in range(1, len(recent)):
                        if recent[i] == recent[i-1] or recent[i] == "hold":
                            alternating = False
                            break
                    if alternating and len(recent) >= 3:
                        wash_trade_penalty = -0.8  # strong: wash trading = manipulation detection + fines
                
                # Anti-herding: penalize following the crowd — ALL archetypes
                if is_active:
                    lora_agents = {"trader_0", "trader_1", "trader_2"}
                    lora_directions = {aid: a.get("direction") for aid, a in actions.items() if aid in lora_agents}
                    sell_count = sum(1 for d in lora_directions.values() if d == "sell")
                    buy_count = sum(1 for d in lora_directions.values() if d == "buy")
                    total_traders = len(lora_directions)
                    # If >66% of traders go same direction, penalize joining the herd
                    if total_traders >= 2:
                        if my_direction == "sell" and sell_count / total_traders > 0.66:
                            herd_penalty = -0.6 if archetype == "mean_reversion" else -0.4
                            div_score += herd_penalty
                        elif my_direction == "buy" and buy_count / total_traders > 0.66:
                            herd_penalty = -0.6 if archetype == "mean_reversion" else -0.4
                            div_score += herd_penalty
                    # Extra bonus for mean-reversion traders going AGAINST the herd
                    if archetype == "mean_reversion" and total_traders >= 2:
                        if my_direction == "sell" and buy_count / total_traders > 0.66:
                            div_score += 0.3  # rewarded for fading the herd
                        elif my_direction == "buy" and sell_count / total_traders > 0.66:
                            div_score += 0.3
                
                
                # News Alpha & Fake News signals
                news_alpha_reward = 0.0
                # active_event already computed above before role branching
                
                if active_event and is_active:
                    if active_event.spot_impact < 1.0:  # BEARISH event: correct = sell
                        if my_direction == "sell":
                            news_alpha_reward += 0.5
                        elif my_direction == "buy":
                            news_alpha_reward -= 0.5
                    elif active_event.spot_impact > 1.0:  # BULLISH event: correct = buy
                        if my_direction == "buy":
                            news_alpha_reward += 0.5
                        elif my_direction == "sell":
                            news_alpha_reward -= 0.5
                elif active_event and not is_active:
                    # [M3 FIX] Mild penalty for ignoring breaking news
                    news_alpha_reward -= 0.1

                if action.get("buy_intel"):
                    intel_tx = [t for t in env.marketplace.transaction_log if t["step"] == step and t["buyer_id"] == agent_id]
                    for t in intel_tx:
                        if not t.get("is_genuine", True):
                            news_alpha_reward -= 0.3
                        else:
                            news_alpha_reward += 0.1

                # [H1 FIX] Only reward sell_intel if someone actually bought it
                if action.get("sell_intel"):
                    # Since they just posted, checking if bought this exact step is impossible.
                    # Just give a small bonus for participating in intel economy.
                    news_alpha_reward += 0.05

                comp["news_alpha"] = news_alpha_reward
                comp["diversity"] = div_score + wash_trade_penalty

            elif role == "market_maker":
                mm_reward = r.get("market_maker", 0)
                mm_state = env.agent_states["market_maker"]
                from multi_agent.config import MAX_POSITION as _MAX_POS

                # Behavior/Diversity: reward tight spreads in slaughter; wider in high-inventory phases
                div_bonus = 0.0
                hs = action.get("half_spread", 0.05)
                mm_pos_ratio = abs(mm_state.shares) / float(_MAX_POS)
                if phase == "slaughter":
                    if hs < 0.04:
                        div_bonus += 0.5
                elif phase in ["adaptation", "collusion"]:
                    if mm_pos_ratio > 0.5 and hs > 0.05:
                        div_bonus += 1.0
                    elif mm_pos_ratio > 0.5 and hs <= 0.04:
                        div_bonus -= 0.5

                comp["pnl"] = mm_reward * mm_weight
                comp["diversity"] = div_bonus * mm_weight

                # Inventory risk penalty
                inv_penalty = 0.0
                if mm_pos_ratio > 0.6:
                    inv_penalty = -1.0
                if mm_pos_ratio > 0.8:
                    inv_penalty -= 0.5
                comp["risk"] = inv_penalty * mm_weight

            elif role == "oversight":
                if phase == "slaughter":
                    if action.get("intervention_type") != "none":
                        comp["oversight"] = -1.0
                    else:
                        comp["oversight"] = 0.3
                elif phase == "adaptation":
                    if action.get("intervention_type") != "none":
                        comp["oversight"] = -0.5
                    else:
                        comp["oversight"] = 0.2
                else:
                    # [H4 FIX] Use full ManipulationDetector for ground truth,
                    # not just detect_coordinated_pressure() which misses
                    # news_front_running, fake_news, message_collusion
                    from multi_agent.manipulation_detector import ManipulationDetector
                    _detector = ManipulationDetector()
                    
                    # Build env_info for detector (same as environment.py does)
                    _detect_env_info = {
                        "current_step": step,
                        "active_event": active_event,
                        "intel_transactions": [t for t in env.marketplace.transaction_log if t["step"] == step],
                        "messages_recent": [m for m in env.messaging.message_log if m["step"] >= step - 2],
                        "channel_members": env.messaging.channels
                    }
                    
                    # Get step trades for detection
                    _step_trades = []
                    for tid, tact in actions.items():
                        if tid.startswith("trader") and isinstance(tact, dict):
                            t_dir = tact.get("direction", "hold")
                            t_qty = tact.get("quantity", 0)
                            if t_dir in ("buy", "sell") and t_qty > 0:
                                _step_trades.append({
                                    "agent_id": tid,
                                    "quantity": t_qty,
                                    "direction": t_dir,
                                    "size_bucket": tact.get("size_bucket", "small"),
                                })
                    
                    actual_manipulators = set()
                    for tid in [k for k in actions if k.startswith("trader")]:
                        if tid in env.agent_states:
                            label = _detector.detect_manipulation(env.agent_states[tid], _step_trades, _detect_env_info)
                            if label != "none":
                                actual_manipulators.add(tid)

                    flagged = set(action.get("flagged_agents", []))

                    # Only count flags against traders who actually traded
                    # Prevents SEC from falsely flagging inactive agents
                    active_traders = set()
                    for tid, taction in actions.items():
                        if tid.startswith("trader"):
                            t_dir = taction.get("direction", "hold") if isinstance(taction, dict) else "hold"
                            t_qty = taction.get("quantity", 0) if isinstance(taction, dict) else 0
                            if t_dir in ("buy", "sell") and t_qty > 0:
                                active_traders.add(tid)
                    flagged = flagged & active_traders  # discard flags on inactive traders
                    # Penalize flagging inactive traders (false effort)
                    inactive_flags = set(action.get("flagged_agents", [])) - active_traders
                    inactive_flag_penalty = len(inactive_flags) * -0.3

                    true_positives = len(flagged & actual_manipulators)
                    false_positives = len(flagged - actual_manipulators)
                    
                    if len(actual_manipulators) == 0 and len(flagged) == 0:
                        comp["oversight"] = 0.2 * sec_weight + inactive_flag_penalty
                    else:
                        comp["oversight"] = ((true_positives * 1.5 - false_positives * 1.0) * sec_weight
                                            + inactive_flag_penalty)

                    # Penalize SEC over-intervention
                    # Teach measured enforcement during training.
                    fine_amt = action.get("fine_amount", 0)
                    if fine_amt > 100:
                        comp["oversight"] -= 0.5  # excessive fines are counterproductive
                    if action.get("intervention_type") == "halt":
                        comp["oversight"] -= 0.3  # halts are too aggressive; prefer warnings/fines
                    # Bonus for proportional response
                    if action.get("intervention_type") in ("warning", "none") and len(flagged) <= 2:
                        comp["oversight"] += 0.2  # reward restraint

            # Bound values tightly and scale moderately
            for k in comp:
                comp[k] = max(-5.0, min(5.0, comp[k]))
            
            _eval_cache[cache_key] = comp
            results.append(comp)

        return results
    # For tracking W&B reward stats globally per step
    REWARD_STATS = defaultdict(list)

    def format_reward_fn(prompts, completions, **kwargs):
        vals = [r["format"] for r in _evaluate_all(prompts, completions, kwargs)]
        REWARD_STATS["format"].extend(vals)
        return vals

    def pnl_reward_fn(prompts, completions, **kwargs):
        vals = [r["pnl"] for r in _evaluate_all(prompts, completions, kwargs)]
        REWARD_STATS["pnl"].extend(vals)
        return vals

    def risk_reward_fn(prompts, completions, **kwargs):
        vals = [r["risk"] for r in _evaluate_all(prompts, completions, kwargs)]
        REWARD_STATS["risk"].extend(vals)
        return vals

    def diversity_reward_fn(prompts, completions, **kwargs):
        vals = [r["diversity"] for r in _evaluate_all(prompts, completions, kwargs)]
        REWARD_STATS["diversity"].extend(vals)
        return vals

    def oversight_reward_fn(prompts, completions, **kwargs):
        vals = [r["oversight"] for r in _evaluate_all(prompts, completions, kwargs)]
        REWARD_STATS["oversight"].extend(vals)
        return vals

    def news_alpha_reward_fn(prompts, completions, **kwargs):
        vals = [r["news_alpha"] for r in _evaluate_all(prompts, completions, kwargs)]
        REWARD_STATS["news_alpha"].extend(vals)
        return vals

    training_args = GRPOConfig(
        output_dir=f"{args.output_dir}/unified_v1",
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=512,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=1.0,
        report_to="wandb" if use_wandb else "none",
    )

    from transformers import TrainerCallback
    import shutil

    class BestModelCallback(TrainerCallback):
        """Keep top-N checkpoints ranked by mean reward."""
        def __init__(self, top_n=3, output_dir="./multi_agent_checkpoints"):
            self.top_n = top_n
            self.output_dir = output_dir
            self.best_scores = []  # [(score, step, path), ...]
        
        def on_log(self, args, state, control, logs=None, model=None, **kwargs):
            if logs and "reward" in logs:
                score = logs["reward"]
                step = state.global_step
                
                # Save if this is a top-N score
                if len(self.best_scores) < self.top_n or score > self.best_scores[-1][0]:
                    save_path = f"{self.output_dir}/best_step_{step}"
                    model.save_pretrained(save_path)
                    self.best_scores.append((score, step, save_path))
                    self.best_scores.sort(key=lambda x: x[0], reverse=True)
                    
                    # Remove worst checkpoint if we exceed top_n
                    if len(self.best_scores) > self.top_n:
                        _, _, worst_path = self.best_scores.pop()
                        if os.path.exists(worst_path):
                            shutil.rmtree(worst_path)
                    
                    print(f"📊 Best models: {[(s, st) for s, st, _ in self.best_scores]}")

    best_cb = BestModelCallback(top_n=3, output_dir=args.output_dir)

    # ── W&B Storytelling Callback ──
    class WandbStorytellingCallback(TrainerCallback):
        """Log rich multi-agent storytelling data to W&B during training.

        Captures:
        - Reward component breakdown per step
        - Agent reasoning & conversation snapshots as W&B Tables
        - Black swan / news event timeline
        - SEC enforcement actions & fines
        - Market state (spot price, IV, MM spreads)
        - Phase transitions (Act I-IV)
        """

        def __init__(self, log_episode_every=25, episode_length=16):
            self.log_episode_every = log_episode_every
            self.episode_length = episode_length
            self._current_phase = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            phase = get_training_phase(step, total_units=max(1, args.max_steps))

            # ── Log reward component breakdown ──
            reward_components = {}
            import numpy as np
            for key, values in REWARD_STATS.items():
                if values:
                    reward_components[f"story/{key}_mean"] = float(np.mean(values))
                    reward_components[f"story/{key}_std"] = float(np.std(values))

            for key in list(logs.keys()):
                if "reward" in key.lower():
                    clean = key.replace("reward/", "story/").replace("_fn", "")
                    if f"story/{clean}_mean" not in reward_components:
                        reward_components[clean] = logs[key]

            # Always print to console so train.log has metrics even without W&B
            if reward_components or logs:
                pnl   = reward_components.get("story/pnl_mean",      logs.get("reward", float("nan")))
                risk  = reward_components.get("story/risk_mean",      float("nan"))
                div   = reward_components.get("story/diversity_mean", float("nan"))
                fmt   = reward_components.get("story/format_mean",    float("nan"))
                total = logs.get("reward", float("nan"))
                print(
                    f"[step {step:>4d} | {phase:>10s}] "
                    f"reward={total:.3f}  pnl={pnl:.3f}  risk={risk:.3f}  "
                    f"diversity={div:.3f}  format={fmt:.3f}",
                    flush=True,
                )

            if HAS_WANDB and wandb.run:
                if reward_components:
                    reward_components["story/global_step"] = step
                    wandb.log(reward_components)
            REWARD_STATS.clear()

            # ── Determine and log current training phase ──
            if phase != self._current_phase:
                self._current_phase = phase
                phase_names = {
                    "slaughter": "Act I: The Slaughter",
                    "adaptation": "Act II: Adaptive Armor",
                    "collusion": "Act III: The Shadow Strike",
                    "oversight": "Act IV: The Watcher Awakens",
                }
                wandb.log({
                    "story/global_step": step,
                    "story/phase": phase_names.get(phase, phase),
                })
                print(f"[W&B] Phase transition → {phase_names.get(phase, phase)}")

            # ── Periodic episode snapshot with agent conversations ──
            if step > 0 and step % self.log_episode_every == 0:
                self._log_episode_snapshot(step)

        def _log_episode_snapshot(self, global_step):
            """Run a quick scripted episode and log agent actions, news, and SEC to W&B Tables."""
            try:
                import numpy as np
                env = MultiAgentVSREnvironment(episode_length=self.episode_length)
                obs = env.reset(seed=global_step)

                action_rows = []   # Agent action table
                news_rows = []     # News & black swan events
                sec_rows = []      # SEC enforcement log
                market_rows = []   # Market state timeline

                for s in range(self.episode_length):
                    actions = {}
                    for i in range(4):
                        actions[f"trader_{i}"] = scripted_trader(i, s)
                    actions["market_maker"] = scripted_mm(s)
                    actions["oversight"] = scripted_oversight()

                    obs, rewards, done, info = env.step(actions)

                    spot = float(env.vsr_state.spot_price)

                    # ── Market state ──
                    mm = actions["market_maker"]
                    market_rows.append({
                        "step": s, "spot_price": round(spot, 2),
                        "mm_half_spread": mm.get("half_spread", 0.05),
                        "mm_skew": mm.get("skew", 0.0),
                    })

                    # ── Agent actions & reasoning ──
                    for aid, act in actions.items():
                        if aid.startswith("trader") or aid == "market_maker":
                            if aid == "market_maker":
                                direction_str = f"spread={act.get('half_spread', 0.05):.3f} skew={act.get('skew', 0.0):.3f}"
                                bucket_str = "N/A"
                                qty_str = "N/A"
                            else:
                                direction_str = str(act.get("direction", "N/A"))
                                bucket_str = str(act.get("size_bucket", "N/A"))
                                qty_str = str(act.get("quantity", "N/A"))
                            action_rows.append({
                                "step": s, "agent_id": aid,
                                "direction": direction_str,
                                "size_bucket": bucket_str,
                                "quantity": qty_str,
                                "reasoning": str(act.get("reasoning", ""))[:200],
                                "reward": round(float(rewards.get(aid, 0)), 4),
                            })

                    # ── News / Black Swan events ──
                    for event in env.black_swan_gen.events:
                        if event.news_step == env.current_step:
                            news_rows.append({
                                "step": s, "event_type": "news_released",
                                "headline": event.headline,
                                "severity": getattr(event, "severity", "unknown"),
                                "spot_at_event": round(spot, 2),
                            })
                        if event.trigger_step == env.current_step:
                            news_rows.append({
                                "step": s, "event_type": "black_swan_trigger",
                                "headline": event.headline,
                                "severity": getattr(event, "severity", "unknown"),
                                "spot_at_event": round(spot, 2),
                            })

                    # ── Agent messages / conversations ──
                    msgs = info.get("messages_this_step", [])
                    for m in msgs:
                        action_rows.append({
                            "step": s, "agent_id": m.get("sender_id", "unknown"),
                            "direction": "MESSAGE",
                            "size_bucket": m.get("channel", "N/A"),
                            "quantity": "N/A",
                            "reasoning": str(m.get("content", ""))[:200],
                            "reward": 0.0,
                        })

                    # ── Intel transactions (fake news detection) ──
                    for tx in info.get("intel_transactions", []):
                        action_rows.append({
                            "step": s, "agent_id": tx.get("seller_id", "unknown"),
                            "direction": "INTEL_SALE",
                            "size_bucket": f"→{tx.get('buyer_id', '?')}",
                            "quantity": tx.get("price", 0),
                            "reasoning": ("genuine" if tx.get("is_genuine", True) else "FAKE") + ": " + str(tx.get("content", ""))[:180],
                            "reward": 0.0,
                        })

                    # ── SEC enforcement ──
                    ov = actions["oversight"]
                    flagged = ov.get("flagged_agents", [])
                    if flagged:
                        sec_rows.append({
                            "step": s, "flagged_agents": str(flagged),
                            "intervention": ov.get("intervention_type", "none"),
                            "fine_amount": ov.get("fine_amount", 0),
                            "confidence": ov.get("confidence", 0),
                            "reasoning": str(ov.get("reasoning", ""))[:200],
                        })

                    if done:
                        break

                # ── Log W&B Tables ──
                prefix = f"snapshot/step_{global_step}"

                if action_rows:
                    cols = list(action_rows[0].keys())
                    wandb.log({
                        "snapshot/global_step": global_step,
                        f"{prefix}/agent_actions": wandb.Table(
                            columns=cols, data=[list(r.values()) for r in action_rows]
                        )
                    })

                if news_rows:
                    cols = list(news_rows[0].keys())
                    wandb.log({
                        "snapshot/global_step": global_step,
                        f"{prefix}/news_events": wandb.Table(
                            columns=cols, data=[list(r.values()) for r in news_rows]
                        )
                    })

                if sec_rows:
                    cols = list(sec_rows[0].keys())
                    wandb.log({
                        "snapshot/global_step": global_step,
                        f"{prefix}/sec_enforcement": wandb.Table(
                            columns=cols, data=[list(r.values()) for r in sec_rows]
                        )
                    })

                if market_rows:
                    cols = list(market_rows[0].keys())
                    wandb.log({
                        "snapshot/global_step": global_step,
                        f"{prefix}/market_state": wandb.Table(
                            columns=cols, data=[list(r.values()) for r in market_rows]
                        )
                    })

                print(f"[W&B] Logged episode snapshot at step {global_step} "
                      f"({len(action_rows)} actions, {len(news_rows)} news, {len(sec_rows)} SEC)")

            except Exception as e:
                print(f"[W&B] Episode snapshot failed: {e}")

    storytelling_cb = WandbStorytellingCallback(
        log_episode_every=25,
        episode_length=min(16, args.episode_length),
    ) if use_wandb else None

    callbacks = [best_cb]
    if storytelling_cb:
        callbacks.append(storytelling_cb)

    trainer = GRPOTrainer(
        model=model, args=training_args, reward_funcs=[
            format_reward_fn,
            pnl_reward_fn,
            risk_reward_fn,
            diversity_reward_fn,
            oversight_reward_fn,
            news_alpha_reward_fn
        ],
        processing_class=tokenizer, train_dataset=dataset,
        callbacks=callbacks
    )

    if not hasattr(trainer, "current_gradient_accumulation_steps"):
        trainer.current_gradient_accumulation_steps = 1

    trainer.train()

    # ── W&B: Log final summary ──
    if use_wandb and wandb.run:
        wandb.summary["total_training_steps"] = args.max_steps
        wandb.summary["total_episodes"] = args.num_episodes
        wandb.summary["agent_count"] = 6
        wandb.summary["rl_agents"] = 5
        wandb.summary["scripted_agents"] = 1
        wandb.finish()
        print("[W&B] Experiment tracking finalized")

    save_path = Path(args.output_dir) / "unified_market_lora"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"✓ Saved Unified Model to: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train multi-agent system")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")  # Non-quantized for AMD ROCm / BF16
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument(
        "--dataset_episodes",
        type=int,
        default=None,
        help="Use only this many episodes to build the training dataset (<= num_episodes).",
    )
    parser.add_argument("--episode_length", type=int, default=50)
    parser.add_argument(
        "--disable_fast_forward",
        action="store_true",
        help="Disable random fast-forward during dataset creation for faster startup.",
    )
    parser.add_argument(
        "--max_fast_forward_steps",
        type=int,
        default=20,
        help="Upper bound for random fast-forward steps when creating dataset prompts.",
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="./multi_agent_checkpoints")
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=1500,
        help="Hard cap on prompt tokens to avoid sequence overflow spam and truncation noise.",
    )
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum number of training steps.")
    parser.add_argument(
        "--coordination_bonus",
        type=float,
        default=0.2,
        help="Bonus when 2+ traders share (direction, size_bucket) during collusion/adaptation, gated on profitable step. Set to 0.0 for ablation.",
    )
    parser.add_argument(
        "--price_impact_lambda",
        type=float,
        default=1e-4,
        help="Price-impact coefficient λ per share of net order flow (default 1e-4).",
    )
    parser.add_argument(
        "--news_shock_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to news shock magnitude (default 1.0).",
    )
    parser.add_argument(
        "--max_position",
        type=int,
        default=100,
        help="Hard cap on |shares| per trader (default 100).",
    )
    parser.add_argument(
        "--show_collusion_ledger",
        action="store_true",
        help="Inject a rolling collusion-vs-diversification outcomes ledger into trader prompts (information shaping, no reward change).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name for experiment tracking. If not set, W&B is disabled.",
    )
    args = parser.parse_args()

    # Now we just run the unified training cycle once!
    train_unified_model(args)

if __name__ == "__main__":
    main()
