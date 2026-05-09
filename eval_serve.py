"""Evaluation script that routes inference through a running serve.py vLLM server.

Drop-in replacement for eval.py when you want to evaluate via the HTTP serving
layer rather than loading the model directly. Requires serve.py to be running.

Usage:
    # Start the server first (in another terminal)
    python serve.py --lora_path ./checkpoints/unified_market_lora

    # Then evaluate against it
    python eval_serve.py \
        --server_url http://localhost:8000 \
        --num_episodes 10 --episode_length 250 \
        --wandb_project "Chaos Economy" --run_name eval-vllm-lora

    # Smoke test (1 episode, 10 steps)
    python eval_serve.py --num_episodes 1 --episode_length 10
"""

import argparse
import os
from collections import Counter

import httpx
import numpy as np

from multi_agent.environment import MultiAgentVSREnvironment
from train_multi_agent_pipeline import (
    TRADER_CONFIGS,
    format_trader_prompt,
    format_mm_prompt,
    format_oversight_prompt,
    parse_json,
    scripted_trader,
    scripted_mm,
    scripted_oversight,
    detect_coordinated_pressure,
    get_position_heatmap,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# vLLM server generation
# ---------------------------------------------------------------------------

def generate_batch_vllm(
    prompts: list[str],
    server_url: str,
    max_tokens: int = 120,
    temperature: float = 0.7,
    timeout: float = 60.0,
) -> list[str]:
    """Send a batch of prompts to serve.py /generate_batch and return text responses."""
    payload = [
        {"prompt": p, "max_tokens": max_tokens, "temperature": temperature}
        for p in prompts
    ]
    resp = httpx.post(f"{server_url}/generate_batch", json=payload, timeout=timeout)
    resp.raise_for_status()
    return [item["text"] for item in resp.json()]


def _check_server(server_url: str) -> None:
    try:
        resp = httpx.get(f"{server_url}/health", timeout=5.0)
        info = resp.json()
        lora = info.get("lora", False)
        print(f"[serve] Connected to {server_url}  lora={lora}")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach serve.py at {server_url}. "
            f"Start it first: python serve.py --lora_path ./checkpoints/unified_market_lora\n{e}"
        )


# ---------------------------------------------------------------------------
# Diversity metric (identical to eval.py)
# ---------------------------------------------------------------------------

def diversity_score(actions: dict) -> float:
    keys = []
    for aid, a in actions.items():
        if not aid.startswith("trader"):
            continue
        d = a.get("direction", "hold")
        b = a.get("size_bucket", "small") if d != "hold" else "hold"
        keys.append((d, b))
    if not keys:
        return 0.0
    counts = Counter(keys)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log(p + 1e-12) for p in probs)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(server_url: str, episode_length: int, seed: int, verbose: bool = False) -> dict:
    env = MultiAgentVSREnvironment(episode_length=episode_length)
    obs = env.reset(seed=seed)

    rewards_total = {f"trader_{i}": 0.0 for i in range(4)}
    rewards_total["market_maker"] = 0.0
    rewards_total["oversight"] = 0.0

    format_hits = 0
    format_attempts = 0
    diversity_per_step = []
    oversight_intervention_steps = 0

    for step in range(episode_length):
        actions = {}

        # Build trader + MM prompts
        prompts, meta = [], []
        for archetype, cfg in TRADER_CONFIGS.items():
            for tid in cfg["trader_ids"]:
                aid = f"trader_{tid}"
                if aid in obs:
                    prompts.append(format_trader_prompt(archetype, aid, obs[aid]))
                    meta.append((aid, "trader", cfg["temperature"]))

        coord = detect_coordinated_pressure(env.agent_states) if hasattr(env, "agent_states") else {}
        prompts.append(format_mm_prompt(obs["market_maker"], coord))
        meta.append(("market_maker", "market_maker", 0.3))

        # Single batch call to serve.py for traders + MM
        outputs = generate_batch_vllm(prompts, server_url, max_tokens=120, temperature=0.7)

        agent_thoughts = {}
        for output, (aid, role, _) in zip(outputs, meta):
            parsed, info = parse_json(output, role=role)
            format_attempts += 1
            if info.get("valid"):
                format_hits += 1
                actions[aid] = parsed
            else:
                if role == "trader":
                    actions[aid] = scripted_trader(int(aid.split("_")[1]), step)
                else:
                    actions[aid] = scripted_mm(step)
            agent_thoughts[aid] = actions[aid].get("reasoning", "")

        # Oversight — separate call (different max_tokens / temperature)
        heatmap = get_position_heatmap(env.agent_states) if hasattr(env, "agent_states") else {}
        ov_prompt = format_oversight_prompt(obs["oversight"], heatmap, coord, agent_thoughts)
        ov_out = generate_batch_vllm([ov_prompt], server_url, max_tokens=140, temperature=0.5)[0]
        ov_parsed, ov_info = parse_json(ov_out, role="oversight")
        format_attempts += 1
        if ov_info.get("valid"):
            format_hits += 1
            actions["oversight"] = ov_parsed
        else:
            actions["oversight"] = scripted_oversight()

        # trader_3 is always scripted
        actions["trader_3"] = scripted_trader(3, step)

        if actions["oversight"].get("intervention_type", "none") != "none":
            oversight_intervention_steps += 1

        diversity_per_step.append(diversity_score(actions))
        obs, rewards, done, _info = env.step(actions)
        for k in rewards_total:
            rewards_total[k] += float(rewards.get(k, 0.0))

        if verbose and step % 20 == 0:
            print(f"  step {step:3d}  pnl_t0={rewards.get('trader_0', 0):+.2f}  "
                  f"div={diversity_per_step[-1]:.2f}  "
                  f"fmt={format_hits}/{format_attempts}")

        if done:
            break

    steps_run = len(diversity_per_step)
    return {
        "rewards_total": rewards_total,
        "format_rate": format_hits / max(1, format_attempts),
        "diversity_mean": float(np.mean(diversity_per_step)),
        "oversight_rate": oversight_intervention_steps / max(1, steps_run),
        "steps": steps_run,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Eval via vLLM serve.py HTTP server")
    p.add_argument("--server_url", default="http://localhost:8000",
                   help="URL of running serve.py instance")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--episode_length", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_name", default=None)
    args = p.parse_args()

    _check_server(args.server_url)

    if args.wandb_project and HAS_WANDB and os.environ.get("WANDB_API_KEY"):
        wandb.init(project=args.wandb_project, name=args.run_name or "eval-vllm",
                   config=vars(args))

    print(f"\n[Eval] {args.num_episodes} episodes x {args.episode_length} steps  "
          f"server={args.server_url}\n")

    agg = {"format_rate": [], "diversity_mean": [], "oversight_rate": [], "rewards_total": []}

    for ep in range(args.num_episodes):
        print(f"--- Episode {ep+1}/{args.num_episodes} ---")
        result = run_episode(args.server_url, args.episode_length,
                             seed=args.seed + ep, verbose=(ep == 0))
        agg["format_rate"].append(result["format_rate"])
        agg["diversity_mean"].append(result["diversity_mean"])
        agg["oversight_rate"].append(result["oversight_rate"])
        agg["rewards_total"].append(result["rewards_total"])

        ep_pnl = sum(result["rewards_total"][f"trader_{i}"] for i in range(4)) / 4
        print(f"  ep_format={result['format_rate']:.2%}  "
              f"ep_diversity={result['diversity_mean']:.3f}  "
              f"ep_oversight={result['oversight_rate']:.2%}  "
              f"ep_pnl_mean={ep_pnl:+.2f}\n")

        if HAS_WANDB and wandb.run:
            wandb.log({
                "story/global_step": ep,
                "story/format_mean": result["format_rate"],
                "story/diversity_mean": result["diversity_mean"],
                "story/oversight_mean": result["oversight_rate"],
                "story/pnl_mean": ep_pnl,
            })

    print("=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"format_rate     mean={np.mean(agg['format_rate']):.2%}  std={np.std(agg['format_rate']):.2%}")
    print(f"diversity_mean  mean={np.mean(agg['diversity_mean']):.3f}  std={np.std(agg['diversity_mean']):.3f}")
    print(f"oversight_rate  mean={np.mean(agg['oversight_rate']):.2%}  std={np.std(agg['oversight_rate']):.2%}")
    for agent in ["trader_0", "trader_1", "trader_2", "trader_3", "market_maker", "oversight"]:
        vals = [r[agent] for r in agg["rewards_total"]]
        print(f"{agent:<14}  cum_reward mean={np.mean(vals):+.2f}  std={np.std(vals):.2f}")

    if HAS_WANDB and wandb.run:
        wandb.summary["format_rate"] = float(np.mean(agg["format_rate"]))
        wandb.summary["diversity_mean"] = float(np.mean(agg["diversity_mean"]))
        wandb.summary["oversight_rate"] = float(np.mean(agg["oversight_rate"]))
        wandb.finish()


if __name__ == "__main__":
    main()
