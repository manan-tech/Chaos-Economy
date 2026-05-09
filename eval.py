"""Inference-only evaluation for the Chaos Economy multi-agent simulator.

Bypasses the GRPO trainer entirely. Loads a base model (optionally with a
saved LoRA adapter) and runs N episodes through the env, logging story/*
metrics to W&B for direct comparison against trained runs.

Usage:
    # Trained LoRA evaluation
    python eval.py --base_model unsloth/Llama-3.2-1B-Instruct \
        --load_lora_path ./checkpoints/unified_market_lora \
        --num_episodes 30 --episode_length 250 \
        --wandb_project "Chaos Economy" --run_name eval-trained-1b

    # Baseline (no adapter)
    python eval.py --base_model unsloth/Llama-3.2-1B-Instruct \
        --num_episodes 30 --episode_length 250 \
        --wandb_project "Chaos Economy" --run_name eval-baseline-1b

    # AWS Bedrock 70B (no local GPU needed)
    python eval.py --use_bedrock \
        --bedrock_model meta.llama3-1-70b-instruct-v1:0 \
        --aws_region us-east-1 \
        --num_episodes 5 --episode_length 50 \
        --wandb_project "Chaos Economy" --run_name eval-bedrock-70b
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def generate_batch_bedrock(prompts, bedrock_client, model_id, max_new_tokens=120, temperature=0.7):
    """Call AWS Bedrock converse API. Returns one response string per prompt."""
    results = []
    for prompt in prompts:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_new_tokens, "temperature": temperature},
        )
        results.append(response["output"]["message"]["content"][0]["text"])
    return results


def generate_batch(prompts, model, tokenizer, device, max_new_tokens=120, temperature=0.7):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=1500).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )
    in_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(out[i][in_len:], skip_special_tokens=True) for i in range(len(prompts))]


def diversity_score(actions):
    """Shannon-entropy-style diversity over (direction, bucket) keys for traders."""
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


def run_episode(model, tokenizer, device, episode_length, seed, use_model=True, verbose=False,
                bedrock_client=None, bedrock_model=None):
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

        if use_model:
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

            outputs = []
            for i, (aid, role, temp) in enumerate(meta):
                if bedrock_client:
                    out = generate_batch_bedrock([prompts[i]], bedrock_client, bedrock_model,
                                                 max_new_tokens=120, temperature=temp)[0]
                else:
                    out = generate_batch([prompts[i]], model, tokenizer, device,
                                         max_new_tokens=120, temperature=temp)[0]
                outputs.append(out)

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

            heatmap = get_position_heatmap(env.agent_states) if hasattr(env, "agent_states") else {}
            ov_prompt = format_oversight_prompt(obs["oversight"], heatmap, coord, agent_thoughts)
            if bedrock_client:
                ov_out = generate_batch_bedrock([ov_prompt], bedrock_client, bedrock_model,
                                                max_new_tokens=140, temperature=0.5)[0]
            else:
                ov_out = generate_batch([ov_prompt], model, tokenizer, device,
                                        max_new_tokens=140, temperature=0.5)[0]
            ov_parsed, ov_info = parse_json(ov_out, role="oversight")
            format_attempts += 1
            if ov_info.get("valid"):
                format_hits += 1
                actions["oversight"] = ov_parsed
            else:
                actions["oversight"] = scripted_oversight()
        else:
            for i in range(3):
                actions[f"trader_{i}"] = scripted_trader(i, step)
            actions["market_maker"] = scripted_mm(step)
            actions["oversight"] = scripted_oversight()

        actions["trader_3"] = scripted_trader(3, step)

        if actions["oversight"].get("intervention_type", "none") != "none":
            oversight_intervention_steps += 1

        diversity_per_step.append(diversity_score(actions))

        obs, rewards, done, _info = env.step(actions)
        for k in rewards_total:
            rewards_total[k] += float(rewards.get(k, 0.0))

        if verbose and step % 20 == 0:
            print(f"  step {step:3d}  pnl_t0={rewards.get('trader_0',0):+.2f}  "
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--load_lora_path", default=None)
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--episode_length", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_name", default=None)
    # AWS Bedrock flags — skips local model loading entirely
    p.add_argument("--use_bedrock", action="store_true")
    p.add_argument("--bedrock_model", default="meta.llama3-1-70b-instruct-v1:0")
    p.add_argument("--aws_region", default="us-east-1")
    args = p.parse_args()

    if args.wandb_project and HAS_WANDB and os.environ.get("WANDB_API_KEY"):
        wandb.init(project=args.wandb_project, name=args.run_name or "eval",
                   config=vars(args))

    bedrock_client = None
    model = tokenizer = device = None

    if args.use_bedrock:
        import boto3
        print(f"[Bedrock] Region={args.aws_region}  Model={args.bedrock_model}")
        bedrock_client = boto3.client("bedrock-runtime", region_name=args.aws_region)
        print("[Bedrock] Client ready — no local model loaded")
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
        if device == "cuda":
            print(f"[Device] {device} — {torch.cuda.get_device_name(0)}  dtype={dtype}")
        else:
            print(f"[Device] {device}  dtype={dtype}")
        if device == "cpu":
            print("[WARNING] No GPU detected — running on CPU will be very slow.")

        print(f"[Model] Loading base: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # device_map="cuda:0" pins the whole model to one GPU — avoids split-device
        # issues on ROCm where device_map="auto" can partially land on CPU.
        load_device_map = "cuda:0" if device == "cuda" else ("mps" if device == "mps" else "cpu")
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=load_device_map, torch_dtype=dtype)
        print(f"[Model] Loaded — first param device: {next(model.parameters()).device}")

        if args.load_lora_path:
            from peft import PeftModel
            print(f"[LoRA] Loading adapter: {args.load_lora_path}")
            model = PeftModel.from_pretrained(model, args.load_lora_path)
        else:
            print("[LoRA] No adapter (baseline mode)")
        model.eval()

    print(f"\n[Eval] {args.num_episodes} episodes x {args.episode_length} steps\n")

    agg = {"format_rate": [], "diversity_mean": [], "oversight_rate": [],
           "rewards_total": []}

    for ep in range(args.num_episodes):
        print(f"--- Episode {ep+1}/{args.num_episodes} ---")
        result = run_episode(model, tokenizer, device, args.episode_length,
                             seed=args.seed + ep, use_model=True, verbose=(ep == 0),
                             bedrock_client=bedrock_client,
                             bedrock_model=args.bedrock_model if args.use_bedrock else None)
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
