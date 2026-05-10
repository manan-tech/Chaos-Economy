"""Generate a rich per-step replay log for the Chaos Economy HF Space frontend.

Runs one episode (default 50 steps) using either a local trained LoRA model
or AWS Bedrock for inference, capturing all per-step detail: spot price,
agent actions + reasoning, messages, news, black swan events, SEC
interventions, intel transactions, and rewards.

Usage:
    # Local trained model (ROCm / no AWS needed)
    python generate_replay.py \\
        --local_model meta-llama/Llama-3.2-1B-Instruct \\
        --lora_path ./checkpoints/chaos_1b_200steps \\
        --episode_length 50 --seed 42 --run_name local_demo

    # AWS Bedrock
    python generate_replay.py \\
        --bedrock_model us.meta.llama4-maverick-17b-instruct-v1:0 \\
        --episode_length 50 --seed 42 --run_name bedrock_demo
"""

import argparse
import json
import os
from pathlib import Path

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

AGENT_ARCHETYPES = {
    "trader_0": "Momentum",
    "trader_1": "Mean Reversion",
    "trader_2": "Vol Timing",
    "trader_3": "Scripted",
}


def generate_bedrock(client, model_id: str, prompt: str, max_tokens: int = 140, temperature: float = 0.7) -> str:
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        print(f"[Bedrock] API error: {e}")
        return ""


def load_local_model(base_model: str, lora_path: str | None):
    """Load base model + optional LoRA adapter for local inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[Local] Loading {base_model}  device={device}  dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device, torch_dtype=dtype)
    if lora_path:
        print(f"[Local] Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    print("[Local] Model ready")
    return model, tokenizer, device


def generate_local(model, tokenizer, device: str, prompt: str, max_tokens: int = 140, temperature: float = 0.7) -> str:
    import torch
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )
    in_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()


def _active_event_info(env, step: int) -> dict | None:
    """Return black swan event dict if one is active this step, else None."""
    for event in env.black_swan_gen.events:
        if event.news_step <= step <= event.trigger_step:
            return {
                "headline": getattr(event, "headline", None),
                "spot_impact": float(getattr(event, "spot_impact", 0.0)),
                "variance_impact": float(getattr(event, "variance_impact", 0.0)),
                "trigger_step": int(event.trigger_step),
            }
    return None


def _active_headline(env, step: int) -> str | None:
    """Return news headline if active this step."""
    for event in env.black_swan_gen.events:
        if event.news_step <= step <= event.trigger_step:
            return getattr(event, "headline", None)
    return None


def run_episode(generate_fn, episode_length: int, seed: int, verbose: bool) -> list[dict]:
    """generate_fn(prompt, max_tokens, temperature) -> str"""
    env = MultiAgentVSREnvironment(episode_length=episode_length)
    obs = env.reset(seed=seed)

    steps_log = []

    for step in range(episode_length):
        actions = {}

        # --- Traders + MM ---
        prompts, meta = [], []
        for archetype, cfg in TRADER_CONFIGS.items():
            for tid in cfg["trader_ids"]:
                aid = f"trader_{tid}"
                if aid in obs and aid != "trader_3":
                    prompts.append(format_trader_prompt(archetype, aid, obs[aid]))
                    meta.append((aid, "trader", cfg["temperature"]))

        coord = detect_coordinated_pressure(env.agent_states) if hasattr(env, "agent_states") else {}
        prompts.append(format_mm_prompt(obs["market_maker"], coord))
        meta.append(("market_maker", "market_maker", 0.3))

        for prompt, (aid, role, temp) in zip(prompts, meta):
            raw = generate_fn(prompt, max_tokens=140, temperature=temp)
            parsed, info = parse_json(raw, role=role)
            if info.get("valid"):
                actions[aid] = parsed
            else:
                actions[aid] = scripted_trader(int(aid.split("_")[1]), step) if role == "trader" else scripted_mm(step)

        # trader_3 always scripted
        actions["trader_3"] = scripted_trader(3, step)

        # --- Oversight ---
        heatmap = get_position_heatmap(env.agent_states) if hasattr(env, "agent_states") else {}
        agent_thoughts = {aid: actions[aid].get("reasoning", "") for aid in actions if aid != "oversight"}
        ov_prompt = format_oversight_prompt(obs["oversight"], heatmap, coord, agent_thoughts)
        ov_raw = generate_fn(ov_prompt, max_tokens=160, temperature=0.5)
        ov_parsed, ov_info = parse_json(ov_raw, role="oversight")
        actions["oversight"] = ov_parsed if ov_info.get("valid") else scripted_oversight()

        # --- Step environment ---
        obs, rewards, done, info = env.step(actions)

        spot = float(env.vsr_state.spot_price)
        headline = _active_headline(env, env.current_step)
        black_swan = _active_event_info(env, env.current_step)

        # Build clean action records preserving message field
        action_records = {}
        for aid in ["trader_0", "trader_1", "trader_2", "trader_3", "market_maker", "oversight"]:
            act = actions.get(aid, {})
            record = {k: v for k, v in act.items()}
            msg = act.get("send_message") or act.get("messages_sent")
            record["message"] = msg if isinstance(msg, dict) else None
            action_records[aid] = record

        # Intel transactions from marketplace
        intel_txns = [
            t for t in env.marketplace.transaction_log
            if t.get("step") == env.current_step
        ]

        # Interventions this step
        interventions = [
            iv for iv in info.get("recent_interventions", [])
            if iv.get("step") == env.current_step
        ]

        step_record = {
            "step": int(env.current_step),
            "spot_price": spot,
            "mm_bid": float(spot * (1.0 - info.get("mm_half_spread", 0.05) + info.get("mm_skew", 0.0))),
            "mm_ask": float(spot * (1.0 + info.get("mm_half_spread", 0.05) + info.get("mm_skew", 0.0))),
            "mm_half_spread": float(info.get("mm_half_spread", 0.05)),
            "mm_skew": float(info.get("mm_skew", 0.0)),
            "news_headline": headline,
            "black_swan": black_swan,
            "collusion_events": sum(
                1 for c in env.collusion_history
                if c["step"] == env.current_step and c["was_collusion"]
            ),
            "sec_fines": float(sum(iv.get("fine_amount", 0.0) for iv in interventions)),
            "actions": action_records,
            "messages": info.get("messages_this_step", []),
            "intel_transactions": intel_txns,
            "sec_interventions": interventions,
            "detected_manipulations": info.get("detected_manipulations", {}),
            "rewards": {k: float(v) for k, v in rewards.items()},
        }
        steps_log.append(step_record)

        if verbose:
            print(f"  step {step+1:3d}  spot={spot:.2f}  "
                  f"fines={step_record['sec_fines']:.0f}  "
                  f"msgs={len(step_record['messages'])}")

        if done:
            print(f"[Episode] terminated early at step {step + 1}")
            break

    return steps_log


def main():
    p = argparse.ArgumentParser(description="Generate Chaos Economy replay log")
    # Local model flags
    p.add_argument("--local_model", default=None,
                   help="HF model ID or path for local inference (skips Bedrock entirely)")
    p.add_argument("--lora_path", default=None,
                   help="Path to trained LoRA adapter (used with --local_model)")
    # Bedrock flags (only used when --local_model is not set)
    p.add_argument("--bedrock_model", default="meta.llama3-1-70b-instruct-v1:0")
    p.add_argument("--aws_region", default="us-east-1")
    # Common flags
    p.add_argument("--episode_length", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default="replay")
    p.add_argument("--output_dir", default="replay_logs")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.local_model:
        model, tokenizer, device = load_local_model(args.local_model, args.lora_path)
        def generate_fn(prompt, max_tokens=140, temperature=0.7):
            return generate_local(model, tokenizer, device, prompt, max_tokens, temperature)
        model_label = args.local_model + (f" + {args.lora_path}" if args.lora_path else "")
    else:
        import boto3
        print(f"[Bedrock] region={args.aws_region}  model={args.bedrock_model}")
        client = boto3.client("bedrock-runtime", region_name=args.aws_region)
        def generate_fn(prompt, max_tokens=140, temperature=0.7):
            return generate_bedrock(client, args.bedrock_model, prompt, max_tokens, temperature)
        model_label = args.bedrock_model

    print(f"[Episode] length={args.episode_length}  seed={args.seed}")
    steps = run_episode(generate_fn, args.episode_length, args.seed, args.verbose)

    out = {
        "meta": {
            "model": model_label,
            "run_name": args.run_name,
            "steps": len(steps),
            "seed": args.seed,
            "episode_length": args.episode_length,
        },
        "archetypes": AGENT_ARCHETYPES,
        "steps": steps,
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / f"replay_{args.run_name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[Done] saved {len(steps)} steps → {out_path}")


if __name__ == "__main__":
    main()
