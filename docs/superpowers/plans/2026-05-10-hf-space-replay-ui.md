# HF Space Replay UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `generate_replay.py` log generator (Bedrock-powered, captures rich per-step data) and a self-contained `index.html` HF Space frontend that lets users scrub through a 50-step episode and see every agent decision, message, news event, and SEC action.

**Architecture:** `generate_replay.py` runs one episode, captures all step-level detail from `env.step()` return values plus direct environment state, and writes a single JSON to `replay_logs/`. `index.html` embeds the replay JSON inline as a JS constant and renders it with Chart.js — no server calls, works in HF Space static serving.

**Tech Stack:** Python 3.10+, boto3 (Bedrock), existing `MultiAgentVSREnvironment`, Chart.js CDN, vanilla JS/HTML/CSS

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `generate_replay.py` | Create | Bedrock-powered log generator |
| `replay_logs/.gitkeep` | Create | Directory placeholder |
| `replay_logs/sample_replay.json` | Create (generated) | Sample log committed for the HF Space demo |
| `index.html` | Create | Self-contained HF Space frontend |

---

## Task 1: Create `generate_replay.py`

**Files:**
- Create: `generate_replay.py`

- [ ] **Step 1: Create the script**

```python
"""Generate a rich per-step replay log for the Chaos Economy HF Space frontend.

Runs one episode (default 50 steps) using AWS Bedrock for inference, capturing
all per-step detail: spot price, agent actions + reasoning, messages, news,
black swan events, SEC interventions, intel transactions, and rewards.

Usage:
    python generate_replay.py --run_name demo --episode_length 50

    # with custom Bedrock model
    python generate_replay.py \
        --bedrock_model meta.llama3-1-70b-instruct-v1:0 \
        --aws_region us-east-1 \
        --episode_length 50 \
        --seed 42 \
        --run_name demo
"""

import argparse
import json
import os
from pathlib import Path

import boto3
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
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return response["output"]["message"]["content"][0]["text"]


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


def run_episode(client, model_id: str, episode_length: int, seed: int, verbose: bool) -> list[dict]:
    env = MultiAgentVSREnvironment(episode_length=episode_length)
    obs = env.reset(seed=seed)

    steps_log = []

    for step in range(episode_length):
        actions = {}
        raw_outputs = {}

        # --- Traders + MM via Bedrock ---
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
            raw = generate_bedrock(client, model_id, prompt, max_tokens=140, temperature=temp)
            raw_outputs[aid] = raw
            parsed, info = parse_json(raw, role=role)
            if info.get("valid"):
                actions[aid] = parsed
            else:
                actions[aid] = scripted_trader(int(aid.split("_")[1]), step) if role == "trader" else scripted_mm(step)

        # trader_3 always scripted
        actions["trader_3"] = scripted_trader(3, step)
        raw_outputs["trader_3"] = "scripted"

        # --- Oversight via Bedrock ---
        heatmap = get_position_heatmap(env.agent_states) if hasattr(env, "agent_states") else {}
        agent_thoughts = {aid: actions[aid].get("reasoning", "") for aid in actions if aid != "oversight"}
        ov_prompt = format_oversight_prompt(obs["oversight"], heatmap, coord, agent_thoughts)
        ov_raw = generate_bedrock(client, model_id, ov_prompt, max_tokens=160, temperature=0.5)
        raw_outputs["oversight"] = ov_raw
        ov_parsed, ov_info = parse_json(ov_raw, role="oversight")
        actions["oversight"] = ov_parsed if ov_info.get("valid") else scripted_oversight()

        # --- Step environment ---
        obs, rewards, done, info = env.step(actions)

        # --- Capture per-step log ---
        spot = float(env.vsr_state.spot_price)
        headline = _active_headline(env, env.current_step)
        black_swan = _active_event_info(env, env.current_step)

        # Build clean action records
        action_records = {}
        for aid in ["trader_0", "trader_1", "trader_2", "trader_3", "market_maker", "oversight"]:
            act = actions.get(aid, {})
            record = {k: v for k, v in act.items()}
            # Normalize message field name
            msg = act.get("send_message") or act.get("messages_sent")
            record["message"] = msg if isinstance(msg, dict) else None
            action_records[aid] = record

        # Collect intel transactions from marketplace
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
            "collusion_events": sum(1 for c in env.collusion_history if c["step"] == env.current_step and c["was_collusion"]),
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
            break

    return steps_log


def main():
    p = argparse.ArgumentParser(description="Generate Chaos Economy replay log via Bedrock")
    p.add_argument("--bedrock_model", default="meta.llama3-1-70b-instruct-v1:0")
    p.add_argument("--aws_region", default="us-east-1")
    p.add_argument("--episode_length", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default="replay")
    p.add_argument("--output_dir", default="replay_logs")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"[Bedrock] region={args.aws_region}  model={args.bedrock_model}")
    client = boto3.client("bedrock-runtime", region_name=args.aws_region)

    print(f"[Episode] length={args.episode_length}  seed={args.seed}")
    steps = run_episode(client, args.bedrock_model, args.episode_length, args.seed, args.verbose)

    out = {
        "meta": {
            "model": args.bedrock_model,
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
```

- [ ] **Step 2: Create replay_logs directory**

```bash
mkdir -p /Users/mananbansal/Desktop/meta/replay_logs
echo "" > /Users/mananbansal/Desktop/meta/replay_logs/.gitkeep
```

- [ ] **Step 3: Verify imports work (no GPU needed)**

```bash
cd /Users/mananbansal/Desktop/meta
python3 -c "
import generate_replay
print('imports OK')
print('TRADER_CONFIGS keys:', list(__import__('train_multi_agent_pipeline').TRADER_CONFIGS.keys()))
"
```

Expected: `imports OK` and archetype names printed.

- [ ] **Step 4: Commit**

```bash
git add generate_replay.py replay_logs/.gitkeep
git commit -m "feat: add generate_replay.py for Bedrock-powered replay log generation"
```

---

## Task 2: Generate the sample replay log

> **Requires AWS credentials with Bedrock access configured.**

- [ ] **Step 1: Run the generator**

```bash
cd /Users/mananbansal/Desktop/meta
python generate_replay.py \
    --bedrock_model meta.llama3-1-70b-instruct-v1:0 \
    --aws_region us-east-1 \
    --episode_length 50 \
    --seed 42 \
    --run_name demo \
    --verbose
```

Expected output ends with: `[Done] saved 50 steps → replay_logs/replay_demo.json`

- [ ] **Step 2: Verify the output**

```bash
python3 -c "
import json
d = json.load(open('replay_logs/replay_demo.json'))
print('steps:', len(d['steps']))
s = d['steps'][0]
print('step keys:', list(s.keys()))
print('spot_price:', s['spot_price'])
print('messages:', len(s['messages']))
print('trader_0 direction:', s['actions']['trader_0'].get('direction'))
print('trader_0 reasoning:', s['actions']['trader_0'].get('reasoning', '')[:60])
"
```

Expected: 50 steps, all keys present, reasoning text visible.

- [ ] **Step 3: Copy to `sample_replay.json` for the frontend**

```bash
cp replay_logs/replay_demo.json replay_logs/sample_replay.json
```

- [ ] **Step 4: Commit**

```bash
git add replay_logs/sample_replay.json
git commit -m "feat: add sample 50-step Bedrock replay log for HF Space demo"
```

---

## Task 3: Build `index.html`

**Files:**
- Create: `index.html`

The page is self-contained. It expects `window.REPLAY_DATA` to be set by an inline `<script>` block embedding the JSON. The HTML itself contains all CSS and JS.

### Layout structure

```
┌─────────────────── HEADER ──────────────────────────────┐
│ 🦈 THE CHAOS ECONOMY  Spot $XXX ▲  [step scrubber] [▶] │
└─────────────────────────────────────────────────────────┘
┌────────── PRICE CHART (left 55%) ──┬── AGENTS (right 45%) ──┐
│ Chart.js line, act-band overlays   │ 4 trader cards          │
│ current step vertical line         │ MM card                 │
│                                    │ Oversight card          │
└────────────────────────────────────┴─────────────────────────┘
┌──── NEWS ────┬──── BLACK SWAN ────┬──── MESSAGES & INTEL ────┐
│ headline     │ event info         │ all messages this step   │
└──────────────┴────────────────────┴──────────────────────────┘
```

- [ ] **Step 1: Create `index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🦈 The Chaos Economy — Replay</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0d0d0d;
  --panel: #161616;
  --border: #272727;
  --text: #f0f0f0;
  --muted: #888;
  --accent: #ffbf00;
  --buy: #3ddc84;
  --sell: #ff5555;
  --hold: #6688aa;
  --act1: rgba(220,50,50,0.08);
  --act2: rgba(220,160,50,0.08);
  --act3: rgba(180,100,220,0.08);
  --act4: rgba(50,150,220,0.08);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: Inter, system-ui, sans-serif; font-size: 13px; min-height: 100vh; }

/* ── Header ── */
.header {
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  padding: 12px 20px; border-bottom: 1px solid var(--border);
  background: #111;
}
.logo { font-size: 18px; font-weight: 700; color: var(--accent); white-space: nowrap; }
.spot-badge { font-size: 15px; font-weight: 600; }
.spot-badge .delta { font-size: 12px; margin-left: 4px; }
.badge { padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.badge-news { background: #2a2200; color: #ffcc44; border: 1px solid #554400; }
.badge-swan { background: #2a0a0a; color: #ff6666; border: 1px solid #551111; }
.badge-collusion { background: #1a0a2a; color: #cc88ff; border: 1px solid #442266; }
.scrubber-wrap { display: flex; align-items: center; gap: 8px; margin-left: auto; }
.scrubber-wrap label { color: var(--muted); font-size: 11px; }
#stepSlider { width: 200px; accent-color: var(--accent); }
#stepLabel { color: var(--accent); font-weight: 700; min-width: 48px; }
#playBtn {
  padding: 4px 14px; border-radius: 6px; border: 1px solid var(--accent);
  background: transparent; color: var(--accent); cursor: pointer; font-size: 12px;
}
#playBtn:hover { background: var(--accent); color: #000; }

/* ── Main grid ── */
.main { display: grid; grid-template-columns: 55% 1fr; gap: 12px; padding: 12px 20px; }

/* ── Chart ── */
.chart-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px;
}
.card-title { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 10px; }
#priceChart { max-height: 280px; }

/* ── Right column ── */
.agents-col { display: flex; flex-direction: column; gap: 8px; }

/* ── Agent card ── */
.agent-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 12px;
}
.agent-card.active-buy { border-color: var(--buy); }
.agent-card.active-sell { border-color: var(--sell); }
.agent-card.active-mm { border-color: #4488ff55; }
.agent-card.active-oversight { border-color: #ff885555; }

.agent-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.agent-name { font-weight: 700; font-size: 12px; }
.archetype-tag { font-size: 10px; color: var(--muted); }
.dir-badge {
  margin-left: auto; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .04em;
}
.dir-buy { background: #0a2a16; color: var(--buy); border: 1px solid #1a5530; }
.dir-sell { background: #2a0a0a; color: var(--sell); border: 1px solid #551515; }
.dir-hold { background: #111a22; color: var(--hold); border: 1px solid #224; }
.agent-stats { display: flex; gap: 10px; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
.agent-stats span { color: var(--text); }
.reasoning { font-size: 11px; color: #aaa; line-height: 1.4; font-style: italic; }
.reward-pill {
  display: inline-block; padding: 1px 7px; border-radius: 999px;
  font-size: 10px; font-weight: 600; margin-left: auto;
}
.reward-pos { background: #0a2a16; color: var(--buy); }
.reward-neg { background: #2a0a0a; color: var(--sell); }
.agent-footer { display: flex; align-items: center; margin-top: 4px; }

/* ── Oversight ── */
.sec-row { display: flex; align-items: center; gap: 8px; font-size: 11px; }
.sec-flag { color: #ff8844; font-weight: 700; }
.sec-fine { color: #ff5555; }
.sec-none { color: var(--muted); font-style: italic; }

/* ── Bottom tray ── */
.tray {
  display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;
  padding: 0 20px 16px;
}
.tray-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 12px; min-height: 90px;
}
.tray-title { font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }
.tray-empty { color: var(--muted); font-style: italic; font-size: 11px; }

/* ── Messages ── */
.msg-item { padding: 4px 0; border-bottom: 1px solid var(--border); font-size: 11px; line-height: 1.4; }
.msg-item:last-child { border-bottom: none; }
.msg-from { font-weight: 700; color: var(--accent); }
.msg-to { color: var(--muted); }
.msg-type { font-size: 10px; padding: 1px 5px; border-radius: 4px; margin-left: 4px; }
.msg-dm { background: #1a2a3a; color: #66aaff; }
.msg-broadcast { background: #2a1a0a; color: #ffaa44; }
.msg-group { background: #1a2a1a; color: #66cc66; }
.msg-intel { background: #2a1a2a; color: #cc66cc; }
.msg-text { color: #ccc; margin-top: 2px; }

/* ── Act band legend ── */
.act-legend { display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
.act-dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; margin-right: 4px; vertical-align: middle; }
.act-legend-item { font-size: 10px; color: var(--muted); display: flex; align-items: center; }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="logo">🦈 THE CHAOS ECONOMY</div>
  <div class="spot-badge" id="spotBadge">Spot $—</div>
  <div id="headerBadges" style="display:flex;gap:6px;"></div>
  <div class="scrubber-wrap">
    <label>STEP</label>
    <input type="range" id="stepSlider" min="0" max="49" value="0" step="1">
    <span id="stepLabel">1 / 50</span>
    <button id="playBtn">▶ Play</button>
  </div>
</div>

<!-- Main -->
<div class="main">
  <!-- Price chart -->
  <div class="chart-card">
    <div class="card-title">📈 Spot Price — 50 Steps</div>
    <canvas id="priceChart"></canvas>
    <div class="act-legend" id="actLegend">
      <div class="act-legend-item"><span class="act-dot" style="background:#dc3232;"></span>Act I: Slaughter</div>
      <div class="act-legend-item"><span class="act-dot" style="background:#dca032;"></span>Act II: Adaptation</div>
      <div class="act-legend-item"><span class="act-dot" style="background:#b464dc;"></span>Act III: Collusion</div>
      <div class="act-legend-item"><span class="act-dot" style="background:#3296dc;"></span>Act IV: Oversight</div>
    </div>
  </div>

  <!-- Agent panels -->
  <div class="agents-col" id="agentsCol">
    <!-- Injected by JS -->
  </div>
</div>

<!-- Bottom tray -->
<div class="tray">
  <div class="tray-card">
    <div class="tray-title">📰 News</div>
    <div id="trayNews" class="tray-empty">No news this step</div>
  </div>
  <div class="tray-card">
    <div class="tray-title">⚡ Black Swan</div>
    <div id="trayBlackSwan" class="tray-empty">No event active</div>
  </div>
  <div class="tray-card">
    <div class="tray-title">📨 Messages & Intel</div>
    <div id="trayMessages" class="tray-empty">No messages this step</div>
  </div>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────
// REPLAY_DATA is injected by the build script (or replaced inline).
// Falls back to a stub if not present.
const DATA = (typeof REPLAY_DATA !== 'undefined') ? REPLAY_DATA : null;
if (!DATA) {
  document.body.innerHTML = '<div style="color:#ff5555;padding:40px;font-size:16px;">No replay data found.<br>Embed REPLAY_DATA in this file or run generate_replay.py first.</div>';
  throw new Error('No REPLAY_DATA');
}

const STEPS = DATA.steps;
const N = STEPS.length;
const ARCHETYPES = DATA.archetypes || {};
const EPISODE_LEN = DATA.meta?.episode_length || N;

// Act boundaries (as step indices, 0-based)
function actFor(stepIdx) {
  const pct = stepIdx / EPISODE_LEN;
  if (pct < 0.24) return 0;
  if (pct < 0.52) return 1;
  if (pct < 0.80) return 2;
  return 3;
}
const ACT_COLORS = ['rgba(220,50,50,0.15)', 'rgba(220,160,50,0.15)', 'rgba(180,100,220,0.15)', 'rgba(50,150,220,0.15)'];
const ACT_NAMES = ['Act I: Slaughter', 'Act II: Adaptation', 'Act III: Collusion', 'Act IV: Oversight'];

// ── Chart ─────────────────────────────────────────────────────────────────
const prices = STEPS.map(s => s.spot_price);
const labels = STEPS.map(s => `Step ${s.step}`);

const linePlugin = {
  id: 'currentLine',
  afterDraw(chart) {
    const ci = chart.currentStepIdx ?? 0;
    const meta = chart.getDatasetMeta(0);
    if (!meta.data[ci]) return;
    const x = meta.data[ci].x;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = '#ffbf00';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(x, chart.chartArea.top);
    ctx.lineTo(x, chart.chartArea.bottom);
    ctx.stroke();
    ctx.restore();
  }
};
Chart.register(linePlugin);

// Background act bands via plugin
const actBandPlugin = {
  id: 'actBands',
  beforeDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea) return;
    ctx.save();
    let lastAct = actFor(0);
    let startX = scales.x.getPixelForValue(0);
    for (let i = 1; i <= N; i++) {
      const act = i < N ? actFor(i) : lastAct + 1; // force close last band
      if (act !== lastAct || i === N) {
        const endX = i < N ? scales.x.getPixelForValue(i) : chartArea.right;
        ctx.fillStyle = ACT_COLORS[lastAct];
        ctx.fillRect(startX, chartArea.top, endX - startX, chartArea.bottom - chartArea.top);
        lastAct = act;
        startX = endX;
      }
    }
    ctx.restore();
  }
};
Chart.register(actBandPlugin);

const chart = new Chart(document.getElementById('priceChart'), {
  type: 'line',
  data: {
    labels,
    datasets: [{
      label: 'Spot Price',
      data: prices,
      borderColor: '#ffbf00',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    }]
  },
  options: {
    responsive: true,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: {
        grid: { color: '#222' },
        ticks: { color: '#888', font: { size: 10 } }
      }
    }
  }
});
chart.currentStepIdx = 0;

// ── Render helpers ────────────────────────────────────────────────────────
function dirBadge(dir) {
  if (!dir || dir === 'hold') return '<span class="dir-badge dir-hold">HOLD</span>';
  if (dir === 'buy') return '<span class="dir-badge dir-buy">BUY</span>';
  return '<span class="dir-badge dir-sell">SELL</span>';
}

function rewardPill(r) {
  const val = typeof r === 'number' ? r : 0;
  const cls = val >= 0 ? 'reward-pos' : 'reward-neg';
  const sign = val >= 0 ? '+' : '';
  return `<span class="reward-pill ${cls}">${sign}${val.toFixed(2)}</span>`;
}

function cardClass(aid, dir) {
  if (aid === 'market_maker') return 'agent-card active-mm';
  if (aid === 'oversight') return 'agent-card active-oversight';
  if (dir === 'buy') return 'agent-card active-buy';
  if (dir === 'sell') return 'agent-card active-sell';
  return 'agent-card';
}

function renderTraderCard(aid, act, reward, archetype) {
  const dir = act?.direction || 'hold';
  const bucket = act?.size_bucket || '—';
  const qty = act?.quantity ?? '—';
  const reasoning = act?.reasoning || '';
  return `
  <div class="${cardClass(aid, dir)}">
    <div class="agent-header">
      <span class="agent-name">${aid.replace('_', ' ').toUpperCase()}</span>
      <span class="archetype-tag">${archetype || ''}</span>
      ${dirBadge(dir)}
    </div>
    ${dir !== 'hold' ? `<div class="agent-stats">Size <span>${bucket}</span>&nbsp;&nbsp;Qty <span>${qty}</span></div>` : ''}
    ${reasoning ? `<div class="reasoning">"${reasoning.slice(0, 120)}${reasoning.length > 120 ? '…' : ''}"</div>` : ''}
    <div class="agent-footer">${rewardPill(reward)}</div>
  </div>`;
}

function renderMMCard(act, reward) {
  const hs = act?.half_spread ?? '—';
  const skew = act?.skew ?? '—';
  const reasoning = act?.reasoning || '';
  return `
  <div class="agent-card active-mm">
    <div class="agent-header">
      <span class="agent-name">MARKET MAKER</span>
      <span class="archetype-tag">Liquidity</span>
    </div>
    <div class="agent-stats">Half-spread <span>${typeof hs === 'number' ? hs.toFixed(4) : hs}</span>&nbsp;&nbsp;Skew <span>${typeof skew === 'number' ? skew.toFixed(4) : skew}</span></div>
    ${reasoning ? `<div class="reasoning">"${reasoning.slice(0, 120)}${reasoning.length > 120 ? '…' : ''}"</div>` : ''}
    <div class="agent-footer">${rewardPill(reward)}</div>
  </div>`;
}

function renderOversightCard(act, reward, interventions) {
  const itype = act?.intervention_type || 'none';
  const flagged = act?.flagged_agents || [];
  const reasoning = act?.reasoning || '';
  const fine = interventions.reduce((s, iv) => s + (iv.fine_amount || 0), 0);
  let secRow = '';
  if (itype !== 'none' && flagged.length > 0) {
    secRow = `<div class="sec-row"><span class="sec-flag">⚖️ ${itype.toUpperCase()}</span> → <span>${flagged.join(', ')}</span>${fine > 0 ? `<span class="sec-fine"> $${fine.toFixed(0)} fine</span>` : ''}</div>`;
  } else {
    secRow = `<div class="sec-none">No intervention this step</div>`;
  }
  return `
  <div class="agent-card active-oversight">
    <div class="agent-header">
      <span class="agent-name">OVERSIGHT (SEC)</span>
      <span class="archetype-tag">Regulatory</span>
    </div>
    ${secRow}
    ${reasoning ? `<div class="reasoning">"${reasoning.slice(0, 120)}${reasoning.length > 120 ? '…' : ''}"</div>` : ''}
    <div class="agent-footer">${rewardPill(reward)}</div>
  </div>`;
}

function renderMessages(messages, intel) {
  const items = [];
  for (const m of (messages || [])) {
    const typeClass = { dm: 'msg-dm', broadcast: 'msg-broadcast', group: 'msg-group' }[m.type] || 'msg-broadcast';
    items.push(`
      <div class="msg-item">
        <span class="msg-from">${m.sender || m.from || '?'}</span>
        <span class="msg-to">→ ${m.to || 'all'}</span>
        <span class="msg-type ${typeClass}">${m.type || 'msg'}</span>
        <div class="msg-text">${(m.text || '').slice(0, 100)}</div>
      </div>`);
  }
  for (const t of (intel || [])) {
    items.push(`
      <div class="msg-item">
        <span class="msg-from">${t.buyer || '?'}</span>
        <span class="msg-to">← intel from ${t.seller || '?'}</span>
        <span class="msg-type msg-intel">intel $${(t.price || 0).toFixed(0)}</span>
        <div class="msg-text">${t.signal || ''}</div>
      </div>`);
  }
  return items.length ? items.join('') : '<div class="tray-empty">No messages this step</div>';
}

// ── Main render ───────────────────────────────────────────────────────────
function render(idx) {
  const s = STEPS[idx];
  if (!s) return;

  // Header
  const spot = s.spot_price;
  const prev = idx > 0 ? STEPS[idx - 1].spot_price : spot;
  const delta = ((spot - prev) / prev * 100).toFixed(2);
  const sign = delta >= 0 ? '▲' : '▼';
  const col = delta >= 0 ? 'var(--buy)' : 'var(--sell)';
  document.getElementById('spotBadge').innerHTML =
    `<span>Spot <strong>$${spot.toFixed(2)}</strong></span> <span class="delta" style="color:${col}">${sign}${Math.abs(delta)}%</span>`;

  // Badges
  const badges = [];
  if (s.news_headline) badges.push(`<span class="badge badge-news">📰 News</span>`);
  if (s.black_swan) badges.push(`<span class="badge badge-swan">⚡ Black Swan</span>`);
  if (s.collusion_events > 0) badges.push(`<span class="badge badge-collusion">🔗 Collusion</span>`);
  document.getElementById('headerBadges').innerHTML = badges.join('');

  // Step label
  document.getElementById('stepLabel').textContent = `${idx + 1} / ${N}`;

  // Chart cursor
  chart.currentStepIdx = idx;
  chart.update('none');

  // Agents
  const col2 = document.getElementById('agentsCol');
  const traders = ['trader_0', 'trader_1', 'trader_2', 'trader_3'];
  let html = traders.map(aid =>
    renderTraderCard(aid, s.actions?.[aid], s.rewards?.[aid], ARCHETYPES[aid])
  ).join('');
  html += renderMMCard(s.actions?.market_maker, s.rewards?.market_maker);
  html += renderOversightCard(s.actions?.oversight, s.rewards?.oversight, s.sec_interventions || []);
  col2.innerHTML = html;

  // Tray — news
  document.getElementById('trayNews').innerHTML = s.news_headline
    ? `<div style="color:#ffcc44;font-size:12px;">${s.news_headline}</div>`
    : '<div class="tray-empty">No news this step</div>';

  // Tray — black swan
  if (s.black_swan) {
    const bs = s.black_swan;
    document.getElementById('trayBlackSwan').innerHTML = `
      <div style="color:#ff6666;font-size:12px;font-weight:700;">⚡ ${bs.headline || 'Event Active'}</div>
      <div style="color:#aaa;font-size:11px;margin-top:4px;">
        Spot impact: ${(bs.spot_impact * 100).toFixed(1)}%&nbsp;&nbsp;
        Vol impact: ${(bs.variance_impact * 100).toFixed(1)}%
      </div>`;
  } else {
    document.getElementById('trayBlackSwan').innerHTML = '<div class="tray-empty">No event active</div>';
  }

  // Tray — messages & intel
  document.getElementById('trayMessages').innerHTML = renderMessages(s.messages, s.intel_transactions);
}

// ── Controls ──────────────────────────────────────────────────────────────
const slider = document.getElementById('stepSlider');
slider.max = N - 1;
slider.addEventListener('input', () => render(parseInt(slider.value)));

let playTimer = null;
const playBtn = document.getElementById('playBtn');
playBtn.addEventListener('click', () => {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    playBtn.textContent = '▶ Play';
  } else {
    playBtn.textContent = '⏸ Pause';
    playTimer = setInterval(() => {
      let cur = parseInt(slider.value);
      if (cur >= N - 1) {
        clearInterval(playTimer);
        playTimer = null;
        playBtn.textContent = '▶ Play';
        return;
      }
      slider.value = cur + 1;
      render(cur + 1);
    }, 900);
  }
});

// ── Init ──────────────────────────────────────────────────────────────────
render(0);
</script>
</body>
</html>
```

- [ ] **Step 2: Commit the base HTML**

```bash
git add index.html
git commit -m "feat: add HF Space replay frontend (index.html)"
```

---

## Task 4: Embed replay data into index.html and test

- [ ] **Step 1: Create embed script**

Create `embed_replay.py`:

```python
"""Embed a replay JSON into index.html as an inline REPLAY_DATA constant.

Usage:
    python embed_replay.py replay_logs/replay_demo.json

Outputs: index.html with REPLAY_DATA injected just before </body>.
The original index.html is read; the output overwrites it.
"""

import json
import sys
from pathlib import Path

replay_path = Path(sys.argv[1] if len(sys.argv) > 1 else "replay_logs/sample_replay.json")
html_path = Path("index.html")

data = json.loads(replay_path.read_text())
html = html_path.read_text()

# Remove any previous injection
import re
html = re.sub(r'<script>\s*const REPLAY_DATA\s*=[\s\S]*?</script>\s*(?=<script>|</body>)', '', html)

injection = f'<script>\nconst REPLAY_DATA = {json.dumps(data, separators=(",", ":"))}\n</script>\n'
html = html.replace('</body>', injection + '</body>')
html_path.write_text(html)
print(f"Embedded {len(data['steps'])} steps into index.html")
```

- [ ] **Step 2: Embed the sample replay**

```bash
python embed_replay.py replay_logs/sample_replay.json
```

Expected: `Embedded 50 steps into index.html`

- [ ] **Step 3: Open in browser and verify**

```bash
open index.html  # macOS
# or: python3 -m http.server 8080 then open http://localhost:8080
```

Verify:
- Spot price shows in header
- Scrubber moves through 50 steps
- Play button animates through steps
- Each step shows trader cards with direction badges
- MM card shows spread and reasoning
- Oversight card shows intervention or "No intervention"
- Bottom tray shows messages, news, black swan where present

- [ ] **Step 4: Commit the embedded version**

```bash
git add index.html embed_replay.py
git commit -m "feat: embed sample replay into index.html for HF Space demo"
```

---

## Task 5: HF Space wiring

The HF Space is already configured as `sdk: docker`. Verify index.html is served correctly.

- [ ] **Step 1: Check Dockerfile**

```bash
cat Dockerfile
```

If the Dockerfile doesn't already serve `index.html`, add a simple Python HTTP server fallback:

```dockerfile
# Add at the end if no web server is already configured:
RUN pip install --no-cache-dir fastapi uvicorn
COPY index.html /app/index.html
```

Or if the Dockerfile already serves a port, just ensure `index.html` is in the right directory.

- [ ] **Step 2: Verify HF Space config in README.md header**

The file header should already have:
```yaml
---
title: The Chaos Economy
sdk: docker
---
```

If missing, add it.

- [ ] **Step 3: Push to trigger HF Space build**

```bash
git push origin main
```

Wait for HF Space build to complete, then open the Space URL and verify the frontend loads with the embedded replay data.

---

## Self-Review Checklist

**Spec coverage:**
- [x] `generate_replay.py` with Bedrock as default — Task 1
- [x] Per-step: spot price, all trader actions + reasoning — captured in `step_record`
- [x] MM spread + reasoning — `actions.market_maker.half_spread/skew/reasoning`
- [x] Oversight action + reasoning — `actions.oversight.intervention_type/flagged_agents/reasoning`
- [x] News headline — `_active_headline()`
- [x] Black swan event — `_active_event_info()`
- [x] Messages (DM, group, broadcast) — `info["messages_this_step"]`
- [x] Intel transactions — `env.marketplace.transaction_log`
- [x] Frontend: spot price current — header badge
- [x] Frontend: trader trade + reasoning cards — `renderTraderCard()`
- [x] Frontend: MM spread card — `renderMMCard()`
- [x] Frontend: Oversight card — `renderOversightCard()`
- [x] Frontend: news tray — `trayNews`
- [x] Frontend: all messages + intel in tray — `renderMessages()`
- [x] HF Space hosting — Task 5

**No placeholders found.**

**Type consistency:** All field names consistent between `generate_replay.py` schema and `index.html` JS accessors (`s.actions.trader_0.direction`, `s.messages`, `s.intel_transactions`, `s.sec_interventions`, `s.news_headline`, `s.black_swan`).
