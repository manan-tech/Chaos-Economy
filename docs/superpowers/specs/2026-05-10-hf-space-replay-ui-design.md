# Design: HuggingFace Space Replay UI

## Overview

Two deliverables:
1. `generate_replay.py` — Bedrock-powered script that runs one simulation episode and saves a rich per-step JSON log
2. `index.html` — Self-contained HF Space frontend that loads the replay JSON and lets users scrub through 50 steps

## Part 1: generate_replay.py

Mirrors `eval.py` structure. Bedrock is the primary inference path (no local GPU needed).

**Per-step log schema:**
```json
{
  "meta": { "model": "meta.llama3-1-70b-instruct-v1:0", "run_name": "...", "steps": 50, "seed": 42 },
  "steps": [
    {
      "step": 1,
      "spot_price": 101.23,
      "mm_bid": 100.9,
      "mm_ask": 101.5,
      "news_headline": "Earnings beat expectations",
      "black_swan": null,
      "collusion_events": 0,
      "sec_fines": 0.0,
      "actions": {
        "trader_0": { "direction": "buy", "size_bucket": "medium", "quantity": 25,
                      "reasoning": "...", "messages_sent": [
                        { "to": "all", "type": "broadcast", "text": "...", "direction": "bullish" }
                      ] },
        "trader_1": { "direction": "hold", "size_bucket": null, "quantity": 0, "reasoning": "...", "messages_sent": [] },
        "trader_2": { "direction": "sell", "size_bucket": "large", "quantity": 50, "reasoning": "...", "messages_sent": [] },
        "trader_3": { "direction": "buy", "size_bucket": "small", "quantity": 5, "reasoning": "scripted", "messages_sent": [] },
        "market_maker": { "half_spread": 0.05, "skew": 0.0, "reasoning": "..." },
        "oversight": { "intervention_type": "fine", "flagged_agents": ["trader_2"], "reasoning": "...", "fine_amount": 50.0 }
      },
      "intel_trades": [
        { "buyer": "trader_1", "seller": "trader_0", "price": 10.0, "signal": "bullish" }
      ],
      "rewards": { "trader_0": 1.2, "trader_1": -0.3, "trader_2": -1.5, "trader_3": 0.1, "market_maker": 0.8, "oversight": 0.2 }
    }
  ]
}
```

**CLI flags:**
- `--bedrock_model` (default: `meta.llama3-1-70b-instruct-v1:0`)
- `--aws_region` (default: `us-east-1`)
- `--episode_length` (default: 50)
- `--seed` (default: 42)
- `--run_name` (default: `replay`)
- `--output_dir` (default: `replay_logs/`)

Output file: `replay_logs/replay_<run_name>.json`

## Part 2: index.html (HF Space Frontend)

Single self-contained HTML file. Embeds the replay JSON inline. No external dependencies except CDN Chart.js.

### Layout (dark theme, matching existing frontend.html)

```
┌──────────────────────────────────────────────────────────────────┐
│  🦈 THE CHAOS ECONOMY          [Step 1 ──●────────── 50]  [▶]   │
│  Spot: $101.23  ▲+1.2%   📰 News Active   ⚡ Black Swan          │
├────────────────────────┬─────────────────────────────────────────┤
│  PRICE CHART (50 steps)│  CURRENT STEP DETAIL                    │
│  Line chart, current   │                                          │
│  step highlighted with │  TRADERS                                 │
│  vertical cursor.      │  ┌ trader_0 [MOMENTUM]  BUY medium 25  ┐│
│  Act bands colored     │  │ "momentum signal confirmed..."       ││
│  (red/yellow/orange/   │  └──────────────────────────────────────┘│
│   blue for 4 acts)     │  ┌ trader_1 [MEAN-REV]  HOLD           ┐│
│                        │  │ "waiting for reversal..."            ││
│                        │  └──────────────────────────────────────┘│
│                        │  ┌ trader_2 [VOL]       SELL large 50  ┐│
│                        │  └──────────────────────────────────────┘│
│                        │  ┌ trader_3 [SCRIPTED]  BUY small 5    ┐│
│                        │  └──────────────────────────────────────┘│
├────────────────────────┤                                          │
│                        │  MARKET MAKER                            │
│                        │  Spread: 0.05  Skew: 0.0                │
│                        │  "tight spread, balanced inventory..."   │
├────────────────────────┴─────────────────────────────────────────┤
│  SEC OVERSIGHT    intervention: fine  →  trader_2  ($50)         │
│  "large coordinated sell detected..."                             │
├──────────────────────────────────────────────────────────────────┤
│  📰 NEWS                ⚡ BLACK SWAN           📨 MESSAGES        │
│  "Earnings beat..."    (none)                 trader_0 → all:    │
│                                               "bullish on momo"  │
│                                               trader_1 → trader_0│
│                                               [intel bought] $10  │
└──────────────────────────────────────────────────────────────────┘
```

### Components

1. **Header bar**: spot price, % change from step 1, news/black swan badge
2. **Timeline scrubber**: range input 1→50, auto-play button (1 step/sec)
3. **Price chart**: Chart.js line chart, all 50 steps, vertical line at current step, act-band color overlays
4. **Traders panel**: 4 cards, each showing direction badge (green BUY / red SELL / grey HOLD), size_bucket, quantity, archetype label, reasoning text
5. **Market Maker panel**: spread + skew + reasoning
6. **Oversight panel**: only visible when intervention_type != "none"; shows flagged agents, fine amounts, reasoning
7. **Bottom tray** (3 columns):
   - News: headline text for this step
   - Black Swan: event type + magnitude if present
   - Messages: all agent messages this step — DMs, group, broadcast, intel buys/sells

### Data loading

The replay JSON is embedded as a `<script>` block (`const REPLAY_DATA = {...}`). No fetch needed. The HTML file is self-contained and works in HF Space's static file serving or as the Docker app's served file.

## Hosting on HF Space

The existing `Dockerfile` serves the app. `index.html` goes in the repo root. The Docker container serves it (or we add a trivial Python HTTP server if needed).

## Files to Create

- `generate_replay.py` — log generator
- `index.html` — frontend (replaces or supplements `frontend.html`)
- `replay_logs/` — directory for output JSONs (gitignored except sample)
