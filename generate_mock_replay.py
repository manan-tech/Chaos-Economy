"""Generate mock replay data for frontend development (no Bedrock needed).

Usage:
    python generate_mock_replay.py
    # Outputs: replay_logs/sample_replay.json
"""

import json
import math
import random
from pathlib import Path

ARCHETYPES = {
    "trader_0": "Momentum",
    "trader_1": "Mean Reversion",
    "trader_2": "Vol Timing",
    "trader_3": "Scripted",
}

REASONINGS = {
    "trader_0": [
        "Strong momentum signal — price breaking above resistance, buying to capture the move.",
        "Trend is accelerating, large position justified by recent returns.",
        "Momentum fading, cutting exposure to medium bucket.",
        "Holding — no clear directional signal this step.",
        "Sell signal triggered — downward momentum confirmed by volume.",
    ],
    "trader_1": [
        "Price extended — mean reversion trade, fading this move.",
        "RSI-equivalent overbought, selling small into the rally.",
        "Holding — price within fair value band.",
        "Price below moving average by 2σ — buying the dip.",
        "Reverting position as price normalizes.",
    ],
    "trader_2": [
        "Volatility spike — large position to capture the move.",
        "Low vol environment — reducing size to small.",
        "Vol expanding, increasing to medium.",
        "Vol regime unclear — holding.",
        "High vol + directional bias = large sell.",
    ],
    "trader_3": [
        "Scripted baseline — buy small.",
        "Scripted baseline — hold.",
        "Scripted baseline — sell small.",
    ],
    "market_maker": [
        "Balanced inventory — maintaining tight spread.",
        "Long inventory bias — widening slightly to attract sellers.",
        "Short inventory — skewing ask lower to balance.",
        "High vol — widening spread to protect against adverse selection.",
        "Stable conditions — tightening spread to capture flow.",
    ],
    "oversight": [
        "No unusual activity detected. All traders within normal parameters.",
        "Monitoring coordinated buy pressure — threshold not yet met.",
        "Wash trading pattern detected in trader_1 — issuing fine.",
        "Spoofing-like behavior from trader_2 — flagging for review.",
        "Market stable. No intervention required.",
    ],
}

HEADLINES = [
    "Federal Reserve signals rate pause amid inflation data",
    "Tech sector earnings beat expectations for third consecutive quarter",
    "Geopolitical tensions drive safe-haven demand",
    "Regulatory probe into high-frequency trading firms announced",
    "Strong GDP data surprises analysts — risk-on sentiment surges",
]

INTEL_TXNS = [
    {"buyer": "trader_1", "seller": "trader_0", "step": 0, "price": 15.0, "signal": "bullish breakout expected"},
    {"buyer": "trader_2", "seller": "trader_1", "step": 0, "price": 10.0, "signal": "bearish reversal within 3 steps"},
]

def mock_episode(n=50):
    spot = 100.0
    steps = []
    directions = ["buy", "sell", "hold"]
    buckets = ["small", "medium", "large"]
    bucket_qty = {"small": 8, "medium": 25, "large": 60}

    for i in range(n):
        pct = i / n
        # Act-based volatility
        if pct < 0.24:
            drift, vol = 0.002, 0.008
        elif pct < 0.52:
            drift, vol = 0.001, 0.012
        elif pct < 0.80:
            drift, vol = 0.003, 0.015
        else:
            drift, vol = -0.001, 0.010

        spot *= math.exp(drift + vol * random.gauss(0, 1))

        # Black swan at step 20
        black_swan = None
        if i == 20:
            black_swan = {
                "headline": "Flash crash triggered by algorithmic cascade",
                "spot_impact": -0.08,
                "variance_impact": 0.45,
                "trigger_step": 22,
            }
            spot *= 0.92

        # News at step 10 and 35
        news = None
        if i == 10:
            news = HEADLINES[0]
        elif i == 35:
            news = HEADLINES[4]

        # Collusion in Act III
        collusion = 1 if 26 <= i <= 40 and random.random() < 0.6 else 0

        # Generate trader actions
        def trader_action(aid, step_i):
            d = random.choices(directions, weights=[0.4, 0.35, 0.25])[0]
            if d == "hold":
                return {"direction": "hold", "size_bucket": None, "quantity": 0,
                        "reasoning": random.choice(REASONINGS[aid]), "message": None}
            b = random.choices(buckets, weights=[0.4, 0.4, 0.2])[0]
            q = bucket_qty[b] + random.randint(-3, 3)
            return {"direction": d, "size_bucket": b, "quantity": q,
                    "reasoning": random.choice(REASONINGS[aid]), "message": None}

        # Act III: force collusion
        if 26 <= i <= 40 and collusion:
            t0_action = {"direction": "buy", "size_bucket": "large", "quantity": 65,
                         "reasoning": random.choice(REASONINGS["trader_0"]), "message": None}
            t1_action = {"direction": "buy", "size_bucket": "large", "quantity": 58,
                         "reasoning": random.choice(REASONINGS["trader_1"]), "message": None}
        else:
            t0_action = trader_action("trader_0", i)
            t1_action = trader_action("trader_1", i)

        hs = round(0.04 + random.uniform(0, 0.06), 4)
        if 26 <= i <= 40:
            hs = round(0.08 + random.uniform(0, 0.04), 4)

        # Interventions
        interventions = []
        sec_fines = 0.0
        intervention_type = "none"
        flagged = []
        if i >= 40 and random.random() < 0.5:
            flagged = ["trader_0", "trader_1"]
            fine = 50.0
            sec_fines = fine * len(flagged)
            intervention_type = "fine"
            interventions = [
                {"step": i + 1, "agent_id": a, "flag_type": "collusion", "fine_amount": fine, "intervention_type": "fine"}
                for a in flagged
            ]

        # Messages — sparse
        msgs = []
        if i in [5, 15, 28, 35, 42]:
            msg_options = [
                {"from": "trader_0", "to": "all", "type": "broadcast", "text": "Bullish momentum — watching for breakout.", "direction": "bullish"},
                {"from": "trader_1", "to": "trader_0", "type": "dm", "text": "Disagree — this looks extended.", "direction": "bearish"},
                {"from": "trader_2", "to": "group_alpha", "type": "group", "text": "Vol regime shifting — size up.", "direction": "bullish"},
            ]
            msgs = [random.choice(msg_options)]

        # Intel — very sparse
        intel = []
        if i == 12:
            t = dict(INTEL_TXNS[0])
            t["step"] = i + 1
            intel = [t]

        # Rewards
        base_r = (spot - 100) / 100 * 5
        rewards = {
            "trader_0": round(base_r + random.gauss(0, 0.5) - (5.0 if "trader_0" in flagged else 0), 3),
            "trader_1": round(base_r * 0.8 + random.gauss(0, 0.4) - (5.0 if "trader_1" in flagged else 0), 3),
            "trader_2": round(base_r * 0.6 + random.gauss(0, 0.6), 3),
            "trader_3": round(random.gauss(0, 0.2), 3),
            "market_maker": round(hs * 10 + random.gauss(0, 0.3), 3),
            "oversight": round((1.0 if interventions else 0.0) + random.gauss(0, 0.2), 3),
        }

        step_record = {
            "step": i + 1,
            "spot_price": round(spot, 4),
            "mm_bid": round(spot * (1 - hs), 4),
            "mm_ask": round(spot * (1 + hs), 4),
            "mm_half_spread": hs,
            "mm_skew": round(random.uniform(-0.01, 0.01), 4),
            "news_headline": news,
            "black_swan": black_swan,
            "collusion_events": collusion,
            "sec_fines": sec_fines,
            "actions": {
                "trader_0": t0_action,
                "trader_1": t1_action,
                "trader_2": trader_action("trader_2", i),
                "trader_3": {"direction": "buy", "size_bucket": "small", "quantity": 5,
                             "reasoning": random.choice(REASONINGS["trader_3"]), "message": None},
                "market_maker": {
                    "half_spread": hs,
                    "skew": round(random.uniform(-0.01, 0.01), 4),
                    "reasoning": random.choice(REASONINGS["market_maker"]),
                },
                "oversight": {
                    "intervention_type": intervention_type,
                    "flagged_agents": flagged,
                    "reasoning": random.choice(REASONINGS["oversight"]),
                },
            },
            "messages": msgs,
            "intel_transactions": intel,
            "sec_interventions": interventions,
            "detected_manipulations": {
                "trader_0": "collusion" if "trader_0" in flagged else "none",
                "trader_1": "collusion" if "trader_1" in flagged else "none",
                "trader_2": "none",
                "trader_3": "none",
            },
            "rewards": rewards,
        }
        steps.append(step_record)

    return steps


def main():
    random.seed(42)
    Path("replay_logs").mkdir(parents=True, exist_ok=True)
    steps = mock_episode(50)
    out = {
        "meta": {
            "model": "mock-data-generator",
            "run_name": "sample",
            "steps": len(steps),
            "seed": 42,
            "episode_length": 50,
        },
        "archetypes": ARCHETYPES,
        "steps": steps,
    }
    out_path = Path("replay_logs/sample_replay.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[Done] Generated {len(steps)} mock steps → {out_path}")


if __name__ == "__main__":
    main()
