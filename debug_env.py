import json
from multi_agent.environment import MultiAgentVSREnvironment

env = MultiAgentVSREnvironment()
obs = env.reset()

actions = {
    "market_maker": {
        "half_spread": 0.05,
        "skew": 0.0,
        "reasoning": "debug"
    },
    "oversight": {
        "flagged_agents": [],
        "flag_type": "none",
        "fine_amount": 0.0,
        "confidence": 0.0,
        "intervention_type": "none",
        "reasoning": "debug"
    }
}
for i in range(4):
    actions[f"trader_{i}"] = {
        "direction": "buy",
        "size_bucket": "medium",
        "quantity": 15,
        "reasoning": "debug"
    }

obs, rewards, done, info = env.step(actions)
print("Step 1 rewards:", rewards)
print("Spot:", round(env.vsr_state.spot_price, 2))
print("Trader 0 shares:", env.agent_states["trader_0"].shares)

obs, rewards, done, info = env.step(actions)
print("Step 2 rewards:", rewards)
print("Trader 0 shares:", env.agent_states["trader_0"].shares)
