import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_agent.environment import MultiAgentVSREnvironment


def _hold_actions():
    actions = {"market_maker": {"half_spread": 0.05, "skew": 0.0, "reasoning": "test"},
               "oversight": {"flagged_agents": [], "flag_type": "none", "fine_amount": 0.0,
                              "confidence": 0.0, "intervention_type": "none", "reasoning": "test"}}
    for i in range(4):
        actions[f"trader_{i}"] = {"direction": "hold", "size_bucket": "small", "quantity": 0, "reasoning": "test"}
    return actions


def test_checkpoint_1():
    print("Initializing environment...")
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=42)

    print("\n--- Testing P0: Black Swan Generator ---")
    events = env.black_swan_gen.events
    assert len(events) > 0, "No Black Swan events generated."
    print(f"Generated {len(events)} events.")
    for e in events:
        print(f"  Event '{e.headline}' at step {e.trigger_step} (news at {e.news_step})")
        assert e.trigger_step > e.news_step, "News must precede trigger."

    print("\n--- Testing P1: News in Observations ---")
    first_event = events[0]

    print(f"Fast forwarding to news step {first_event.news_step}...")
    for _ in range(first_event.news_step - 1):
        actions = _hold_actions()
        obs, _, done, _ = env.step(actions)
        if done:
            break

    print(f"At step {env.current_step}, checking for news...")
    obs, _, _, _ = env.step(_hold_actions())

    assert obs["trader_0"].news_headline == first_event.headline, (
        f"Expected '{first_event.headline}', got '{obs['trader_0'].news_headline}'"
    )
    print("SUCCESS: News headline appeared in observation.")

    print("\n--- Testing P2: News Marketplace ---")
    actions = _hold_actions()
    actions["trader_1"] = {"direction": "hold", "size_bucket": "small", "quantity": 0,
                           "reasoning": "test", "sell_intel": {"content": "Tech breakthrough!", "price": 10.0}}
    obs, _, _, _ = env.step(actions)

    listings = obs["trader_0"].market_stats.get("available_intel_listings", [])
    assert len(listings) > 0, "Intel listing not found in market stats."
    listing_id = listings[0]["listing_id"]
    print(f"SUCCESS: Intel listed with ID: {listing_id}")

    # trader_2 buys the intel
    actions = _hold_actions()
    actions["trader_2"] = {"direction": "hold", "size_bucket": "small", "quantity": 0,
                           "reasoning": "test", "buy_intel": listing_id}
    obs, _, _, _ = env.step(actions)

    private_intel = obs["trader_2"].private_intel
    assert len(private_intel) > 0, "Trader 2 did not receive private intel."
    print("SUCCESS: Intel successfully purchased.")

    print("\n--- Testing P3: Agent Messaging ---")
    actions = _hold_actions()
    actions["trader_0"] = {"direction": "hold", "size_bucket": "small", "quantity": 0,
                           "reasoning": "test",
                           "send_message": {"to": "trader_1", "type": "dm", "text": "Let's coordinate on large buys!", "direction": "bullish"}}
    obs, _, _, _ = env.step(actions)

    assert len(obs["trader_1"].inbox) > 0, "Trader 1 did not receive DM."
    assert obs["trader_1"].inbox[0]["sender"] == "trader_0"
    print(f"SUCCESS: Trader 1 received DM: {obs['trader_1'].inbox[0].get('text', '')}")

    assert len(obs["trader_2"].inbox) == 0, "Trader 2 should not have received the DM."
    print("SUCCESS: DM privacy maintained (Trader 2 did not receive it).")

    print("\n--- Testing P4: Oversight Manipulation Detection ---")
    # Fast-forward past the first event trigger_step so we are NOT in any news window
    while any(e.news_step <= env.current_step <= e.trigger_step for e in events):
        obs, _, done, _ = env.step(_hold_actions())
        if done:
            break

    # trader_0 sends a message to trader_1
    actions = _hold_actions()
    actions["trader_0"] = {"direction": "hold", "size_bucket": "small", "quantity": 0,
                           "reasoning": "test",
                           "send_message": {"to": "trader_1", "type": "dm", "text": "Buy large now!", "direction": "bullish"}}
    env.step(actions)

    # trader_1 sends back AND makes a large trade (≥30 shares) → message_collusion
    actions = _hold_actions()
    actions["trader_1"] = {"direction": "buy", "size_bucket": "large", "quantity": 50,
                           "reasoning": "test",
                           "send_message": {"to": "trader_0", "type": "dm", "text": "Confirmed, buying large!", "direction": "bullish"}}
    obs, _, _, info = env.step(actions)

    ground_truth = info.get("detected_manipulations", {})
    print("Ground truth:", ground_truth)
    assert ground_truth.get("trader_1") == "message_collusion", (
        f"Expected message_collusion, got {ground_truth.get('trader_1')}"
    )
    print("SUCCESS: Oversight successfully detected message collusion.")

    print("\nAll Checkpoint 1 tests passed successfully!")


if __name__ == "__main__":
    test_checkpoint_1()
