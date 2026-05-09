NUM_TRADERS = 4
EPISODE_LENGTH = 250
INITIAL_CASH = 10_000.0
MM_INITIAL_CASH = 100_000.0

# Stock specifics
SPOT_INITIAL = 100.0
MAX_POSITION = 100  # max |shares| per trader

# Size buckets: (min_shares, max_shares) inclusive
SIZE_BUCKETS = {
    "small":  (1, 10),
    "medium": (11, 40),
    "large":  (41, 100),
}

# Bucket representative quantities for scripted agents
BUCKET_QTY = {"small": 5, "medium": 20, "large": 60}

# Price-impact lambda: each net share moves log-price by this much
PRICE_IMPACT_LAMBDA = 1e-4

# Market maker
MM_BASE_HALF_SPREAD = 0.05

# Message retention window (steps)
MESSAGE_HISTORY_WINDOW = 20

# Archetype index mapping
ARCHETYPE_TO_IDX = {
    "momentum": 0,
    "mean_reversion": 1,
    "vol_timing": 2,
    "scripted": 3,
}
