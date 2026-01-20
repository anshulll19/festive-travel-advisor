import random
import os
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

FESTIVALS = {
    "Diwali": 0.95,
    "Holi": 0.75,
    "Durga Puja": 0.85,
    "Chhath Puja": 0.90,
    "Eid-ul-Fitr": 0.80
}

TRANSPORT_MODES = {
    "Train": 1.0,
    "Bus": 0.7,
    "Flight": 0.4
}

CITY_TIERS = [1, 2, 3]  # 1=Metro, 2=Tier-2, 3=Tier-3


def get_rush_level(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    return "High"


rows = []
NUM_ROWS = 7000

for _ in range(NUM_ROWS):
    festival = random.choice(list(FESTIVALS.keys()))
    days_before = random.randint(0, 30)
    src_tier = random.choice(CITY_TIERS)
    dst_tier = random.choice(CITY_TIERS)
    distance = random.randint(100, 1800)
    mode = random.choice(list(TRANSPORT_MODES.keys()))

    base = FESTIVALS[festival] * 100
    days_effect = (30 - days_before) * random.uniform(1.2, 1.8)
    distance_effect = (distance / 1800) * random.uniform(10, 25)
    tier_effect = (4 - src_tier + 4 - dst_tier) * 6
    mode_effect = TRANSPORT_MODES[mode] * random.uniform(10, 25)

    rush_index = base * 0.3 + days_effect + distance_effect + tier_effect + mode_effect
    rush_index = max(0, min(100, rush_index))

    rows.append({
        "festival": festival,
        "days_before_festival": days_before,
        "source_city_tier": src_tier,
        "destination_city_tier": dst_tier,
        "route_distance_km": distance,
        "transport_mode": mode,
        "historical_rush_index": round(rush_index, 2),
        "rush_level": get_rush_level(rush_index),
        "booking_risk_score": round(min(1, rush_index / 100), 2)
    })

df = pd.DataFrame(rows)
os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/festive_travel_data.csv", index=False)

print("Dataset generated successfully")
