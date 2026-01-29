import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# Enhanced festival data with typical travel patterns
FESTIVALS = {
    "Diwali": {"rush_multiplier": 0.95, "peak_days": [3, 4, 5], "duration": 5},
    "Holi": {"rush_multiplier": 0.75, "peak_days": [1, 2], "duration": 3},
    "Durga Puja": {"rush_multiplier": 0.85, "peak_days": [5, 6, 7], "duration": 10},
    "Chhath Puja": {"rush_multiplier": 0.90, "peak_days": [1, 2], "duration": 4},
    "Eid-ul-Fitr": {"rush_multiplier": 0.80, "peak_days": [1], "duration": 3},
    "Christmas": {"rush_multiplier": 0.70, "peak_days": [24, 25], "duration": 3},
    "Pongal": {"rush_multiplier": 0.72, "peak_days": [1, 2], "duration": 4}
}

# Popular routes with typical rush patterns
POPULAR_ROUTES = [
    {"from": "Delhi", "to": "Patna", "tier_from": 1, "tier_to": 2, "distance": 1000, "base_rush": 0.85},
    {"from": "Mumbai", "to": "Kolkata", "tier_from": 1, "tier_to": 1, "distance": 2000, "base_rush": 0.75},
    {"from": "Bangalore", "to": "Chennai", "tier_from": 1, "tier_to": 1, "distance": 350, "base_rush": 0.65},
    {"from": "Delhi", "to": "Lucknow", "tier_from": 1, "tier_to": 2, "distance": 500, "base_rush": 0.80},
    {"from": "Mumbai", "to": "Ahmedabad", "tier_from": 1, "tier_to": 2, "distance": 500, "base_rush": 0.70},
]

TRAIN_CLASSES = ["Sleeper", "3AC", "2AC", "1AC", "General"]
TRAIN_TYPES = ["Express", "Superfast", "Rajdhani", "Shatabdi", "Duronto", "Mail"]
QUOTAS = ["General", "Tatkal", "Ladies", "Senior Citizen", "Premium Tatkal"]

def generate_waitlist_confirmation_probability(waitlist_pos, days_to_journey, train_type, quota):
    """Calculate realistic confirmation probability based on multiple factors"""
    base_prob = 0.8
    
    # Waitlist position impact (exponential decay)
    if waitlist_pos <= 10:
        wl_factor = 0.9
    elif waitlist_pos <= 50:
        wl_factor = 0.6
    elif waitlist_pos <= 100:
        wl_factor = 0.3
    else:
        wl_factor = 0.1
    
    # Days to journey impact
    if days_to_journey > 30:
        days_factor = 0.9
    elif days_to_journey > 15:
        days_factor = 0.7
    elif days_to_journey > 7:
        days_factor = 0.5
    else:
        days_factor = 0.3
    
    # Train type impact
    train_factors = {
        "Rajdhani": 0.9, "Duronto": 0.85, "Shatabdi": 0.85,
        "Superfast": 0.75, "Express": 0.65, "Mail": 0.60
    }
    train_factor = train_factors.get(train_type, 0.7)
    
    # Quota impact
    quota_factors = {"Tatkal": 0.95, "Premium Tatkal": 0.98, "General": 0.70, 
                     "Ladies": 0.85, "Senior Citizen": 0.80}
    quota_factor = quota_factors.get(quota, 0.75)
    
    prob = base_prob * wl_factor * days_factor * train_factor * quota_factor
    return min(0.98, max(0.05, prob))


def generate_enhanced_dataset(num_samples=10000):
    rows = []
    
    for _ in range(num_samples):
        # Select festival and route
        festival = random.choice(list(FESTIVALS.keys()))
        festival_data = FESTIVALS[festival]
        route = random.choice(POPULAR_ROUTES)
        
        # Time-based features
        days_before_festival = random.randint(0, 60)
        days_to_journey = days_before_festival  # Assuming journey on festival day
        booking_hour = random.randint(0, 23)  # Important for tatkal
        
        # Train details
        train_class = random.choice(TRAIN_CLASSES)
        train_type = random.choice(TRAIN_TYPES)
        quota = random.choice(QUOTAS)
        
        # Current booking status
        is_waitlisted = random.random() < 0.6  # 60% of bookings are waitlisted during festivals
        
        if is_waitlisted:
            current_waitlist_position = random.randint(1, 200)
            ticket_status = "WL"
        else:
            current_waitlist_position = 0
            ticket_status = random.choice(["CNF", "RAC"])
        
        # Calculate features
        peak_day_proximity = min([abs(days_before_festival - pd) 
                                  for pd in festival_data["peak_days"]])
        
        # Base rush calculation
        base_rush = route["base_rush"] * 100
        festival_rush = festival_data["rush_multiplier"] * 100
        
        # Rush components
        time_rush = (60 - days_before_festival) * 1.5
        peak_rush = max(0, 30 - peak_day_proximity * 3)
        distance_factor = (route["distance"] / 2000) * 15
        tier_factor = (3 - route["tier_from"] + 3 - route["tier_to"]) * 4
        
        # Train class factor (sleeper has most rush)
        class_factors = {"General": 25, "Sleeper": 20, "3AC": 15, "2AC": 10, "1AC": 5}
        class_rush = class_factors.get(train_class, 15)
        
        historical_rush_index = (
            base_rush * 0.25 +
            festival_rush * 0.25 +
            time_rush * 0.2 +
            peak_rush * 0.15 +
            distance_factor +
            tier_factor +
            class_rush
        )
        
        historical_rush_index = max(10, min(100, historical_rush_index + random.uniform(-5, 5)))
        
        # Determine rush level
        if historical_rush_index >= 75:
            rush_level = "High"
        elif historical_rush_index >= 45:
            rush_level = "Medium"
        else:
            rush_level = "Low"
        
        # Calculate confirmation probability
        confirmation_probability = generate_waitlist_confirmation_probability(
            current_waitlist_position, days_to_journey, train_type, quota
        )
        
        # Optimal booking window (days before festival when rush is manageable)
        if festival_data["rush_multiplier"] > 0.85:
            optimal_booking_min = 45
            optimal_booking_max = 60
        elif festival_data["rush_multiplier"] > 0.75:
            optimal_booking_min = 30
            optimal_booking_max = 45
        else:
            optimal_booking_min = 20
            optimal_booking_max = 35
        
        # Alternative mode viability
        flight_price_ratio = route["distance"] / 400  # Rough flight/train price ratio
        bus_available = route["distance"] < 1000
        
        rows.append({
            # Festival & Route Info
            "festival": festival,
            "route": f"{route['from']}-{route['to']}",
            "source_city": route["from"],
            "destination_city": route["to"],
            "route_distance_km": route["distance"],
            "source_city_tier": route["tier_from"],
            "destination_city_tier": route["tier_to"],
            
            # Time Features
            "days_before_festival": days_before_festival,
            "days_to_journey": days_to_journey,
            "peak_day_proximity": peak_day_proximity,
            "booking_hour": booking_hour,
            
            # Train Details
            "train_class": train_class,
            "train_type": train_type,
            "quota": quota,
            
            # Current Status
            "ticket_status": ticket_status,
            "current_waitlist_position": current_waitlist_position,
            "is_waitlisted": is_waitlisted,
            
            # Target Variables
            "historical_rush_index": round(historical_rush_index, 2),
            "rush_level": rush_level,
            "confirmation_probability": round(confirmation_probability, 3),
            
            # Recommendations Data
            "optimal_booking_window_min": optimal_booking_min,
            "optimal_booking_window_max": optimal_booking_max,
            "flight_price_ratio": round(flight_price_ratio, 2),
            "bus_available": bus_available,
            
            # Risk Score
            "booking_risk_score": round(historical_rush_index / 100, 2)
        })
    
    return pd.DataFrame(rows)


# Generate and save
if __name__ == "__main__":
    import os
    
    df = generate_enhanced_dataset(10000)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/enhanced_festive_travel_data.csv", index=False)
    
    print(f"✅ Generated {len(df)} samples")
    print(f"\n📊 Dataset Summary:")
    print(df.describe())
    print(f"\n🎯 Rush Level Distribution:")
    print(df['rush_level'].value_counts())
    print(f"\n🚂 Train Class Distribution:")
    print(df['train_class'].value_counts())