import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

# Load enhanced dataset
df = pd.read_csv("data/processed/enhanced_festive_travel_data.csv")

print(f"📊 Loaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}\n")

# ===============================
# PREPROCESSING
# ===============================

# Separate categorical and numerical columns
categorical_cols = [
    "festival", "train_class", "train_type", "quota", 
    "ticket_status", "source_city", "destination_city"
]

numerical_cols = [
    "days_before_festival", "days_to_journey", "route_distance_km",
    "source_city_tier", "destination_city_tier", "peak_day_proximity",
    "booking_hour", "current_waitlist_position", "historical_rush_index"
]

# Create a copy for encoding
df_encoded = df.copy()

# Label encode categorical columns ONLY
label_encoders = {}
for col in categorical_cols:
    if col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# DON'T scale yet - we'll scale per model
print("✅ Label encoding completed\n")

# ===============================
# MODEL 1: RUSH LEVEL CLASSIFIER
# ===============================

print("=" * 60)
print("TRAINING MODEL 1: Rush Level Classification")
print("=" * 60)

# Encode target
rush_encoder = LabelEncoder()
y_rush = rush_encoder.fit_transform(df['rush_level'])

# Features for rush prediction
rush_features = [
    "festival", "days_before_festival", "route_distance_km",
    "source_city_tier", "destination_city_tier", "peak_day_proximity",
    "train_class", "train_type", "historical_rush_index"
]

X_rush = df_encoded[rush_features]

# Scale features for this model
scaler_rush = StandardScaler()
X_rush = pd.DataFrame(
    scaler_rush.fit_transform(X_rush),
    columns=X_rush.columns
)

# Train-test split
X_train_rush, X_test_rush, y_train_rush, y_test_rush = train_test_split(
    X_rush, y_rush, test_size=0.2, random_state=42, stratify=y_rush
)

# Train Random Forest with hyperparameter tuning
rf_rush = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_rush.fit(X_train_rush, y_train_rush)
# Save feature importance for explainability
rush_feature_importance = {
    "features": list(X_rush.columns),
    "importance": rf_rush.feature_importances_.tolist()
}

import json
with open("ml/models/rush_feature_importance.json", "w") as f:
    json.dump(rush_feature_importance, f, indent=2)


# Evaluate
y_pred_rush = rf_rush.predict(X_test_rush)
rush_accuracy = accuracy_score(y_test_rush, y_pred_rush)

print(f"\n📈 Rush Level Model Accuracy: {rush_accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test_rush, y_pred_rush, 
                          target_names=rush_encoder.classes_))

# Feature importance
feature_importance_rush = pd.DataFrame({
    'feature': rush_features,
    'importance': rf_rush.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Top 5 Important Features for Rush Prediction:")
print(feature_importance_rush.head())

# ===============================
# MODEL 2: CONFIRMATION PROBABILITY REGRESSOR
# ===============================

print("\n" + "=" * 60)
print("TRAINING MODEL 2: Confirmation Probability Prediction")
print("=" * 60)

# Features for confirmation prediction
confirm_features = [
    "current_waitlist_position", "days_to_journey", "train_type",
    "quota", "train_class", "historical_rush_index", "ticket_status"
]

X_confirm = df_encoded[confirm_features]

# Scale features for this model
scaler_confirm = StandardScaler()
X_confirm = pd.DataFrame(
    scaler_confirm.fit_transform(X_confirm),
    columns=X_confirm.columns
)

y_confirm = df['confirmation_probability']

# Train-test split
X_train_conf, X_test_conf, y_train_conf, y_test_conf = train_test_split(
    X_confirm, y_confirm, test_size=0.2, random_state=42
)

# Train Gradient Boosting Regressor
gb_confirm = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    min_samples_split=5,
    random_state=42
)

gb_confirm.fit(X_train_conf, y_train_conf)

# Evaluate
y_pred_conf = gb_confirm.predict(X_test_conf)
mae_conf = mean_absolute_error(y_test_conf, y_pred_conf)
r2_conf = r2_score(y_test_conf, y_pred_conf)

print(f"\n📊 Confirmation Probability Model:")
print(f"  - Mean Absolute Error: {mae_conf:.4f}")
print(f"  - R² Score: {r2_conf:.4f}")

# Feature importance
feature_importance_conf = pd.DataFrame({
    'feature': confirm_features,
    'importance': gb_confirm.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Top 5 Important Features for Confirmation Prediction:")
print(feature_importance_conf.head())

# ===============================
# MODEL 3: OPTIMAL BOOKING TIME PREDICTOR
# ===============================

print("\n" + "=" * 60)
print("TRAINING MODEL 3: Optimal Booking Window Prediction")
print("=" * 60)

# Use midpoint of optimal booking window as target
df['optimal_booking_days'] = (df['optimal_booking_window_min'] + 
                               df['optimal_booking_window_max']) / 2

booking_features = [
    "festival", "route_distance_km", "source_city_tier",
    "destination_city_tier", "train_class", "historical_rush_index"
]

X_booking = df_encoded[booking_features]

# Scale features for this model
scaler_booking = StandardScaler()
X_booking = pd.DataFrame(
    scaler_booking.fit_transform(X_booking),
    columns=X_booking.columns
)

y_booking = df['optimal_booking_days']

# Train-test split
X_train_book, X_test_book, y_train_book, y_test_book = train_test_split(
    X_booking, y_booking, test_size=0.2, random_state=42
)

# Train Gradient Boosting
gb_booking = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

gb_booking.fit(X_train_book, y_train_book)

# Evaluate
y_pred_book = gb_booking.predict(X_test_book)
mae_book = mean_absolute_error(y_test_book, y_pred_book)
r2_book = r2_score(y_test_book, y_pred_book)

print(f"\n📅 Optimal Booking Window Model:")
print(f"  - Mean Absolute Error: {mae_book:.2f} days")
print(f"  - R² Score: {r2_book:.4f}")

# ===============================
# SAVE ALL MODELS & ARTIFACTS
# ===============================

os.makedirs("ml/models", exist_ok=True)

# Save models
joblib.dump(rf_rush, "ml/models/rush_classifier.pkl")
joblib.dump(gb_confirm, "ml/models/confirmation_regressor.pkl")
joblib.dump(gb_booking, "ml/models/booking_window_regressor.pkl")

# Save encoders and scalers
joblib.dump(label_encoders, "ml/models/label_encoders.pkl")
joblib.dump(rush_encoder, "ml/models/rush_target_encoder.pkl")

# Save individual scalers for each model
joblib.dump(scaler_rush, "ml/models/rush_scaler.pkl")
joblib.dump(scaler_confirm, "ml/models/confirm_scaler.pkl")
joblib.dump(scaler_booking, "ml/models/booking_scaler.pkl")

# Save feature lists
joblib.dump(rush_features, "ml/models/rush_features.pkl")
joblib.dump(confirm_features, "ml/models/confirm_features.pkl")
joblib.dump(booking_features, "ml/models/booking_features.pkl")

print("\n" + "=" * 60)
print("✅ ALL MODELS SAVED SUCCESSFULLY")
print("=" * 60)
print("\nSaved files:")
print("  - ml/models/rush_classifier.pkl")
print("  - ml/models/confirmation_regressor.pkl")
print("  - ml/models/booking_window_regressor.pkl")
print("  - ml/models/label_encoders.pkl")
print("  - ml/models/rush_target_encoder.pkl")
print("  - ml/models/rush_scaler.pkl")
print("  - ml/models/confirm_scaler.pkl")
print("  - ml/models/booking_scaler.pkl")
print("  - ml/models/*_features.pkl")
