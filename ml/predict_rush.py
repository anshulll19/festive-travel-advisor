import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("ml/rush_model.pkl")
label_encoders = joblib.load("ml/label_encoders.pkl")

def predict_rush(
    festival: str,
    transport_mode: str,
    distance_km: float,
    historical_rush_index: float,
):
    """
    Predict festive travel rush level
    """

    # Create input dataframe (MATCH training features exactly)
    data = pd.DataFrame([{
        "festival": festival,
        "transport_mode": transport_mode,
        "distance_km": distance_km,
        "historical_rush_index": historical_rush_index,
    }])

    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in data.columns:
            data[col] = encoder.transform(data[col])

    # Predict
    prediction = model.predict(data)[0]
    confidence = model.predict_proba(data).max()

    # Decode prediction back to label
    rush_label = label_encoders["rush_level"].inverse_transform([prediction])[0]

    return {
        "predicted_rush_level": rush_label,
        "confidence": round(confidence * 100, 2)
    }


# ----------- Test Run -----------
if __name__ == "__main__":
    result = predict_rush(
        festival="Diwali",
        transport_mode="Train",
        distance_km=1200,
        historical_rush_index=92
    )

    print("\n📊 Prediction Result")
    for k, v in result.items():
        print(f"{k}: {v}")
