import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

class FestiveTravelAdvisor:
    """
    Complete system for predicting travel rush and providing recommendations
    """

    def __init__(self, model_dir="ml/models"):
        """Load all trained models and encoders"""

        # ML models
        self.rush_model = joblib.load(f"{model_dir}/rush_classifier.pkl")
        self.confirm_model = joblib.load(f"{model_dir}/confirmation_regressor.pkl")
        self.booking_model = joblib.load(f"{model_dir}/booking_window_regressor.pkl")

        # Encoders
        self.label_encoders = joblib.load(f"{model_dir}/label_encoders.pkl")
        self.rush_encoder = joblib.load(f"{model_dir}/rush_target_encoder.pkl")

        # Scalers
        self.rush_scaler = joblib.load(f"{model_dir}/rush_scaler.pkl")
        self.confirm_scaler = joblib.load(f"{model_dir}/confirm_scaler.pkl")
        self.booking_scaler = joblib.load(f"{model_dir}/booking_scaler.pkl")

        # Feature lists
        self.rush_features = joblib.load(f"{model_dir}/rush_features.pkl")
        self.confirm_features = joblib.load(f"{model_dir}/confirm_features.pkl")
        self.booking_features = joblib.load(f"{model_dir}/booking_features.pkl")

        # Explainability (optional - create if doesn't exist)
        feature_importance_path = f"{model_dir}/rush_feature_importance.json"
        if os.path.exists(feature_importance_path):
            with open(feature_importance_path) as f:
                self.rush_feature_info = json.load(f)
        else:
            # Generate feature importance from the model if JSON doesn't exist
            if hasattr(self.rush_model, 'feature_importances_'):
                importances = self.rush_model.feature_importances_
                self.rush_feature_info = {
                    "features": self.rush_features,
                    "importance": importances.tolist()
                }
                # Save it for future use
                with open(feature_importance_path, 'w') as f:
                    json.dump(self.rush_feature_info, f, indent=2)
                print(f"✅ Created {feature_importance_path}")
            else:
                # Fallback if model doesn't have feature importances
                self.rush_feature_info = {
                    "features": self.rush_features,
                    "importance": [1.0] * len(self.rush_features)
                }

        print("✅ All models & explainability loaded successfully")

    def _prepare_features(self, input_data, feature_list, scaler):
        """Encode and scale features"""

        df = pd.DataFrame([input_data])

        for col in self.label_encoders:
            if col in df.columns:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except:
                    df[col] = 0

        df = df[feature_list]
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        return df_scaled

    def predict_rush_level(self, festival, days_before_festival, route_distance_km,
                          source_city_tier, destination_city_tier, 
                          train_class, train_type, historical_rush_index=None,
                          peak_day_proximity=None):

        if historical_rush_index is None:
            historical_rush_index = self._estimate_historical_rush(
                festival, route_distance_km, source_city_tier,
                destination_city_tier, train_class
            )

        if peak_day_proximity is None:
            peak_day_proximity = max(0, 5 - abs(days_before_festival - 3))

        input_data = {
            "festival": festival,
            "days_before_festival": days_before_festival,
            "route_distance_km": route_distance_km,
            "source_city_tier": source_city_tier,
            "destination_city_tier": destination_city_tier,
            "peak_day_proximity": peak_day_proximity,
            "train_class": train_class,
            "train_type": train_type,
            "historical_rush_index": historical_rush_index
        }

        X = self._prepare_features(input_data, self.rush_features, self.rush_scaler)

        prediction = self.rush_model.predict(X)[0]
        probabilities = self.rush_model.predict_proba(X)[0]

        rush_level = self.rush_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))

        # 🔍 Explainability
        features = self.rush_feature_info["features"]
        importance = self.rush_feature_info["importance"]

        top_factors = sorted(
            zip(features, importance),
            key=lambda x: x[1],
            reverse=True
        )[:4]

        explanations = [f.replace("_", " ").title() for f, _ in top_factors]

        return {
            "rush_level": rush_level,
            "confidence": round(confidence, 3),
            "top_factors": explanations,
            "probabilities": {
                self.rush_encoder.inverse_transform([i])[0]: round(prob, 3)
                for i, prob in enumerate(probabilities)
            }
        }

    def predict_confirmation_probability(self, current_waitlist_position, 
                                        days_to_journey, train_type, quota,
                                        train_class, historical_rush_index, 
                                        ticket_status="WL"):

        input_data = {
            "current_waitlist_position": current_waitlist_position,
            "days_to_journey": days_to_journey,
            "train_type": train_type,
            "quota": quota,
            "train_class": train_class,
            "historical_rush_index": historical_rush_index,
            "ticket_status": ticket_status
        }

        X = self._prepare_features(input_data, self.confirm_features, self.confirm_scaler)
        probability = self.confirm_model.predict(X)[0]
        return round(max(0, min(1, probability)), 3)

    def predict_optimal_booking_window(self, festival, route_distance_km,
                                       source_city_tier, destination_city_tier,
                                       train_class, historical_rush_index=None):

        if historical_rush_index is None:
            historical_rush_index = self._estimate_historical_rush(
                festival, route_distance_km, source_city_tier,
                destination_city_tier, train_class
            )

        input_data = {
            "festival": festival,
            "route_distance_km": route_distance_km,
            "source_city_tier": source_city_tier,
            "destination_city_tier": destination_city_tier,
            "train_class": train_class,
            "historical_rush_index": historical_rush_index
        }

        X = self._prepare_features(input_data, self.booking_features, self.booking_scaler)
        optimal_days = self.booking_model.predict(X)[0]

        return {
            "optimal_min": int(optimal_days - 5),
            "optimal_max": int(optimal_days + 5),
            "recommended": int(optimal_days)
        }

    def get_complete_advisory(self, festival, days_before_festival, source_city,
                             destination_city, route_distance_km, source_city_tier,
                             destination_city_tier, train_class, train_type,
                             current_waitlist_position=0, quota="General"):
        """
        Get complete travel advisory including rush level, confirmation probability,
        and optimal booking window
        """
        
        # Predict rush level
        rush_info = self.predict_rush_level(
            festival=festival,
            days_before_festival=days_before_festival,
            route_distance_km=route_distance_km,
            source_city_tier=source_city_tier,
            destination_city_tier=destination_city_tier,
            train_class=train_class,
            train_type=train_type
        )
        
        # Get historical rush index from rush prediction
        historical_rush_index = self._estimate_historical_rush(
            festival, route_distance_km, source_city_tier,
            destination_city_tier, train_class
        )
        
        # Predict confirmation probability if waitlisted
        confirmation_prob = None
        if current_waitlist_position > 0:
            confirmation_prob = self.predict_confirmation_probability(
                current_waitlist_position=current_waitlist_position,
                days_to_journey=days_before_festival,
                train_type=train_type,
                quota=quota,
                train_class=train_class,
                historical_rush_index=historical_rush_index
            )
        
        # Get optimal booking window
        booking_window = self.predict_optimal_booking_window(
            festival=festival,
            route_distance_km=route_distance_km,
            source_city_tier=source_city_tier,
            destination_city_tier=destination_city_tier,
            train_class=train_class,
            historical_rush_index=historical_rush_index
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            rush_info["rush_level"],
            days_before_festival,
            booking_window,
            confirmation_prob,
            train_class
        )
        
        return {
            "route": {
                "from": source_city,
                "to": destination_city,
                "distance_km": route_distance_km
            },
            "festival": festival,
            "days_before_festival": days_before_festival,
            "rush_analysis": rush_info,
            "confirmation_probability": confirmation_prob,
            "optimal_booking_window": booking_window,
            "recommendations": recommendations,
            "train_details": {
                "class": train_class,
                "type": train_type,
                "quota": quota
            }
        }
    
    def _generate_recommendations(self, rush_level, days_before, booking_window, 
                                 confirmation_prob, train_class):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Rush-based recommendations
        if rush_level == "High":
            recommendations.append("⚠️ Very high rush expected. Book as early as possible.")
            recommendations.append("Consider booking Tatkal if regular quota is full.")
        elif rush_level == "Medium":
            recommendations.append("📊 Moderate rush expected. Book within optimal window.")
        else:
            recommendations.append("✅ Low rush expected. Normal booking should work.")
        
        # Booking window recommendations
        if days_before > booking_window["optimal_max"]:
            recommendations.append(f"⏰ Book within {booking_window['optimal_min']}-{booking_window['optimal_max']} days before festival.")
        elif days_before < booking_window["optimal_min"]:
            recommendations.append("🚨 You're booking late! Consider alternate options.")
        else:
            recommendations.append("✅ You're in the optimal booking window!")
        
        # Confirmation probability recommendations
        if confirmation_prob is not None:
            if confirmation_prob < 0.3:
                recommendations.append("❌ Low confirmation chances. Consider alternate trains or dates.")
            elif confirmation_prob < 0.7:
                recommendations.append("⚠️ Moderate confirmation chances. Have backup plans ready.")
            else:
                recommendations.append("✅ Good confirmation chances!")
        
        # Class-based recommendations
        if train_class in ["General", "Sleeper"] and rush_level == "High":
            recommendations.append("💡 Consider upgrading to AC classes for better availability.")
        
        return recommendations

    def _estimate_historical_rush(self, festival, distance, src_tier, dst_tier, train_class):
        """Estimate historical rush index based on route and festival characteristics"""
        
        festival_weights = {
            "Diwali": 95, "Chhath Puja": 90, "Durga Puja": 85,
            "Eid-ul-Fitr": 80, "Holi": 75, "Christmas": 70, "Pongal": 72
        }

        class_weights = {
            "General": 85, "Sleeper": 80, "3AC": 70, "2AC": 60, "1AC": 50
        }

        base = festival_weights.get(festival, 70)
        class_factor = class_weights.get(train_class, 70)
        distance_factor = min(20, distance / 100)
        tier_factor = (4 - src_tier + 4 - dst_tier) * 3

        return min(100, base * 0.5 + class_factor * 0.3 + distance_factor + tier_factor)
