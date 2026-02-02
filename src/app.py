from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys
import os

# Add the current directory to sys.path to import advisor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advisor import FestiveTravelAdvisor

app = Flask(__name__)
CORS(app)

# Initialize advisor with models from the correct directory
# Since we run from the project root, ml/models is correct
advisor = FestiveTravelAdvisor(model_dir="ml/models")

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Required fields mapping
        params = {
            "festival": data.get("festival"),
            "days_before_festival": data.get("days_before_festival"),
            "source_city": data.get("source_city"),
            "destination_city": data.get("destination_city"),
            "route_distance_km": data.get("route_distance_km"),
            "source_city_tier": data.get("source_city_tier"),
            "destination_city_tier": data.get("destination_city_tier"),
            "train_class": data.get("train_class"),
            "train_type": data.get("train_type"),
            "current_waitlist_position": data.get("current_waitlist_position", 0),
            "quota": data.get("quota", "General")
        }

        # Get complete advisory
        result = advisor.get_complete_advisory(**params)

        # Extract rush info
        rush_info = result['rush_analysis']

        # Estimate historical rush index for the frontend
        historical_rush_index = advisor._estimate_historical_rush(
            params['festival'],
            params['route_distance_km'],
            params['source_city_tier'],
            params['destination_city_tier'],
            params['train_class']
        )

        # Format response for index.html
        response = {
            "success": True,
            "data": {
                "predictions": {
                    "rush_level": rush_info['rush_level'],
                    "rush_confidence": rush_info['confidence'],
                    "historical_rush_index": round(historical_rush_index, 1),
                    "confirmation_probability": result['confirmation_probability']
                },
                "recommendations": {
                    "risk_level": rush_info['rush_level'],
                    "primary_advice": result['recommendations'][0] if result['recommendations'] else "No primary advice",
                    "booking_timing": result['recommendations'][1] if len(result['recommendations']) > 1 else "No timing advice",
                    "action_items": result['recommendations'][2:] if len(result['recommendations']) > 2 else [],
                    "alternatives": [
                        "Consider flying if distance > 1000km",
                        "Check for special festival trains",
                        "Try changing travel dates by 1-2 days"
                    ]
                }
            }
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index.html'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
