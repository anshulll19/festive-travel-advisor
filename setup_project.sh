#!/bin/bash

# Setup script for Festive Travel Advisor
# This script initializes the data and trains the models

set -e

echo "🚀 Initializing Festive Travel Advisor..."

# 1. Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# 2. Generate Dataset
echo "📊 Generating synthetic dataset..."
python src/generate_enhanced_dataset.py

# 3. Train Models
echo "🧠 Training machine learning models..."
python src/train_enhanced_models.py

# 4. Verify Setup
echo "🧪 Running verification tests..."
python tests/test_predictions.py

echo "✅ Setup complete! You can now run the application."
echo "   - For Streamlit: streamlit run src/streamlit_app.py"
echo "   - For Flask API: python src/app.py"
