# 🚂 Festive Travel Advisor

An AI-powered system designed to predict travel rush during major Indian festivals and provide actionable travel recommendations, including confirmation probabilities and optimal booking windows.

## 🌟 Features
- **Rush Prediction:** Predicts Low, Medium, or High rush levels based on historical patterns.
- **Confirmation Probability:** Real-time estimation of waitlist confirmation chances.
- **Optimal Booking Window:** Recommends the best time to book tickets to ensure a confirmed seat.
- **Dual Interface:** Choose between a sleek Streamlit dashboard and a traditional Flask-based web application.

## 📂 Project Structure
```text
.
├── data/               # Processed datasets
├── ml/
│   ├── models/         # Trained model artifacts (.pkl files)
│   └── legacy/         # Outdated/legacy scripts
├── src/
│   ├── advisor.py      # Core logic and prediction engine
│   ├── app.py          # Flask API backend
│   ├── streamlit_app.py # Interactive Streamlit dashboard
│   ├── generate_enhanced_dataset.py # Data generation script
│   └── train_enhanced_models.py    # Model training script
├── index.html          # Frontend for the Flask app
└── requirements.txt    # Project dependencies
```

## 🛠️ Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Initialize the System
The system requires trained models to function. Run the following scripts in order:
```bash
# Generate synthetic training data
python src/generate_enhanced_dataset.py

# Train the machine learning models
python src/train_enhanced_models.py
```

### 3. Verify the Installation (Optional)
Run the test suite to ensure everything is set up correctly:
```bash
python tests/test_predictions.py
```

## 🚀 Running the Application

### Option A: Streamlit Dashboard (Recommended)
Launch the interactive dashboard for a rich visual experience:
```bash
streamlit run src/streamlit_app.py
```

### Option B: Flask API & Web Frontend
Run the Flask server to serve the API and the static frontend:
```bash
python src/app.py
```
Then, open `index.html` in your browser (or visit `http://localhost:3000` if served by Flask).

## 📊 How it Works
The advisor uses a combination of **Random Forest** and **Gradient Boosting** models trained on a synthetic dataset that simulates real-world Indian Railway booking patterns during festivals like Diwali, Holi, and Durga Puja. It considers factors such as city tiers, route distance, train classes, and peak day proximity.

---
© 2026 Festive Travel Advisor | AI-Powered Railway Intelligence
