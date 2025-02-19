import pickle
import traceback
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Expected features
EXPECTED_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded properly."}), 500
    
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Invalid input. Expected JSON with 'features' key."}), 400
        
        features = data["features"]
        
        # Validate features
        if not isinstance(features, dict):
            return jsonify({"error": "Features should be a dictionary."}), 400
        
        missing_features = [f for f in EXPECTED_FEATURES if f not in features]
        extra_features = [f for f in features if f not in EXPECTED_FEATURES]
        
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        if extra_features:
            return jsonify({"error": f"Unexpected features: {extra_features}"}), 400
        
        # Convert input features to a DataFrame with proper column names
        feature_values = [features[feature] for feature in EXPECTED_FEATURES]
        features_df = pd.DataFrame([feature_values], columns=EXPECTED_FEATURES)
        
        # Make prediction
        prediction = model.predict(features_df).tolist()
        return jsonify({"prediction": prediction})
    
    
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500



@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"})

if __name__ == "__main__":
    app.run(debug=True, port=6000, host="0.0.0.0")

