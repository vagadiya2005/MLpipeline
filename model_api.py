import pickle
import traceback
import logging
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Expected features
EXPECTED_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        logging.error("Model is not loaded properly.")
        return jsonify({"error": "Model is not loaded properly."}), 500
    
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")
        
        if not data or "features" not in data:
            logging.warning("Invalid input. Expected JSON with 'features' key.")
            return jsonify({"error": "Invalid input. Expected JSON with 'features' key."}), 400
        
        features = data["features"]
        
        # Validate features
        if not isinstance(features, dict):
            logging.warning("Features should be a dictionary.")
            return jsonify({"error": "Features should be a dictionary."}), 400
        
        missing_features = [f for f in EXPECTED_FEATURES if f not in features]
        extra_features = [f for f in features if f not in EXPECTED_FEATURES]
        
        if missing_features:
            logging.warning(f"Missing features: {missing_features}")
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        if extra_features:
            logging.warning(f"Unexpected features: {extra_features}")
            return jsonify({"error": f"Unexpected features: {extra_features}"}), 400
        
        # Convert input features to a DataFrame with proper column names
        feature_values = [features[feature] for feature in EXPECTED_FEATURES]
        features_df = pd.DataFrame([feature_values], columns=EXPECTED_FEATURES)
        
        # Make prediction
        prediction = model.predict(features_df).tolist()
        logging.info(f"Prediction: {prediction}")
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    logging.info("Health check endpoint called.")
    return jsonify({"status": "API is running"})

if __name__ == "__main__":
    app.run(debug=True, port=6000, host="0.0.0.0")