import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import joblib
import pickle
import logging
import argparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_dataset():
    """Load the California housing dataset and return it as a DataFrame."""
    try:
        house_price_dataset = sklearn.datasets.fetch_california_housing()
        df = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
        df["price"] = house_price_dataset.target
        logging.info("Dataset successfully loaded.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def preprocess_data(df):
    """Perform basic preprocessing and return feature and target variables."""
    if df.isnull().sum().sum() > 0:
        logging.warning("Dataset contains missing values.")
        df = df.dropna()
    
    X = df.drop(["price"], axis=1)
    Y = df["price"]
    return X, Y

def train_model(X_train, Y_train):
    """Train an XGBoost model and return the trained model."""
    try:
        model = XGBRegressor()
        model.fit(X_train, Y_train)
        logging.info("Model training completed.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def evaluate_model(model, X, Y, dataset_type="Test"):
    """Evaluate model performance and log results."""
    predictions = model.predict(X)
    r2_score = metrics.r2_score(Y, predictions)
    mae = metrics.mean_absolute_error(Y, predictions)

    logging.info(f"{dataset_type} Data - R² Score: {r2_score:.4f}, Mean Absolute Error: {mae:.4f}")
    
    if dataset_type == "Train":
        plt.scatter(Y, predictions)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual Price vs Predicted Price")
        plt.show()

def save_model(model, filename="model.pkl"):
    """Save trained model as a pickle file."""
    try:
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved as {filename}.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="House Price Prediction Model")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model")
    args = parser.parse_args()

    df = load_dataset()
    X, Y = preprocess_data(df)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    logging.info(f"Data split: Train shape {X_train.shape}, Test shape {X_test.shape}")

    if args.train:
        model = train_model(X_train, Y_train)
        save_model(model)

    if args.evaluate:
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            logging.info("Loaded saved model for evaluation.")
            evaluate_model(model, X_test, Y_test, dataset_type="Test")
        else:
            logging.error("No saved model found. Train the model first.")

if __name__ == "__main__":
    main()
