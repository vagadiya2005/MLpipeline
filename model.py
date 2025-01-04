# linear_regression_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import pickle
import os

# 1. Load the Dataset
data = {
    "area": [1400, 1600, 1700, 1875, 1100, 1550],
    "bedrooms": [3, 3, 4, 3, 2, 3],
    "age": [20, 15, 18, 30, 8, 25],
    "price": [245000, 312000, 279895, 308000, 199000, 219000],
}
df = pd.DataFrame(data)

# 2. Preprocess Data
X = df[["area", "bedrooms", "age"]]
y = df["price"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

 # Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print("Model trained and saved as model.pkl")
