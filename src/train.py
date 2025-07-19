import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import os

print("--- Training Script Started ---")

# --- 1. Loading Configuration ---
config_path = '../config/config.json'
print(f"Loading configuration from: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)
print("Configuration loaded.")
print(f"Model Parameters: {config}")

# --- 2. Loading Dataset ---
print("Loading digits dataset...")
digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset loaded with {len(X)} samples.")

# --- 3. Training Model ---
print("Training Logistic Regression model...")
model = LogisticRegression(**config)
model.fit(X, y)
print("Model training complete.")

# --- 4. Saving Model ---
model_output_path = '../model_train.pkl'
print(f"Saving trained model to: {model_output_path}")
joblib.dump(model, model_output_path)

print("--- Training Script Finished Successfully ---")