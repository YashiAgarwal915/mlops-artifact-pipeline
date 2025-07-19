import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import os

def train_and_save_model(config_path, model_output_path):
    """
    Loads data and config, trains a model, and saves it.
    """
    # --- 1. Loading Configuration ---
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
    print(f"Saving trained model to: {model_output_path}")
    joblib.dump(model, model_output_path)
    print("Model saved.")

    return model

# This block ensures the script runs only when executed directly
if __name__ == '__main__':
    print("--- Training Script Started ---")
    
    # Use robust pathing to define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config', 'config.json')
    model_output_path = os.path.join(script_dir, '..', 'model_train.pkl')
    
    # Run the main function
    train_and_save_model(config_path, model_output_path)
    
    print("--- Training Script Finished Successfully ---")