import joblib
import os
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

print("--- Inference Script Started ---")


script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, '..', 'model_train.pkl')

print(f"Looking for model at: {model_path}")


if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

print("Loading trained model...")
model = joblib.load(model_path)
print("Model loaded successfully.")


print("Loading dataset for inference...")
digits = load_digits()

X_inference = digits.data
y_true = digits.target 
print(f"Dataset loaded with {len(X_inference)} samples.")


print("Generating predictions...")
predictions = model.predict(X_inference)
print("Predictions generated.")


print("\n--- Model Performance Report ---")
report = classification_report(y_true, predictions)
print(report)
print("--------------------------------\n")

print("--- Inference Script Finished Successfully ---")