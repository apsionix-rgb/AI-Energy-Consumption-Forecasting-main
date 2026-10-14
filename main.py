import os
import joblib
import matplotlib.pyplot as plt

from src.data_loader import load_all_data
from src.preprocess import preprocess_data
from src.feature_engineering import create_features
from src.model import train_model
from src.evaluate import evaluate_model

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load ALL data
data = load_all_data("data")

# Preprocess
data = preprocess_data(data)

# Feature engineering
data = create_features(data)

# Train model
model, X_test, y_test = train_model(data)

# Evaluate
predictions, rmse, r2 = evaluate_model(model, X_test, y_test)

print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

# Save model
joblib.dump(model, "models/energy_model.pkl")

# Visualization

# 1. Time series
plt.figure(figsize=(12,5))
data['Energy'].plot(title="Energy Consumption (All Regions)")
plt.savefig("outputs/time_series.png")

# 2. Actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(predictions[:200], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.savefig("outputs/actual_vs_pred.png")

print("✅ Done! Check outputs/")