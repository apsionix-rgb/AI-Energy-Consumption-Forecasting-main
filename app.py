from flask import Flask, request, jsonify
import joblib
import numpy as np
import datetime

app = Flask(__name__)

# Load model
try:
    model = joblib.load("models/energy_model.pkl")
except:
    print("❌ Model not found! Run main.py first")
    exit()

@app.route("/")
def home():
    return {
        "message": "⚡ Energy Forecasting API Running",
        "status": "OK"
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = ["hour", "day", "month", "lag_1", "lag_24", "rolling_mean_24"]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is missing"}), 400

        features = np.array([[
            data["hour"],
            data["day"],
            data["month"],
            data["lag_1"],
            data["lag_24"],
            data["rolling_mean_24"]
        ]])

        prediction = model.predict(features)

        return jsonify({
            "predicted_energy": float(prediction[0]),
            "timestamp": str(datetime.datetime.now()),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        })

if __name__ == "__main__":
    app.run(debug=True)