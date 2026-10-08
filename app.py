
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("heart_model.pkl")

features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Heart Disease Prediction",
        "status": "ok"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    if any(key not in data for key in features):
        return jsonify({
            "error": "All 13 input features are required"
        }), 400

    try:
        values = [float(data[key]) for key in features]
        input_data = pd.DataFrame([values], columns=features)
        prediction = int(model.predict(input_data)[0])

        return jsonify({
            "prediction_code": prediction,
            "prediction": (
                "Disease class (1)" if prediction == 1
                else "Non-disease class (0)"
            )
        })
    except (ValueError, TypeError):
        return jsonify({
            "error": "Input features must be numeric"
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
