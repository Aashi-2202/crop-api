from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "Crop Recommendation API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]])

        prediction = model.predict(features)
        crop = str(prediction[0])
        return jsonify({"crop": crop, "status": "success"})

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}", "status": "fail"}), 400
    except Exception as e:
        return jsonify({"error": str(e), "status": "fail"}), 500

if __name__ == "__main__":
    app.run(debug=True)