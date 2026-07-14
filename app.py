from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)


vectorizer = joblib.load("models/vectorizer.pkl")


attributes = [
    "category",
    "color",
    "fabric",
    "neckline",
    "sleeve",
    "length",
    "silhouette",
    "embellishment"
]

models = {}

for attribute in attributes:
    models[attribute] = joblib.load(
        f"models/{attribute}_model.pkl"
    )

@app.route("/")
def home():
    return "Welcome to home"

@app.route("/extract", methods=["POST"])
def extract():
    data = request.get_json()

    if not data or "description" not in data:
        return jsonify({"error": "Missing 'description' field"}), 400

    description = data["description"].strip()

    if not description:
        return jsonify({"error": "Description cannot be empty"}), 400

    vector = vectorizer.transform([description])

    result = {}

    for attribute, model in models.items():
        result[attribute] = model.predict(vector)[0]

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)