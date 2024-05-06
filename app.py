from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.metrics import precision_score
import numpy as np

app = Flask("household_occupancy")
app.secret_key = "super_secret_key"

# Load and validate the model
try:
    model = joblib.load("model/lightgbm_model.pkl")
    # Validate the model by making a prediction
    model.predict(np.zeros((1, 28)))
    print("Model loaded and validated successfully.")
except Exception as e:
    model = None
    print(f"Error loading or validating model: {e}")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", precision=None)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if model and file:
        # Read the uploaded file as a CSV
        data = pd.read_csv(file)

        if "multiple_occupancy" not in data.columns or data.shape[1] != 29:  # Validate input data
            return render_template("index.html", precision="Input does not have the correct number of features.")

        # Separate features and ground truth labels
        X = data.drop(columns=["multiple_occupancy"])
        y_true = data["multiple_occupancy"]

        # Make predictions
        y_pred = model.predict(X)

        # Calculate precision
        precision = precision_score(y_true, y_pred)
        return render_template("index.html", precision=precision)
    else:
        return render_template("index.html", precision="Model not available.")

if __name__ == "__main__":
    app.run(debug=True)
