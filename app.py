from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import precision_score

app = Flask("household_occupancy")
app.secret_key = "super_secret_key"

# Load the model
try:
    model = joblib.load("model/lightgbm_model.pkl")
    # Ensure the model is fitted properly
    model.predict(np.zeros((1, 5)))
except (FileNotFoundError, NotFittedError):
    print("Model not properly fitted or not found. Train and save the model first.")
    model = None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", precision=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", precision="Model not available.")

    file = request.files["file"]
    if file:
        # Read the uploaded file as a CSV
        data = pd.read_csv(file)
        # Make predictions
        y_pred = model.predict(data)

        # Assuming some ground truth labels for precision calculation
        # This is a placeholder; in real use cases, ground truth should be known
        ground_truth_labels = [1, 0] * (len(y_pred) // 2) + [1] * (len(y_pred) % 2)
        
        # Calculate precision
        precision = precision_score(ground_truth_labels[:len(y_pred)], y_pred)
        return render_template("index.html", precision=precision)
    else:
        return render_template("index.html", precision=None)

if __name__ == "__main__":
    app.run(debug=True)
