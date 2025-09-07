from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained machine learning model from the pickle file
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(
        "Error: model.pkl not found. Please run model.py first to create the model file."
    )
    exit()


app = Flask(__name__)


@app.route("/")
def home():
    """Renders the home page with the diabetes prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles the prediction request from the HTML form.
    It takes the form data, processes it, and returns a prediction.
    """
    try:
        # Get the input values from the form and convert them to floats
        features = [
            float(request.form.get("glucose")),
            float(request.form.get("blood_pressure")),
            float(request.form.get("skin_thickness")),
            float(request.form.get("insulin")),
            float(request.form.get("bmi")),
            float(request.form.get("diabetes_pedigree_function")),
            float(request.form.get("age")),
        ]

        # Convert the list of features into a NumPy array, reshaping for the model
        final_features = np.array(features).reshape(1, -1)

        # Use the loaded model to make a prediction
        prediction = model.predict(final_features)[0]

        # Return a user-friendly string with the prediction result
        if prediction == 1:
            return "Based on the data, the patient is predicted to have diabetes."
        else:
            return "Based on the data, the patient is predicted to be non-diabetic."

    except Exception as e:
        # Provide a more informative error message to the user
        return f"An error occurred: {str(e)}. Please check your input."


if __name__ == "__main__":
    app.run(debug=True)
