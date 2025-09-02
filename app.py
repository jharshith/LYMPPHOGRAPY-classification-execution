from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Load Model and Scaler ---
# It's good practice to load these once when the app starts.
try:
    with open('lymphography_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    # Handle the error gracefully if files are missing
    model = None
    scaler = None
    print("Error: Model or scaler file not found. Please ensure 'lymphography_model.pkl' and 'scaler.pkl' are in the same directory.")


# --- Define Feature Order and Class Mapping ---
# This list is crucial to ensure the data is in the same order as the training data.
# I've extracted these names from the form in your GitHub repository.
FEATURE_ORDER = [
    'lymphatics', 'block_of_affere', 'bl_of_lymph_c', 'bl_of_lymph_s',
    'by_pass', 'extravasates', 'regeneration_of', 'early_uptake_in',
    'lym_nodes_dimin', 'lym_nodes_enlar', 'changes_in_lym', 'defect_in_node',
    'changes_in_node', 'changes_in_stru', 'special_forms', 'dislocation_of',
    'exclusion_of_no', 'no_of_nodes_in'
]

CLASS_MAPPING = {
    1: "Normal",
    2: "Metastases",
    3: "Malign lymph",
    4: "Fibrosis"
}

# --- Application Routes ---

# Route for the landing page
@app.route('/')
def landing():
    """Renders the main landing page (index.html)."""
    return render_template('index.html')

# Route for the form page (both GET and POST)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handles both displaying the form and processing the prediction."""
    prediction_result = None # Initialize prediction as None

    if request.method == 'POST':
        if model and scaler:
            try:
                # 1. Get the data from the form
                form_data = request.form.to_dict()

                # 2. Create a DataFrame from form data
                input_df = pd.DataFrame([form_data])

                # 3. Convert data to numeric, handling potential errors
                input_df = input_df.apply(pd.to_numeric, errors='coerce')

                # 4. Reorder DataFrame columns to match the model's expected input
                input_df = input_df[FEATURE_ORDER]

                # 5. Scale the input data
                data_scaled = scaler.transform(input_df)

                # 6. Make prediction
                prediction_raw = model.predict(data_scaled)
                prediction_result = CLASS_MAPPING.get(prediction_raw[0], "Unknown class")

            except Exception as e:
                # Handle potential errors during prediction
                print(f"An error occurred during prediction: {e}")
                prediction_result = "Error during prediction. Please check input."
        else:
            prediction_result = "Model is not loaded. Cannot make a prediction."

    # Render the form page. If a prediction was made, it will be passed to the template.
    return render_template('form.html', prediction=prediction_result)


if __name__ == '__main__':
    # Use debug=True for development to see detailed errors in the browser.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(debug=True)
