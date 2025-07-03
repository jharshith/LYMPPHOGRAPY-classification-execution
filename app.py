from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
with open('lymphography_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the class mapping
class_mapping = {
    1: "Normal",
    2: "Metastases",
    3: "Malign lymph",
    4: "Fibrosis"
}

# Route for the landing page
@app.route('/')
def landing():
    return render_template('index.html')

# Route for the form page
@app.route('/form')
def form_page():
    return render_template('form.html')

# Route for form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form.to_dict()
    data = pd.DataFrame([data], columns=data.keys())
    data = data.astype(float)  # Ensure the data is in the correct format
    
    # Scale the input data
    data_scaled = scaler.transform(data)
    
    # Make prediction
    prediction = model.predict(data_scaled)
    prediction_class = class_mapping[prediction[0]]
    
    return render_template('form.html', prediction=prediction_class)

if __name__ == '__main__':
    app.run(debug=False)
