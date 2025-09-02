import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Lymphography Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

# --- Caching the Model and Scaler ---
# Use st.cache_resource to load the model and scaler only once
@st.cache_resource
def load_resources():
    """Loads the pickled model and scaler."""
    try:
        with open('lymphography_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'lymphography_model.pkl' and 'scaler.pkl' are in the root directory.")
        return None, None

model, scaler = load_resources()

# --- Feature Order and Class Mapping ---
# This ensures the data is in the correct order for the model
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
    3: "Malignant Lymphoma",
    4: "Fibrosis"
}

# --- Main Application UI ---
st.title("🩺 Lymphography Disease Prediction")
st.write("This app predicts the lymphography diagnosis based on 18 features. Please provide the patient's data below.")


# --- Input Form ---
# st.form is used to group inputs and submit them all at once
with st.form("prediction_form"):
    st.header("Patient Feature Input")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        lymphatics = st.selectbox('Lymphatics', [1, 2, 3, 4], help="Normal, Arched, Deformed, Displaced")
        block_of_affere = st.selectbox('Block of Afferent', [1, 2], help="No, Yes")
        bl_of_lymph_c = st.selectbox('Block of Lymphatic Channels (c)', [1, 2], help="No, Yes")
        bl_of_lymph_s = st.selectbox('Block of Lymphatic Sinuses (s)', [1, 2], help="No, Yes")
        by_pass = st.selectbox('By-pass', [1, 2], help="No, Yes")
        extravasates = st.selectbox('Extravasates', [1, 2], help="No, Yes")
        regeneration_of = st.selectbox('Regeneration of Lymphatics', [1, 2], help="No, Yes")
        early_uptake_in = st.selectbox('Early Uptake in Lymphatic Vessels', [1, 2], help="No, Yes")
        lym_nodes_dimin = st.selectbox('Lymph Nodes Diminished', [1, 2, 3], help="0-3")

    with col2:
        lym_nodes_enlar = st.selectbox('Lymph Nodes Enlarged', [1, 2, 3, 4], help="1-4")
        changes_in_lym = st.selectbox('Changes in Lymphatics', [1, 2, 3, 4], help="Bean, Oval, Round, Irregular")
        defect_in_node = st.selectbox('Defect in Node', [1, 2, 3, 4], help="No, Lacunar, Lac. Marginal, Lac. Central")
        changes_in_node = st.selectbox('Changes in Node', [1, 2, 3], help="No, Lacunar, Granular")
        changes_in_stru = st.selectbox('Changes in Structure', [1, 2, 3, 4], help="No, Coarse, Fine, Irregular")
        special_forms = st.selectbox('Special Forms', [1, 2, 3], help="No, Chalices, Lesions")
        dislocation_of = st.selectbox('Dislocation of Chain', [1, 2], help="No, Yes")
        exclusion_of_no = st.selectbox('Exclusion of Node', [1, 2], help="No, Yes")
        no_of_nodes_in = st.number_input('Number of Nodes in Region', min_value=0, max_value=20, value=5, step=1)

    # Submit button for the form
    submitted = st.form_submit_button("Predict Diagnosis")


# --- Prediction Logic ---
if submitted:
    if model and scaler:
        # 1. Collect form data into a dictionary
        input_data = {
            'lymphatics': lymphatics,
            'block_of_affere': block_of_affere,
            'bl_of_lymph_c': bl_of_lymph_c,
            'bl_of_lymph_s': bl_of_lymph_s,
            'by_pass': by_pass,
            'extravasates': extravasates,
            'regeneration_of': regeneration_of,
            'early_uptake_in': early_uptake_in,
            'lym_nodes_dimin': lym_nodes_dimin,
            'lym_nodes_enlar': lym_nodes_enlar,
            'changes_in_lym': changes_in_lym,
            'defect_in_node': defect_in_node,
            'changes_in_node': changes_in_node,
            'changes_in_stru': changes_in_stru,
            'special_forms': special_forms,
            'dislocation_of': dislocation_of,
            'exclusion_of_no': exclusion_of_no,
            'no_of_nodes_in': no_of_nodes_in
        }

        # 2. Create a DataFrame in the correct feature order
        input_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)

        # 3. Scale the data
        scaled_features = scaler.transform(input_df)

        # 4. Make prediction
        prediction_raw = model.predict(scaled_features)
        prediction_class = CLASS_MAPPING.get(prediction_raw[0], "Unknown")

        # 5. Display the result
        st.success(f"**Predicted Diagnosis:** {prediction_class}")
        st.balloons()
    else:
        st.error("The prediction model is not loaded. Cannot proceed.")

st.info("Note: This is a predictive model and not a substitute for professional medical advice.")

