import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Path Configuration for Deployment ---
def is_kaggle_environment():
    return os.path.exists("/kaggle/input")

if is_kaggle_environment():
    BASE_PATH = "/kaggle/input/accident-prediction-model/"
else:
    BASE_PATH = "" 

# --- Load Resources (cached) ---
@st.cache_resource
def load_model_and_resources():
    try:
        model = joblib.load(BASE_PATH + "lgb_model.pkl")
        thresholds = joblib.load(BASE_PATH + "thresholds.pkl")
        scaler = joblib.load(BASE_PATH + "scaler.pkl")
        le = joblib.load(BASE_PATH + "label_encoder.pkl")
        freq_maps = joblib.load(BASE_PATH + "freq_maps.pkl")
        return model, thresholds, scaler, le, freq_maps

    except FileNotFoundError as e:
        st.error(f"Missing File: {e.filename}")
        st.error(f"Looking in: {BASE_PATH}")
        st.stop()

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

model, thresholds, scaler, le, freq_maps = load_model_and_resources()

# --- Mapping Dictionaries for UI and Prediction ---
urban_rural_options = {
    "Urban ": 1,
    "Rural ": 2,
    "Unallocated ": 3
}

light_desc_to_code = {
    'Daylight: Street light present': 4,
    'Darkness: Street lights present and lit': 3,
    'Darkness: Street lighting unknown': 2,
    'Darkness: Street lights present but unlit': 1,
    'Darkness: No street lighting': 0, 
}
surface_desc_to_code = {
    'Dry': 4, 'Wet/Damp': 3, 'Frost/Ice': 2, 'Snow': 1, 'Flood (Over 3cm of water)': 0,
}

# --- UI Inputs ---
st.title("Accident Severity Prediction")

speed_limit = st.number_input("Speed Limit (mph)", min_value=20, max_value=70, value=30)

selected_urban_rural_desc = st.selectbox("Urban or Rural Area", list(urban_rural_options.keys()))
urban_rural_value = urban_rural_options[selected_urban_rural_desc]

selected_light_desc = st.selectbox("Light Conditions", list(light_desc_to_code.keys()))
light_condition_value = light_desc_to_code[selected_light_desc]

selected_road_surface_desc = st.selectbox("Road Surface Conditions", list(surface_desc_to_code.keys()))
road_surface_value = surface_desc_to_code[selected_road_surface_desc]

# --- Prediction Logic ---
input_df = pd.DataFrame({
    "Speed_limit": [speed_limit],
    "Urban_or_Rural_Area": [urban_rural_value],
    "Light_Conditions": [light_condition_value],
    "Road_Surface_Conditions": [road_surface_value],
})

input_df['Speed_Urban_Rural'] = input_df['Urban_or_Rural_Area'] * input_df['Speed_limit']
input_df['Light_Road_Interaction'] = input_df['Light_Conditions'] * input_df['Road_Surface_Conditions']

for col in model.feature_name_:
    if col not in input_df.columns:
        input_df[col] = 0

for col in freq_maps:
    if col in input_df.columns:
        input_df[col] = input_df[col].map(freq_maps[col]).fillna(0)

probs = model.predict_proba(input_df)
pred = np.zeros(len(probs), dtype=int)
for i, th in enumerate(thresholds):
    mask = probs[:, i] >= th
    pred[mask] = i

mask_no_class = ~np.any(probs >= thresholds, axis=1)
pred[mask_no_class] = np.argmax(probs[mask_no_class], axis=1)

pred_labels = le.inverse_transform(pred)
st.write("Predicted Accident Severity:", pred_labels[0])
