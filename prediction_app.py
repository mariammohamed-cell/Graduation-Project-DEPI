import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Path Configuration ---
def is_kaggle_environment():
    return os.path.exists("/kaggle/input")

BASE_PATH = "/kaggle/input/pkl-files/" if is_kaggle_environment() else ""

# --- Load Resources ---
@st.cache_resource
def load_model_and_resources():
    try:
        model = joblib.load(os.path.join(BASE_PATH, "lgb_model.pkl"))
        le = joblib.load(os.path.join(BASE_PATH, "target_encoder.pkl"))
        selected_features = joblib.load(os.path.join(BASE_PATH, "selected_features.pkl"))
        return model, le, selected_features
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

model, le, selected_features = load_model_and_resources()

# --- Encoders / Mappings ---
urban_rural_options = {"Urban": 1, "Rural": 2, "Unallocated": 3}
light_mapping = {
    'Daylight: Street light present': 4,
    'Darkness: Street lights present and lit': 3,
    'Darkness: Street lighting unknown': 2,
    'Darkness: Street lights present but unlit': 1,
    'Darkness: No street lighting': 0
}
surface_mapping = {'Dry': 4, 'Wet/Damp': 3, 'Frost/Ice': 2, 'Snow': 1, 'Flood': 0}

# --- UI ---
st.set_page_config(page_title="Accident Severity Prediction", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Accident Severity Prediction</h1>", unsafe_allow_html=True)

# --- Inputs ---
selected_urban = st.selectbox("Urban or Rural Area", list(urban_rural_options.keys()))
speed_limit = st.slider(
    "Speed Limit (mph)", 20, 60 if selected_urban=="Urban" else 70, 30 if selected_urban=="Urban" else 40, step=5
)
selected_light = st.selectbox("Light Conditions", list(light_mapping.keys()))
surface_options = ['Dry', 'Wet/Damp', 'Frost/Ice', 'Snow', 'Flood'] if "Darkness" in selected_light else ['Dry', 'Wet/Damp']
selected_surface = st.selectbox("Road Surface Conditions", surface_options)

# --- Build Input DataFrame ---
input_df = pd.DataFrame({
    "Speed_limit": [speed_limit],
    "Urban_or_Rural_Area": [urban_rural_options[selected_urban]],
    "Light_Conditions": [light_mapping[selected_light]],
    "Road_Surface_Conditions": [surface_mapping[selected_surface]],
})

# Feature Engineering
input_df['Speed_Urban_Rural'] = input_df['Urban_or_Rural_Area'] * input_df['Speed_limit']
input_df['Light_Road_Interaction'] = input_df['Light_Conditions'] * input_df['Road_Surface_Conditions']

# Add missing features & ensure order
for col in selected_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[selected_features]

# Convert all to float
input_df = input_df.astype(float)

# --- Prediction ---
try:
    probs = model.predict_proba(input_df)
    pred = np.argmax(probs, axis=1)
    pred_label_raw = le.inverse_transform(pred)[0]

    # --- Smarter Rule-Based Logic ---
    pred_label = "NOT SEVERE"  # default

    # Base prediction from model
    if pred_label_raw in ["High Severity", "3", 3]:
        pred_label = "SEVERE"

    # 1. Daylight, Dry, low speed
    if selected_surface == "Dry" and "Daylight" in selected_light and speed_limit < 40:
        pred_label = "NOT SEVERE"

    # 2. Urban area, low speed
    if selected_urban == "Urban" and speed_limit <= 40:
        pred_label = "NOT SEVERE"

    # 3. Rural area, high speed
    if selected_urban == "Rural" and speed_limit > 60:
        pred_label = "SEVERE"

    # 4. Darkness and slippery surface
    if "Darkness" in selected_light and selected_surface in ["Wet/Damp", "Frost/Ice", "Snow", "Flood"]:
        pred_label = "SEVERE"

except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# --- Display Prediction ---
st.markdown(
    f"""
    <div style='text-align:center; margin-top:40px;'>
        <div style='font-size:32px; font-weight:bold; color:#E74C3C; animation: pulse 1.5s infinite;'>{pred_label}</div>
    </div>
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1); color:#E74C3C; }}
        50% {{ transform: scale(1.25); color:#F1C40F; }}
        100% {{ transform: scale(1); color:#E74C3C; }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)
