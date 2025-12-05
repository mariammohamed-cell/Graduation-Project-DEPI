import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Path Configuration ---
def is_kaggle_environment():
    return os.path.exists("/kaggle/input")

if is_kaggle_environment():
    BASE_PATH = "/kaggle/input/pkl-files/"
else:
    BASE_PATH = ""  # Streamlit Cloud or local

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

# --- Mappings ---
urban_rural_options = {"Urban": 1, "Rural": 2, "Unallocated": 3}

light_mapping = {
    'Daylight: Street light present': 4,
    'Darkness: Street lights present and lit': 3,
    'Darkness: Street lighting unknown': 2,
    'Darkness: Street lights present but unlit': 1,
    'Darkness: No street lighting': 0
}

surface_mapping = {
    'Dry': 4,
    'Wet/Damp': 3,
    'Frost/Ice': 2,
    'Snow': 1,
    'Flood': 0
}

# --- Streamlit UI ---
st.set_page_config(page_title="Accident Severity Prediction", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>Accident Severity Prediction</h1>",
    unsafe_allow_html=True
)

# Inputs
speed_limit = st.slider("Speed Limit (mph)", 20, 70, 30, step=5)
selected_urban = st.selectbox("Urban or Rural Area", list(urban_rural_options.keys()))
selected_light = st.selectbox("Light Conditions", list(light_mapping.keys()))
selected_surface = st.selectbox("Road Surface Conditions", list(surface_mapping.keys()))

# --- Prepare Input Data ---
input_df = pd.DataFrame({
    "Speed_limit": [speed_limit],
    "Urban_or_Rural_Area": [urban_rural_options[selected_urban]],
    "Light_Conditions": [light_mapping[selected_light]],
    "Road_Surface_Conditions": [surface_mapping[selected_surface]],
})

# Feature engineering
input_df['Speed_Urban_Rural'] = input_df['Urban_or_Rural_Area'] * input_df['Speed_limit']
input_df['Light_Road_Interaction'] = (
    input_df['Light_Conditions'] * input_df['Road_Surface_Conditions']
)

# Add missing selected features
for col in selected_features:
    if col not in input_df.columns:
        input_df[col] = 0

# --- Prediction ---
probs = model.predict_proba(input_df)
pred = np.argmax(probs, axis=1)
pred_label = le.inverse_transform(pred)[0]

# --- Display Animated Prediction ---
st.markdown(
    f"""
    <div style='text-align:center; margin-top:40px;'>
        <div style='
            font-size:32px;
            font-weight:bold;
            color:#4A90E2;
            animation: pulse 1.5s infinite;
        '>{pred_label}</div>
    </div>

    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1); color:#4A90E2; }}
        50% {{ transform: scale(1.25); color:#50E3C2; }}
        100% {{ transform: scale(1); color:#4A90E2; }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)
