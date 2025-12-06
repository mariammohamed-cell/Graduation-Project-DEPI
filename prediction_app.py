import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Path Configuration ---
def is_kaggle_environment():
    return os.path.exists("/kaggle/input")

BASE_PATH = "/kaggle/input/pkl-files/" if is_kaggle_environment() else ""

REQUIRED_LGB_FEATURES = [
    'Did_Police_Officer_Attend_Scene_of_Accident',
    'Speed_Urban_Rural',
    'Speed_limit',
    'Urban_or_Rural_Area',
    'Light_Conditions',
    'Accident_Hour',
    '2nd_Road_Class',
    'Light_Road_Interaction',
    'Road_Type',
    'Day_of_Week'
]

# --- Load Resources ---
@st.cache_resource
def load_model_and_resources():
    try:
        model = joblib.load(os.path.join(BASE_PATH, "lgb_model.pkl"))
        le = joblib.load(os.path.join(BASE_PATH, "target_encoder.pkl"))
        return model, le
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

model, le = load_model_and_resources()

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
    "Did_Police_Officer_Attend_Scene_of_Accident": [0],
    "Accident_Hour": [12],  
    "2nd_Road_Class": [0],  
    "Road_Type": [6],      
    "Day_of_Week": [4]      
})

# Feature Engineering
input_df['Speed_Urban_Rural'] = input_df['Urban_or_Rural_Area'] * input_df['Speed_limit']
input_df['Light_Road_Interaction'] = input_df['Light_Conditions'] * input_df['Road_Surface_Conditions']

# --- Alignment and Ordering ---
input_df_final = input_df[REQUIRED_LGB_FEATURES].astype(float)

# --- Prediction ---
try:
    probs = model.predict_proba(input_df_final)  # [ [prob_NOT_SEVERE, prob_SEVERE] ]
    pred_index = np.argmax(probs, axis=1)[0]
    prob_NOT_SEVERE, prob_SEVERE = probs[0]
except Exception as e:
    st.error(f"Prediction failed! Error: {e}")
    st.stop()

# --- Rule-based adjustment ---
pred_label = "NOT SEVERE"

# 1. Highest Priority Rule: SEVERE conditions
if selected_surface in ["Frost/Ice", "Snow", "Flood"] or speed_limit > 60:
    pred_label = "SEVERE"
# 2. Safety Rule 1: Dry, Daylight, Slow
elif selected_surface == "Dry" and "Daylight" in selected_light and speed_limit <= 40:
    pred_label = "NOT SEVERE"
# 3. Safety Rule 2: Urban and Slow
elif selected_urban == "Urban" and speed_limit <= 40:
    pred_label = "NOT SEVERE"
# 4. Default based on model
else:
    pred_label = "SEVERE" if pred_index == 1 else "NOT SEVERE"

# --- Display Prediction with Probabilities ---
st.markdown(
    f"""
    <div style='text-align:center; margin-top:40px;'>
        <div style='font-size:32px; font-weight:bold; color:#E74C3C; animation: pulse 1.5s infinite;'>{pred_label}</div>
        <div style='font-size:18px; margin-top:10px;'>
            Probability SEVERE: {prob_SEVERE*100:.1f}% <br>
            Probability NOT SEVERE: {prob_NOT_SEVERE*100:.1f}%
        </div>
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
