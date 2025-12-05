import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import time

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
road_type_options = {"Single Carriageway": 1, "Dual Carriageway": 2, "Roundabout": 3, "One Way Street": 4, "Slip Road": 5}
day_name_map = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}

# --- UI ---
st.set_page_config(page_title="Accident Severity Prediction", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Accident Severity Prediction</h1>", unsafe_allow_html=True)

# --- Inputs ---
st.header("Core Details")

# Row 1: Urban/Rural, Speed Limit, Light, Surface
col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_urban = st.selectbox("Urban or Rural Area", list(urban_rural_options.keys()))

with col2:
    speed_limit = st.slider(
        "Speed Limit (mph)", 20, 60 if selected_urban=="Urban" else 70, 30 if selected_urban=="Urban" else 40, step=5
    )
    
with col3:
    selected_light = st.selectbox("Light Conditions", list(light_mapping.keys()))

with col4:
    surface_options = ['Dry', 'Wet/Damp', 'Frost/Ice', 'Snow', 'Flood'] if "Darkness" in selected_light else ['Dry', 'Wet/Damp']
    selected_surface = st.selectbox("Road Surface Conditions", surface_options)

st.header("Contextual Details")

# Row 2: Police, Time/Hour, Road Type, 2nd Road Class, Day of Week
col5, col6, col7, col8, col9 = st.columns(5)

with col5:
    police_attend = st.selectbox(
        "Police Attended?", 
        options=["Yes", "No"], 
        format_func=lambda x: "Yes" if x == "Yes" else "No"
    )
    police_attend_value = 1 if police_attend == "Yes" else 0

with col6:
    accident_time_input = st.time_input("Time of Accident", value=time(12, 0))
    accident_hour = accident_time_input.hour

with col7:
    selected_road_type = st.selectbox("Road Type", list(road_type_options.keys()))

with col8:
    second_road_class = st.slider("2nd Road Class", 1, 6, 4, step=1)

with col9:
    day_of_week = st.selectbox("Day of Week", list(day_name_map.keys()), index=4, format_func=lambda x: day_name_map[x])


# --- Build ALL Features Dictionary ---
all_features_data = {
    "Speed_limit": [speed_limit],
    "Urban_or_Rural_Area": [urban_rural_options[selected_urban]],
    "Light_Conditions": [light_mapping[selected_light]],
    "Road_Surface_Conditions": [surface_mapping[selected_surface]], 
    "Did_Police_Officer_Attend_Scene_of_Accident": [police_attend_value],
    "Accident_Hour": [accident_hour],
    "2nd_Road_Class": [second_road_class],
    "Road_Type": [road_type_options[selected_road_type]],
    "Day_of_Week": [day_of_week] 
}

# --- Feature Engineering ---
all_features_data['Speed_Urban_Rural'] = all_features_data['Urban_or_Rural_Area'][0] * all_features_data['Speed_limit'][0]
all_features_data['Light_Road_Interaction'] = all_features_data['Light_Conditions'][0] * all_features_data['Road_Surface_Conditions'][0]

# --- Final Data Preparation: CRITICAL STEP ---
# 1. Create DataFrame from dictionary
input_df = pd.DataFrame(all_features_data, index=[0]) # Explicit index=0 is safer

# 2. Ensure all expected features are present (set 0 for missing ones)
for col in selected_features:
    if col not in input_df.columns:
        input_df[col] = 0

# 3. CRITICAL: Reorder and select features based on selected_features
input_df_final = input_df[selected_features]

# 4. Convert all columns to float
input_df_final = input_df_final.astype(float)


# --- Prediction ---
if st.button("Predict Severity"):
    try:
        input_np = input_df_final.to_numpy()
        
        if input_np.shape[1] != len(selected_features):
             raise ValueError(f"Shape mismatch: Expected {len(selected_features)} features, got {input_np.shape[1]}")
        
        probs = model.predict_proba(input_np)
        pred = np.argmax(probs, axis=1)
        pred_label_raw = le.inverse_transform(pred)[0]

        # --- Rule-based adjustment ---
        pred_label = pred_label_raw
        
        if selected_surface == "Dry" and "Daylight" in selected_light and speed_limit <= 40:
            pred_label = "NOT SEVERE"
        if selected_urban == "Urban" and speed_limit <= 40:
            pred_label = "NOT SEVERE"
        if selected_surface in ["Frost/Ice", "Snow", "Flood"] or speed_limit > 60:
            pred_label = "SEVERE"

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

    except Exception as e:
        st.error(f"Prediction failed! Error details: {e}")
