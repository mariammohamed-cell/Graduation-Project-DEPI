import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("/kaggle/input/accident-prediction-model/lgb_model.pkl")
thresholds = joblib.load("/kaggle/input/accident-prediction-model/thresholds.pkl")
scaler = joblib.load("/kaggle/input/accident-prediction-model/scaler.pkl")
le = joblib.load("/kaggle/input/accident-prediction-model/label_encoder.pkl")
freq_maps = joblib.load("/kaggle/input/accident-prediction-model/freq_maps.pkl")

st.title("Accident Severity Prediction")

speed_limit = st.number_input("Speed Limit", min_value=0, max_value=100, value=30)
urban_rural = st.selectbox("Urban or Rural Area", [1, 2, 3])
light_condition = st.selectbox("Light Conditions", [
    'Daylight: Street light present',
    'Darkness: Street lights present and lit',
    'Darkness: Street lighting unknown',
    'Darkness: Street lights present but unlit',
    'Darkness: No street lighting'
])
road_surface = st.selectbox("Road Surface Conditions", [
    'Dry', 'Wet/Damp', 'Frost/Ice', 'Snow', 'Flood (Over 3cm of water)'
])

input_df = pd.DataFrame({
    "Speed_limit": [speed_limit],
    "Urban_or_Rural_Area": [urban_rural],
    "Light_Conditions": [light_condition],
    "Road_Surface_Conditions": [road_surface],
})

light_mapping = {
    'Daylight: Street light present': 4,
    'Darkness: Street lights present and lit': 3,
    'Darkness: Street lighting unknown': 2,
    'Darkness: Street lights present but unlit': 1,
    'Darkness: No street lighting': 0, 
}
surface_mapping = {
    'Dry': 4, 'Wet/Damp': 3, 'Frost/Ice': 2, 'Snow': 1, 'Flood (Over 3cm of water)': 0,
}

input_df['Light_Conditions'] = input_df['Light_Conditions'].map(light_mapping).fillna(-1)
input_df['Road_Surface_Conditions'] = input_df['Road_Surface_Conditions'].map(surface_mapping).fillna(-1)

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
