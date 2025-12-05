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

# --- Load Resources with Debug ---
@st.cache_resource
def load_model_and_resources():
    expected_files = {
        "lgb_model.pkl": None,
        "target_encoder.pkl": None,
        "selected_features.pkl": None
    }

    for file in expected_files:
        file_path = os.path.join(BASE_PATH, file)
        if not os.path.exists(file_path):
            st.error(f"❌ Missing File: {file} in {BASE_PATH}")
        else:
            try:
                expected_files[file] = joblib.load(file_path)
                st.success(f"✅ Loaded {file}")
            except Exception as e:
                st.error(f"⚠️ Error loading {file}: {e}")

    # تأكد كل الملفات اتحملت
    if None in expected_files.values():
        st.error("❌ Some files could not be loaded. Please check paths and filenames.")
        st.stop()

    return (
        expected_files["lgb_model.pkl"],
        expected_files["target_encoder.pkl"],
        expected_files["selected_features.pkl"]
    )

# --- Load Model ---
model, le, selected_features = load_model_and_resources()

# --- UI Mappings ---
urban_rural_options = {"Urban": 1, "Rural": 2, "Unallocated": 3}

light_mapping = {
    'Daylight: Street light present': 4,
    'Darkness: Street lights present and lit': 3,
    'Darkness: Street lighting unknown': 2,
    'Darkness: Street lights present but unlit': 1,
    'Darkness: No street lighting': 0
}

surface_mapping = {
    'Dry': 4, 'Wet/Damp': 3, 'Frost/Ice': 2, 'Snow': 1, 'Flood (Over 3cm of water)': 0
}

# --- Streamlit UI ---
st.set_page_config(page_title="Accident Severity Prediction", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>Accident Severity Prediction</h1>
    """, unsafe_allow_html=True
)

# Inputs
speed_limit = st.slider("Speed Limit (mph)", 20, 70, 30, step=5)
selected_urban = st.selectbox("Urban or Rural Area", list(urban_rural_options.keys()))
selected_light = st.selectbox("Light Conditions", list(light_mapping.keys()))
selected_surface = st.selectbox("Road Surface Conditions", list(surface_mapping.keys()))

# --- Prepare Input Data ---
# --- Prepare Input Data ---
input_df = pd.DataFrame({
    "Speed_limit": [speed_limit],
    "Urban_or_Rural_Area": [urban_rural_options[selected_urban]],
    "Light_Conditions": [light_mapping[selected_light]],
    "Road_Surface_Conditions": [surface_mapping[selected_surface]],
})

input_df['Speed_Urban_Rural'] = input_df['Urban_or_Rural_Area'] * input_df['Speed_limit']
input_df['Light_Road_Interaction'] = input_df['Light_Conditions'] * input_df['Road_Surface_Conditions']

# --- Align with selected_features ---
for col in selected_features:
    if col not in input_df.columns:
        input_df[col] = 0  # add missing column

input_df = input_df[selected_features]  # re-order to match training


# --- Prediction ---
probs = model.predict_proba(input_df)
pred = np.argmax(probs, axis=1)
pred_label = le.inverse_transform(pred)[0]

# --- Animated Result ---
st.markdown(
    f"""
    <div style='
        text-align:center;
        font-size:30px;
        font-weight:bold;
        color:#4A90E2;
        animation: pulse 1.5s infinite;
    '>{pred_label}</div>
    
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.2); color:#50E3C2; }}
        100% {{ transform: scale(1); }}
    }}
    </style>
    """, unsafe_allow_html=True
)


# %% [code] {"execution":{"iopub.status.busy":"2025-12-05T21:35:43.214397Z","iopub.execute_input":"2025-12-05T21:35:43.214733Z","iopub.status.idle":"2025-12-05T21:35:43.221146Z","shell.execute_reply.started":"2025-12-05T21:35:43.214707Z","shell.execute_reply":"2025-12-05T21:35:43.220219Z"}}
%%writefile requirements.txt
pandas==2.1.1
scikit-learn==1.3.0
lightgbm==4.0.0
catboost==1.2.2
imbalanced-learn==1.2.1
streamlit==1.25.0
joblib==1.3.2


# %% [code]
