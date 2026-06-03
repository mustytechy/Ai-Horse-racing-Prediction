import sys
import os
import requests
import streamlit as st
import pandas as pd
import joblib
from utils import engineer_features

# Ensure Python can find the 'scripts' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'scripts'))

st.set_page_config(page_title="Racing AI Dashboard", layout="wide")
st.title("🏇 AI Race Predictor Dashboard")

# --- Helper: Find columns automatically ---
def find_col(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key in col.lower():
                return col
    return None

# --- Helper: Extract name from object objects ---
def extract_name(val):
    if isinstance(val, dict) and 'name' in val:
        return val['name']
    return val

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('models/race_predictor.pkl')

model = load_model()

# --- Data Loading ---
if 'df' not in st.session_state: st.session_state['df'] = None

if st.sidebar.button("📡 Fetch Today's Cards (API)"):
    try:
        response = requests.get("https://api.theracingapi.com/v1/racecards/free", 
                                auth=('PjicO3P5s7worIWnolo5eN6Z', 'NMuYJpB7OJRZIjZg9w2MTos1'))
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                for k in data:
                    if isinstance(data[k], list): data = data[k]; break
            st.session_state['df'] = pd.json_normalize(data)
            st.sidebar.success("Loaded from API")
    except Exception as e: st.sidebar.error(f"Error: {e}")

uploaded_file = st.sidebar.file_uploader("Or Upload CSV", type="csv")
if uploaded_file: st.session_state['df'] = pd.read_csv(uploaded_file)

# --- Process Data ---
if st.session_state['df'] is not None:
    df = st.session_state['df'].copy()
    
    # Identify key columns automatically
    HORSE_COL = find_col(df, ['horse', 'runner']) or 'runner_horse'
    COURSE_COL = find_col(df, ['course', 'track', 'meeting']) or 'race_course'
    TIME_COL = find_col(df, ['time', 'off']) or 'race_off_time'

    # Apply the name extraction fix to the horse column
    if HORSE_COL in df.columns:
        df[HORSE_COL] = df[HORSE_COL].apply(extract_name)

    # Engineering and Prediction
    X = engineer_features(df)
    for feat in ['or', 'wgt', 'jockey_win_rate', 'days_since_run']:
        if feat not in X.columns: X[feat] = 0
        
    df['win_probability'] = model.predict_proba(X[['or', 'wgt', 'jockey_win_rate', 'days_since_run']])[:, 1]
    
    # --- UI Filters ---
    st.sidebar.header("Filters")
    threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.55)
    
    selected_course = st.sidebar.selectbox("Course", ["All"] + sorted(df[COURSE_COL].unique().tolist()))
    
    results = df[df['win_probability'] >= threshold]
    if selected_course != "All": results = results[results[COURSE_COL] == selected_course]
    
    # --- Display ---
    rename_map = {HORSE_COL: 'Horse', 'win_probability': 'Win Probability', COURSE_COL: 'Course', TIME_COL: 'Time'}
    display_df = results[[HORSE_COL, 'win_probability', COURSE_COL, TIME_COL]].rename(columns=rename_map)
    display_df['Win Probability'] = (display_df['Win Probability'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.bar_chart(results.set_index(HORSE_COL)['win_probability'])
