import sys
import os
import requests
import io

# Ensure Python can find the 'scripts' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'scripts'))

import streamlit as st
import pandas as pd
import joblib
from utils import engineer_features

st.set_page_config(page_title="Racing AI Dashboard", layout="wide")

st.title("🏇 AI Race Predictor Dashboard")
st.sidebar.header("Settings")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('models/race_predictor.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the model. Have you run `train_model.py` yet?")
    st.stop()

# ---------------------------------------------------------
# DATA SOURCE LOGIC (API + FALLBACK)
# ---------------------------------------------------------
st.sidebar.markdown("### 1. Load Data")

# Initialize session state to hold our dataframe
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Option A: API Fetch
if st.sidebar.button("📡 Fetch Today's Cards (API)"):
    with st.spinner("Connecting to The Racing API..."):
        try:
            # Replace with the exact endpoint URL from your API documentation
            # Often, APIs allow you to request CSV format by adding ?format=csv to the URL
            api_url = "https://api.theracingapi.com/v1/racecards/pro?format=csv" 
            
            response = requests.get(
                api_url, 
                auth=('PjicO3P5s7worIWnolo5eN6Z', 'NMuYJpB7OJRZIjZg9w2MTos1')
            )
            
            if response.status_code == 200:
                # Read the response text as a CSV into pandas
                st.session_state['df'] = pd.read_csv(io.StringIO(response.text))
                st.sidebar.success("✅ API Data loaded successfully!")
            else:
                st.sidebar.error(f"❌ API Failed: Error {response.status_code}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Connection Error: {e}")

# Option B: File Upload (Fallback)
uploaded_file = st.sidebar.file_uploader("Or Upload CSV (Fallback)", type="csv")
if uploaded_file:
    st.session_state['df'] = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ CSV loaded successfully!")

# ---------------------------------------------------------
# PREDICTION & FILTERING (Only runs if data is loaded)
# ---------------------------------------------------------
if st.session_state['df'] is not None:
    df = st.session_state['df'].copy()
    
    st.sidebar.markdown("### 2. Predict")
    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.55)
    
    # Process and Predict for the whole file
    X = engineer_features(df)
    features = ['or', 'wgt', 'jockey_win_rate', 'days_since_run']
    df['win_probability'] = model.predict_proba(X[features])[:, 1]
    
    # Race Filters
    st.sidebar.markdown("---")
    st.sidebar.header("Race Filters")
    
    COURSE_COL = 'race_course'  
    TIME_COL = 'race_off_time'      
    
    # Course Filter
    if COURSE_COL in df.columns:
        courses = ["All Courses"] + sorted(df[COURSE_COL].dropna().unique().tolist())
        selected_course = st.sidebar.selectbox("Select Course", courses)
    else:
        selected_course = "All Courses"
        st.sidebar.warning(f"Column '{COURSE_COL}' not found. Filters disabled.")

    # Time Filter
    if TIME_COL in df.columns:
        if selected_course != "All Courses":
            available_times = sorted(df[df[COURSE_COL] == selected_course][TIME_COL].dropna().unique().tolist())
        else:
            available_times = sorted(df[TIME_COL].dropna().unique().tolist())
            
        times = ["All Times"] + available_times
        selected_time = st.sidebar.selectbox("Select Time", times)
    else:
        selected_time = "All Times"

    # Filter Results
    results = df[df['win_probability'] >= threshold]
    if selected_course != "All Courses":
        results = results[results[COURSE_COL] == selected_course]
    if selected_time != "All Times":
        results = results[results[TIME_COL] == selected_time]
        
    results = results.sort_values('win_probability', ascending=False)
    
    display_title = f"Predictions ({selected_course} at {selected_time})" 
    if selected_course == "All Courses" and selected_time == "All Times":
        display_title = "Overall Top Picks"
        
    st.write(f"### {display_title} (Confidence > {threshold:.2f})")
    
    display_cols = ['runner_horse', 'win_probability']
    if COURSE_COL in df.columns: display_cols.insert(0, COURSE_COL)
    if TIME_COL in df.columns: display_cols.insert(1, TIME_COL)
    
    if not results.empty:
        rename_map = {
            'runner_horse': 'Horse',
            'win_probability': 'Win Probability',
            COURSE_COL: 'Course',
            TIME_COL: 'Time'
        }
        display_df = results[display_cols].rename(columns=rename_map)
        display_df['Win Probability'] = (display_df['Win Probability'] * 100).round(1).astype(str) + '%'
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.bar_chart(results.set_index('runner_horse')['win_probability'])
    else:
        st.info("No horses met the confidence threshold for this specific race/course.")

else:
    st.info("👈 Please click 'Fetch API' or upload a CSV file in the sidebar to begin.")
