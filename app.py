import sys
import os

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

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload Daily Racecard (CSV)", type="csv")

# Threshold Slider
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.55)

if uploaded_file:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # Process and Predict for the whole file
    X = engineer_features(df)
    features = ['or', 'wgt', 'jockey_win_rate', 'days_since_run']
    df['win_probability'] = model.predict_proba(X[features])[:, 1]
    
    # ---------------------------------------------------------
    # DYNAMIC FILTERING LOGIC
    # ---------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("Race Filters")
    
    # Updated to your exact CSV column names
    COURSE_COL = 'race_course'  
    TIME_COL = 'race_off_time'      
    
    # 1. Course Filter
    if COURSE_COL in df.columns:
        courses = ["All Courses"] + sorted(df[COURSE_COL].dropna().unique().tolist())
        selected_course = st.sidebar.selectbox("Select Course", courses)
    else:
        selected_course = "All Courses"
        st.sidebar.warning(f"Column '{COURSE_COL}' not found in CSV. Filters disabled.")

    # 2. Time Filter (Dynamically updates based on selected course)
    if TIME_COL in df.columns:
        if selected_course != "All Courses":
            available_times = sorted(df[df[COURSE_COL] == selected_course][TIME_COL].dropna().unique().tolist())
        else:
            available_times = sorted(df[TIME_COL].dropna().unique().tolist())
            
        times = ["All Times"] + available_times
        selected_time = st.sidebar.selectbox("Select Time", times)
    else:
        selected_time = "All Times"

    # ---------------------------------------------------------
    # APPLY FILTERS & DISPLAY RESULTS
    # ---------------------------------------------------------
    
    # Filter by Threshold first
    results = df[df['win_probability'] >= threshold]
    
    # Apply Course and Time filters
    if selected_course != "All Courses":
        results = results[results[COURSE_COL] == selected_course]
    if selected_time != "All Times":
        results = results[results[TIME_COL] == selected_time]
        
    # Sort the highest probability to the top
    results = results.sort_values('win_probability', ascending=False)
    
    # Header update based on selection
    display_title = f"Predictions ({selected_course} at {selected_time})" 
    if selected_course == "All Courses" and selected_time == "All Times":
        display_title = "Overall Top Picks"
        
    st.write(f"### {display_title} (Confidence > {threshold:.2f})")
    
    # Build the display columns dynamically
    display_cols = ['runner_horse', 'win_probability']
    if COURSE_COL in df.columns: display_cols.insert(0, COURSE_COL)
    if TIME_COL in df.columns: display_cols.insert(1, TIME_COL)
    
    if not results.empty:
        # --- NEW CLEAN FORMATTING ---
        # Map the old ugly names to clean, capitalized names
        rename_map = {
            'runner_horse': 'Horse',
            'win_probability': 'Win Probability',
            COURSE_COL: 'Course',
            TIME_COL: 'Time'
        }
        
        # Create a display-only dataframe with the new names
        display_df = results[display_cols].rename(columns=rename_map)
        
        # Format probability as a clean percentage
        display_df['Win Probability'] = (display_df['Win Probability'] * 100).round(1).astype(str) + '%'
        
        # Show the clean dataframe
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Keep the bar chart working with the original numeric data
        st.bar_chart(results.set_index('runner_horse')['win_probability'])
    else:
        st.info("No horses met the confidence threshold for this specific race/course.")

else:
    st.info("Please upload a racecard CSV file in the sidebar to begin.")