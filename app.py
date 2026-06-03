import sys
import os
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
st.sidebar.header("Settings")

# --- 1. SAFE MODEL LOADER ---
@st.cache_resource
def load_model():
    """Checks both root and models directory for the shrunken brain file"""
    root_path = 'race_predictor.pkl'
    folder_path = os.path.join('models', 'race_predictor.pkl')
    
    if os.path.exists(root_path):
        return joblib.load(root_path)
    elif os.path.exists(folder_path):
        return joblib.load(folder_path)
    else:
        # Final fallback
        return joblib.load('models/race_predictor.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the model. Have you run `train_model.py` yet?")
    st.stop()

# ---------------------------------------------------------
# DATA SOURCE LOGIC (FILE UPLOAD ONLY)
# ---------------------------------------------------------
st.sidebar.markdown("### 1. Load Data")

# Initialize session state to hold our dataframe
if 'df' not in st.session_state:
    st.session_state['df'] = None

# File Upload 
uploaded_file = st.sidebar.file_uploader("Upload Daily Race Card", type=["csv", "xlsx", "xls"])
if uploaded_file:
    try:
        # Check if Excel
        if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            st.session_state['df'] = pd.read_excel(uploaded_file)
            st.sidebar.success("✅ Excel loaded successfully!")
        # Otherwise handle as CSV with a robust fallback system
        else:
            try:
                st.session_state['df'] = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Rewind file pointer
                st.session_state['df'] = pd.read_csv(uploaded_file, encoding='latin1')
            st.sidebar.success("✅ CSV loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ File Processing Error: {e}")

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
    
    # Dynamically identifies matching columns if headers vary across files
    COURSE_COL = next((c for c in ['race_course', 'course', 'track', 'venue', 'course_name'] if c in df.columns), 'race_course')
    TIME_COL = next((t for t in ['race_off_time', 'time', 'off_time', 'race_time'] if t in df.columns), 'race_off_time')
    RUNNER_COL = next((r for r in ['runner_horse', 'horse', 'horse_name', 'runner'] if r in df.columns), 'runner_horse')
    
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
    
    display_cols = [RUNNER_COL, 'win_probability']
    if COURSE_COL in df.columns: display_cols.insert(0, COURSE_COL)
    if TIME_COL in df.columns: display_cols.insert(1, TIME_COL)
    
    if not results.empty:
        rename_map = {
            RUNNER_COL: 'Horse',
            'win_probability': 'Win Probability',
            COURSE_COL: 'Course',
            TIME_COL: 'Time'
        }
        display_df = results[display_cols].rename(columns=rename_map)
        display_df['Win Probability'] = (display_df['Win Probability'] * 100).round(1).astype(str) + '%'
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.bar_chart(results.set_index(RUNNER_COL)['win_probability'])
    else:
        st.info("No horses met the confidence threshold for this specific race/course.")

else:
    st.info("👈 Please upload a daily race card file in the sidebar to begin.")
