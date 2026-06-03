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

# Configure Page
st.set_page_config(page_title="Racing AI Dashboard", layout="wide")
st.title("🏇 AI Race Predictor Dashboard")

# 1. Load Model
@st.cache_resource
def load_model():
    return joblib.load('race_predictor.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Define Required Features
# IMPORTANT: These MUST match the headers in your input file exactly.
REQUIRED_FEATURES = ['or', 'wgt', 'jockey_win_rate', 'days_since_run']

# 3. File Upload
st.sidebar.markdown("### 1. Load Data")
uploaded_file = st.sidebar.file_uploader("Upload Daily Race Card (CSV/XLSX)", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Load File
    try:
        if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.success("✅ File loaded successfully!")
        
        # 4. Data Validation (The "Make it Right" step)
        # Check if all required columns exist in the file
        missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Input file missing columns: {missing_cols}")
            st.info(f"Please ensure your file has these exact headers: {REQUIRED_FEATURES}")
            st.stop() # Stop execution until headers are fixed
            
        # 5. Prediction
        st.sidebar.markdown("### 2. Run Prediction")
        threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.55)
        
        if st.button("Generate Predictions"):
            # Engineer features using the validated dataframe
            X = engineer_features(df)
            
            # Predict
            df['win_probability'] = model.predict_proba(X[REQUIRED_FEATURES])[:, 1]
            
            # 6. Display Results
            results = df[df['win_probability'] >= threshold].sort_values('win_probability', ascending=False)
            
            st.write(f"### Top Picks (Confidence > {threshold:.2f})")
            
            if not results.empty:
                # Assuming you have columns like 'runner_horse' and 'race_course'
                display_cols = [c for c in ['runner_horse', 'race_course', 'race_off_time', 'win_probability'] if c in results.columns]
                results['win_probability'] = (results['win_probability'] * 100).round(1).astype(str) + '%'
                st.dataframe(results[display_cols], use_container_width=True)
                st.bar_chart(results.set_index('runner_horse')['win_probability'].str.replace('%', '').astype(float))
            else:
                st.info("No horses met the confidence threshold.")
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("👈 Please upload a CSV or Excel file to get started.")
