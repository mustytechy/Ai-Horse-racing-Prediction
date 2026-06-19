import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import plotly.express as px
import re
import os
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Racing Predictor", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_model_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'models', 'xgb_model.joblib')
    mapping_path = os.path.join(BASE_DIR, 'models', 'categorical_mappings.json')
    stats_path = os.path.join(BASE_DIR, 'models', 'horse_stats.csv')

    # --- DIAGNOSTIC BLOCK: If file is missing, tell us exactly what's wrong ---
    if not os.path.exists(model_path):
        st.error(f"CRITICAL: Could not find model at {model_path}")
        st.write("Files found in the 'models' directory:")
        if os.path.exists(os.path.join(BASE_DIR, 'models')):
            st.write(os.listdir(os.path.join(BASE_DIR, 'models')))
        else:
            st.write(f"The 'models' folder does not exist at {os.path.join(BASE_DIR, 'models')}")
        st.stop()
    # --------------------------------------------------------------------------

    model = joblib.load(model_path)
    
    with open(mapping_path, 'r') as f:
        mappings = json.load(f)
        
    try:
        horse_stats = pd.read_csv(stats_path)
    except FileNotFoundError:
        horse_stats = pd.DataFrame(columns=['horse', 'career_runs', 'win_strike_rate'])
        
    return model, mappings, horse_stats

# --- DATA PROCESSING ---
def process_new_data(df, mappings, horse_stats):
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {'course': 'racecourse', 'ran': 'runners', 'type': 'race type', 'horse name': 'horse', 'name': 'horse', 'race time': 'time'}
    df = df.rename(columns=rename_map)
    
    def clean_name(name):
        name = str(name)
        name = re.sub(r'\([^)]*\)', '', name)
        name = re.sub(r'[^\w\s]', '', name)
        name = ' '.join(name.split()).lower()
        return name
    
    df['horse_clean'] = df['horse'].apply(clean_name)
    horse_stats['horse_clean'] = horse_stats['horse'].apply(clean_name)
    horse_stats = horse_stats.drop_duplicates(subset=['horse_clean'], keep='last')
    
    # Merge logic
    df = pd.merge(df, horse_stats.drop(columns=['horse'], errors='ignore'), on='horse_clean', how='left')
    
    # Feature columns
    features_needed = ['draw', 'runners', 'career_runs', 'win_strike_rate']
    for col in features_needed:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['course_code'] = df['racecourse'].map(mappings['course']).fillna(0)
    df['going_code'] = df['going'].map(mappings['going']).fillna(0)
    return df

# --- UI START ---
st.title("🏇 AI Horse Racing Prediction Hub")

try:
    model, mappings, horse_stats = load_model_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# --- SIDEBAR: DATA INGESTION ---
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Today's Racecard (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    processed_df = process_new_data(raw_df, mappings, horse_stats)
    
    feature_cols = ['course_code', 'going_code', 'draw', 'runners', 'career_runs', 'win_strike_rate']
    processed_df['win_prob'] = (model.predict_proba(processed_df[feature_cols])[:, 1] * 100).round(2)
    
    # Simple display
    st.dataframe(processed_df[['horse', 'win_prob']])
else:
    st.info("Please upload a file.")
