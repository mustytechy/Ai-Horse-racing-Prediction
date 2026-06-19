import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import plotly.express as px
import re
import os  # <-- ADD THIS NEW IMPORT

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Racing Predictor", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_model_assets():
    # Find the exact absolute path of the folder this script is sitting in
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    model = xgb.XGBClassifier()
    # Safely join the paths together
    model.load_model(os.path.join(BASE_DIR, 'models', 'xgb_model.json'))
    
    with open(os.path.join(BASE_DIR, 'models', 'categorical_mappings.json'), 'r') as f:
        mappings = json.load(f)
        
    # Load the new Horse Encyclopedia
    try:
        horse_stats = pd.read_csv(os.path.join(BASE_DIR, 'models', 'horse_stats.csv'))
    except FileNotFoundError:
        horse_stats = pd.DataFrame(columns=['horse', 'career_runs', 'win_strike_rate'])
        
    return model, mappings, horse_stats

# ... (the rest of your script remains exactly the same)
