import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import plotly.express as px
import re
import os
import joblib  # Added for loading the .joblib file

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Racing Predictor", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_model_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # LOAD THE JOBLIB MODEL
    model_path = os.path.join(BASE_DIR, 'models', 'xgb_model.joblib')
    model = joblib.load(model_path)
    
    with open(os.path.join(BASE_DIR, 'models', 'categorical_mappings.json'), 'r') as f:
        mappings = json.load(f)
        
    try:
        horse_stats = pd.read_csv(os.path.join(BASE_DIR, 'models', 'horse_stats.csv'))
    except FileNotFoundError:
        horse_stats = pd.DataFrame(columns=['horse', 'career_runs', 'win_strike_rate'])
        
    return model, mappings, horse_stats

# ... [Keep your process_new_data function exactly as it was] ...
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
    if 'horse' not in df.columns: df['horse'] = "Unknown"
    if 'time' not in df.columns: df['time'] = "TBA"
    for col in ['career_runs', 'win_strike_rate']:
        if col in df.columns: df = df.drop(columns=[col])
    df = pd.merge(df, horse_stats.drop(columns=['horse']), on='horse_clean', how='left')
    features_needed = ['draw', 'runners', 'career_runs', 'win_strike_rate']
    for col in features_needed:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['win_strike_rate'] = df['win_strike_rate'].round(2)
    df['course_code'] = df['racecourse'].map(mappings['course']).fillna(0)
    df['going_code'] = df['going'].map(mappings['going']).fillna(0)
    df = df.drop(columns=['horse_clean'])
    return df

# --- UI START ---
st.title("🏇 AI Horse Racing Prediction Hub")
model, mappings, horse_stats = load_model_assets()

# --- SIDEBAR: DATA INGESTION ---
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Today's Racecard (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'): raw_df = pd.read_csv(uploaded_file)
    else: raw_df = pd.read_excel(uploaded_file)
    processed_df = process_new_data(raw_df, mappings, horse_stats)
    processed_df = processed_df.drop_duplicates(subset=['horse', 'time'])
    feature_cols = ['course_code', 'going_code', 'draw', 'runners', 'career_runs', 'win_strike_rate']
    processed_df['win_prob'] = (model.predict_proba(processed_df[feature_cols])[:, 1] * 100).round(2)
    processed_df['racecourse'] = processed_df['racecourse'].astype(str).str.title()
    processed_df['time'] = processed_df['time'].astype(str)
    processed_df['race_display'] = processed_df['racecourse'] + " - " + processed_df['time']
    
    st.markdown("### 🏆 Top Predictions for Today")
    top_picks = processed_df.sort_values('win_prob', ascending=False).head(5)
    cols = st.columns(len(top_picks))
    for idx in range(len(top_picks)):
        row = top_picks.iloc[idx]
        cols[idx].metric(label=row.get('horse', 'Unknown'), value=f"{row.get('win_prob', 0)}%", delta_color="off")

    st.divider()
    selected_race = st.selectbox("Select a Race to Visualize", sorted(processed_df['race_display'].unique()))
    race_subset = processed_df[processed_df['race_display'] == selected_race].sort_values('win_prob', ascending=True)
    st.plotly_chart(px.bar(race_subset, x='win_prob', y='horse', orientation='h', title=f"Win Probabilities: {selected_race}", text='win_prob'), use_container_width=True)
