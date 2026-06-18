import os
import json
import re
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st

# Set elegant page layout
st.set_page_config(page_title="AI Horse Racing Predictor", page_icon="🎯", layout="wide")

# Custom CSS to make it look stunning on mobile devices
st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 10px; border: 1px solid #e9ecef; }
    div[data-testid="stExpander"] { border: none; box-shadow: 0px 2px 8px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allowed_html=True)

MODEL_DIR = '/content/drive/MyDrive/HorseRacingModel'

# --- HELPER FUNCTIONS ---
def clean_horse_name(name):
    if pd.isna(name): return ""
    s = str(name).upper().strip()
    s = re.sub(r'\s*\(.*?\)\s*', '', s)
    s = re.sub(r"[^A-Z0-9\s]", "", s)
    return " ".join(s.split())

def intelligent_schema_match(val, target_mapping):
    if pd.isna(val): return -1
    val_clean = str(val).strip().lower().replace('-', ' ')
    if val_clean in target_mapping: return target_mapping[val_clean]
    for key, mapped_int in target_mapping.items():
        key_clean = str(key).strip().lower()
        if key_clean and (key_clean in val_clean or val_clean in key_clean): return mapped_int
    val_tokens = set(re.findall(r'\w+', val_clean))
    for key, mapped_int in target_mapping.items():
        key_tokens = set(re.findall(r'\w+', str(key).strip().lower()))
        if val_tokens.intersection(key_tokens): return mapped_int
    return -1

def match_race_type(val, target_mapping):
    if pd.isna(val): return target_mapping.get('flat', -1)
    v = str(val).lower()
    if 'hurdle' in v: return target_mapping.get('hurdle', -1)
    elif 'chase' in v or 'steeplechase' in v: return target_mapping.get('chase', -1)
    elif 'nh flat' in v or 'bumper' in v: return target_mapping.get('nh flat', -1)
    return target_mapping.get('flat', -1)

# --- CORE DATA PIPELINE (CACHED FOR SPEED) ---
@st.cache_data(ttl=600)
def load_and_process_predictions():
    if not os.path.exists(MODEL_DIR):
        return None, "Model directory not found."
    
    all_files = os.listdir(MODEL_DIR)
    todays_files = [f for f in all_files if '2026-06-18' in f and (f.endswith('.csv') or f.endswith('.xlsx'))]
    if not todays_files:
        return None, "No data file found for today."
        
    excel_files = [f for f in todays_files if f.lower().endswith('.xlsx')]
    live_card_filename = excel_files[0] if excel_files else todays_files[0]
    live_card_path = os.path.join(MODEL_DIR, live_card_filename)
    
    if live_card_filename.lower().endswith('.xlsx'):
        xl = pd.ExcelFile(live_card_path)
        races_sheets = [s for s in xl.sheet_names if 'races' in s.lower()]
        df_live = pd.read_excel(live_card_path, sheet_name=races_sheets[0] if races_sheets else xl.sheet_names[0])
    else:
        df_live = pd.read_csv(live_card_path)

    df_live.columns = df_live.columns.str.strip().str.lower()
    
    # Standardize time
    if pd.api.types.is_datetime64_any_dtype(df_live['time']):
        df_live['time'] = df_live['time'].dt.strftime('%H:%M')
    else:
        df_live['time'] = df_live['time'].astype(str).str.strip()
        df_live['time'] = df_live['time'].apply(lambda x: x.split()[1] if ' ' in x else x)
        df_live['time'] = df_live['time'].apply(lambda x: ':'.join(x.split(':')[:2]) if ':' in x else x)

    # Load Model & Mappings
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, 'xgb_model.json'))
    historical_profiles = pd.read_csv(os.path.join(MODEL_DIR, 'horse_historical_profiles.csv'))
    
    with open(os.path.join(MODEL_DIR, 'categorical_mappings.json'), 'r') as f:
        categorical_encodings = json.load(f)

    course_map = {str(k).lower(): v for k, v in categorical_encodings.get('course', {}).items()}
    going_map = {str(k).lower(): v for k, v in categorical_encodings.get('going', {}).items()}
    type_map = {str(k).lower(): v for k, v in categorical_encodings.get('type', {}).items()}

    df_live['course_code'] = df_live['racecourse'].apply(lambda x: intelligent_schema_match(x, course_map))
    df_live['going_code'] = df_live['going'].apply(lambda x: intelligent_schema_match(x, going_map))
    df_live['type_code'] = df_live['race type'].apply(lambda x: match_race_type(x, type_map))

    df_live['join_name'] = df_live['horse name'].apply(clean_horse_name)
    historical_profiles.columns = historical_profiles.columns.str.strip().str.lower()
    hist_horse_col = [c for c in historical_profiles.columns if 'name' in c or 'horse' in c][0]
    historical_profiles['join_name'] = historical_profiles[hist_horse_col].apply(clean_horse_name)
    historical_profiles = historical_profiles.drop_duplicates(subset=['join_name'])

    df_live = df_live.merge(historical_profiles, on='join_name', how='left', suffixes=('', '_hist'))

    # Feature enforcement
    df_live['draw'] = pd.to_numeric(df_live.get('draw', 0), errors='coerce').fillna(0).astype(int)
    df_live['ran'] = pd.to_numeric(df_live.get('runners', 10), errors='coerce').fillna(10).astype(int)
    df_live['career_runs'] = df_live.get('career_runs', 0).fillna(0)
    df_live['win_strike_rate'] = df_live.get('win_strike_rate', 0.0).fillna(0.0)
    df_live['place_strike_rate'] = df_live.get('place_strike_rate', 0.0).fillna(0.0)
    df_live['avg_beaten_distance'] = df_live.get('avg_beaten_distance', 6.0).fillna(6.0)

    booster = model.get_booster()
    feature_cols = booster.feature_names if booster.feature_names else ['course_code', 'going_code', 'type_code', 'draw', 'ran', 'career_runs', 'win_strike_rate', 'place_strike_rate', 'avg_beaten_distance']
    
    for f in feature_cols:
        if f not in df_live.columns: df_live[f] = 0.0

    df_live['raw_score'] = model.predict_proba(df_live[feature_cols].astype(float))[:, 1]

    processed_races = []
    for (course, r_time), group in df_live.groupby(['racecourse', 'time']):
        g = group.copy()
        sum_scores = g['raw_score'].sum()
        g['Win Probability %'] = (g['raw_score'] / sum_scores * 100) if sum_scores > 0 else (100.0 / len(g))
        processed_races.append(g)
    
    final_df = pd.concat(processed_races)
    final_df['racecourse'] = final_df['racecourse'].astype(str).str.title()
    final_df['horse name'] = final_df['horse name'].astype(str).str.title()
    return final_df, None

# --- APP INTERFACE ---
st.title("🎯 Predictive Racing Analytics Dashboard")
st.caption("Production Engine v4.1 • Optimised for Mobile Lookups")

df, error = load_and_process_predictions()

if error:
    st.error(error)
else:
    # --- INTERACTIVE FILTER AREA ---
    st.write("### 🔍 Select Your Race Card")
    
    # 1. Select Racecourse
    sorted_courses = sorted(df['racecourse'].unique())
    selected_course = st.selectbox("📍 Choose Racecourse:", sorted_courses)
    
    # Filter times based on chosen course
    course_df = df[df['racecourse'] == selected_course]
    sorted_times = sorted(course_df['time'].unique())
    
    # 2. Select Time
    selected_time = st.selectbox("⏰ Choose Race Time:", sorted_times)
    
    # Filter to final unique selection
    race_df = course_df[course_df['time'] == selected_time].sort_values(by='Win Probability %', ascending=False)
    
    # Context data for the selected race
    current_going = race_df['going'].iloc[0].title() if 'going' in race_df.columns else "Unknown"
    current_type = race_df['race type'].iloc[0].title() if 'race type' in race_df.columns else "Flat"
    total_runners = len(race_df)

    # --- VISUAL CARDS SECTION ---
    st.markdown("---")
    st.write(f"### 🏁 {selected_course} ({selected_time}) — *{current_type} / Going: {current_going}*")
    
    # Top 3 visual pods
    top_3 = race_df.head(3)
    cols = st.columns(3)
    
    medals = ["🥇 Top Pick", "🥈 2nd Alternative", "🥉 3rd Alternative"]
    for idx, (_, runner) in enumerate(top_3.iterrows()):
        with cols[idx]:
            st.markdown(f"#### {medals[idx]}")
            st.metric(
                label=runner['horse name'], 
                value=f"{runner['Win Probability %']:.2f}%", 
                delta=f"Draw {runner['draw']}"
            )
            st.caption(f"Historic Win Strike Rate: **{(runner['win_strike_rate']*100):.1f}%**")

    # --- DETAILED COMPARISON CHART ---
    st.write("### 📊 Complete Field Probabilities")
    chart_data = race_df.set_index('horse name')['Win Probability %'].sort_values(ascending=True)
    st.bar_chart(chart_data, horizontal=True, color="#4A90E2")

    # --- RAW DATA GRID ---
    with st.expander("📋 View Full Grid & Auxiliary Metrics"):
        display_grid = race_df[['horse name', 'Win Probability %', 'draw', 'win_strike_rate', 'career_runs', 'avg_beaten_distance']].copy()
        display_grid['Win Probability %'] = display_grid['Win Probability %'].map('{:.2f}%'.format)
        display_grid['win_strike_rate'] = (display_grid['win_strike_rate'] * 100).map('{:.1f}%'.format)
        
        display_grid.columns = ['Runner Name', 'AI Win Prob', 'Stall/Draw', 'Historic Win %', 'Total Career Runs', 'Avg Beaten Distance']
        st.dataframe(display_grid, use_container_width=True, hide_index=True)
        
